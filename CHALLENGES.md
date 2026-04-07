# Drift-LLM：核心挑战与设计方案

## 背景

图像 Drifting 在 ImageNet 上成功的三大支柱：
1. **连续输出** — Generator 输出连续像素/latent，天然可微
2. **好的 kernel 空间** — MAE 特征空间里，语义相似的图片距离近
3. **充足的正样本** — ImageNet 每类有上千张图，Memory Bank 按类提供 ~32 张同类正样本

把 Drifting 搬到 Language，这三根支柱**全部断裂**。以下逐一分析并提出设计方案。

---

## 问题 1：离散性 — 梯度链路断裂

### 问题描述

图像 Drifting 的梯度链路：
```
Generator → 连续像素 → 冻结 MAE → 特征 → kernel → drift loss
                                                        ↓ 梯度
全程连续，梯度畅通无阻 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

Language 的梯度链路：
```
Generator → logits → argmax → 离散 token → Feature Model → 特征 → kernel → drift loss
                       ↑
                  梯度 = 0，链路断裂
```

argmax 不可微，drift loss 的梯度无法回传到 Generator。

### 可能的方案

#### 方案 A：Gumbel-Softmax 桥接（IDEA.md 当前方案）

```
Generator → logits → Gumbel-Softmax(τ) → soft_probs → @ Embedding → soft_embedding → Feature Model → drift loss
```

- **优点**：端到端可微，概念简单
- **风险**：
  - Feature Model 训练时只见过 hard embedding（真实 token），从未见过"所有 token 的加权平均"
  - τ 两难：τ 大 → 输入模糊，特征质量差；τ 小 → 接近 one-hot，梯度消失
  - 计算量大：每个候选需要 Feature Model 的完整 forward + backward
- **缓解措施**：
  - τ 退火（训练初期 τ 大探索，后期 τ 小精确）
  - 可以先在小规模实验中验证 Feature Model 对 soft embedding 输入的鲁棒性

#### 方案 B：REINFORCE / Score Function Estimator

```
Generator → logits → 采样离散 token → Feature Model → drift loss
                                                         ↓
                                            用 REINFORCE 估计梯度
```

```python
# 伪代码
sampled_tokens = Categorical(logits).sample()       # 离散采样，不可微
features = feature_model(sampled_tokens)             # 正常输入
loss = drift_loss(features, pos_features)
log_prob = Categorical(logits).log_prob(sampled_tokens).sum()
reinforce_loss = (loss.detach() * log_prob).mean()   # REINFORCE
reinforce_loss.backward()                            # 梯度回传到 logits
```

- **优点**：Feature Model 拿到的是真实 token，特征质量有保障
- **风险**：方差大，训练不稳定，需要 baseline 减方差
- **缓解措施**：多候选 (G 个) 天然提供 baseline，可以用候选间平均 loss 作为 baseline

#### 方案 C：Straight-Through Estimator (STE)

```
前向：argmax(logits) → 离散 token → Feature Model → drift loss
反向：假装 argmax 是 identity，梯度直接穿过
```

- **优点**：前向时 Feature Model 看到的是真实 token（和训练分布一致）
- **风险**：梯度是有偏的近似，理论上不保证收敛
- **实际**：在很多离散 VAE 和 VQ 工作中被证明 work

#### 方案 D：完全不用外部 Feature Model — 直接在 logits/embedding 空间做 kernel

```
Generator → logits → softmax → prob_vector [B, T, V]
GT →                           one_hot     [B, T, V]

kernel 直接在 prob_vector 空间上计算，无需 Feature Model
```

- **优点**：没有离散瓶颈，没有额外模型，计算轻量
- **风险**：R^V 空间里任意两个 one-hot 的 L2 距离都是 √2，没有语义结构
- **改进**：投影到 embedding 空间 — `soft_emb = softmax(logits) @ E`，在 R^D 空间做 kernel
  - embedding 空间有一定语义结构（相似词距离近）
  - 但这个结构是 token 级别的，不是序列级别的

### 推荐策略（基于文献更新）

**方案 A（Gumbel-Softmax）已有文献支撑（SofT-GRPO, GRADE），应作为主方案。**

实验递进：
1. **Level 1**：方案 D（embedding 空间 kernel）— 最快验证 drift loss 信号
2. **Level 2**：方案 A（Gumbel-Softmax + frozen Feature LLM）— 文献证明可行，提供上下文感知特征
3. **Level 3**：方案 C（STE + sentence embedder）— 备选，GRADE 证明 STE 梯度方差低

---

## 问题 2：Kernel 空间 — 什么距离对语言有意义？

### 问题描述

图像 Drifting 的 kernel：
```
k(x, y) = exp(-||MAE(x) - MAE(y)|| / τ)
```
MAE 特征空间里，两只"差不多的金鱼"距离近，"金鱼"和"卡车"距离远。
**L2 距离有清晰的语义含义。**

语言 kernel 的困难：
- Token 空间：离散，无距离概念
- One-hot 空间：任意两个不同 token 距离相同 (√2)，无语义梯度
- LLM Hidden state 空间：为 next-token prediction 训练，不是为语义相似度训练
- 序列级别：不同长度、不同表述的序列怎么比？

### 可能的方案

#### 方案 A：Token Embedding 空间（最轻量）

```
特征 = softmax(logits) @ Embedding_matrix    # [B, T, D]
kernel 在每个 token position 独立计算
```

- **语义结构**：embedding 空间有基本的词级语义（"king" 和 "queen" 距离近）
- **局限**：没有上下文信息，position 5 的 embedding 不知道 position 1-4 是什么
- **适用场景**：作为 baseline，最小验证

#### 方案 B：Sentence Embedding 模型（专为语义距离训练）

用专门训练过语义相似度的模型（如 E5、GTE、BGE）做特征提取：
```
特征 = SentenceEmbedder(token_sequence)    # [B, D_sent]
kernel 在序列级别计算
```

- **优点**：这类模型的训练目标就是让语义相似的文本在特征空间里距离近，和 MAE 在图像 Drifting 中的角色完美对应
- **风险**：需要解决离散瓶颈（问题 1），因为这些模型期望离散 token 输入
- **计算量**：sentence embedding 模型通常较小（~300M），可接受

#### 方案 C：Reward Model 特征空间

```
特征 = RewardModel.get_features(token_sequence)    # [B, D_rm]
```

- **优点**：RM 的特征空间编码了"回答质量"的梯度，这可能是最有意义的距离
- **风险**：同样需要解决离散瓶颈；RM 特征可能过于关注 preference 而忽略多样性
- **有趣方向**：drift loss 中 RM 特征作为一个尺度，token embedding 作为另一个尺度（多尺度）

#### 方案 D：自训练 Feature Encoder（类比图像 Drifting 自训练 MAE）

图像 Drifting 论文 Table 3 表明，**自训练的 MAE 效果最好**（优于现成的 SimCLR、MoCo）。
类比 language：可以训练一个专门的 encoder 来做 drift loss 的特征提取。

```
训练目标：在 SFT 数据上做 masked language modeling（或对比学习）
→ 得到一个特征空间，其中"好回答"的特征紧凑，"坏回答"的特征远离
```

- **优点**：feature space 和训练任务完全对齐
- **风险**：额外训练开销；需要先验证思路再投入
- **时机**：方案 A/B 跑通后再考虑

### 推荐策略

**多尺度组合**（图像 Drifting 在 MAE 不同层上做多尺度 drift loss，语言版同理）：

```python
# 尺度 1：Token embedding 空间（局部、词级）
feat_emb = softmax(logits) @ E                    # [B, T, D]

# 尺度 2：Sentence embedding（全局、语义级）— 如果能解决离散瓶颈
feat_sent = sentence_embedder(generated_tokens)    # [B, D_sent]

# 每个尺度独立算 drift loss，加权求和
loss = w1 * drift_loss(feat_emb) + w2 * drift_loss(feat_sent)
```

先从尺度 1 开始，验证有效后逐步加入更高级的尺度。

---

## 问题 3：正样本 — 没有 category 时条件生成怎么做？

### 问题描述

图像 Drifting 的 conditional 机制：
```
条件 = class label（如 "goldfish"）
正样本 = Memory Bank 里 32 张金鱼图的特征
→ drift force 描述的是"金鱼这个类别的分布形状"
```

LLM 没有类别：
```
条件 = prompt（每个 prompt 都是唯一的）
正样本 = ？？？
→ 不存在"同一个 prompt 的 32 条不同 GT response"
```

### 可能的方案

#### 方案 A：只用自己的 GT（1 个正样本，DESIGN.md 当前方案）

```python
pos_feat = feature(ground_truth_response)    # [B, 1, D]
gen_feat = feature(generated_candidates)     # [B, G, D]
drift_loss(gen=gen_feat, pos=pos_feat)
```

- **吸引力** → 退化为"往 GT 方向拉"，类似 regression
- **自排斥力** → G 个候选互相排斥，防止模式坍缩
- **实际意义**：drift loss ≈ 带自排斥的 regression loss

#### 方案 B：Batch 内所有 GT 作为正样本（IDEA.md 方案）

```python
all_gt_feat = feature(all_gt_in_batch)       # [B, D]
pos_feat = all_gt_feat.unsqueeze(0).expand(B, B, -1)  # [B, B, D]
gen_feat = feature(generated_candidates)     # [B, G, D]
drift_loss(gen=gen_feat, pos=pos_feat)
```

- 对于 prompt_i，所有 B 条 GT 都是正样本
- 对应的 GT 在 feature space 里最近 → 吸引力最强
- 其他 GT 提供"好回答的流形"的背景吸引力
- **风险**：不同 prompt 的 GT 在特征空间里不一定应该被拉到一起

#### 方案 C：基于语义相似度的 soft grouping

既然没有 hard category，可以用特征距离做 soft 分组：

```python
all_gt_feat = feature(all_gt_in_batch)               # [B, D]
similarity = cosine_similarity(all_gt_feat, all_gt_feat)  # [B, B]
# 每个样本的正样本权重由语义相似度决定
pos_weights = softmax(similarity / τ_group, dim=-1)   # [B, B]
# 相似的 GT 贡献更大的正样本吸引力
```

- **效果**：数学题的生成看到更多数学题 GT 的吸引力，代码题看到更多代码题的
- **类比**：soft version 的 ImageNet per-class memory bank
- **零额外标注**

### 推荐策略

从 **方案 A** 开始（最简单，1 个正样本 = GT），验证 drift loss 的自排斥力是否对 LLM 有价值。
如果有信号，升级到 **方案 C**（soft grouping），模拟 ImageNet 的 per-class 正样本机制。

---

## 问题 4：计算量 — G 个候选的开销

### 问题描述

图像 Drifting 用 gen_per_label = 16，即每个类别生成 16 个候选。
LLM 的 gen_per_prompt = G 意味着：

```
每个 prompt 需要 G 次完整 forward pass（每次用不同噪声/mask）
如果 Generator 是 7B，G=4：
  → 4 × 7B forward + backward = 28B 参数的 forward + backward
如果还要过 Feature LLM（冻结 7B，梯度穿过）：
  → 4 × 7B forward（Feature LLM）+ 4 × 7B backward（梯度穿过）
  → 总计 ~84B 参数量的计算
```

这比普通 SFT（7B 一次 forward + backward）贵 6-12 倍。

### 可能的方案

#### 方案 A：小 G + 大 batch

G=2（最小有意义值：至少 2 个候选才有自排斥），增大 batch size 补偿。

#### 方案 B：不用外部 Feature Model

直接在 softmax/embedding 空间做 kernel（问题 2 方案 A），省掉 Feature Model 的开销。
计算量回到 G × Generator forward+backward，没有额外 Feature Model 的开销。

#### 方案 C：分阶段训练

1. **Phase 1**：纯 CE loss 训练（标准 MDLM/LLaDA），训到收敛
2. **Phase 2**：加入 drift loss 微调，少量 steps，G=2
   - Phase 2 不需要从头训，只需要 drift loss 在 CE 基础上做分布优化

#### 方案 D：候选共享 KV Cache

G 个候选共享相同的 prompt，prompt 部分的 KV cache 可以复用：
```
prompt 部分：1 次计算，G 次复用
response 部分：G 次独立计算
→ 如果 prompt 占序列的 50%，计算量减少 ~25-35%
```

### 推荐策略

Phase 1 纯 CE，Phase 2 加 drift loss（G=2），不用外部 Feature Model。
这是计算量最低的组合，先验证 drift loss 是否有增益。

---

## 问题 5：Drift Loss vs CE Loss 的关系

### 问题描述

DESIGN.md 当前设计是 `total_loss = CE + λ * drift_loss`。
但 drift loss 和 CE loss 可能冲突：

- CE loss 让每个候选独立地往 GT 逼近（每个候选都被拉向 argmax = GT token）
- Drift loss 的自排斥力让候选互相远离
- 如果 CE loss 太强，所有候选坍缩到同一个 argmax，drift 的排斥力白费
- 如果 drift loss 太强，候选为了互相远离而偏离 GT，CE 变差

### 可能的方案

#### 方案 A：λ 退火

```
训练早期：λ 小（CE 主导，先学对基本的 token prediction）
训练后期：λ 增大（drift 介入，在质量基础上做分布优化）
```

#### 方案 B：只在部分 token 上加 drift loss

不是所有 token 都需要多样性。对于确定性 token（如格式、标点），CE 就够了。
只在高不确定性的 token 上加 drift loss：

```python
entropy = -sum(p * log(p), dim=-1)            # [B, G, T]
drift_mask = (entropy > threshold)             # 只在高熵 token 上加 drift
drift_loss = drift_loss_fn(gen_feat[drift_mask], pos_feat[drift_mask])
```

#### 方案 C：不用 CE，纯 drift

图像 Drifting 没有 CE，纯靠 drift loss。能否在 language 上也这样？
- **风险极高**：图像是连续空间，drift force 可以"推一小步"；语言是离散的，drift force 必须在 softmax 空间里跨越 token boundary
- **可能需要**：方案 D（embedding 空间 kernel）才能让纯 drift 有意义

### 推荐策略

从 **方案 A（λ 退火）** 开始：Phase 1 纯 CE 训到收敛，Phase 2 逐渐引入 drift loss。

---

## 问题 6：Feature LLM 能否接受 soft embedding？— 文献综述

### 原始担忧

预训练 LLM 只见过 hard embedding（单个 token 的查表结果），Gumbel-Softmax 产生的 soft embedding
（所有 token 的加权平均）是 OOD 输入，Feature LLM 可能输出垃圾特征。

### 文献证据：soft embedding 通过预训练 LLM 已被证明可行

#### SofT-GRPO (2024, arXiv:2511.06411)
**直接做了 IDEA.md 描述的事情**：Gumbel-Softmax → soft embedding → 送入 LLM 继续前向。
- 在 1.5B-7B 模型上，soft-thinking（连续 embedding）**优于**离散 token Chain-of-Thought
- 证明预训练 LLM 对 soft embedding 输入有鲁棒性

#### GRADE (2025, arXiv:2601.11574)
用 Gumbel-Softmax + STE 替代 PPO 做 LLM alignment：
- 比 PPO 提升 50%，梯度方差降低 14×
- 证明通过 Gumbel-Softmax / STE 反向传播到 LLM 参数是可行的

#### DLM-One (2025, arXiv:2506.00290)
**一步文本生成**：score distillation 在连续 embedding 空间对齐 student 和 teacher DLM：
- ~500× 推理加速，质量接近多步 teacher
- 基于 continuous diffusion (DiffuSeq)，不是 masked diffusion
- 证明一步文本生成在 embedding 空间是可行的

#### DiMO (2025, ICCV, arXiv:2503.15457)
蒸馏 masked diffusion model → one-step generator：
- Token-level distribution matching + token initialization strategy
- 应用于 class-conditional 和 text-conditional **图像**生成
- 方法可迁移到语言（masked diffusion → one-step 的通用方案）

#### dParallel (2025, arXiv:2509.26488)
LLaDA-8B 解码步数 256 → 24-30（~10× 加速）：
- certainty-forcing distillation
- 不是一步生成，但证明 LLaDA 可大幅减步

### 修正后的结论

| 原始担忧 | 文献结论 |
|---|---|
| soft embedding 喂给预训练 LLM 会出垃圾 | SofT-GRPO：**不会**，甚至优于离散 token |
| STE 梯度有偏不能训 | GRADE：**能训**，比 PPO 好 50% |
| 一步文本生成不可行 | DLM-One：**可行**，500× 加速 |
| Masked diffusion 不能蒸馏到一步 | DiMO：**可以**（图像上已验证） |

**Feature LLM 方案（Gumbel-Softmax → soft embedding → 冻结 LLM）有充分的文献支撑。**

### Feature LLM 方案的完整梯度链路

```
Gen 端（有梯度）：
  Generator(prompt + noise) → logits
    → Gumbel-Softmax(τ) → soft_probs [B,T,V]
    → soft_probs @ Feature_LLM.embedding → soft_emb [B,T,D]
    → Feature_LLM(soft_emb) → gen_hidden_states [B,T,D]
    → gen_feat（有梯度，可回传到 Generator）

GT 端（无梯度，正常输入）：
  gt_tokens → Feature_LLM(gt_tokens) → gt_hidden_states [B,T,D]
    → pos_feat（stop_gradient，正常 LLM 前向，特征质量有保障）

Drift loss：
  drift_loss(gen_feat, pos_feat) → 梯度回传到 Generator

关键：GT 端完全正常（离散 token 正常输入 LLM），无任何问题。
     Gen 端通过 Gumbel-Softmax 桥接，SofT-GRPO 已证明可行。
```

---

## 推荐架构：One-Step Text Generator — 完整详解

### 系统组成：2 个模型（推理时只需 1 个）

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  模型 1: Generator（可训练，~1-8B）                                  │
│  ═══════════════════════════════════                                 │
│  是什么：双向 Transformer（LLaDA 架构）                               │
│  做什么：noise + prompt → 一步生成完整 response 的 logits            │
│  初始化：LLaDA 预训练权重                                            │
│  训练时：参数更新（drift loss 的梯度回传到这里）                      │
│  推理时：唯一需要的模型                                               │
│                                                                     │
│  模型 2: Feature Encoder（冻结，~300M-1B）                           │
│  ═════════════════════════════════════════                           │
│  是什么：sentence embedding 模型（如 E5、BGE、GTE）                  │
│         或小型 LLM（如 LLM2Vec）                                     │
│  做什么：文本/soft embedding → 语义特征向量                           │
│  初始化：现成预训练权重（或在 SFT 数据上自训练）                      │
│  训练时：完全冻结，只做前向传播（梯度穿过但参数不更新）               │
│  推理时：不需要（只在训练时用）                                       │
│  类比图像 Drifting：冻结 MAE                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 图像 Drifting vs 语言版：逐组件对应

| 组件 | 图像 Drifting | Drift-LLM |
|---|---|---|
| **Generator** | DiT (~400M) | LLaDA Transformer (~1-8B) |
| Generator 输入 | noise [H,W,C] + class_embed | noise [T_r,D] + prompt_emb |
| Generator 输出 | 连续图片 [H,W,C] | logits [T_r,V] |
| **Feature Encoder** | 冻结 MAE-ResNet (~50-100M) | 冻结 sentence embedder (~300M-1B) |
| Feature Encoder 训练 | ImageNet 上 self-supervised (MAE) | 现成预训练 / SFT 数据上自训练 |
| Gen → Feature 的桥接 | 直接（图片是连续的） | Gumbel-Softmax → soft_emb（桥接离散） |
| GT → Feature | 真实图片直接过 MAE | 离散 GT token 正常过 Feature Encoder |
| **Drift Loss 特征** | MAE 多层多尺度特征 | Feature Encoder hidden states 多尺度 |
| 特征尺度 | patch_mean_2, patch_mean_4, global_mean, layer1-4 | per-token, chunk_mean, global_mean, 不同层 |
| **正样本** | Memory Bank 同类 ~32 张图 | 当前 GT (1 个) |
| **负样本** | Memory Bank 随机类图 + 自身 old_gen | 自身 old_gen（G 个候选互相排斥） |
| **CE Loss** | 无（纯 drift） | 可选辅助 |
| **推理** | noise + class → 一次前向 → 图片 | noise + prompt → 一次前向 → argmax → 文本 |
| 推理需要 Feature Encoder? | 不需要 | 不需要 |

### 各模型的具体规格

#### Generator

```
架构:     LLaDA (双向 Transformer, 非自回归)
          bidirectional self-attention, 所有位置互相可见
参数量:   1B (快速实验) / 8B (最终版)
初始化:   LLaDA 预训练权重
输入:     [prompt_embedding, noise_vectors]  shape: [B, T_p+T_r, D]
          prompt 部分: 正常 token embedding (查表)
          response 部分: i.i.d. Gaussian noise N(0, I)
输出:     logits [B, T_r, V]  (只在 response 位置)
训练:     参数可更新
推理:     noise + prompt → logits → argmax → tokens
```

#### Feature Encoder

```
方案 A — 现成 sentence embedder（先试这个）:
  模型:     E5-large (~335M) 或 BGE-large (~335M) 或 GTE-large (~335M)
  特点:     专门为语义相似度训练，embedding 距离 = 语义距离
  输入:     token sequence [B, T] 或 soft_emb [B, T, D_feat]
  输出:     hidden_states [B, T, D_feat]（多层可提取多尺度）
  训练:     完全冻结
  类比:     图像 Drifting 用现成 SimCLR/MoCo (Table 3, FID 8-11)

方案 B — LLM2Vec（把 decoder LLM 转成 encoder）:
  模型:     LLaDA-1B 经 LLM2Vec 3 步适配 → 双向 encoder
  特点:     和 Generator 同架构/同 tokenizer，特征空间自然对齐
  输入/输出: 同上
  类比:     图像 Drifting 用相关架构的 MAE

方案 C — 自训练 encoder（预期效果最好）:
  模型:     小型 Transformer (~300M-1B)
  训练:     在 SFT 数据上做 masked language modeling 或对比学习
  类比:     图像 Drifting 自训练 latent-MAE (Table 3, FID 3.36, 最佳)
```

### 完整训练数据流（一个 training step，7 步）

```
输入数据：B 个 (prompt, gt_response) 对
         prompt: "What is 2+2?"
         gt_response: "The answer is 4."


══════════════════════════════════════════════════════════════════════
 STEP 1: 构造 Generator 输入
══════════════════════════════════════════════════════════════════════

  prompt tokens: [What, is, 2, +, 2, ?]
       ↓ Generator.embedding 查表
  prompt_emb: [B, T_p, D]               D = hidden dim, 如 4096

  noise: N(0, I)  →  [B*G, T_r, D]      G 个候选，每个不同的噪声
                                         T_r = response 最大长度

  拼接：input_emb = [prompt_emb | noise]  →  [B*G, T_p + T_r, D]


══════════════════════════════════════════════════════════════════════
 STEP 2: Generator 前向（一次，非迭代）
══════════════════════════════════════════════════════════════════════

  input_emb [B*G, T_p+T_r, D]
       ↓
  ┌─────────────────────────────────┐
  │  Generator                       │
  │  (双向 Transformer, LLaDA 架构)  │
  │  bidirectional self-attention    │
  │  所有位置互相看到                 │
  │  参数量: ~1-8B                   │
  │  状态: 可训练                    │
  └─────────────────────────────────┘
       ↓
  hidden_states [B*G, T_p+T_r, D]     ← Generator 的中间表示
       ↓ LM Head (线性层 D→V)
  logits [B*G, T_r, V]                 ← 只取 response 位置
                                        V = vocab size, 如 128000

  输出：每个 response 位置上对 V 个 token 的预测分数


══════════════════════════════════════════════════════════════════════
 STEP 3: 构造 Gen 端的 drift 特征（有梯度）
══════════════════════════════════════════════════════════════════════

  logits [B*G, T_r, V]
       ↓ Gumbel-Softmax(τ)
  soft_probs [B*G, T_r, V]             ← 可微的"软"概率分布
       ↓ @ Feature_Encoder.embedding    矩阵乘法 [V] × [V, D_feat] → [D_feat]
  soft_emb [B*G, T_r, D_feat]          ← "软 embedding"，落在 token embedding 的凸包内
       ↓                                D_feat = Feature Encoder 的 hidden dim
  ┌─────────────────────────────────┐
  │  Feature Encoder（冻结）         │
  │  (sentence embedding 模型       │
  │   如 E5-large ~335M             │
  │   或 LLM2Vec ~1B)               │
  │  状态: 完全冻结，参数不更新      │
  │  但梯度穿过计算图               │
  └─────────────────────────────────┘
       ↓
  gen_hidden [B*G, T_r, D_feat]        ← 有上下文的语义特征

  提取多尺度特征（类比图像 Drifting 的多层 MAE）：
    gen_feat_token  = gen_hidden                              # [B*G, T_r, D_feat]  per-token
    gen_feat_chunk  = chunk_mean_pool(gen_hidden, chunk=32)   # [B*G, T_r/32, D_feat]  段落级
    gen_feat_global = mean_pool(gen_hidden)                   # [B*G, D_feat]  全局

  reshape 成 drift loss 需要的格式：
    gen_feat_token = rearrange('(b g) t d -> (b t) g d', g=G)  # [B*T_r, G, D_feat]


══════════════════════════════════════════════════════════════════════
 STEP 4: 构造 GT 端的 drift 特征（无梯度）
══════════════════════════════════════════════════════════════════════

  gt_response tokens: [The, answer, is, 4, .]
       ↓ 正常输入（离散 token，和预训练时完全一致）
  ┌─────────────────────────────────┐
  │  Feature Encoder（冻结）         │
  │  （同一个模型，同一组参数）       │
  │  输入是正常的离散 token          │
  │  → 特征质量有保障                │
  └─────────────────────────────────┘
       ↓ stop_gradient（不回传梯度）
  gt_hidden [B, T_r, D_feat]          ← 正常的 hidden states

  同样提取多尺度特征：
    pos_feat_token  = gt_hidden                               # [B, T_r, D_feat]
    pos_feat_chunk  = chunk_mean_pool(gt_hidden, chunk=32)    # [B, T_r/32, D_feat]
    pos_feat_global = mean_pool(gt_hidden)                    # [B, D_feat]

  reshape:
    pos_feat_token = rearrange('b t d -> (b t) 1 d')          # [B*T_r, 1, D_feat]


══════════════════════════════════════════════════════════════════════
 STEP 5: 多尺度 Drift Loss
══════════════════════════════════════════════════════════════════════

  对每个尺度独立计算 drift loss：

  尺度 1 — per-token（类比图像 Drifting 的 per-patch feature）:
    drift_loss(
      gen  = gen_feat_token,    # [B*T_r, G, D_feat]  G 个候选在每个 token 位置的特征
      pos  = pos_feat_token,    # [B*T_r, 1, D_feat]  GT 在每个 token 位置的特征
      R_list = (0.02, 0.05, 0.2)
    )

  尺度 2 — chunk-level（类比图像的 patch_mean_2, patch_mean_4）:
    drift_loss(gen=gen_feat_chunk, pos=pos_feat_chunk, ...)

  尺度 3 — global（类比图像的 global_mean）:
    drift_loss(gen=gen_feat_global, pos=pos_feat_global, ...)

  total_drift_loss = w1 * loss_token + w2 * loss_chunk + w3 * loss_global


══════════════════════════════════════════════════════════════════════
 STEP 6: 可选 CE Loss
══════════════════════════════════════════════════════════════════════

  ce_loss = cross_entropy(logits, gt_response_tokens)

  total_loss = total_drift_loss + λ_ce * ce_loss


══════════════════════════════════════════════════════════════════════
 STEP 7: 反向传播 — 梯度流向
══════════════════════════════════════════════════════════════════════

  total_loss.backward()

  梯度流向：
    total_loss
      ↓
    drift_loss
      ↓
    gen_feat (Feature Encoder 的输出)
      ↓ 穿过冻结的 Feature Encoder（参数不更新，但梯度穿过计算图）
    soft_emb (soft_probs @ Feature_Encoder.embedding)
      ↓
    soft_probs (Gumbel-Softmax 的输出)
      ↓ Gumbel-Softmax 是可微的
    logits (Generator 的输出)
      ↓
    Generator 参数  ← 这里被更新

  不更新的：Feature Encoder 的参数（完全冻结）
```

### 推理（只需 Generator）

```
推理只需要 Generator，不需要 Feature Encoder。

  用户输入: "What is 2+2?"
       ↓ embedding 查表
  prompt_emb [1, T_p, D]

  随机噪声: N(0, I) → noise [1, T_r, D]

  input = [prompt_emb | noise]
       ↓
  Generator 一次前向
       ↓
  logits [1, T_r, V]
       ↓ argmax
  token_ids [1, T_r]
       ↓ decode
  "The answer is 4."

  不同 noise → 不同回答（多样性）
  一次前向，无迭代，无 diffusion steps
```

### 训练伪代码 (PyTorch)

```python
# 初始化
generator = LLaDA.from_pretrained("llada-1b")          # 可训练
feature_encoder = SentenceTransformer("e5-large")       # 冻结
feature_encoder.eval()
for p in feature_encoder.parameters():
    p.requires_grad = False

G = 4                    # 每个 prompt 的噪声候选数
tau = 1.0                # Gumbel-Softmax 温度（可退火）
lambda_ce = 0.1          # CE loss 权重

optimizer = AdamW(generator.parameters(), lr=1e-4)

for prompt, gt_response in dataloader:
    B, T_p = prompt.shape
    T_r = gt_response.shape[1]

    # ═══ Step 1: Generator 前向 ═══
    prompt_emb = generator.embed(prompt)                         # [B, T_p, D]
    prompt_rep = prompt_emb.repeat_interleave(G, dim=0)          # [B*G, T_p, D]
    noise = torch.randn(B * G, T_r, D)                          # [B*G, T_r, D]
    input_emb = torch.cat([prompt_rep, noise], dim=1)            # [B*G, T_p+T_r, D]
    logits = generator(inputs_embeds=input_emb).logits[:, T_p:]  # [B*G, T_r, V]

    # ═══ Step 2: Gen 端特征（有梯度）═══
    soft_probs = gumbel_softmax(logits, tau=tau, hard=False)     # [B*G, T_r, V]
    soft_emb = soft_probs @ feature_encoder.embedding.weight     # [B*G, T_r, D_feat]
    gen_hidden = feature_encoder.forward_from_embeds(soft_emb)   # [B*G, T_r, D_feat]

    # ═══ Step 3: GT 端特征（无梯度）═══
    with torch.no_grad():
        gt_hidden = feature_encoder(gt_response)                 # [B, T_r, D_feat]

    # ═══ Step 4: 多尺度 drift loss ═══
    # 尺度 1: per-token
    gen_tok = rearrange(gen_hidden, '(b g) t d -> (b t) g d', g=G)
    pos_tok = rearrange(gt_hidden,  'b t d -> (b t) 1 d')
    loss_tok, _ = drift_loss(gen=gen_tok, fixed_pos=pos_tok)

    # 尺度 2: global mean pooling
    gen_glob = rearrange(gen_hidden.mean(dim=1), '(b g) d -> b g d', g=G)
    pos_glob = gt_hidden.mean(dim=1).unsqueeze(1)                # [B, 1, D_feat]
    loss_glob, _ = drift_loss(gen=gen_glob, fixed_pos=pos_glob)

    total_drift = loss_tok.mean() + loss_glob.mean()

    # ═══ Step 5: 可选 CE loss ═══
    logits_for_ce = rearrange(logits, '(b g) t v -> b g t v', g=G)[:, 0]
    ce_loss = F.cross_entropy(logits_for_ce.reshape(-1, V), gt_response.reshape(-1))

    # ═══ Step 6: 反向传播 ═══
    total_loss = total_drift + lambda_ce * ce_loss
    total_loss.backward()   # 梯度穿过冻结的 Feature Encoder → soft_emb → logits → Generator
    optimizer.step()
    optimizer.zero_grad()
```

### 关键设计决策总结

| 决策 | 选择 | 原因 |
|---|---|---|
| Generator 架构 | LLaDA (双向 Transformer) | 支持并行生成所有 token（非自回归）|
| Generator 初始化 | LLaDA 预训练权重 | 不从头训，利用已有的语言能力 |
| Feature Encoder | 冻结 sentence embedder | 提供语义距离，不参与训练（类比冻结 MAE）|
| Gen → Feature 桥接 | Gumbel-Softmax | SofT-GRPO 已验证可行 |
| GT → Feature | 正常离散 token 输入 | 无问题，和预训练分布一致 |
| 多尺度特征 | per-token + chunk + global | 类比图像 Drifting 的 patch + global |
| 正样本 | 当前 GT (1 个) | LLM 无类别分组 |
| 负样本 | G 个候选自排斥 | 防止忽略噪声 → 模式坍缩 |
| CE loss | 可选辅助 | 提供 per-token 精确监督 |
| 推理 | 只需 Generator | Feature Encoder 只在训练时使用 |

### 为什么一步生成下 drift loss ≠ MSE

在 masked diffusion 里（之前讨论的架构 A），G 个候选看到不同的 mask，自然就不同。
自排斥推它们更不同 → 但推离 GT → 有害。

**在一步生成里完全不一样：**

```
G=4 个候选，每个用不同噪声，但都看到同一个 prompt：
  候选 1：noise_1 + "What is 2+2?" → 模型输出 ???
  候选 2：noise_2 + "What is 2+2?" → 模型输出 ???
  候选 3：noise_3 + "What is 2+2?" → 模型输出 ???
  候选 4：noise_4 + "What is 2+2?" → 模型输出 ???

如果没有自排斥（纯 MSE/CE）：
  模型学会忽略噪声 → 4 个候选输出完全一样 → 模式坍缩
  无论给什么 noise，"2+2" 永远输出 "4"
  → 对于 "2+2" 这没问题，但对于 "写一首诗" 就丧失了多样性

有自排斥（drift loss）：
  4 个候选被迫不同 → 模型必须利用噪声来决定输出什么
  noise_1 → "The answer is 4."
  noise_2 → "2+2 equals 4."
  noise_3 → "It's 4."
  noise_4 → "Four."
  → 不同噪声 → 不同的合法表达方式
```

**这和图像 Drifting 的价值命题完全一致：**
- 图像：没有自排斥 → 忽略噪声 → 同一个 class 永远生成同一张图
- 语言：没有自排斥 → 忽略噪声 → 同一个 prompt 永远生成同一个回答

### Kernel 空间方案：三层递进

基于文献调研，Feature LLM 方案是可行的。按复杂度递进：

**Level 1：Embedding 投影（无 Feature Encoder，最快验证）**
```
gen_feat = softmax(logits) @ E     # [B, G, T, D]
pos_feat = E[gt_tokens]            # [B, 1, T, D]
drift_loss 在 R^D 空间，per-token 独立
```
优点：零额外开销，无离散瓶颈
缺点：无上下文，无序列级语义

**Level 2：Feature Encoder + Gumbel-Softmax（有文献支撑）**
```
Gen: logits → Gumbel-Softmax → soft_emb → Feature_Encoder → gen_feat [B,G,T,D]
GT:  gt_tokens → Feature_Encoder → pos_feat [B,1,T,D]（正常输入，无问题）
drift_loss 在 Feature Encoder hidden state 空间，有上下文
```
优点：有上下文感知的丰富特征，可做多尺度（类比 MAE 多层）
缺点：额外计算量（Feature Encoder forward + backward for gen）
文献支撑：SofT-GRPO 证明 soft embedding 通过 LLM 可行

**Level 3：多尺度组合（类比图像 Drifting 的多层 MAE feature）**
```
尺度 1: per-token embedding 投影（Level 1）
尺度 2: Feature Encoder 浅层 hidden states（语法/词法）
尺度 3: Feature Encoder 深层 hidden states（语义/推理）
尺度 4: chunk-level pooled features（段落级）
尺度 5: global pooled features（全局）

每个尺度独立算 drift loss，加权求和（和图像 Drifting 完全一致）
```

### 和竞品的差异化

| | DLM-One | DiMO | dParallel | Drift-LLM（ours） |
|---|---|---|---|---|
| 基础模型 | continuous diffusion (DiffuSeq) | masked diffusion (图像) | LLaDA | LLaDA / masked diffusion |
| 方法 | score distillation | distribution matching | certainty-forcing | drift loss（粒子漂移） |
| 需要 teacher? | 是 | 是 | 是 | **否**（纯 drift，不需要 teacher） |
| 目标步数 | 1 步 | 1 步 | 24-30 步 | 1 步 |
| 应用 | 文本 | 图像 | 文本 | 文本 |

**Drift-LLM 的独特卖点：不需要 teacher model 的一步文本生成。**
其他方法都是蒸馏（需要先训好一个多步 teacher），Drifting 从头训一步 generator。

### 可能的失败点

1. **Transformer 能否从纯噪声一步生成连贯文本？**
   - 图像 DiT 可以从纯噪声一步生成图片，是因为像素空间连续
   - 语言最终要 argmax 到离散 token，每个位置必须"选对一个 token"
   - 一步生成没有迭代修正的机会，错一个 token 可能影响全局连贯性
   - **这是最大的风险**
   - 但 DLM-One 已在 embedding 空间证明一步文本生成可行

2. **Gumbel-Softmax 温度 τ 的选择**
   - τ 大：soft embedding 模糊，Feature Encoder 特征不精确，但梯度更平滑
   - τ 小：soft embedding 接近 hard，特征好，但梯度可能消失
   - SofT-GRPO 的经验：可以用 τ 退火 + straight-through 结合

3. **CE 和 Drift 的关系？**
   - 纯 drift（像图像一样）：最符合框架，但语言是离散的，可能不够
   - CE + drift：CE 提供 per-token 精确监督，drift 提供分布匹配 + 防坍缩
   - 需要实验确定最佳组合

---

## 验证实验方案

所有实验都以 one-step generation 为目标。

### Exp 0：Baseline — LLaDA 各步数下的性能

**目标**：确认 one-step 需要弥补多大的 gap

```
模型：LLaDA-1B（快速迭代）/ LLaDA-8B（最终对比）
数据：Alpaca / UltraChat SFT 数据
训练：标准 masked diffusion + CE loss

评估（不同推理步数）：
  步数:  128步  64步  16步  4步  1步
  指标:  MMLU, GSM8K, human eval, coherence, perplexity

这个实验回答：
  - LLaDA 1步推理有多差？→ 这是 drift one-step generator 需要超越的下界
  - LLaDA 128步推理有多好？→ 这是 one-step generator 的质量上界
```

### Exp 1：Drift Loss 信号验证（最关键，成本最低）

**目标**：drift loss 在 embedding 空间对语言有没有有意义的梯度？

```
不训练，只做前向分析。

用 LLaDA 的 embedding matrix E 构造特征：

  1. 取 100 条 (prompt, response) 对
  2. 对每条，用 G=8 个不同噪声构造输入：
     [prompt_emb, noise_1], [prompt_emb, noise_2], ...
  3. 过 LLaDA（当作 one-step generator）得到 logits
  4. gen_feat = softmax(logits) @ E     # [8, T, D]
     pos_feat = E[gt_tokens]            # [1, T, D]
  5. 计算 drift_loss(gen_feat, pos_feat)

  观察：
  a. loss 值是否有意义？（不是 nan/inf）
  b. 梯度方向分析：
     - ∂loss/∂logits 是否把 logits 往 GT token 方向推？
     - ∂loss/∂logits 是否让不同候选互相远离？
  c. 对比：把 GT 替换为随机 token → loss 梯度方向应该变化
     如果不变 → embedding kernel 无区分能力 → 需要换 kernel 空间
  d. 不同 R 值的效果
  e. embedding 空间 vs softmax 空间 vs generator hidden states

预计耗时：几小时
如果失败 → 需要重新设计 kernel 空间后再继续
```

### Exp 2：One-Step Generator 训练（核心实验）

**目标**：验证 one-step text generation + drift loss 是否可行

```
模型：LLaDA-1B 架构（双向 Transformer），可用 LLaDA 预训练权重初始化
数据：Alpaca-52k

训练配置：
  输入：[prompt_emb, noise] → Generator → logits [B*G, T_r, V]
  G = 4（每个 prompt 4 个噪声候选）
  Kernel 空间 = embedding 投影（softmax(logits) @ E）

对比组：
  A: One-step + CE only              → 纯 CE 训练一步生成器
  B: One-step + drift only           → 纯 drift（对标图像 Drifting）
  C: One-step + CE + drift           → 组合
  D: One-step + MSE(gen_emb, gt_emb) → 直接 embedding 回归（对照组）
  E: LLaDA 1步推理                   → 不训练一步生成，直接 1步推理

关键对比：
  B vs D：drift loss vs MSE 在一步生成下的区别
    → B 有自排斥，D 没有
    → 如果 B > D，证明自排斥对一步生成有用（防止忽略噪声）
    → 如果 B ≈ D，说明 drift 在语言上退化为 MSE

  B vs E：drift 训练的一步生成 vs LLaDA 直接跳到 1 步
    → 如果 B > E，drift one-step generator 有独立价值
    → 如果 B ≤ E，不需要专门训一步生成器

  A vs B vs C：CE, drift, 还是组合效果最好？

评估指标：
  - 生成文本连贯性（人工打分 1-5）
  - 回答正确率（GSM8K, MMLU）
  - 多样性：对同一 prompt 用不同 noise 生成 10 次，计算 self-BLEU
    → 多样性高 = 模型在利用噪声，drift 的自排斥在起作用
    → 多样性 ≈ 0 = 模型忽略噪声，模式坍缩
  - 和 LLaDA 多步推理的质量对比
```

### Exp 3：消融实验（理解 drift loss 各组件的作用）

**目标**：drift loss 的哪个组件对一步生成最关键？

```
前提：Exp 2 中 drift loss 有正向信号

3a. G 的影响：
  G=2 vs G=4 vs G=8 vs G=16
  → G 越大自排斥越强，但计算量也越大
  → 找到 cost-effective 的 G

3b. 自排斥 vs 吸引力分离：
  只有吸引力（G=1，drift = MSE）vs 完整 drift（G=4）
  → 定量衡量自排斥的边际收益

3c. R_list 温度选择：
  (0.02, 0.05, 0.2) vs (0.1, 0.5, 2.0) vs (0.5, 2.0, 10.0)
  → 图像的 R 值不一定适合语言的 embedding 空间尺度

3d. CE 辅助的比例：
  λ_ce = {0, 0.01, 0.1, 1.0, 10.0}
  → 纯 drift 是否可行？还是必须有 CE 兜底？

3e. 噪声注入方式：
  方式 1: response 位置全部替换为 N(0,I)
  方式 2: response 位置 = GT_emb + N(0,σ)（加噪而非替换）
  方式 3: 额外的 noise token 拼接在 prompt 后面
  → 图像用方式 1（纯噪声），但语言可能需要不同策略
```

### Exp 4：Kernel 空间探索（含 Feature LLM）

**目标**：找到语言最优的 kernel 空间

```
前提：Exp 2 确认 drift loss 对一步生成有用

固定 one-step 架构和其他超参，只换 kernel 空间：

  A: Embedding 投影 — softmax(logits) @ E → R^D（Exp 2 的 baseline）
     优点：有词级语义，无离散瓶颈
     缺点：无上下文，无序列级信息

  B: Feature LLM + Gumbel-Softmax（文献已验证可行）
     Gen: Gumbel-Softmax → soft_emb → frozen Feature LLM → hidden states
     GT:  gt_tokens → frozen Feature LLM → hidden states（正常输入）
     优点：有上下文感知特征，可做多层/多尺度
     缺点：额外计算量
     Feature LLM 候选：LLaDA 自身 / LLM2Vec / 小型 LLM (1B)

  C: 多尺度（B 的增强版，类比图像 Drifting 多层 MAE）
     per-token embedding + Feature LLM 浅层 + 深层 + chunk-pooled + global-pooled
     每个尺度独立算 drift loss

  D: STE + sentence embedder（GRADE 验证 STE 可行）
     前向：argmax → 离散 token → sentence embedder → R^D_sent
     反向：STE（梯度穿过 argmax，GRADE 证明方差低于 REINFORCE 14×）
     优点：sentence embedder 拿到真实 token，有序列级语义

  E: REINFORCE + sentence embedder（对照组）
     和 D 相同前向，用 REINFORCE 替代 STE 估计梯度
     对比 D vs E 确认 STE 是否优于 REINFORCE

关键对比：
  A vs B：加 Feature LLM 是否显著改善 kernel 质量？
  B vs C：多尺度是否进一步提升？
  B vs D：Gumbel-Softmax vs STE 哪个更好？

评估：同 Exp 2 的指标
```

---

## 实验优先级排序

```
Exp 0 (Baseline)          ← 必须先做，确认 LLaDA 1步有多差
  ↓
Exp 1 (信号验证)          ← 成本最低（不训练），信息量最大
  ↓                          失败 → 换 kernel 空间
Exp 2 (One-step 训练)     ← 核心实验，直接验证一步生成 + drift
  ↓                          失败 → 可能一步生成对语言不可行
Exp 3 + Exp 4 (并行)      ← 理解各组件作用，优化 kernel 空间
```

**总预计时间**：
- Exp 0: 1-2 天（跑标准 LLaDA baseline）
- Exp 1: 半天（只做前向 + 梯度分析）
- Exp 2: 5-7 天（多组对比训练）
- Exp 3+4: 5-7 天（消融 + kernel 探索，可并行）

**Go/No-Go 决策点**：
- Exp 1 之后：如果 embedding kernel 无区分能力 → 必须先找到好的 kernel 空间
- Exp 2 B vs D：如果 drift ≈ MSE → 自排斥对语言一步生成没用，方向可能不对
- Exp 2 B vs E：如果 drift one-step ≤ LLaDA 1步 → 不需要专门训一步生成器

---

## 可能的失败模式与应对

| 失败模式 | 如何检测 | 应对 |
|---|---|---|
| Kernel 在 embedding 空间无区分能力 | Exp 1: 换随机 GT 后 drift 梯度方向不变 | 换 kernel 空间（sentence embedder + STE） |
| 自排斥和 CE 冲突 | Exp 2: 加 drift 后 CE loss 反而升高 | 降低 λ，或只在高熵 token 上加 drift |
| 温度 R 对语言不合适 | Exp 2: drift loss 不下降或梯度爆炸 | 在 Exp 1 中先分析特征距离的尺度，据此设定 R |
| G 个候选计算量太大 | Exp 2: 同样计算量分给大 batch 效果更好 | 对比 G=2 vs batch_size×2，确认 drift 的边际收益 |
| One-step generation 对语言根本不可行 | Exp 2: 即使 drift loss 有效，一步生成仍然不连贯 | 退回到"少步生成"（4-8步），drift 做加速辅助（dParallel 证明可行） |
| Embedding kernel 太弱 | Exp 2: drift ≈ MSE | 升级到 Feature LLM kernel（Exp 4B，文献已验证可行） |

---

## 总结：图像 → 语言的断裂与应对

| 图像 Drifting 的支柱 | 语言的挑战 | 应对策略 | 文献支撑 |
|---|---|---|---|
| 连续输出 → 可微链路 | 离散 token → 梯度断裂 | Gumbel-Softmax 桥接（或 STE） | SofT-GRPO, GRADE |
| MAE 特征空间 → 好的 kernel | LLM hidden states 不是为距离训的 | 冻结 LLM 做 Feature Model + 多尺度 | SofT-GRPO 证明 soft emb 通过 LLM 可行 |
| 冻结 MAE 接受生成的图片 | Feature LLM 能否接受 soft emb？ | **能**（SofT-GRPO 已验证） | SofT-GRPO |
| 按类别分组 → 充足正样本 | 没有类别，每个 prompt 独一无二 | 单 GT + 自排斥力；后续可加 soft grouping | — |
| 多步 diffusion → 一步 generator | 一步文本生成是否可行？ | noise + prompt → one-step Transformer | DLM-One, DiMO, dParallel |
| 不需要 teacher model | 竞品都需要蒸馏 teacher | **Drift 不需要 teacher**（独特卖点） | — |
