# One-Step Text Generation via Drifting

## 核心问题

Diffusion LLM（LLaDA, MDLM, Dream 等）推理需要 64-128 步迭代去掩码，非常慢。
图像领域，Drifting 把多步扩散成功蒸馏到**一步**生成（ImageNet FID 1.54）。
我们要做同样的事情：**一步文本生成**。

---

## 1. Generator 到底是什么？（不是 MDLM！）

**最重要的一点：图像 Drifting 的 Generator 根本不是 Diffusion Model。**

```python
# 图像 Drifting 的 Generator（drifting/models/generator.py 第 601-622 行）
def __call__(self, c, cfg_scale=1.0, temp=1.0):
    x = random.normal(rng, (B, H, W, C))     # ← 输入是随机高斯噪声
    cond = self.embed(c) + self.cfg_embed(cfg_scale)  # ← 条件 = 类别
    samples = self.dit(x, cond)               # ← 一次前向传播
    return samples                            # ← 直接输出完整图片
```

**没有去噪循环，没有迭代，不是 diffusion model。**
Generator = 纯前馈网络：noise + condition → 一步 → 完整输出。
多样性来自**随机噪声**：不同 noise → 不同图片。

### 所以 LLM 版的 Generator 也不应该是 MDLM

```
错误设计（之前的版本）:
  输入: prompt + [MASK][MASK][MASK]           ← MDLM 风格
  → 一次前向 → logits
  问题 1: 没有随机性，同一个 prompt 永远输出同一个结果
  问题 2: MDLM 是训练来"看部分猜部分"的，不是"看零猜全部"
  问题 3: 加 drift loss 不能改变模型的能力范围

正确设计（和图像版一致）:
  输入: prompt + 随机噪声 embedding           ← 和图像版一样
  → 一次前向 → logits → 所有 token
  → 不同噪声 → 不同回答（多样性）
  → 从一开始就按一步训练，不做 MDLM 式部分 mask
```

### Generator 架构

```
输入: prompt_embeddings + noise_embeddings
      ↑                    ↑
      prompt 正常编码       response 位置填充随机噪声
      [B, T_p, D]         noise ~ N(0, I), shape [B, T_r, D]

  ┌──────────────────────────┐
  │  Generator (θ)            │
  │  = Transformer            │
  │  (可以用 LLaDA 架构初始化) │
  └──────────────────────────┘
      ↓
  logits [B, T_r, V]  → softmax → soft_tokens
```

和图像版完全对应：
- 图像: noise [B, H, W, C] + class_embed → DiT → 图片 [B, H, W, C]
- 文本: noise [B, T_r, D]  + prompt_embed → Transformer → logits [B, T_r, V]

多样性来源相同：**不同随机噪声 → 不同生成结果**。

---

## 2. 为什么还需要 Feature LLM？Generator 已经有 soft tokens 了

### 问题：soft tokens 本身不是好的"特征"

Generator 输出的 soft tokens 是 R^V 空间里的概率向量（V = 128k）。
我们之前讨论过可以直接在这个空间上做 drift loss。

但 Generator 的输入是 prompt + 噪声，response 部分没有任何真实 token。
每个位置的 softmax 是"盲猜"的——不知道相邻位置预测了什么。
直接用这些孤立的 softmax 做 drift loss，特征质量太差。

**Feature LLM 的作用：把这些孤立的预测"读"一遍，让每个位置感知全局。**

### Feature LLM 做了什么

```
Generator 输出: soft_tokens [B, T, V]  （每个位置一个概率分布，彼此无关）
                    ↓
        soft_tokens @ Feature_LLM.embedding  → soft_embeddings [B, T, D]
        （把概率分布投影成 embedding 向量）
                    ↓
          ┌─────────────────────┐
          │  Feature LLM (冻结)  │   ← 不训练，只用来"读"生成的序列
          │  (pretrained LLM)    │
          └─────────────────────┘
                    ↓
        contextualized features [B, T, D]
        （每个位置的特征融合了整个序列的信息）
```

Feature LLM 把 Generator 产出的孤立的 soft tokens "读"了一遍，
让每个位置的特征都感知到序列里其他位置的内容。

### 类比图像 Drifting

```
图像 Drifting:
  Generator → 生成图片（像素）                     ← 像素本身不是好特征
  → 冻结 MAE → 多层特征（有语义结构、多尺度）       ← 这才是好特征

文本 Drifting:
  Generator → soft tokens（概率分布）               ← 概率分布本身不是好特征
  → 冻结 LLM → hidden states（有上下文、多层）      ← 这才是好特征
```

**关键：Generator 产出原始内容，Feature LLM 把它变成有意义的特征。
就像你画了一幅画（Generator），然后让一个艺术评论家（Feature LLM）
来分析这幅画的构图、色彩、主题（特征）。**

---

## 2. 两个模型的完整对比

```
┌─────────────────────────────────────────────────────────────┐
│                     Generator (θ)                            │
│                                                              │
│  角色: 画家                                                   │
│  做什么: 从 prompt + [MASK]s 一步生成完整回答                   │
│  是否训练: ✓ 是（drift loss 的梯度回传到这里）                  │
│  输入: prompt + [MASK][MASK]...[MASK]                         │
│  输出: logits → softmax → soft_tokens [B, T, V]              │
│                                                              │
│  初始化: 预训练的 diffusion LLM（如 LLaDA）                    │
│  参数量: 和基础模型一样（如 7B）                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Feature LLM (冻结)                         │
│                                                              │
│  角色: 评论家                                                 │
│  做什么: 读生成的序列，提取有意义的特征                          │
│  是否训练: ✗ 否（完全冻结，只做特征提取）                       │
│  输入: soft_tokens 经过 embedding 投影后的连续序列               │
│  输出: 多层 hidden states → 全局/chunk 池化特征                 │
│                                                              │
│  初始化: 同一个预训练 LLM（或另一个预训练 LLM）                 │
│  参数量: 和基础模型一样（如 7B）                                │
│  注意: 只做前向传播，不存梯度，显存开销可控                      │
│                                                              │
│  类比: 图像 Drifting 中的冻结 MAE                              │
└─────────────────────────────────────────────────────────────┘
```

### 梯度流向

```
drift_loss
    ↓ 梯度
Feature LLM 的输出特征
    ↓ 梯度（Feature LLM 本身冻结，但梯度穿过它的计算图）
soft_embeddings = soft_tokens @ Feature_LLM.embedding
    ↓ 梯度（Gumbel-Softmax 保持可微）
soft_tokens = Gumbel-Softmax(logits)
    ↓ 梯度
logits = Generator(prompt + [MASK]s)
    ↓ 梯度
Generator 参数 θ   ← 这里被更新
```

**Feature LLM 的参数不更新，但梯度穿过它。** 就像一面固定的透镜——
光（梯度）穿过透镜（Feature LLM）到达底片（Generator），
透镜本身不动，但决定了光怎么聚焦。

---

## 3. 为什么不直接用 Generator 自身的 hidden states？

你可能会问：Generator 也有 hidden states，为什么不直接用？

```
方案 A（不用 Feature LLM）:
  Generator(prompt + [MASK]s) → hidden_states → 直接做 drift loss

方案 B（用 Feature LLM）:
  Generator → soft_tokens → Feature LLM → features → drift loss
```

方案 A 的问题：

**Generator 的 hidden states 是看着 [MASK] 算出来的，不是看着"正常文本"算出来的。**

```
Generator 看到的:       [MASK] [MASK] [MASK] [MASK] [MASK]
Generator hidden state:  来自全是 MASK 的上下文，质量差

Feature LLM 看到的:      soft("The") soft("answer") soft("is") soft("42") soft(".")
Feature LLM hidden state: 来自近似正常文本的上下文，质量好
```

Generator 的 hidden states 编码的是"面对一堆 MASK 时的困惑"，
Feature LLM 的 hidden states 编码的是"读了一段（软）文本后的理解"。

后者才是有意义的特征，可以和真实文本的特征做比较。

**正样本的特征怎么来的？**

```
真实文本: "The answer is 42."
→ Feature LLM("The answer is 42.") → hidden_states → 全局池化 → pos_feat
```

Feature LLM 读真实文本得到 pos_feat，读生成的 soft 文本得到 gen_feat。
两者在同一个模型、同一个特征空间里，可以直接比距离。

如果用 Generator 的 hidden states（方案 A），gen 和 pos 的特征来自不同条件
（MASK 输入 vs 真实 token 输入），比较不公平。

---

## 4. 多尺度特征（和图像 Drifting 完全类比）

图像 Drifting 用 MAE 的不同层和不同池化方式提取多尺度特征。
文本 Drifting 完全照搬这个思路：

```
Feature LLM 输出的 hidden states: [B, T, D]  （T 个 token，每个 D 维）

Level 1 — 全局均值/标准差（类比 global_mean / global_std）:
  feat_mean = hidden.mean(dim=T)          # [B, 1, D]
  feat_std  = hidden.std(dim=T)           # [B, 1, D]
  → 捕捉"这个回答整体讲了什么"

Level 2 — Chunk 均值（类比 patch_mean_2 / patch_mean_4）:
  chunks = split(hidden, chunk_size=32)
  feat_chunk = chunks.mean(dim=chunk_dim)  # [B, T/32, D]
  → 捕捉"回答的每一段讲了什么"

Level 3 — 多层特征（类比 layer1, layer2, layer3, layer4）:
  feat_layer4  = Feature_LLM.layer[4].hidden   → 浅层：语法/词法
  feat_layer8  = Feature_LLM.layer[8].hidden   → 中层：短语/句法
  feat_layer12 = Feature_LLM.layer[12].hidden  → 深层：语义/推理

每种特征独立跑 drift loss，最后加起来（和图像 Drifting 完全一样）。
```

**不做 token 级 drift loss**（因为不同序列的 token 位置不对齐）。
只用全局和 chunk 级特征，这些在跨序列比较时是有意义的：
两个数学回答的全局 hidden state 均值是相似的，即使逐 token 完全不同。

---

## 5. 正/负样本从哪来 — 不需要 Memory Bank

### 图像 Drifting 需要 Memory Bank 是因为：
- ImageNet 每张图自带 class_id，Memory Bank 按 class 索引
- 一个 batch 里同类图片太少（batch=64，goldfish 可能只有 1-2 张）
- 需要 bank 凑够 32 张同类正样本

### LLM SFT 数据没有类别：
```
dllm 的 SFT 数据格式（Alpaca / UltraChat / OPC 等）:
{
    "messages": [
        {"role": "user", "content": "What is gravity?"},
        {"role": "assistant", "content": "Gravity is..."}
    ]
}
→ 没有 class_id，没有 topic 标签
→ 手动分类不现实，聚类引入额外复杂度
```

### 解决方案：直接用当前 batch 当正样本池

不需要 Memory Bank，不需要分类。当前 batch 本身就是正样本池：

```
一个 batch 有 32 条 (prompt, response):
  feat_1:  物理回答的全局特征
  feat_2:  代码回答的全局特征
  feat_3:  诗歌回答的全局特征
  ...
  feat_32: 地理回答的全局特征

对于 prompt_1（物理题），Generator 的 G 个候选:
  gen:  [G, D]        ← G 个候选的全局特征
  pos:  [32, D]       ← batch 内所有 32 条真实 response 的特征

为什么这样合理？

在全局特征空间里:
  - feat_1（物理回答）离 gen 候选最近 → 吸引力最强（对应的 GT）
  - feat_2（代码回答）稍远 → 吸引力弱，但提供"好回答的大致方向"
  - 所有真实回答共同构成"好 response 的流形"
  - drift force 把生成结果推向这个流形

类比图像 Drifting:
  正样本 = 同类图片（强吸引）+ unconditioned 图片（弱背景吸引）
  batch 内 = 对应 GT（强吸引）+ 其他 response（弱背景吸引）
```

### 负样本怎么来？

几种选择，都不需要 Memory Bank：

```
方案 1（最简单）: 不用负样本
  drift_loss(gen=gen_feat, fixed_pos=batch_feat, fixed_neg=None)
  只靠正样本吸引 + 自排斥，先跑通再说

方案 2: Batch 内 shuffle 作为负样本
  把 response 和错误的 prompt 配对 → "答非所问"的特征作为负样本
  neg_feat = batch_feat[torch.randperm(B)]
  → 排斥力让模型远离"和 prompt 不匹配的回答风格"

方案 3: CFG 风格（和图像 Drifting 一致）
  图像版的负样本就是 unconditional 生成（不给 class label）
  LLM 版：不给 prompt，直接从 [MASK]s 生成 → "无条件回答"的特征
  → 排斥力让模型远离"没有针对性的泛泛回答"
```

### 什么时候才需要 Memory Bank？

```
需要 Memory Bank 的场景（未来扩展）:
  - 数据集本身有类别标签（如 MMLU 的 subject）
  - 同一个 prompt 有多条合法回答（RLHF 数据集）
  - 数据集非常大，需要缓存特征避免重复计算

不需要的场景（我们先做这个）:
  - 标准 SFT 数据集（Alpaca, UltraChat）
  - 每个 prompt 只有一条 GT
  - Batch 本身提供足够的正样本信号
```

---

## 6. 完整训练流程

```
初始化:
  generator    = LLaDA(pretrained)         # 可训练
  feature_llm  = LLaDA(pretrained, frozen) # 冻结，用于特征提取
  G = 8                                    # gen_per_prompt

for step in range(total_steps):

    # ① 读取 batch: B 个 (prompt, response) 对
    prompt, response = next(dataloader)      # response: [B, T]

    # ② 预计算正样本特征（batch 内所有真实 response，无梯度）
    with torch.no_grad():
        real_emb = feature_llm.embedding(response)          # [B, T, D]
        real_hidden = feature_llm(inputs_embeds=real_emb)    # [B, T, D]
        pos_feat = real_hidden.mean(dim=1)                   # [B, D]
        # 每个样本看到整个 batch 作为正样本
        pos_feat = pos_feat.unsqueeze(0).expand(B, B, -1)   # [B, B, D]

    # ③ G 个候选，每个用不同随机噪声（和图像 Drifting 一样）
    prompt_emb = generator.embed(prompt)                      # [B, T_p, D]
    noise = torch.randn(B * G, T_r, D)                       # 随机噪声 ← 多样性来源
    prompt_rep = repeat(prompt_emb, 'b t d -> (b g) t d', g=G)
    input_emb = cat([prompt_rep, noise], dim=1)               # [B*G, T_p+T_r, D]
    logits = generator(inputs_embeds=input_emb)               # [B*G, T_r, V]

    # ④ Gumbel-Softmax → soft embeddings → Feature LLM（梯度穿过）
    soft_tokens = Gumbel_Softmax(logits, tau=τ)              # [B*G, T, V]
    soft_emb = soft_tokens @ feature_llm.embedding.weight    # [B*G, T, D]
    gen_hidden = feature_llm(inputs_embeds=soft_emb)         # [B*G, T, D]

    # ⑤ 提取全局特征（不做 token 级）
    gen_feat = gen_hidden.mean(dim=1)                         # [B*G, D]
    gen_feat = rearrange(gen_feat, '(b g) d -> b g d', g=G)  # [B, G, D]

    # ⑥ Drift Loss — 无需 Memory Bank
    d_loss = drift_loss(
        gen = gen_feat,    # [B, G, D]  — G 个候选，有梯度
        pos = pos_feat,    # [B, B, D]  — batch 内 B 条真实 response，无梯度
        # neg = None       — 先不用负样本，靠自排斥 + 正样本吸引
        R_list = (0.02, 0.05, 0.2),
    )
    # 对每种特征尺度（mean, std, chunk, 不同层）分别算，加起来

    # ⑦ 反向传播
    d_loss.backward()
    optimizer.step()

推理:
    noise ~ N(0, I)
    [prompt_emb, noise] → Generator → argmax → 完整回答（一步，一次前向传播）
```

---

## 7. 和图像 Drifting 的完整对应关系

| 图像 Drifting | One-Step Drift-LLM |
|---|---|
| Generator: (label, noise) → 像素/latent | Generator: (prompt, noise) → soft tokens |
| 冻结 MAE: 像素 → 多层特征 | 冻结 LLM: soft tokens → 多层 hidden states |
| 为什么需要 MAE: 像素不是好特征 | 为什么需要 Feature LLM: 孤立 softmax 不是好特征 |
| 多尺度: patch_mean, global_mean, layer_k | 多尺度: chunk_mean, global_mean, layer_k |
| Memory Bank: 按类别存图片特征 | Memory Bank: 按 topic 存回答特征 |
| gen_per_label = 16 | gen_per_prompt = 8 |
| 噪声输入: N(0, I) → 不同图片 | 噪声输入: N(0, I) → 不同回答 |
| Gumbel-Softmax: 不需要（输出本身连续） | Gumbel-Softmax: 必须（离散 token → 连续） |
| 推理: noise + label → 一步出图 | 推理: noise + prompt → 一步出文 |
| 训练: ~100k steps | 训练: 待实验 |

---

## 8. 论文贡献

1. **首次将粒子漂移框架（Drifting）应用于文本生成**
2. **实现一步文本生成**：Diffusion LLM 从 128 步降到 1 步
3. **设计了文本特征提取方案**：Gumbel-Softmax + 冻结 LLM + 全局/chunk 特征
   - 解决了 token 级位置不对齐的问题
   - 解决了离散 → 连续的可微性问题
4. **无需对抗训练 / 无需 reward model**：纯粹基于粒子动力学的分布匹配

---

## 9. 预期实验

| 实验 | 目的 |
|------|------|
| One-step vs multi-step diffusion LLM | 证明一步生成质量可接受 |
| Drift vs CE for one-step training | 证明 CE 不够，drift 更好 |
| Drift vs GAN-based text generation | 证明训练更稳定 |
| Ablation: feature scales | 哪些尺度（global/chunk/layer）最重要 |
| Ablation: G (gen_per_prompt) | 多少候选最优 |
| Ablation: R_list | 温度参数敏感性 |
| 推理速度对比 | 一步 vs 多步的实际加速比 |
