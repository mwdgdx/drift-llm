# Drift-LLM：用 Drifting 框架训练语言模型

## 1. 一句话概括

**Drifting** 原本是一个图像生成方法：在特征空间中，把生成样本当作"粒子"，用吸引力（靠近真实数据）和排斥力（远离彼此/负样本）驱动它们"漂移"到真实数据分布上。

**Drift-LLM** 的目标：把同样的思路搬到 masked diffusion LLM（如 LLaDA）上，让模型在**连续表示空间**里做分布级优化，而不仅仅是逐 token 最大似然。

---

## 2. 直觉：图像 Drifting 在做什么？

想象一个二维平面上有一些蓝色点（真实图片）和一些红色点（生成图片）：

```
训练开始：                     训练结束：
  ●              ○             ● ○          ○
     ○                            ● ○
  ●     ●                      ●     ●
              ○ ○                 ○ ○
     ●                            ●

  ● = 真实数据（蓝色）          红色点"漂移"到了蓝色点附近
  ○ = 生成样本（红色）          但彼此之间保持距离（不重叠）
```

每一步训练：
1. 看看每个红点周围有哪些蓝点（**吸引力**）和其他红点（**排斥力**）
2. 算一个"目标位置"（当前位置 + 合力方向）
3. 训练生成器，让它的输出靠近这个目标位置
4. 重复，红点逐渐漂移到蓝点分布上

---

## 3. 核心概念解释

### 3.1 Feature（特征）= 粒子在哪个空间里移动

在图像 Drifting 中，"粒子"不是在像素空间里漂移，而是在**特征空间**里。

**为什么不直接用像素？** 因为像素空间太原始了——两张语义相同但位移一个像素的图片，在像素空间里距离很大，但语义上几乎一样。特征空间能捕捉更有意义的相似性。

```
图像 Drifting:
  图片 → 冻结的 MAE (特征提取器) → 特征向量 [D维]
         ↑                           ↑
    不训练，只用来                 粒子就在这个
    提取有意义的表示               D维空间里漂移

LLM Drifting:
  token序列 → LLM 的 hidden states / softmax(logits) @ E → 特征向量 [D维]
              ↑                                              ↑
         模型前向传播                                    粒子在这个空间里漂移
```

**对于 LLM，什么是好的"特征"？** 有三个天然选择，都完全可微：

```
选项 1: softmax(logits)       → [B, T, V]   概率分布本身就是粒子
选项 2: softmax(logits) @ E   → [B, T, D]   投影到 embedding 空间
选项 3: hidden_states         → [B, T, D]   transformer 中间层表示
```

**选项 1 最直接**：生成的 softmax 概率分布就是粒子，ground truth 的 one-hot
向量就是正样本。drift loss 直接在 R^V 空间上把概率分布往 one-hot 方向推。
不需要任何额外投影。缺点是维度大（V=128k），但粒子数量小，实际可接受。

**选项 2 加了语义结构**：embedding 空间里语义相近的 token 距离近，
drift force 会考虑"差一点也不算太远"。选项 1 里任何两个不同 token 的
one-hot 距离都是 √2，没有这种结构。

**选项 3 最丰富**：不同层的 hidden states 编码不同层级的信息：
- 浅层：语法、词法（像图像的纹理）
- 深层：语义、推理（像图像的高层结构）
- 多层组合 = 多尺度特征，和图像版在 MAE 不同层上做 drift loss 完全类比

三种选项可以**单独用，也可以组合**（每种跑一个独立的 drift loss，加起来）。

### 3.2 正样本从哪来 — LLM 不需要 Memory Bank

图像 Drifting 需要 Memory Bank 是因为：
- 同一个类别（如"金鱼"）有无数种合法的样子
- 需要 32 张参考金鱼图来展示"金鱼分布有多宽"
- 位置 (5,5) 在任何金鱼图片里都代表大致相同的空间区域，可以跨图比较

```
图像 Drifting 的 Memory Bank（有意义）:
┌─────────────────────────────────────────────┐
│  金鱼:   [img1] [img2] ... [img64]          │  ← 32 张金鱼都是合法的正样本
│  鲨鱼:   [img1] [img2] ... [img64]          │     在位置(5,5)上可以跨图比较
└─────────────────────────────────────────────┘
```

**LLM 不一样**：

- 每个 token 位置的"正确答案"取决于它自己的上下文（前面所有 token）
- 别的序列在相同位置上的 token 和当前序列毫无关系
- "The capital of France is [MASK]" 在位置5的正样本只能是 "Paris"
  "I love eating [MASK] for breakfast" 在位置5的正样本只能是 "eggs"
  它们在位置5上不可比

```
LLM 正样本（不需要 Memory Bank）:
  每个位置的正样本就是当前序列自己的 ground truth token。
  一个就够了。

  唯一例外：如果数据集里同一个 prompt 有多条合法回答（如 RLHF 数据），
  这些多回答才是有意义的多正样本。标准 SFT 不需要。
```

**所以对于 LLM Drifting，drift loss 的价值不在于跨序列比较（不像图像），
而在于同一序列的多个候选之间的自排斥力 → 鼓励多样性。**

### 3.3 正样本 vs 负样本 = 吸引 vs 排斥

```
                    drift loss 中的三类粒子

  ┌─────────────┐
  │  Generated   │  ← 生成粒子（gen），模型当前产出，有梯度
  │  ○ ○ ○ ○    │     训练目标：让它们漂移到正确的位置
  └─────────────┘
        │
        ├── 被吸引向 ──→  ┌─────────────┐
        │                  │  Positive    │  ← 正样本（fixed_pos），从 memory bank 采样
        │                  │  ● ● ● ●    │     同类/高质量的真实数据
        │                  └─────────────┘     "这是好的，靠过去"
        │
        └── 被排斥离 ──→  ┌─────────────┐
                           │  Negative    │  ← 负样本（fixed_neg），也从 memory bank 采样
                           │  ✕ ✕ ✕ ✕    │     不同类/无关的数据
                           └─────────────┘     "这是不相关的，远离它"
                           ┌─────────────┐
                           │  Self        │  ← 自身（old_gen），stop_gradient 的自己
                           │  ○'○'○'○'   │     "别和其他生成样本重叠"
                           └─────────────┘     防止模式坍缩
```

### 3.4 多温度核 R_list = 从粗到细的力场

drift loss 用不同的"温度" R 来控制力的作用范围：

```
R = 0.02（小温度，局部精细）        R = 0.2（大温度，全局粗糙）
  只关注最近的几个邻居                关注整个分布的大趋势
  ┌───────────┐                     ┌───────────┐
  │  ○        │                     │  ○        │
  │ ↗ ●      │                     │ ↗ ↗ ● ↗  │
  │   ●       │                     │ ↗  ● ↗   │
  │     ●     │← 远处的●没有力      │   ↗ ● ↗  │← 远处的●也有力
  └───────────┘                     └───────────┘

默认 R_list = (0.02, 0.05, 0.2)
  - 0.02: 精确匹配局部细节（token 级别的精确性）
  - 0.05: 中等范围的语义连贯性
  - 0.2:  全局分布形状匹配
```

三个尺度的力加在一起，保证从宏观到微观都在往正确方向漂移。

---

## 4. Drift-LLM 模型结构

```
                          ┌─────────────────────────────────┐
                          │          Drift-LLM 训练         │
                          └─────────────────────────────────┘

输入：(prompt, ground_truth_response)
同一个 prompt 生成 G 个候选（G = gen_per_prompt，如 4）
从 memory bank 采样 P 条同类参考序列（P = pos_per_sample，如 8）

Step 1: MDLM 随机掩码 — 每个候选独立掩码
────────────────────────────────────────────────────────────

  原始:     [The] [answer] [is]  [42] [.]
  候选 1:   [The] [MASK]   [is]  [MASK] [.]     ← mask pattern 1
  候选 2:   [The] [MASK]   [MASK] [42]  [.]     ← mask pattern 2
  候选 3:   [MASK] [answer] [is]  [MASK] [.]    ← mask pattern 3
  候选 4:   [The] [answer] [MASK] [MASK] [.]    ← mask pattern 4

  每个候选用不同的随机 mask，所以同一个位置上不同候选的预测是不同的。
  这就是"多个粒子"的来源 — 和图像 Drifting 的 gen_per_label=16 完全类比。

Step 2: Generator 批量前向传播
────────────────────────────────────────────────────────────

  把 G 个候选拼成一个大 batch，一次前向：

  [B*G, T] → ┌───────────────────────┐ → logits [B*G, T, V]
              │   Generator (θ)        │
              │   (LLaDA / 可训练LLM)  │
              └───────────────────────┘

  reshape → [B, G, T, V]
  每个 prompt 有 G 个候选，每个候选在每个位置都有一个 softmax 概率分布。

Step 3: 组织粒子 — 每个 token 位置独立（和图像 Drifting 完全相同）
────────────────────────────────────────────────────────────

  图像 Drifting 的 reshape 逻辑：
    rearrange('b x f d -> (b f) x d')
    b=batch, x=图片数(粒子), f=空间位置, d=特征维度
    → 每个空间位置独立跑 drift loss，粒子 = 同位置的不同图片

  LLM 完全类比：
    rearrange('b g t v -> (b t) g v')
    b=batch, g=候选数(粒子), t=token位置, v=特征维度
    → 每个 token 位置独立跑 drift loss，粒子 = 同位置的不同候选

  gen 粒子（有梯度）：
    gen_softmax = softmax(logits)                        # [B, G, T, V]
    gen_feat = rearrange(gen_softmax, 'b g t v -> (b t) g v')  # [B*T, G, V]
    → 在 token 位置 t=0: G 个候选各自的 softmax 向量就是 G 个粒子
    → 在 token 位置 t=1: 同理
    → ...

  pos 粒子（无梯度，从 memory bank 采样的 P 条参考序列）：
    pos_onehot = one_hot(pos_tokens, V)                  # [B, P, T, V]
    pos_feat = rearrange(pos_onehot, 'b p t v -> (b t) p v')  # [B*T, P, V]
    → 每个 token 位置有 P 个 one-hot 向量作为正样本粒子

  不做任何 mean pooling，不压缩序列，每个位置独立。

Step 4: Drift Loss — 每个位置独立计算
────────────────────────────────────────────────────────────

  drift_loss(
      gen  = gen_feat,      # [B*T, G, V]  有梯度
      pos  = pos_feat,      # [B*T, P, V]  无梯度
      neg  = neg_feat,      # [B*T, N, V]  无梯度
      R_list = (0.02, 0.05, 0.2),
  )

  在 token 位置 t=5 上发生的事情：
    - G=4 个候选的 softmax 分布是 4 个粒子
    - P=8 个参考序列在 t=5 的真实 token（one-hot）是 8 个正样本粒子
    - drift force 把 4 个生成分布往 8 个 one-hot 方向推（吸引）
    - 同时 4 个生成分布互相排斥（不能都预测一样的东西 → 多样性）

Step 5: 总 Loss
────────────────────────────────────────────────────────────

  CE Loss（标准 MDLM，token 级别监督）:
    ce_loss = CrossEntropy(logits, ground_truth_tokens, mask=masked_positions)

  Drift Loss（每个位置上的分布级优化）:
    d_loss = drift_loss(gen_feat, pos_feat, neg_feat)

  Total Loss:
    loss = λ_ce * ce_loss + λ_drift * d_loss

  CE loss 管"每个 token 预测得准不准"
  Drift loss 管"多个候选的预测分布要像真实数据，且彼此不同"
```

---

## 5. Drift Loss 是怎么算的（图解）

以一个简化例子说明，假设有 3 个生成粒子和 4 个正样本粒子，在 2D 空间里：

```
Step 1: 计算距离矩阵
───────────────────

  gen (3个)  vs  targets (3个self + 4个pos = 7个)

  距离矩阵 dist[i][j] = ||gen_i - target_j||

        self₀  self₁  self₂  pos₀  pos₁  pos₂  pos₃
  gen₀ [ ∞      0.5    0.8 │  0.3   0.9   1.2   0.7 ]
  gen₁ [ 0.5    ∞      0.6 │  0.8   0.2   0.5   1.0 ]
  gen₂ [ 0.8    0.6    ∞   │  1.0   0.4   0.1   0.3 ]
         ↑                     ↑
         对角线设为 ∞           正样本距离
         (不能自己吸引自己)

Step 2: 对每个温度 R，计算 softmax 亲和力
───────────────────────────────────────

  affinity = softmax(-dist / R)

  R 小 → 只有最近的邻居有大权重 → 局部力
  R 大 → 远处的邻居也有权重     → 全局力

Step 3: 算合力方向
─────────────────

  对于 gen₂，假设离 pos₂ 最近 (dist=0.1):

  吸引力 (来自 pos):
    force_attract = Σ  aff_pos[j] * (pos[j] - gen₂)
                    j                    ↑
                          指向正样本的方向

  排斥力 (来自 self + neg):
    force_repel = Σ  -aff_neg[j] * (neg[j] - gen₂)
                  j                     ↑
                         远离负样本/其他生成样本的方向

  合力 = 吸引力 + 排斥力

Step 4: 确定目标位置
─────────────────

  goal = gen₂_当前位置 + 合力方向
  (goal 全程 stop_gradient，不传梯度)

Step 5: L2 Loss
────────────────

  loss = || gen₂_新位置 - goal ||²

  训练 generator，让 gen₂ 的输出靠近 goal
  ← 梯度只从这个 L2 差值回传到 generator
```

---

## 6. 完整训练流程

```
初始化:
  generator = LLaDA(pretrained)    # 可训练
  G = 4                            # gen_per_prompt: 每个 prompt 几个候选

for step in range(total_steps):

    # ① 读取 batch: B 个 (prompt, response) 对
    prompt, response = next(dataloader)           # response: [B, T]

    # ② MDLM 掩码 — 每个候选独立随机掩码
    input_repeated = repeat(response, 'b t -> (b g) t', g=G)    # [B*G, T]
    t = uniform(ε, 1.0, size=B*G)                 # 每个候选独立采样时间步
    masks = bernoulli(1 - α(t))                    # G 个不同的 mask pattern
    masked_inputs = where(masks, MASK_TOKEN, input_repeated)  # [B*G, T]

    # ③ Generator 一次批量前向（G 个候选一起跑）
    logits = generator(masked_inputs).logits       # [B*G, T, V]
    logits = rearrange(logits, '(b g) t v -> b g t v', g=G)  # [B, G, T, V]

    # ④ CE Loss（标准 MDLM，在所有候选上平均）
    ce_loss = cross_entropy(logits, response, mask=masks).mean()

    # ⑤ 组织粒子 — 每个 token 位置独立
    gen_feat = softmax(logits)                                     # [B, G, T, V]
    gen_feat = rearrange(gen_feat, 'b g t v -> (b t) g v')        # [B*T, G, V]

    # 正样本: 就是自己的 ground truth，每个位置一个 one-hot
    gt_onehot = one_hot(response, V).float()                       # [B, T, V]
    pos_feat = repeat(gt_onehot, 'b t v -> (b t) 1 v')            # [B*T, 1, V]

    # ⑥ Drift Loss — 每个 token 位置独立
    d_loss = drift_loss(
        gen = gen_feat,     # [B*T, G, V] — G 个候选的预测，互相排斥
        pos = pos_feat,     # [B*T, 1, V] — 1 个正样本：自己的 GT
        R_list = (0.02, 0.05, 0.2),
    ).mean()

    # ⑦ 反向传播
    total_loss = ce_loss + λ * d_loss
    total_loss.backward()
    optimizer.step()
```

---

## 7. 和纯 MDLM (LLaDA) 的对比

```
                    纯 MDLM                          Drift-LLM
                    ──────                           ─────────
优化目标:        逐 token CE loss                CE loss + Drift loss
                 "每个 mask 位置猜对 token"       "猜对 token" + "整体分布像真实数据"

多样性:          无显式机制                       自排斥力防止模式坍缩
                 不同 mask 可能收敛到                不同生成结果互相排斥
                 完全相同的输出                      鼓励多样但合理的输出

全局 vs 局部:    纯局部（每个 token 独立优化）    局部（CE）+ 全局（Drift）
                                                  Drift 在序列/chunk 级别做分布匹配

负样本利用:      无                               排斥力远离无关/低质量模式

额外开销:        0                                G 倍前向传播（gen_per_prompt）
                                                  + drift loss 计算（轻量）
```

---

## 8. 文件结构规划

```
drift-llm/
├── dllm/                          # 现有 dLLM 库（不改动）
│   └── dllm/
│       ├── core/
│       │   ├── trainers/
│       │   │   ├── mdlm.py        # 现有 MDLM trainer
│       │   │   └── drift.py       # [新增] DriftLLMTrainer（继承 MDLMTrainer）
│       │   └── losses/
│       │       └── drift_loss.py  # [新增] PyTorch 版 drift loss
│       └── ...
├── drifting/                      # 现有图像 Drifting（参考）
│   ├── drift_loss.py              # JAX 版 drift loss（移植参考）
│   └── ...
├── examples/
│   └── drift/
│       └── sft.py                 # [新增] Drift-LLM SFT 训练脚本
└── DESIGN.md                      # ← 你正在看的这个文件
```

---

## 9. 关键设计决策总结

| 问题 | 决策 | 原因 |
|------|------|------|
| 生成粒子用什么表示？ | softmax(logits) 概率向量（R^V 空间） | 最直接，和 CE loss 哲学一致 |
| 正样本从哪来？ | 当前序列自己的 ground truth（one-hot） | 文本位置没有跨序列可比性，不需要 memory bank |
| 需要 memory bank 吗？ | 标准 SFT 不需要；多回答数据集可选 | 图像需要是因为同类有多种"长法"，文本一般一个 GT |
| gen_per_prompt？ | 至少 4（必须 > 1 才有自排斥） | 自排斥是 drift loss 对 LLM 的核心价值 |
| CE loss 还留吗？ | 留，作为主 loss | Drift loss 是补充，不是替代 |
| 主要额外开销？ | G 倍前向传播 | 可通过 batch 并行缓解 |
