# Drift-LLM 实验计划

## 1. 实验目标

验证核心假设：**在一步文本生成中，drift loss + Feature LLM 比 CE loss 更好。**

对标 **FLM (Flow Map Language Models, arXiv 2602.16813)** 的实验设定，
在相同的数据集 (OpenWebText) 和相近的模型规模 (~170M) 上对比。

---

## 2. 两个对比方法

### 方法 A：CE-only（一步 baseline）

- **训练**: prompt(BOS) + Gaussian noise → TextGenerator → logits → CE(logits, GT)
- **推理**: noise → 一次前向 → argmax → 完整序列
- **特点**: 每个 token 位置独立优化，没有跨 token 协调信号

### 方法 B：Drift + Feature LLM（我们的方法）

- **训练**: prompt(BOS) + noise → TextGenerator → logits
  - Bridge: Gumbel-Softmax(logits) → soft_probs @ GPT2.embedding → soft_embeddings
  - Feature LLM: GPT-2(frozen)(soft_embeddings) → contextualized features
  - 正样本: GPT-2(frozen)(GT tokens) → contextualized features (stop gradient)
  - Loss: drift_loss(gen_features, gt_features) + 0.1 × CE(logits, GT)
  - G=4 个候选互相排斥
- **推理**: noise → 一次前向 → argmax（和 CE-only 完全相同）
- **优势**: Feature LLM 的 attention 提供跨 token 上下文 → 序列级训练信号

---

## 3. 实验配置（8×H100, ~5 hours）

### 模型

| 组件 | 规格 | 参数量 |
|------|------|--------|
| Generator (TextGenerator) | 12 层 Transformer, d=768, 12 heads, RoPE | **~170M** |
| Feature LLM | GPT-2 (frozen) | 124M |
| PPL Judge | GPT-2 Large (frozen, 评估用) | 774M |
| Tokenizer | GPT-2 (V=50,257) | — |

### 数据

| 项目 | 配置 |
|------|------|
| 数据集 | **OpenWebText** (Skylion007/openwebtext, streaming) |
| 任务 | 无条件语言建模 (unconditional LM) |
| 序列长度 | 128 tokens (packed sequences) |
| Prompt | 1 token (BOS), response = 整个 128 token 序列 |

### 训练超参

| 参数 | CE-only | Drift+Feat | 说明 |
|------|---------|------------|------|
| GPU | 8×H100 | 8×H100 | torchrun DDP |
| batch_size (per GPU) | 32 | 32 | 总 BS=256 |
| max_steps | **120,000** | **80,000** | CE 快所以多跑 |
| learning_rate | 3e-4 | 3e-4 | cosine decay |
| warmup_steps | 2,500 | 2,500 | |
| G | 1 | 4 | 候选数 |
| lambda_drift | 0.0 | 1.0 | drift loss 权重 |
| lambda_ce | 1.0 | 0.1 | CE loss 权重 |
| tau (Gumbel-Softmax) | — | 1.0→0.5 | 线性退火 |
| R_list | — | [0.5, 1.0, 2.0] | drift kernel 温度 |
| dropout | 0.0 | 0.0 | |

### 时间预估

| 阶段 | 时间 |
|------|------|
| CE-only (120k steps) | ~1 hr |
| Drift+FeatureLLM (80k steps, G=4) | ~3.5 hr |
| 评估 (200 samples, Gen.PPL) | ~20 min |
| **总计** | **~5 hr** |

### 与 FLM 论文对比

| 维度 | FLM (2602.16813) | 我们 | 备注 |
|------|------------------|------|------|
| 模型 | 179M DiT | ~170M Transformer | 同量级 |
| 数据 | OpenWebText | OpenWebText | 相同 |
| 序列长度 | 1024 | 128 | 我们短 8x |
| 训练步数 | 1M + 100k 蒸馏 | 80-120k | 少 10x |
| BS | 512 | 256 | 少 2x |
| 方法 | continuous flow → flow map 蒸馏 | drift loss 直接一步训练 | 不同范式 |
| 评估 | Gen.PPL + Entropy | Gen.PPL + Entropy | 对齐 |

---

## 4. 评估指标

| 指标 | 说明 | 来源 |
|------|------|------|
| **Gen. PPL ↓** | GPT-2 Large 打分的生成困惑度 | FLM 论文主指标 |
| **Entropy ↑** | 平均 per-sample unigram 信息熵 | 检测重复/mode collapse |
| Token Accuracy | argmax(logits) vs GT | 辅助指标 |
| Distinct-1/2 | 词汇多样性 | 辅助指标 |
| Repetition | 连续重复 token 比例 | 辅助指标 |

### FLM 论文参考数值 (OpenWebText, 1024 steps FLM / 1 step FMLM)

```
FLM  (1024 step): Gen.PPL ~62.60,  Entropy ~5.37
FMLM (1 step):    Gen.PPL ~148.72, Entropy ~5.14
MDLM baseline:    Gen.PPL ~95+,    Entropy ~5.3
```

注意：我们用 L=128 (vs 他们 L=1024)，Gen.PPL 数值不直接可比，
但 **Drift vs CE-only 的相对差距**是核心观察。

---

## 5. 预期结果

```
Method                Mode     Gen.PPL   Entropy   TokAcc   Rep
──────────────────────────────────────────────────────────────
CE-only(1step)        1-step   高        低-中     中       高
Drift+Feat(1step)     1-step   较低      中        中-高    低    ← 核心对比
```

**成功标准**:
- Drift+Feat 的 Gen.PPL **明显低于** CE-only（生成质量更好）
- Drift+Feat 的 Entropy **接近或高于** CE-only（没有 mode collapse）
- Drift+Feat 的 Repetition **低于** CE-only（跨 token 协调减少重复）

---

## 6. 运行方式

### 一键运行（推荐）

```bash
git clone <repo> drift-llm && cd drift-llm
bash drift_llm/v2/setup_and_run.sh
```

自动完成：创建 venv → 安装依赖 → 检测 GPU 数量 → 启动实验。

### 手动运行

```bash
cd drift-llm/drift_llm/v2
bash run_experiment.sh
```

### 自定义配置

```bash
# 快速测试（单卡, ~30 min）
NPROC=1 BATCH_SIZE=8 CE_STEPS=5000 DRIFT_STEPS=3000 bash run_experiment.sh

# 完整实验（8卡）
NPROC=8 BATCH_SIZE=32 CE_STEPS=120000 DRIFT_STEPS=80000 bash run_experiment.sh

# 启用 WandB
WANDB_PROJECT=drift-llm bash run_experiment.sh
```

### 单独运行某个方法

```bash
cd drift-llm/drift_llm/v2

# Drift + Feature LLM (8 GPUs)
torchrun --nproc_per_node=8 drift_train.py \
    --dataset owt --feature_model gpt2 \
    --d_model 768 --n_layers 12 --n_heads 12 \
    --G 4 --lambda_drift 1.0 --lambda_ce 0.1 \
    --batch_size 32 --max_steps 80000 \
    --output_dir runs/v2_drift_feat

# CE-only baseline (8 GPUs)
torchrun --nproc_per_node=8 drift_train.py \
    --dataset owt \
    --d_model 768 --n_layers 12 --n_heads 12 \
    --G 1 --lambda_drift 0.0 --lambda_ce 1.0 \
    --batch_size 32 --max_steps 120000 \
    --output_dir runs/v2_ce_only
```

---

## 7. 代码结构

```
drift_llm/v2/
├── setup_and_run.sh      # 一键安装+运行（自动检测 GPU 数量）
├── run_experiment.sh     # 运行 2 个实验 + 评估对比（torchrun DDP）
│
├── drift_train.py        # 核心训练脚本（DDP, OWT 支持）
│                         #   --lambda_drift 0 --G 1            → CE-only
│                         #   --lambda_drift 1 --feature_model gpt2 → Drift + Feature LLM
│
├── diffusion_train.py    # Diffusion-LM baseline（DDP, 可选）
├── eval_compare.py       # 统一评估（Gen.PPL + Entropy + 辅助指标）
│
├── transformer.py        # TextGenerator: 双向 Transformer, RoPE, ~170M
├── data.py               # 数据加载（OpenWebText streaming + Alpaca）
└── runs/                 # 训练输出
    ├── v2_ce_only/
    └── v2_drift_feat/
```

上层依赖（`drift_llm/`）：
- `drift_loss.py` — PyTorch 版 drift loss

---

## 8. 训练流程图解

### CE-only

```
BOS ──→ Embedding ──→ ┐
                       ├── [bos_emb, noise] ──→ TextGenerator(170M) ──→ logits
noise ~ N(0,I) ──→ ───┘                                                  │
                                                                          ↓
                                                         CE(logits, GT) = loss
```

### Drift + Feature LLM

```
BOS ──→ Embedding ──→ ┐
                       ├── [bos_emb, noise] ──→ TextGenerator(170M) ──→ logits
noise ~ N(0,I) ──→ ───┘                                                  │
                                                           ┌──────────────┤
                                                           ↓              ↓
                                                 Gumbel-Softmax     0.1 × CE
                                                           │
                                                 soft_probs @ GPT2.embed
                                                           │
                                                 GPT-2(frozen) forward
                                                           │
                                              gen_features [B×G, 128, 768]
                                                           │
                                       ┌───────────────────┤
                                       ↓                   ↓
                          extract_features          extract_features (GT)
                          (global, chunk)           (global, chunk)
                                │                       │
                                └──── drift_loss ───────┘
                                           │
                                     1.0 × drift
                                           │
                          total_loss = drift + 0.1 × CE
                                           │
                                      backward()
                                           │
                     ┌─────────────────────┤
                     ↓                     ↓
              更新 TextGenerator     GPT-2 不更新
              (唯一训练的模型)       (梯度穿过但冻结)
```

---

## 9. 如果实验失败怎么办

| 现象 | 可能原因 | 下一步 |
|------|---------|--------|
| Drift 和 CE 没区别 | Feature LLM 太小 | 换 gpt2-medium (355M) |
| Drift 和 CE 没区别 | R_list 不合适 | 尝试 (0.02, 0.05, 0.2) |
| Drift 不收敛 | 梯度太大 | 调 lambda_drift，加 grad clip |
| 全生成垃圾 | 步数不够 | 增到 200k+ steps |
| OOM | Feature LLM + G=4 | 减 batch_size 或 G |
| 速度太慢 | Feature LLM 瓶颈 | 减 G 到 2 |

---

## 10. 后续实验（如果成功）

| 阶段 | 实验 | 目的 |
|------|------|------|
| **消融** | Drift-Embed vs Drift+Feat | 验证 Feature LLM 必要性 |
| **消融** | G = 2, 4, 8, 16 | 最优候选数 |
| **消融** | R_list 组合 | kernel 温度敏感性 |
| **扩大** | L=1024 (和 FLM 对齐) | 长序列验证 |
| **扩大** | Generator 350M-870M | 规模效应 |
| **扩大** | Feature LLM → gpt2-large | 更强特征空间 |
| **对比** | 复现 FLM 1-step 结果 | 直接数值对比 |
| **应用** | 推理延迟 benchmark | 实际加速比 |
