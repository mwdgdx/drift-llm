# Drift-LLM

One-step text generation via particle drifting.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start: Signal Validation (Exp 1)

Test if drift loss produces meaningful gradients — no training needed:

```bash
# Quick test with GPT-2 (CPU ok, ~5 min)
python exp1_signal.py --model_name gpt2 --n_samples 5 --G 4 --T_r 32

# Real experiment with LLaDA (needs GPU)
python exp1_signal.py --model_name GSAI-ML/LLaDA-8B-Base --n_samples 20 --G 8 --T_r 64
```

## Training

```bash
# Level 1: Embedding projection (no Feature Encoder, cheapest)
python train.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --bridge_mode embedding \
    --G 4 \
    --lambda_ce 0.1 \
    --lambda_drift 1.0 \
    --batch_size 2 \
    --max_steps 5000

# Level 2: Gumbel-Softmax + external Feature Encoder
python train.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --feature_encoder_name intfloat/e5-large-v2 \
    --bridge_mode gumbel \
    --G 4 \
    --batch_size 2
```

## Architecture

```
Training:
  prompt + noise → Generator (LLaDA) → logits
    → Gumbel-Softmax → soft_probs @ Embedding → Feature Encoder → gen_feat
  GT tokens → Feature Encoder → gt_feat
  drift_loss(gen_feat, gt_feat) → backward → Generator params

Inference:
  prompt + noise → Generator → argmax → text
  (Feature Encoder not needed)
```

## Files

- `drift_loss.py` — Drift loss (ported from JAX)
- `bridge.py` — Gumbel-Softmax / MLP bridges
- `feature_encoder.py` — Feature Encoder wrappers (embedding / transformer)
- `generator.py` — One-step generator wrapper
- `train.py` — Training script
- `exp1_signal.py` — Signal validation experiment
