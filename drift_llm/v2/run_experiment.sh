#!/bin/bash
set -e

# ============================================================
# Drift-LLM v2: Quick validation experiment
#
# Trains both Drift and Diffusion-LM from scratch on same arch,
# then compares 1-step and multi-step generation quality.
#
# Expected time: ~2-3 hours total on single A100
# ============================================================

# --- Databricks PyPI proxy (remove if not on Databricks) ---
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi-proxy.dev.databricks.com/simple/}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi-proxy.dev.databricks.com/simple/}"

# --- Config (small model for fast iteration) ---
TOKENIZER="${TOKENIZER:-meta-llama/Llama-2-7b-hf}"
D_MODEL="${D_MODEL:-512}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-3e-4}"
G="${G:-4}"
WANDB="${WANDB_PROJECT:-}"
MAX_PROMPT="${MAX_PROMPT:-128}"
MAX_RESP="${MAX_RESP:-64}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Drift-LLM v2: Quick Validation"
echo "  Model: d=${D_MODEL}, L=${N_LAYERS}, H=${N_HEADS}"
echo "  Steps: ${MAX_STEPS}"
echo "  Batch: ${BATCH_SIZE}"
echo "  G (drift): ${G}"
echo "============================================"

# --- Install deps ---
pip install -q torch transformers datasets wandb 2>/dev/null || true

COMMON_ARGS="--tokenizer $TOKENIZER --d_model $D_MODEL --n_layers $N_LAYERS --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE --max_steps $MAX_STEPS --lr $LR \
    --max_prompt_len $MAX_PROMPT --max_response_len $MAX_RESP \
    --log_every 50 --eval_every 500 --save_every $MAX_STEPS"

WANDB_ARG=""
if [ -n "$WANDB" ]; then
    WANDB_ARG="--wandb_project $WANDB"
fi

# ============================================================
# Step 1: Train Drift model
# ============================================================
echo ""
echo ">>> [1/3] Training DRIFT model..."
echo ""
python drift_train.py $COMMON_ARGS \
    --G $G \
    --lambda_drift 1.0 --lambda_ce 0.1 \
    --output_dir runs/v2_drift \
    $WANDB_ARG

# ============================================================
# Step 2: Train Diffusion-LM baseline
# ============================================================
echo ""
echo ">>> [2/3] Training DIFFUSION-LM baseline..."
echo ""
python diffusion_train.py $COMMON_ARGS \
    --diffusion_steps 50 \
    --output_dir runs/v2_diffusion \
    $WANDB_ARG

# ============================================================
# Step 3: Compare
# ============================================================
echo ""
echo ">>> [3/3] Evaluating both models..."
echo ""
python eval_compare.py \
    --checkpoints runs/v2_drift/step_${MAX_STEPS}.pt runs/v2_diffusion/step_${MAX_STEPS}.pt \
    --labels "Drift" "Diffusion-LM" \
    --tokenizer $TOKENIZER \
    --n_eval 50 \
    --diffusion_steps 50

echo ""
echo "============================================"
echo "DONE! Check logs above for comparison table."
echo "============================================"
