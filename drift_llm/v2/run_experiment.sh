#!/bin/bash
set -e

# ============================================================
# Drift-LLM v2: Scaled Experiment (8×H100, ~5 hours total)
#
# Trains 2 methods on the same 170M architecture + OpenWebText:
#   1. CE-only:          one-step, trained with CE loss only      (~1 hr)
#   2. Drift+FeatureLLM: one-step, drift loss in GPT-2 space     (~3.5 hr)
#
# Then evaluates with Gen. PPL (GPT-2 Large) + Entropy.
#
# Usage:
#   torchrun --nproc_per_node=8 drift_train.py ...   (multi-GPU)
#   bash run_experiment.sh                            (orchestrates all)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Config ---
TOKENIZER="${TOKENIZER:-gpt2}"
FEATURE_MODEL="${FEATURE_MODEL:-gpt2}"
DATASET="${DATASET:-owt}"
D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"           # per-GPU
SEQ_LEN="${SEQ_LEN:-128}"
NPROC="${NPROC:-8}"                      # number of GPUs
WANDB="${WANDB_PROJECT:-}"

# Step budget: CE-only is ~4x faster (no Feature LLM, G=1)
CE_STEPS="${CE_STEPS:-120000}"
DRIFT_STEPS="${DRIFT_STEPS:-80000}"
DIFF_STEPS="${DIFF_STEPS:-50}"           # diffusion inference steps

# Drift hyperparams
G="${G:-4}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-2500}"
SAVE_EVERY="${SAVE_EVERY:-20000}"

echo "============================================"
echo "Drift-LLM v2: Scaled Experiment"
echo "  GPUs:         ${NPROC}"
echo "  Generator:    d=${D_MODEL}, L=${N_LAYERS}, H=${N_HEADS}"
echo "  Feature LLM:  ${FEATURE_MODEL}"
echo "  Dataset:      ${DATASET}, seq_len=${SEQ_LEN}"
echo "  CE steps:     ${CE_STEPS}"
echo "  Drift steps:  ${DRIFT_STEPS}"
echo "  Batch/GPU:    ${BATCH_SIZE} (total: $((BATCH_SIZE * NPROC)))"
echo "  G (drift):    ${G}"
echo "============================================"

COMMON_ARGS="--tokenizer $TOKENIZER --dataset $DATASET \
    --d_model $D_MODEL --n_layers $N_LAYERS --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE --max_response_len $SEQ_LEN \
    --lr $LR --warmup_steps $WARMUP --dropout 0.0 \
    --log_every 100 --eval_every 5000 --save_every $SAVE_EVERY"

WANDB_ARG=""
if [ -n "$WANDB" ]; then
    WANDB_ARG="--wandb_project $WANDB"
fi

TORCHRUN="torchrun --nproc_per_node=$NPROC"

# ============================================================
# Step 1: CE-only baseline (one-step, no drift) — ~1 hr
# ============================================================
echo ""
echo ">>> [1/3] Training CE-ONLY baseline (one-step)..."
echo ""
$TORCHRUN drift_train.py $COMMON_ARGS \
    --G 1 \
    --lambda_drift 0.0 --lambda_ce 1.0 \
    --max_steps $CE_STEPS \
    --output_dir runs/v2_ce_only \
    $WANDB_ARG

# ============================================================
# Step 2: Drift + Feature LLM (our method) — ~3.5 hr
# ============================================================
echo ""
echo ">>> [2/3] Training DRIFT + Feature LLM..."
echo ""
$TORCHRUN drift_train.py $COMMON_ARGS \
    --feature_model "$FEATURE_MODEL" \
    --G $G \
    --lambda_drift 1.0 --lambda_ce 0.1 \
    --max_steps $DRIFT_STEPS \
    --output_dir runs/v2_drift_feat \
    $WANDB_ARG

# ============================================================
# Step 3: Evaluate both
# ============================================================
echo ""
echo ">>> [3/3] Evaluating all models..."
echo ""

# Find the last checkpoint for each run
CE_CKPT="runs/v2_ce_only/step_${CE_STEPS}.pt"
DRIFT_CKPT="runs/v2_drift_feat/step_${DRIFT_STEPS}.pt"

EVAL_WANDB_ARG=""
if [ -n "$WANDB" ]; then
    EVAL_WANDB_ARG="--wandb_project $WANDB"
fi

python eval_compare.py \
    --checkpoints "$CE_CKPT" "$DRIFT_CKPT" \
    --labels "CE-only(1step)" "Drift+Feat(1step)" \
    --tokenizer $TOKENIZER \
    --dataset $DATASET \
    --ppl_model gpt2-large \
    --n_eval 200 \
    --diffusion_steps $DIFF_STEPS \
    $EVAL_WANDB_ARG

echo ""
echo "============================================"
echo "DONE! Check logs above for comparison table."
echo "============================================"
