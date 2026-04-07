#!/bin/bash
# ============================================================
# Drift-LLM: Training launch script
# Assumes prepare.sh has been run already.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# --- Defaults (override via environment variables) ---
GENERATOR="${GENERATOR:-$MODEL_DIR/LLaDA-8B-Base}"
FEATURE_ENCODER="${FEATURE_ENCODER:-}"        # empty = no Feature LLM (Level 1)
BRIDGE_MODE="${BRIDGE_MODE:-embedding}"        # "embedding" or "gumbel"
DATASET="${DATASET:-tatsu-lab/alpaca}"
G="${G:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-5000}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-128}"
LAMBDA_CE="${LAMBDA_CE:-0.1}"
LAMBDA_DRIFT="${LAMBDA_DRIFT:-1.0}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/runs/drift_llm}"
WANDB_PROJECT="${WANDB_PROJECT:-drift-llm}"
SEED="${SEED:-42}"

echo "============================================"
echo "Drift-LLM: Training"
echo "  Generator:       $GENERATOR"
echo "  Feature Encoder: ${FEATURE_ENCODER:-none (Level 1)}"
echo "  Bridge:          $BRIDGE_MODE"
echo "  G:               $G"
echo "  Batch size:      $BATCH_SIZE"
echo "  Max steps:       $MAX_STEPS"
echo "  Lambda CE:       $LAMBDA_CE"
echo "  Lambda Drift:    $LAMBDA_DRIFT"
echo "  Output:          $OUTPUT_DIR"
echo "  WandB project:   $WANDB_PROJECT"
echo "============================================"

# --- Build command ---
CMD="python $PROJECT_DIR/train.py \
    --model_name $GENERATOR \
    --dataset_name $DATASET \
    --bridge_mode $BRIDGE_MODE \
    --G $G \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --response_length $RESPONSE_LENGTH \
    --lambda_ce $LAMBDA_CE \
    --lambda_drift $LAMBDA_DRIFT \
    --lr $LR \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT \
    --seed $SEED \
    --bf16"

if [ -n "$FEATURE_ENCODER" ]; then
    CMD="$CMD --feature_encoder_name $FEATURE_ENCODER"
fi

echo ""
echo "Running: $CMD"
echo ""
exec $CMD
