#!/bin/bash
# ============================================================
# Drift-LLM: Prepare environment and download models/data
# Run this ONCE on a machine with internet access.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"

echo "============================================"
echo "Drift-LLM: Prepare"
echo "  Project:  $PROJECT_DIR"
echo "  Models:   $MODEL_DIR"
echo "  Data:     $DATA_DIR"
echo "============================================"

# --- Databricks PyPI proxy (MCT clusters block PyPI directly) ---
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi-proxy.dev.databricks.com/simple/}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi-proxy.dev.databricks.com/simple/}"

# --- 1. Install dependencies ---
echo "[1/5] Installing Python dependencies (via $PIP_INDEX_URL)..."
pip install -r "$PROJECT_DIR/requirements.txt"

# --- 2. WandB login ---
echo "[2/5] WandB login..."
if python -c "import wandb; assert wandb.api.default_entity is not None" 2>/dev/null; then
    echo "  Already logged in as: $(python -c 'import wandb; print(wandb.api.default_entity)')"
else
    echo "  Not logged in. Running: wandb login"
    wandb login
fi

# --- 3. Download Generator (LLaDA) ---
echo "[3/5] Downloading Generator model..."
GENERATOR="${GENERATOR:-GSAI-ML/LLaDA-8B-Base}"
GEN_LOCAL="$MODEL_DIR/$(basename $GENERATOR)"
if [ -d "$GEN_LOCAL" ]; then
    echo "  Already exists: $GEN_LOCAL"
else
    echo "  Downloading $GENERATOR → $GEN_LOCAL"
    huggingface-cli download "$GENERATOR" --local-dir "$GEN_LOCAL"
fi

# --- 4. Download Feature Encoder (optional) ---
echo "[4/5] Downloading Feature Encoder..."
FEATURE_ENCODER="${FEATURE_ENCODER:-intfloat/e5-large-v2}"
FE_LOCAL="$MODEL_DIR/$(basename $FEATURE_ENCODER)"
if [ -d "$FE_LOCAL" ]; then
    echo "  Already exists: $FE_LOCAL"
else
    echo "  Downloading $FEATURE_ENCODER → $FE_LOCAL"
    huggingface-cli download "$FEATURE_ENCODER" --local-dir "$FE_LOCAL"
fi

# --- 5. Download dataset ---
echo "[5/5] Downloading dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('tatsu-lab/alpaca', split='train', cache_dir='$DATA_DIR')
print(f'  Dataset loaded: {len(ds)} examples')
"

echo ""
echo "============================================"
echo "Done! Models and data are ready."
echo ""
echo "Generator:       $GEN_LOCAL"
echo "Feature Encoder: $FE_LOCAL"
echo "Data:            $DATA_DIR"
echo ""
echo "Next: run scripts/run.sh"
echo "============================================"
