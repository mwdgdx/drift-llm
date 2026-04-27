#!/bin/bash
set -e

# ============================================================
# Drift-LLM: One-click setup & run on remote GPU machine
#
# Usage:
#   git clone <repo> && cd drift-llm
#   bash drift_llm/v2/setup_and_run.sh
#
# Optionally override config:
#   NPROC=4 DRIFT_STEPS=50000 bash drift_llm/v2/setup_and_run.sh
#
# Requirements: CUDA GPUs, Python 3.10+, pip
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================"
echo "Drift-LLM: Setup & Run"
echo "  Project dir: $PROJECT_DIR"
echo "============================================"

# --- Step 1: Create venv if not exists ---
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo ">>> Python: $(python --version) at $(which python)"

# --- Step 2: Install dependencies ---
echo ">>> Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers datasets einops wandb

# --- Step 3: Verify GPU ---
python -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs detected: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    [{i}] {name} ({mem:.1f} GB)')
if n == 0:
    print('  WARNING: No GPU detected! Training will be very slow.')
"

# Auto-detect number of GPUs if not set
if [ -z "$NPROC" ]; then
    export NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
    echo ">>> Auto-detected $NPROC GPUs"
fi

# --- Step 4: Run the experiment ---
echo ""
echo ">>> Starting experiment (${NPROC} GPUs)..."
cd "$SCRIPT_DIR"
exec bash run_experiment.sh
