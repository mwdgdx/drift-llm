#!/bin/bash
# v3 Text Drifting — run on mcli instance (GPUs 2-7)
set -e

cd "$(dirname "$0")/.."   # drift_llm/

# ---- Install extra deps ----
pip install -q datasets scikit-learn 2>/dev/null || true

# ---- Step 1: Preprocess (single GPU) ----
if [ ! -f data/v3_cache/preprocessed.pt ]; then
    echo "=== Preprocessing (features + clusters) ==="
    CUDA_VISIBLE_DEVICES=2 python -m v3.train --mode preprocess \
        --seq_len 32 --num_clusters 256 --cache_dir data/v3_cache
fi

# ---- Step 2: Train (6 GPUs via DDP) ----
echo "=== Training v3 (6×GPU DDP) ==="
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun \
    --nproc_per_node=6 \
    --master_port=29500 \
    -m v3.train \
    --mode train \
    --seq_len 32 \
    --num_clusters 256 \
    --d_model 768 \
    --n_layers 8 \
    --n_heads 8 \
    --G 16 \
    --P 32 \
    --N 16 \
    --cluster_batch 8 \
    --R_list 0.02 0.05 0.2 \
    --feature_mode direct \
    --lambda_diversity 5.0 \
    --lambda_reg 1.0 \
    --lambda_intra 2.0 \
    --lr 1e-4 \
    --warmup_steps 1000 \
    --max_steps 50000 \
    --log_every 50 \
    --eval_every 2000 \
    --save_every 10000 \
    --output_dir runs/v3 \
    --cache_dir data/v3_cache \
    --wandb_project drift-llm-v3
