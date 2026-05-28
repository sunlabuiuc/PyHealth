#!/usr/bin/env bash
# Submit Table 2 jobs for mlp — 3 fixed seeds, IllinoisComputes-GPU (A100/H200).
# Seeds are shared across all lab members for aligned comparisons.
# Usage: bash scripts/slurm/submit_mlp.sh
set -euo pipefail

SEEDS=(267573289 1872967241 706384748)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ACCOUNT="${SLURM_ACCOUNT:-jimeng-ic}"
PARTITION="${SLURM_PARTITION:-IllinoisComputes-GPU}"

cd "${PROJECT_DIR}"
mkdir -p logs/slurm

for seed in "${SEEDS[@]}"; do
    job=$(sbatch \
        --job-name="t2_mlp_s${seed}" \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --gres=gpu:1 --time=6:00:00 \
        --output="logs/slurm/table2_mlp_seed${seed}_%j.out" \
        --error="logs/slurm/table2_mlp_seed${seed}_%j.err" \
        --export=ALL,CACHE_DIR=/u/rianatri/pyhealth_cache,MODEL=mlp,SEED="${seed}",TABLE2_BS_MLP=16 \
        scripts/slurm/run_table2.sh | awk '{print $NF}')
    echo "  Submitted mlp seed=${seed} → ${job}"
done
