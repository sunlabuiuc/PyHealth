#!/usr/bin/env bash
# Submit Table 2 jobs for rnn — 3 fixed seeds, sunlab cluster (tmux sessions).
# Seeds are shared across all lab members for aligned comparisons.
# Usage: bash scripts/sunlab/run_rnn.sh
set -euo pipefail

# SEEDS=(267573289 1872967241 706384748)
SEEDS=(42)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
USER="${USER:-wp14}"
CONDA_ENV="${CONDA_ENV:-pyhealth2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/home/${USER}/pyhealth_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/${USER}/output/table2}"

cd "${PROJECT_DIR}"
mkdir -p logs/sunlab

for seed in "${SEEDS[@]}"; do
    session="rnn_seed${seed}"
    echo "  Launching rnn seed=${seed} in tmux session: ${session}"
    tmux new-session -d -s "${session}" "bash -lc '
        conda activate ${CONDA_ENV} && \
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
            --ehr-root ${EHR_ROOT} \
            --cache-dir ${CACHE_DIR} \
            --task icd_labs \
            --model rnn \
            --embedding-dim 128 \
            --hidden-dim 128 \
            --heads 4 \
            --num-layers 2 \
            --dropout 0.1 \
            --epochs 20 \
            --batch-size 16 \
            --weight-decay 1e-5 \
            --patience 5 \
            --seed ${seed} \
            --output-dir ${OUTPUT_BASE}/rnn_seed${seed} \
            2>&1 | tee logs/sunlab/rnn_seed${seed}.log; exec bash'"
    echo "  Launched rnn seed=${seed} → tmux session: ${session}"
done
