#!/usr/bin/env bash
# Submit Table 2 jobs for transformer — 3 fixed seeds, sunlab cluster (tmux sessions).
# Seeds are shared across all lab members for aligned comparisons.
# Usage: bash scripts/sunlab/run_transformer.sh
set -euo pipefail

# SEEDS=(267573289 1872967241 706384748)
SEEDS=(44)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
USER="${USER:-wp14}"
CONDA_ENV="${CONDA_ENV:-pyhealth2}"
CUDA_DEVICE="${CUDA_DEVICE:-4}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/home/${USER}/pyhealth_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/${USER}/output/table2}"

cd "${PROJECT_DIR}"

for seed in "${SEEDS[@]}"; do
    session="transformer_seed${seed}"
    log_dir="logs/sunlab/$(date +%Y%m%d)"
    mkdir -p "${log_dir}"
    tmux new-session -d -s "${session}" "bash -lc '
        conda activate ${CONDA_ENV} && \
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
            --ehr-root ${EHR_ROOT} \
            --cache-dir ${CACHE_DIR} \
            --task icd_labs \
            --model transformer \
            --embedding-dim 64 \
            --hidden-dim 64 \
            --heads 2 \
            --num-layers 1 \
            --dropout 0.1 \
            --epochs 20 \
            --batch-size 4 \
            --weight-decay 1e-5 \
            --patience 5 \
            --seed ${seed} \
            --output-dir ${OUTPUT_BASE}/$(date +%Y%m%d) \
            2>&1 | tee ${log_dir}/transformer_seed${seed}.log; exec bash'"
    echo "  Launched transformer seed=${seed} → tmux session: ${session}, log: ${log_dir}/transformer_seed${seed}.log"
done
