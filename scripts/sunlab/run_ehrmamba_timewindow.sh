#!/usr/bin/env bash
# Run ehrmamba jobs for multiple seeds in tmux sessions
# Usage: bash scripts/run_ehrmamba.sh
set -euo pipefail

SEEDS=(267573289 1872967241 706384748)
TIME_WINDOWS=(24 48 96)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
USER=""

cd "${PROJECT_DIR}"
mkdir -p logs

for seed in "${SEEDS[@]}"; do
    for tw in "${TIME_WINDOWS[@]}"; do
        echo "  Launching ehrmamba seed=${seed} time_window=${tw} in tmux"
        tmux new -s "ehrmamba_${seed}_observationtimewindow_${tw}" "bash -lc '
            conda activate pyhealth2 && \
            CUDA_VISIBLE_DEVICES=4 python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
                --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \
                --cache-dir /home/${USER}/pyhealth_cache \
                --task icd_labs \
                --model ehrmamba \
                --embedding-dim 128 \
                --num-layers 2 \
                --observation-window-hours ${tw} \
                --mamba-state-size 16 \
                --mamba-conv-kernel 4 \
                --epochs 20 \
                --batch-size 8 \
                --seed ${seed} \
                --output-dir /home/${USER}/output/table2/ehrmamba_seed${seed}_observationtimewindow_${tw} \
                2>&1 | tee logs/ehrmamba_seed${seed}_observationtimewindow_${tw}.log; exec bash'"
        echo "  Launched ehrmamba seed=${seed} time_window=${tw} → tmux session: ehrmamba_${seed}_observationtimewindow_${tw}"
    done
done
