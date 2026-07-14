#!/usr/bin/env bash
# SLURM runner — Table 2, one model + one seed per job.
# Submitted by submit_table2_random.sh via sbatch with --export=MODEL=...,SEED=...
# Resource flags (#SBATCH) are passed on the sbatch command line, not here,
# because heavy vs light models have different memory/time requirements.
set -euo pipefail

MODEL="${MODEL:?MODEL env var required}"
SEED="${SEED:?SEED env var required}"

CONDA_ENV="${CONDA_ENV:-pyhealth2}"
PROJECT_DIR="${PROJECT_DIR:-/u/rianatri/PyHealth}"
EHR_ROOT="${EHR_ROOT:-/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/u/${USER}/pyhealth_cache}"
TABLE2_EPOCHS="${TABLE2_EPOCHS:-20}"
TABLE2_NUM_WORKERS="${TABLE2_NUM_WORKERS:-2}"
TABLE2_DEV_MODE="${TABLE2_DEV_MODE:-0}"
TABLE2_TASK="${TABLE2_TASK:-clinical_notes_icd_labs}"
TABLE2_WINDOW_HOURS="${TABLE2_WINDOW_HOURS:-24}"
TABLE2_OUTPUT_DIR="${TABLE2_OUTPUT_DIR:-output/table2}"
TABLE2_RUN_LABEL="${TABLE2_RUN_LABEL:-full}"
TABLE2_FREEZE_ENCODER="${TABLE2_FREEZE_ENCODER:-0}"
TABLE2_ICD_CODES="${TABLE2_ICD_CODES:-0}"

# ── Activate conda ────────────────────────────────────────────────
module load miniconda3/24.9.2
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

JOB_TAG="${MODEL}_seed${SEED}_j${SLURM_JOB_ID:-local}"

# Use SLURM local scratch if available (--tmp= requested), else shared home.
# Never use /tmp — it fills up and corrupts shared parquet cache.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export DASK_TEMPORARY_DIRECTORY="${SLURM_TMPDIR}/dask-${JOB_TAG}"
else
    export DASK_TEMPORARY_DIRECTORY="/u/${USER}/dask_tmp/dask-${JOB_TAG}"
fi
mkdir -p "${DASK_TEMPORARY_DIRECTORY}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYHEALTH_DISABLE_DASK_DISTRIBUTED="${PYHEALTH_DISABLE_DASK_DISTRIBUTED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

if ! python -c "import pyhealth" >/dev/null 2>&1; then
    echo "ERROR: pyhealth is not importable. Run: bash setup.sh" >&2
    exit 1
fi

echo "========================================================"
echo "  Table 2 run  |  label=${TABLE2_RUN_LABEL}"
echo "  Model     : ${MODEL}"
echo "  Seed      : ${SEED}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  SLURM job : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Conda env : ${CONDA_ENV}"
echo "  EHR root  : ${EHR_ROOT}"
echo "  Note root : ${NOTE_ROOT}"
echo "  Cache dir : ${CACHE_DIR}"
echo "  Dask temp : ${DASK_TEMPORARY_DIRECTORY}"
echo "  Epochs    : ${TABLE2_EPOCHS}"
echo "  Workers   : ${TABLE2_NUM_WORKERS}"
echo "  Dev mode  : ${TABLE2_DEV_MODE}"
echo "  Output dir: ${TABLE2_OUTPUT_DIR}"
echo "  Task      : ${TABLE2_TASK}"
echo "  Window    : ${TABLE2_WINDOW_HOURS}h"
echo "  Patience  : 5 (early stopping)"
echo "  Freeze enc: ${TABLE2_FREEZE_ENCODER} (1=freeze BERT)"
echo "  ICD codes : ${TABLE2_ICD_CODES} (1=include, ablation only)"
echo "========================================================"

COMMON=(
    --ehr-root "${EHR_ROOT}"
    --note-root "${NOTE_ROOT}"
    --cache-dir "${CACHE_DIR}"
    --task "${TABLE2_TASK}"
    --observation-window-hours "${TABLE2_WINDOW_HOURS}"
    --model "${MODEL}"
    --embedding-dim 128
    --hidden-dim 128
    --heads 4
    --num-layers 2
    --dropout 0.1
    --epochs "${TABLE2_EPOCHS}"
    --batch-size 16
    --weight-decay 1e-5
    --num-workers "${TABLE2_NUM_WORKERS}"
    --output-dir "${TABLE2_OUTPUT_DIR}"
    --patience 5
)

if [[ "${TABLE2_DEV_MODE}" == "1" ]]; then
    COMMON+=(--dev)
fi

if [[ "${TABLE2_FREEZE_ENCODER}" == "1" ]]; then
    COMMON+=(--freeze-encoder)
fi

if [[ "${TABLE2_ICD_CODES}" == "1" ]]; then
    COMMON+=(--icd-codes)
fi

if [[ "${TABLE2_DRY_RUN:-0}" == "1" ]]; then
    echo "Dry-run complete."
    exit 0
fi

case "${MODEL}" in
    mlp)
        # BERT encoder dominates VRAM even for lightweight heads.
        # A10 24GB: bs=4 safe. A100/H200: bs=16 default is fine.
        COMMON+=(--batch-size "${TABLE2_BS_MLP:-4}")
        ;;
    rnn)
        COMMON+=(--batch-size "${TABLE2_BS_RNN:-4}")
        ;;
    transformer)
        # Default bs=1 for A10 24GB; override via TABLE2_BS_TRANSFORMER for larger GPUs.
        COMMON+=(
            --batch-size "${TABLE2_BS_TRANSFORMER:-1}"
            --embedding-dim 64
            --hidden-dim 64
            --heads 2
            --num-layers 1
        )
        ;;
    bottleneck_transformer)
        COMMON+=(
            --batch-size "${TABLE2_BS_BOTTLENECK:-1}"
            --embedding-dim 96
            --hidden-dim 96
            --heads 2
            --num-layers 1
            --max-grad-norm 0.5
            --bottlenecks-n 4
            --fusion-startidx 1
        )
        ;;
    ehrmamba)
        COMMON+=(
            --batch-size "${TABLE2_BS_EHRMAMBA:-2}"
            --embedding-dim 96
            --hidden-dim 96
            --mamba-state-size 16
            --mamba-conv-kernel 4
        )
        ;;
    jambaehr)
        COMMON+=(
            --batch-size "${TABLE2_BS_JAMBAEHR:-1}"
            --embedding-dim 64
            --hidden-dim 64
            --jamba-transformer-layers 1
            --jamba-mamba-layers 2
            --mamba-state-size 16
            --mamba-conv-kernel 4
        )
        ;;
esac

python examples/mortality_prediction/unified_embedding_e2e_mimic4.py "${COMMON[@]}" --seed "${SEED}"

echo "========================================================"
echo "  Completed label=${TABLE2_RUN_LABEL} model=${MODEL} seed=${SEED}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"
