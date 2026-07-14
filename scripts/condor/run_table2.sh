#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: run_table2.sh <model> <seed>}"
SEED="${2:?usage: run_table2.sh <model> <seed>}"

CONDA_ENV="${CONDA_ENV:-pyhealth2}"
PROJECT_DIR="${PROJECT_DIR:-/home/rianatri/PyHealth}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/home/rianatri/pyhealth_cache}"
CONDA_SH="${CONDA_SH:-}"
PYHEALTH_DISABLE_DASK_DISTRIBUTED="${PYHEALTH_DISABLE_DASK_DISTRIBUTED:-1}"
TABLE2_EPOCHS="${TABLE2_EPOCHS:-20}"
TABLE2_NUM_WORKERS="${TABLE2_NUM_WORKERS:-2}"
TABLE2_DEV_MODE="${TABLE2_DEV_MODE:-0}"
TABLE2_TASK="${TABLE2_TASK:-clinical_notes_icd_labs}"
TABLE2_WINDOW_HOURS="${TABLE2_WINDOW_HOURS:-24}"
TABLE2_OUTPUT_DIR="${TABLE2_OUTPUT_DIR:-output/table2}"
TABLE2_RUN_LABEL="${TABLE2_RUN_LABEL:-full}"
TABLE2_FREEZE_ENCODER="${TABLE2_FREEZE_ENCODER:-0}"
TABLE2_ICD_CODES="${TABLE2_ICD_CODES:-0}"
TABLE2_INCLUDE_VITALS="${TABLE2_INCLUDE_VITALS:-0}"
TABLE2_BALANCED_SAMPLING="${TABLE2_BALANCED_SAMPLING:-0}"
TABLE2_BALANCED_RATIO="${TABLE2_BALANCED_RATIO:-1.0}"

resolve_conda_sh() {
    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        echo "${CONDA_SH}"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
            echo "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
        if command -v module >/dev/null 2>&1; then
            module load miniconda3 >/dev/null 2>&1 || true
            module load anaconda3 >/dev/null 2>&1 || true
            if command -v conda >/dev/null 2>&1; then
                local mod_base
                mod_base="$(conda info --base 2>/dev/null || true)"
                if [[ -n "${mod_base}" && -f "${mod_base}/etc/profile.d/conda.sh" ]]; then
                    echo "${mod_base}/etc/profile.d/conda.sh"
                    return 0
                fi
            fi
        fi
    fi
    local user_name home_dir
    user_name="${USER:-$(id -un 2>/dev/null || true)}"
    home_dir="${HOME:-/home/${user_name}}"
    local candidates=(
        "${home_dir}/miniconda3/etc/profile.d/conda.sh"
        "/home/${user_name}/miniconda3/etc/profile.d/conda.sh"
        "${home_dir}/anaconda3/etc/profile.d/conda.sh"
        "/home/${user_name}/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )
    local c
    for c in "${candidates[@]}"; do
        if [[ -f "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done

    local found=""
    found="$(find "${home_dir}" /opt /usr/local /shared -maxdepth 6 -type f -path '*/etc/profile.d/conda.sh' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${found}" && -f "${found}" ]]; then
        echo "${found}"
        return 0
    fi

    return 1
}

CONDA_SH="$(resolve_conda_sh || true)"
if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: conda.sh not found. Set CONDA_SH explicitly." >&2
    exit 1
fi
source "${CONDA_SH}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

JOB_TAG="${MODEL}_seed${SEED}_c${_CONDOR_CLUSTER_ID:-local}_p${_CONDOR_PROCNO:-0}"
JOB_CACHE_DIR="${CACHE_DIR}/${JOB_TAG}"
mkdir -p "${JOB_CACHE_DIR}"

if [[ -n "${_CONDOR_SCRATCH_DIR:-}" ]]; then
    export DASK_TEMPORARY_DIRECTORY="${_CONDOR_SCRATCH_DIR}/dask-${JOB_TAG}"
else
    export DASK_TEMPORARY_DIRECTORY="/tmp/dask-${JOB_TAG}"
fi
mkdir -p "${DASK_TEMPORARY_DIRECTORY}"
# Ensure local repo package is importable even if not installed into env.
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYHEALTH_DISABLE_DASK_DISTRIBUTED
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
echo "  Conda env: ${CONDA_ENV}"
echo "  Conda sh : ${CONDA_SH}"
echo "  EHR root : ${EHR_ROOT}"
echo "  Note root: ${NOTE_ROOT}"
echo "  Cache dir: ${CACHE_DIR}"
echo "  Job cache : ${JOB_CACHE_DIR}"
echo "  Dask temp: ${DASK_TEMPORARY_DIRECTORY}"
echo "  Dask dist: ${PYHEALTH_DISABLE_DASK_DISTRIBUTED} (1=local scheduler)"
echo "  Epochs   : ${TABLE2_EPOCHS}"
echo "  Workers  : ${TABLE2_NUM_WORKERS}"
echo "  Dev mode : ${TABLE2_DEV_MODE}"
echo "  Task      : ${TABLE2_TASK}"
echo "  Window    : ${TABLE2_WINDOW_HOURS}h"
echo "  Output dir: ${TABLE2_OUTPUT_DIR}"
echo "  Patience  : 5 (early stopping)"
echo "  Freeze enc: ${TABLE2_FREEZE_ENCODER} (1=freeze BERT)"
echo "  ICD codes : ${TABLE2_ICD_CODES} (1=include, ablation only)"
echo "  Vitals    : ${TABLE2_INCLUDE_VITALS} (1=include chartevents)"
echo "  Balanced  : ${TABLE2_BALANCED_SAMPLING} (1=undersample negatives)"
echo "  Bal ratio : ${TABLE2_BALANCED_RATIO}"
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
    --patience "${TABLE2_PATIENCE:-5}"
)

if [[ "${TABLE2_DEV_MODE}" == "1" ]]; then
    COMMON+=(--dev "${TABLE2_DEV_COUNT:-1000}")
fi

if [[ "${TABLE2_FREEZE_ENCODER}" == "1" ]]; then
    COMMON+=(--freeze-encoder)
fi

if [[ "${TABLE2_ICD_CODES}" == "1" ]]; then
    COMMON+=(--icd-codes)
fi

if [[ "${TABLE2_INCLUDE_VITALS}" == "1" ]]; then
    COMMON+=(--include-vitals)
fi

if [[ "${TABLE2_BALANCED_SAMPLING}" == "1" ]]; then
    COMMON+=(--balanced-sampling --balanced-ratio "${TABLE2_BALANCED_RATIO}")
fi

if [[ "${TABLE2_DRY_RUN:-0}" == "1" ]]; then
    echo "Dry-run complete: conda activation and argument assembly succeeded."
    exit 0
fi

case "${MODEL}" in
    mlp)
        # No dim overrides needed; A6000/A100 handles bs=16 at embedding-dim=128.
        COMMON+=(--batch-size "${TABLE2_BS_MLP:-16}")
        ;;
    rnn)
        COMMON+=(--batch-size "${TABLE2_BS_RNN:-16}")
        ;;
    transformer)
        # Bumped batch-size 1→2: at bs=1 ~9000s/epoch × 20 = 50h, exceeds deadline.
        # embedding-dim=64 is tiny; bs=2 is safe on A6000 (48GB) and A100 (80GB).
        COMMON+=(
            --batch-size 2
            --embedding-dim 64
            --hidden-dim 64
            --heads 2
            --num-layers 1
        )
        ;;
    bottleneck_transformer)
        # Same timing issue as transformer; bs=2 is safe given embedding-dim=96.
        COMMON+=(
            --batch-size 2
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
            --batch-size 2
            --embedding-dim 96
            --hidden-dim 96
            --mamba-state-size 16
            --mamba-conv-kernel 4
        )
        ;;
    jambaehr)
        # Bumped batch-size 1→2 for same timing reason as transformer.
        # Reduced jamba-mamba-layers 4→2: 3 total layers still a valid Jamba model,
        # saves ~30-40% per-step cost.
        COMMON+=(
            --batch-size 2
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
