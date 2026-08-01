#!/usr/bin/env bash
# HTCondor executable — labs-only RNN mortality run.
#
# Condor equivalent of scripts/will/sunlab/tmux_run_labs_only_rnn_variant.py:
# same task (labs), same model (rnn), same hyperparameter defaults. GPU
# selection (nvidia-smi / CUDA_VISIBLE_DEVICES) is dropped since Condor
# assigns the GPU via request_gpus / cgroups.
#
# usage: run_labs_only_rnn.sh <seed>
# to remove logs: rm -rf logs/condor/*
# to remove cache: rm -rf /shared/eng/wp14/pyhealth_cache_labs/*
set -euo pipefail

SEED="${1:?usage: run_labs_only_rnn.sh <seed>}"

CONDA_ENV="${CONDA_ENV:-pyhealth2}"
PROJECT_DIR="${PROJECT_DIR:-/home/wp14/PyHealth}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
CACHE_DIR="${CACHE_DIR:-/shared/eng/wp14/pyhealth_cache_labs}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wp14/output}"
CONDA_SH="${CONDA_SH:-}"

DEV_MODE="${DEV_MODE:-0}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
RNN_TYPE="${RNN_TYPE:-GRU}"
RNN_LAYERS="${RNN_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
PATIENCE="${PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"

# Condor GPU cgroups can expose a truncated CUDA_VISIBLE_DEVICES UUID that
# distributed's NVML diagnostics can't resolve, crashing LocalCluster startup.
# Skip the distributed Dask cluster (falls back to the plain local scheduler)
# to avoid touching NVML during event-dataframe preprocessing.
export PYHEALTH_DISABLE_DASK_DISTRIBUTED="${PYHEALTH_DISABLE_DASK_DISTRIBUTED:-1}"

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pyhealth-multimodal-labs-only}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

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
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

JOB_TAG="rnn_labs_s${SEED}_c${_CONDOR_CLUSTER_ID:-local}_p${_CONDOR_PROCNO:-0}"

echo "========================================================"
echo "  Labs-only RNN run  |  ${JOB_TAG}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Conda env : ${CONDA_ENV}"
echo "  EHR root  : ${EHR_ROOT}"
echo "  Cache dir : ${CACHE_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Seed      : ${SEED}"
echo "  Dev mode  : ${DEV_MODE}"
echo "========================================================"

if ! python -c "import pyhealth" >/dev/null 2>&1; then
    echo "ERROR: pyhealth is not importable. Run: bash condor_setup.sh" >&2
    exit 1
fi

COMMON=(
    --ehr-root "${EHR_ROOT}"
    --cache-dir "${CACHE_DIR}"
    --task labs
    --model rnn
    --embedding-dim "${EMBEDDING_DIM}"
    --hidden-dim "${HIDDEN_DIM}"
    --rnn-type "${RNN_TYPE}"
    --rnn-layers "${RNN_LAYERS}"
    --dropout "${DROPOUT}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --patience "${PATIENCE}"
    --num-workers "${NUM_WORKERS}"
    --pos-weight "${POS_WEIGHT}"
    --seed "${SEED}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${DEV_MODE}" == "1" ]]; then
    COMMON+=(--dev)
fi

if [[ "${USE_WANDB}" == "1" ]]; then
    COMMON+=(--wandb --wandb-project "${WANDB_PROJECT}")
    if [[ -n "${WANDB_RUN_NAME}" ]]; then
        COMMON+=(--wandb-run-name "${WANDB_RUN_NAME}")
    fi
fi

python examples/mortality_prediction/unified_embedding_e2e_mimic4.py "${COMMON[@]}"

echo "========================================================"
echo "  Completed ${JOB_TAG}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"
