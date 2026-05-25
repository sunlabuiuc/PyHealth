#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: run_table2.sh <model>}"

CONDA_ENV="${CONDA_ENV:-pyhealth2}"
PROJECT_DIR="${PROJECT_DIR:-/home/rianatri/PyHealth}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/shared/eng/pyhealth_cache}"
CONDA_SH="${CONDA_SH:-}"

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

# Isolate per-job cache/temp paths so concurrent Condor jobs do not race
# on the same tmp directory tree.
JOB_TAG="${MODEL}_c${_CONDOR_CLUSTER_ID:-local}_p${_CONDOR_PROCNO:-0}"
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

if ! python -c "import pyhealth" >/dev/null 2>&1; then
    echo "ERROR: pyhealth is not importable. Run: bash setup.sh" >&2
    exit 1
fi

echo "========================================================"
echo "  Table 2 run  |  model=${MODEL}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Conda env: ${CONDA_ENV}"
echo "  Conda sh : ${CONDA_SH}"
echo "  EHR root : ${EHR_ROOT}"
echo "  Note root: ${NOTE_ROOT}"
echo "  Cache dir: ${CACHE_DIR}"
echo "  Job cache: ${JOB_CACHE_DIR}"
echo "  Dask temp: ${DASK_TEMPORARY_DIRECTORY}"
echo "========================================================"

COMMON=(
    --ehr-root "${EHR_ROOT}"
    --note-root "${NOTE_ROOT}"
    --cache-dir "${JOB_CACHE_DIR}"
    --task clinical_notes_icd_labs
    --model "${MODEL}"
    --embedding-dim 128
    --hidden-dim 128
    --heads 4
    --num-layers 2
    --dropout 0.1
    --epochs 30
    --batch-size 32
    --weight-decay 1e-5
    --num-workers 4
    --output-dir output/table2
)

if [[ "${TABLE2_DRY_RUN:-0}" == "1" ]]; then
    echo "Dry-run complete: conda activation and argument assembly succeeded."
    exit 0
fi

case "${MODEL}" in
    bottleneck_transformer)
        COMMON+=(--max-grad-norm 0.5 --bottlenecks-n 4 --fusion-startidx 1)
        ;;
    ehrmamba)
        COMMON+=(--mamba-state-size 16 --mamba-conv-kernel 4)
        ;;
    jambaehr)
        COMMON+=(--jamba-transformer-layers 2 --jamba-mamba-layers 6 --mamba-state-size 16 --mamba-conv-kernel 4)
        ;;
esac

# Sample 3 distinct random seeds each run.
mapfile -t SEEDS < <(
    python - <<'PY'
import random

for seed in random.sample(range(1, 2_147_483_647), 3):
    print(seed)
PY
)

if [[ "${#SEEDS[@]}" -ne 3 ]]; then
    echo "ERROR: failed to generate 3 random seeds." >&2
    exit 1
fi

echo "  Seeds    : ${SEEDS[*]}"
echo "========================================================"

for SEED in "${SEEDS[@]}"; do
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py "${COMMON[@]}" --seed "${SEED}"
done

echo "========================================================"
echo "  Completed model=${MODEL}"
echo "  Seeds used: ${SEEDS[*]}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"
