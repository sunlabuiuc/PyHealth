#!/usr/bin/env bash
#SBATCH --job-name=table2_cachewarm
#SBATCH --account=jimeng-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/table2_cachewarm_%j.out
#SBATCH --error=logs/slurm/table2_cachewarm_%j.err
# Requests 1 GPU even though cachewarm is CPU-only — eng-research-gpu
# partition requires --gres=gpu to schedule. GPU will sit idle.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pyhealth2}"
PROJECT_DIR="${PROJECT_DIR:-/u/rianatri/PyHealth}"
EHR_ROOT="${EHR_ROOT:-/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/u/${USER}/pyhealth_cache}"
NUM_WORKERS="${TABLE2_NUM_WORKERS:-4}"

module load miniconda3/24.9.2
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYHEALTH_DISABLE_DASK_DISTRIBUTED="${PYHEALTH_DISABLE_DASK_DISTRIBUTED:-1}"
# Never use /tmp for dask temp — it fills up and corrupts shared parquet cache.
export DASK_TEMPORARY_DIRECTORY="${SLURM_TMPDIR:-/u/${USER}/dask_tmp}/dask-cachewarm-${SLURM_JOB_ID:-local}"
mkdir -p "${DASK_TEMPORARY_DIRECTORY}"

echo "========================================================"
echo "  Table 2 cache warm"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Node      : $(hostname)"
echo "  EHR root  : ${EHR_ROOT}"
echo "  Note root : ${NOTE_ROOT}"
echo "  Cache dir : ${CACHE_DIR}"
echo "  Workers   : ${NUM_WORKERS}"
echo "========================================================"

TABLE2_SHARED_CACHE_ROOT="${CACHE_DIR}" \
EHR_ROOT="${EHR_ROOT}" \
NOTE_ROOT="${NOTE_ROOT}" \
TABLE2_CACHE_WARM_NUM_WORKERS="${NUM_WORKERS}" \
TABLE2_DEV_MODE="0" \
    python scripts/condor/warm_table2_cache.py
# Exit 0 even if cache was already warm — ensures --dependency=afterok works.
exit 0

echo "========================================================"
echo "  Cache warm complete"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"
