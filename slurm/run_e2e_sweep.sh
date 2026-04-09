#!/bin/bash
# ============================================================
# SLURM array sweep: 4 model heads x 3 seeds on full MIMIC-IV
# 12 jobs total (array indices 0-11)
#
# Submit from ~/PyHealth:
#   sbatch slurm/run_e2e_sweep.sh
# Monitor:
#   squeue -u $USER
#   tail -f logs/e2e_sweep_<JOBID>_<ARRAYID>.out
# ============================================================

#SBATCH --job-name=pyhealth_e2e_sweep
#SBATCH --account=jimeng-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11
#SBATCH --output=/u/%u/PyHealth/logs/e2e_sweep_%A_%a.out
#SBATCH --error=/u/%u/PyHealth/logs/e2e_sweep_%A_%a.err

# ---- Environment setup ----------------------------------------
module load miniconda3/24.9.2
eval "$(conda shell.bash hook)"
conda activate pyhealth2

cd ~/PyHealth

# ---- Data paths -----------------------------------------------
EHR_ROOT="/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2"

# Use local scratch for cache + Dask temp — /u/ is NFS and doesn't support flock
CACHE_DIR="/scratch/${USER}/pyhealth_cache"
OUTPUT_DIR="/u/${USER}/pyhealth_output/e2e_sweep"

# Point Dask scratch to local disk too (avoids flock errors on NFS)
export DASK_TEMPORARY_DIRECTORY="/scratch/${USER}/dask_tmp"
export TMPDIR="/scratch/${USER}/tmp"
mkdir -p "${CACHE_DIR}" "${DASK_TEMPORARY_DIRECTORY}" "${TMPDIR}" "${OUTPUT_DIR}"

# ---- Map array index -> (model, seed) -------------------------
# Index layout:
#   0-2   mlp                    seeds 42 43 44
#   3-5   rnn                    seeds 42 43 44
#   6-8   transformer            seeds 42 43 44
#   9-11  bottleneck_transformer seeds 42 43 44
MODELS=("mlp" "rnn" "transformer" "bottleneck_transformer")
SEEDS=(42 43 44)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "================================================"
echo "Job array ID : ${SLURM_ARRAY_TASK_ID}"
echo "Model        : ${MODEL}"
echo "Seed         : ${SEED}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================"

# ---- Run ------------------------------------------------------
python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
    --ehr-root      "${EHR_ROOT}" \
    --cache-dir     "${CACHE_DIR}" \
    --output-dir    "${OUTPUT_DIR}" \
    --task          stagenet \
    --model         "${MODEL}" \
    --seed          "${SEED}" \
    --epochs        20 \
    --batch-size    64 \
    --lr            1e-3 \
    --embedding-dim 128 \
    --hidden-dim    128 \
    --heads         4 \
    --num-layers    2 \
    --dropout       0.1 \
    --num-workers   "${SLURM_CPUS_PER_TASK}" \
    --device        cuda

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
