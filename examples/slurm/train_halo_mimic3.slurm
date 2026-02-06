#!/bin/bash
#SBATCH --job-name=halo_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/halo_train_%j.out

# Canonical SLURM script for training HALO on MIMIC-III
# Adjust paths, partition names, and resource allocations for your cluster

# Navigate to working directory
cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Starting HALO training at: $(date)"
echo "========================================"

# Activate your Python environment
# Example: conda activate pyhealth
# Example: source venv/bin/activate

# Set Python path if needed
# export PYTHONPATH=/path/to/PyHealth:${PYTHONPATH}

# Training script
python examples/halo_mimic3_training.py \
    --mimic3_dir /u/jalenj4/pehr_scratch/data_files_train/ \
    --output_dir /scratch/jalenj4/halo_results/ \
    --epochs 80 \
    --batch_size 48 \
    --learning_rate 0.0001 \
    --save_best \
    --save_final

echo "========================================"
echo "Training completed at: $(date)"
