#!/bin/bash
#SBATCH --job-name=halo_generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/logs/halo_generate_%j.out

# Canonical SLURM script for generating synthetic data with HALO
# Adjust paths, partition names, and resource allocations for your cluster

# Navigate to working directory
cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Starting HALO generation at: $(date)"
echo "========================================"

# Activate your Python environment
# Example: conda activate pyhealth
# Example: source venv/bin/activate

# Set Python path if needed
# export PYTHONPATH=/path/to/PyHealth:${PYTHONPATH}

# Generation script
python examples/generate_synthetic_mimic3_halo.py \
    --checkpoint_dir /scratch/jalenj4/halo_results/ \
    --checkpoint_file halo_model_best \
    --output_pkl /scratch/jalenj4/halo_results/synthetic/halo_synthetic_10k.pkl \
    --output_csv /scratch/jalenj4/halo_results/synthetic/halo_synthetic_10k.csv \
    --n_samples 10000

echo "========================================"
echo "Generation completed at: $(date)"
