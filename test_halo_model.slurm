#!/bin/bash
#SBATCH --account=ethanmr3-ic
#SBATCH --job-name=pyhealth-halo-testing
#SBATCH --output=halo-testing-logs/halo_test_%j.out
#SBATCH --error=halo-testing-logs/halo_test_%j.err
#SBATCH --partition=IllinoisComputes-GPU              # Change to appropriate partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Change to the directory where you submitted the job
cd "$SLURM_SUBMIT_DIR"

# Print useful Slurm environment variables for debugging
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Optional: check what GPU(s) is/are actually visible
echo "Running nvidia-smi to confirm GPU availability:"
nvidia-smi

# Load modules or activate environment
# module load python/3.10
# module load cuda/11.7
# conda activate your-env

# Run your Python training script
python /u/ethanmr3/halo/PyHealth/halo_testing_script.py