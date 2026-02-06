#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --partition=ic-express
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=promptehr_gen_10k
#SBATCH --output=logs/promptehr_gen_10k_%j.out
#SBATCH --error=logs/promptehr_gen_10k_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2@gmail.com

# Exit on error
set -e
set -o pipefail

# Load modules
module purge
module load gcc/11.2.0 || true
module load cuda/12.6

# Environment setup
VENV_PATH="/u/jalenj4/pehr_scratch/venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated environment: $VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Change to project directory
cd /u/jalenj4/final/PyHealth

# Create logs directory
mkdir -p logs

# Print environment info
echo "==================== Environment Info ===================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# Run generation for 10,000 patients with optimal parameters
# Checkpoint: best_model_fixed.pt (20 epochs, Jan 29 2026)
# Optimal params from diagnostic: alpha=2.0, temp=1.0 (R²=0.61)
python3 examples/promptehr_mimic3.py \
    --mimic3_root /u/jalenj4/pehr_scratch/data_files \
    --output_dir ./promptehr_outputs \
    --checkpoint /scratch/jalenj4/promptehr_checkpoints/best_model_fixed.pt \
    --generate_only \
    --num_synthetic 10000 \
    --num_patients 1000 \
    --temperature 1.0 \
    --alpha 2.0 \
    --device cuda

echo "Generation completed at $(date)"
