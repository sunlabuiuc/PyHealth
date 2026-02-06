#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=promptehr_holdout
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --output=logs/promptehr_train_%j.out
#SBATCH --error=logs/promptehr_train_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

################################################################################
# PromptEHR Training Script with Holdout Set
#
# This script trains the PromptEHR model on 45,520 patients (excluding 1k holdout)
# from MIMIC-III data using the PyHealth framework.
#
# Data Split:
#   - Training: 45,520 patients in /u/jalenj4/pehr_scratch/data_files_train
#   - Holdout: 1,000 patients in /u/jalenj4/pehr_scratch/data_files_holdout
#
# Usage:
#   sbatch examples/promptehr_train_holdout.slurm
#
# Expected Runtime: ~5-6 hours for 20 epochs
################################################################################

# Print job information
echo "========================================"
echo "PromptEHR Training Job (with Holdout)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "========================================"

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"
echo "Working from: $(pwd)"

# Show GPU information
echo ""
echo "GPU Information:"
nvidia-smi
echo "========================================"

# Create necessary directories
mkdir -p logs
mkdir -p promptehr_outputs
mkdir -p promptehr_outputs/checkpoints

# Activate virtual environment
VENV_PATH="/u/jalenj4/pehr_scratch/venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated environment: $VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Print Python and package versions
echo ""
echo "Environment Information:"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import pyhealth; print(f'PyHealth version: {pyhealth.__version__}')" 2>/dev/null || echo "PyHealth: (version unknown)"
echo "========================================"

# Configuration - Using TRAINING data only (45,520 patients)
MIMIC3_ROOT="/u/jalenj4/pehr_scratch/data_files_train"
OUTPUT_DIR="./promptehr_outputs"
NUM_PATIENTS=45520  # Training set only (1k holdout excluded)
BATCH_SIZE=16
NUM_EPOCHS=20
LEARNING_RATE=1e-5
DEVICE="cuda"

echo ""
echo "Training Configuration:"
echo "  MIMIC-III Root: $MIMIC3_ROOT (TRAINING SET)"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Number of Patients: $NUM_PATIENTS (45,520 train / 1,000 holdout)"
echo "  Batch Size: $BATCH_SIZE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo "========================================"

# Run training
echo ""
echo "Starting PromptEHR training on TRAINING SET..."
echo ""

python examples/promptehr_mimic3.py \
    --mimic3_root "$MIMIC3_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_patients $NUM_PATIENTS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device "$DEVICE" \
    --num_synthetic 200 \
    --temperature 0.7

# Capture exit code
EXIT_CODE=$?

# Print completion information
echo ""
echo "========================================"
echo "Training Job Complete"
echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo ""
echo "Outputs:"
echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  Logs: $OUTPUT_DIR/logs/"
echo "  Synthetic Data: $OUTPUT_DIR/synthetic_patients_200.csv"
echo ""
echo "Data Split Info:"
echo "  Training patients: 45,520 ($MIMIC3_ROOT)"
echo "  Holdout patients: 1,000 (/u/jalenj4/pehr_scratch/data_files_holdout)"
echo "========================================"

# Show final checkpoint size
if [ -f "$OUTPUT_DIR/checkpoints/final_model.pt" ]; then
    echo ""
    echo "Final model checkpoint:"
    ls -lh "$OUTPUT_DIR/checkpoints/final_model.pt"
fi

# Show GPU memory usage at end
echo ""
echo "Final GPU Status:"
nvidia-smi

exit $EXIT_CODE
