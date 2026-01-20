#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=promptehr_synthetic
#SBATCH --output=logs/promptehr_synthetic_%j.out
#SBATCH --error=logs/promptehr_synthetic_%j.err
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd "$SLURM_SUBMIT_DIR"

cd /u/jalenj4/PyHealth
source pyhealth/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth:$PYTHONPATH

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
mkdir -p logs promptehr_synthetic_output

pip install 'accelerate>=0.26.0' --quiet

echo "Starting PromptEHR pipeline"
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export OUTPUT_DIR="./promptehr_synthetic_output_${TIMESTAMP}"
export MIMIC_ROOT="./data_files"
export MODE="train_and_generate"
export N_SYNTHETIC="1000"
export EPOCHS="1"
export BATCH_SIZE="8"
export LEARNING_RATE="1e-4"
export MAX_VISITS="10"
export MIN_VISITS="2"


if [ "$MODE" = "preprocess_only" ]; then
    python examples/promptehr_mimic3_synthetic_generation.py \
        --mode preprocess_only \
        --mimic_root $MIMIC_ROOT \
        --output_dir $OUTPUT_DIR \
        --max_visits $MAX_VISITS \
        --min_visits $MIN_VISITS \
        --code_vocab_threshold 5 \
        --train_ratio 0.8

elif [ "$MODE" = "generate_only" ]; then
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: MODEL_PATH required for generate_only mode"
        exit 1
    fi
    python examples/promptehr_mimic3_synthetic_generation.py \
        --mode generate_only \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --n_synthetic $N_SYNTHETIC \
        --temperature 1.0

else
    python examples/promptehr_mimic3_synthetic_generation.py \
        --mode train_and_generate \
        --mimic_root $MIMIC_ROOT \
        --output_dir $OUTPUT_DIR \
        --max_visits $MAX_VISITS \
        --min_visits $MIN_VISITS \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --device cuda \
        --n_synthetic $N_SYNTHETIC \
        --temperature 1.0 \
        --code_vocab_threshold 5 \
        --train_ratio 0.8 \
        --include_procedures \
        --include_medications
fi

if [ "$MODE" = "preprocess_only" ]; then
    echo "Preprocessing completed"
    echo "Data in: $OUTPUT_DIR"
elif [ "$MODE" = "generate_only" ]; then
    echo "Generation completed"
    echo "Files in: $OUTPUT_DIR"
else
    echo "Pipeline completed"
    echo "Output: $OUTPUT_DIR"
    echo "Files:"
    echo "  $OUTPUT_DIR/synthetic/promptehr_synthetic_binary.npy"
    echo "  $OUTPUT_DIR/synthetic/promptehr_synthetic_raw.pkl"
    echo "  $OUTPUT_DIR/synthetic/code_mapping.pkl"
    echo "  $OUTPUT_DIR/synthetic/generation_stats.json"
    echo "  $OUTPUT_DIR/synthetic/synthetic_patients_summary.csv"
    echo "  $OUTPUT_DIR/synthetic/synthetic_code_frequencies.csv"
    echo "  $OUTPUT_DIR/synthetic/synthetic_patient_diagnoses_sparse.csv"
fi