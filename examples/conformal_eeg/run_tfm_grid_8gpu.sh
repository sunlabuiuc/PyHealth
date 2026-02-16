#!/usr/bin/env bash
# Run full TFM conformal grid on 8 GPUs (4 alphas × 2 datasets × 4 methods = 32 jobs, 4 waves of 8).
# Usage: set TOK, TUEV_CLF, TUAB_CLF below, then: bash run_tfm_grid_8gpu.sh
# From repo root: bash examples/conformal_eeg/run_tfm_grid_8gpu.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# --- PATHS (structure: .../TFM_Tokenizer_multiple_finetuned_on_TUEV/TFM_Tokenizer_multiple_finetuned_on_TUEV_{seed}/best_model.pth) ---
TOK="${TOK:-/srv/local/data/arjunc4/tfm_tokenizer_last.pth}"
TUEV_CLF="${TUEV_CLF:-/srv/local/data/arjunc4/TFM_Tokenizer_multiple_finetuned_on_TUEV/TFM_Tokenizer_multiple_finetuned_on_TUEV_{seed}/best_model.pth}"
TUAB_CLF="${TUAB_CLF:-/srv/local/data/arjunc4/TFM_Tokenizer_multiple_finetuned_on_TUAB/TFM_Tokenizer_multiple_finetuned_on_TUAB_{seed}/best_model.pth}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1"
  local method="$2"
  local dataset="$3"
  local alpha="$4"
  local script
  case "$method" in
    naive)  script="tuev_naive_cp_conformal.py" ;;
    kde)    script="tuev_kde_cp_conformal.py" ;;
    kmeans) script="tuev_kmeans_conformal.py" ;;
    ncp)    script="tuev_ncp_conformal.py" ;;
    *) echo "Unknown method $method"; exit 1 ;;
  esac
  local clf_path
  [ "$dataset" = tuev ] && clf_path="$TUEV_CLF" || clf_path="$TUAB_CLF"
  CUDA_VISIBLE_DEVICES=$gpu python "examples/conformal_eeg/$script" \
    --dataset "$dataset" --model tfm --alpha "$alpha" \
    --tfm-tokenizer-checkpoint "$TOK" \
    --tfm-classifier-checkpoint "$clf_path" \
    --tfm-skip-train --seeds 1,2,3,4,5 --split-seed 0 \
    --cache-dir "$LOG_DIR/cache_gpu${gpu}" \
    --log-file "$LOG_DIR/${method}_${dataset}_alpha${alpha}.log"
}

# Build list of jobs: method dataset alpha (32 jobs)
jobs=()
for method in naive kde kmeans ncp; do
  for dataset in tuev tuab; do
    for alpha in 0.2 0.1 0.05 0.01; do
      jobs+=("$method $dataset $alpha")
    done
  done
done

# Run 8 at a time (one per GPU)
total=${#jobs[@]}
for ((i = 0; i < total; i += 8)); do
  pids=()
  for ((j = 0; j < 8 && i + j < total; j++)); do
    read -r method dataset alpha <<< "${jobs[i+j]}"
    run_one "$j" "$method" "$dataset" "$alpha" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "Finished batch $((i/8 + 1))/4"
done
echo "All 32 jobs done. Logs in $LOG_DIR"
