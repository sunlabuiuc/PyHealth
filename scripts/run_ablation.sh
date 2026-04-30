#!/usr/bin/env bash
# run_ablation.sh — run the seg_weight (λ) ablation sweep on a prepared
# dataset with dataset-appropriate defaults.
#
# Usage:
#   scripts/run_ablation.sh hippocampus [EPOCHS] [LAMBDAS]
#   scripts/run_ablation.sh spleen      [EPOCHS] [LAMBDAS]
#
# Examples:
#   scripts/run_ablation.sh hippocampus                # defaults: 20 epochs, λ∈{0,0.5,1,2}
#   scripts/run_ablation.sh spleen 3 1.0               # 3-epoch smoke at λ=1
#   EPOCHS=30 scripts/run_ablation.sh spleen           # longer run
#
# All ablation knobs can be overridden via env vars. See
# examples/retina_unet_seg_weight_ablation.py for the full list.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-}"
EPOCHS_ARG="${2:-}"
LAMBDAS_ARG="${3:-}"

PYTHON="python"

# Dataset-specific defaults. These match the configs documented in
# IMPLEMENTATION_NOTES.md for the reported results.
case "$DATASET" in
  hippocampus)
    DATA_ROOT="${RUN_DATA_ROOT:-examples/data/hippocampus}"
    HU_WINDOW="${RUN_HU_WINDOW:-}"                  # MR: no windowing
    MIN_SIZE="${RUN_MIN_SIZE:-128}"
    MAX_SIZE="${RUN_MAX_SIZE:-128}"
    ANCHORS="${RUN_ANCHORS:-4,8,16,32,64}"
    OUT_TAG="${RUN_OUT_TAG:-}"                      # default file name
    ;;
  spleen)
    DATA_ROOT="${RUN_DATA_ROOT:-examples/data/spleen}"
    HU_WINDOW="${RUN_HU_WINDOW:--160,240}"          # abdominal CT window
    MIN_SIZE="${RUN_MIN_SIZE:-256}"
    MAX_SIZE="${RUN_MAX_SIZE:-256}"
    ANCHORS="${RUN_ANCHORS:-16,32,64,128,256}"
    OUT_TAG="${RUN_OUT_TAG:-spleen}"
    ;;
  luna16)
    # Lung-nodule detection (subset of LIDC-IDRI, most commonly used
    # benchmark in the Retina U-Net paper's follow-ups). Foreground
    # nodules occupy well under 1% of pixels, so the default
    # seg_pos_weight lifts the seg head out of the all-background
    # collapse that plain BCE falls into at this imbalance.
    DATA_ROOT="${RUN_DATA_ROOT:-examples/data/luna16}"
    HU_WINDOW="${RUN_HU_WINDOW:--1000,400}"         # lung CT window
    MIN_SIZE="${RUN_MIN_SIZE:-256}"
    MAX_SIZE="${RUN_MAX_SIZE:-256}"
    ANCHORS="${RUN_ANCHORS:-4,8,16,32,64}"          # nodules are small
    # Export the pos-weight default so the ablation script picks it up
    # unless the caller explicitly overrides RUN_SEG_POS_WEIGHT.
    export RUN_SEG_POS_WEIGHT="${RUN_SEG_POS_WEIGHT:-100}"
    OUT_TAG="${RUN_OUT_TAG:-luna16}"
    ;;
  *)
    echo "Usage: $0 {hippocampus|spleen|luna16} [EPOCHS] [LAMBDAS]" >&2
    exit 1
    ;;
esac

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] data root not found: $DATA_ROOT" >&2
  echo "[hint]  run: scripts/download_data.sh $DATASET" >&2
  exit 1
fi

EPOCHS="${EPOCHS:-${EPOCHS_ARG:-20}}"
LAMBDAS="${LAMBDAS:-${LAMBDAS_ARG:-0,0.5,1.0,2.0}}"

export RUN_DATA_ROOT="$DATA_ROOT"
export RUN_HU_WINDOW="$HU_WINDOW"
export RUN_EPOCHS="$EPOCHS"
export RUN_LAMBDAS="$LAMBDAS"
export RUN_BATCH_SIZE="${RUN_BATCH_SIZE:-8}"
export RUN_LR="${RUN_LR:-5e-5}"
export RUN_SEED="${RUN_SEED:-42}"
export RUN_MIN_SIZE="$MIN_SIZE"
export RUN_MAX_SIZE="$MAX_SIZE"
export RUN_ANCHORS="$ANCHORS"
export RUN_EVAL_IOU="${RUN_EVAL_IOU:-0.3}"
export RUN_SCORE_THRESH="${RUN_SCORE_THRESH:-0.005}"
export RUN_OUT_TAG="$OUT_TAG"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "[run] dataset=$DATASET  epochs=$EPOCHS  lambdas=$LAMBDAS"
echo "[run] min/max_size=$MIN_SIZE/$MAX_SIZE  anchors=$ANCHORS  hu_window=${HU_WINDOW:-<none>}"

LOG_NAME="ablation_run${OUT_TAG:+_$OUT_TAG}.log"
LOG_DIR="${LOG_DIR:-examples/figures}"
LOG_PATH="$LOG_DIR/$LOG_NAME"
mkdir -p "$LOG_DIR"

# Dry-run hook: print config and exit without spawning Python. Used by
# tests/scripts/test_run_ablation.py so the suite doesn't depend on a
# specific Python stub being on PATH or exiting cleanly.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] skipping python invocation (would run: $PYTHON examples/retina_unet_seg_weight_ablation.py)"
  exit 0
fi

"$PYTHON" examples/retina_unet_seg_weight_ablation.py 2>&1 | tee "$LOG_PATH"
echo "[done] log saved to $LOG_PATH"
