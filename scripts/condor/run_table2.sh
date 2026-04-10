#!/usr/bin/env bash
# scripts/condor/run_table2.sh
# Condor executable for Table 2: Structured EHR + Clinical Notes.
#
# Invoked by batch_table2.sub as:
#   run_table2.sh <model>
#
# Trains the given backbone for 3 seeds (42, 43, 44) sequentially on the
# full MIMIC-IV dataset (no --dev flag).  VRAM usage and per-epoch timing
# are captured in the Condor .out file via the trainer's built-in logging.
#
# ── Fill in these paths before submitting ────────────────────────────────
CONDA_SH="/home/YOUR_USERNAME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="pyhealth"                       # conda env with pyhealth installed
PROJECT_DIR="/home/YOUR_USERNAME/Multimodal-PyHealth"
EHR_ROOT="/local/data/mimic-iv/2.2"       # MIMIC-IV EHR tables (diagnoses_icd, labevents, …)
NOTE_ROOT="/local/data/mimic-iv/note"      # MIMIC-IV notes (discharge, radiology)
# ─────────────────────────────────────────────────────────────────────────

MODEL="${1:?usage: run_table2.sh <model>}"

set -euo pipefail

# Activate conda (Condor jobs have no shell environment)
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Navigate to project root (Condor jobs run in a temp dir)
cd "${PROJECT_DIR}"

echo "========================================================"
echo "  Table 2 run  |  model=${MODEL}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  EHR root : ${EHR_ROOT}"
echo "  Note root: ${NOTE_ROOT}"
echo "  Full MIMIC-IV dataset (no --dev)"
echo "========================================================"

# ── Shared hyperparameters ────────────────────────────────────────
# These are held fixed across all backbones to isolate architecture
# effects (see paper Section 3.6 / Table 2 experimental design).
COMMON=(
    --ehr-root   "${EHR_ROOT}"
    --note-root  "${NOTE_ROOT}"
    --task       clinical_notes_icd_labs
    --model      "${MODEL}"
    --embedding-dim  128
    --hidden-dim     128
    --heads          4
    --num-layers     2
    --dropout        0.1
    --epochs         30
    --batch-size     32
    --weight-decay   1e-5
    --num-workers    4
    --output-dir     "output/table2"
    # --dev is intentionally omitted → full MIMIC-IV
)

# ── Model-specific overrides ──────────────────────────────────────
case "${MODEL}" in
    bottleneck_transformer)
        # Keep BT-specific stability overrides aligned with runner defaults.
        COMMON+=(--max-grad-norm 0.5 --bottlenecks-n 4 --fusion-startidx 1)
        ;;
    ehrmamba)
        COMMON+=(--mamba-state-size 16 --mamba-conv-kernel 4)
        ;;
    jambaehr)
        COMMON+=(--jamba-transformer-layers 2 --jamba-mamba-layers 6
                 --mamba-state-size 16 --mamba-conv-kernel 4)
        ;;
esac

# ── Train 3 seeds sequentially ────────────────────────────────────
# Sequential (not parallel) to avoid OOM on long-sequence models.
# With 80 GB VRAM a single seed of EHRMamba / JambaEHR can use
# 30–50 GB on the full notes corpus.
for SEED in 42 43 44; do
    echo ""
    echo "── seed ${SEED} ──────────────────────────────────────────"
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
        "${COMMON[@]}" --seed "${SEED}"
    echo "── seed ${SEED} done ─────────────────────────────────────"
done

echo ""
echo "========================================================"
echo "  All seeds complete  |  model=${MODEL}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Results: output/table2/${MODEL}_seed{42,43,44}/"
echo "========================================================"
