#!/usr/bin/env bash
# One-time setup for PyHealth on the UIUC Campus Cluster.
# Run this once from the login node after cloning the repo:
#
#   ssh rianatri@cc-login.campuscluster.illinois.edu
#   cd ~/PyHealth
#   bash scripts/slurm/setup_cc.sh
#
# Pass "clean" to nuke and rebuild the conda env from scratch:
#   bash scripts/slurm/setup_cc.sh clean
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pyhealth2}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CACHE_DIR="${CACHE_DIR:-/u/${USER}/pyhealth_cache}"

cd "${PROJECT_DIR}"

echo "========================================================"
echo "  PyHealth CC setup"
echo "  Conda env : ${CONDA_ENV}"
echo "  Project   : ${PROJECT_DIR}"
echo "  Cache dir : ${CACHE_DIR}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"

# ── Load conda ────────────────────────────────────────────────────
module load miniconda3/24.9.2
eval "$(conda shell.bash hook)"

# ── Clean if requested ────────────────────────────────────────────
if [[ "${1:-}" == "clean" ]]; then
    echo "[clean] Removing existing conda env ${CONDA_ENV}..."
    conda env remove -n "${CONDA_ENV}" -y 2>/dev/null || true
fi

# ── Create env if it doesn't exist ───────────────────────────────
if ! conda env list | grep -q "^${CONDA_ENV}[[:space:]]"; then
    echo "[1/4] Creating conda env ${CONDA_ENV} (Python 3.12)..."
    conda create -n "${CONDA_ENV}" python=3.12 -y
else
    echo "[1/4] Conda env ${CONDA_ENV} already exists, skipping create."
fi

conda activate "${CONDA_ENV}"

# ── Install PyTorch with CUDA 12.x (matches A10 driver on CC) ────
echo "[2/4] Installing PyTorch 2.7.x + CUDA 12.1..."
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── Install pyhealth and all dependencies in editable mode ───────
echo "[3/4] Installing PyHealth (editable) + all deps..."
pip install -e ".[full]" --quiet 2>/dev/null || pip install -e . --quiet

# Install any extras not covered by pyproject.toml
pip install platformdirs filelock --quiet

# ── Pre-download Bio_ClinicalBERT so jobs don't race to HuggingFace ──
echo "[4/4] Pre-caching Bio_ClinicalBERT tokenizer and weights..."
python - <<'PY'
from transformers import AutoTokenizer, AutoModel
print("  Downloading Bio_ClinicalBERT...")
AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("  Done.")
PY

# ── Create cache and log dirs ─────────────────────────────────────
mkdir -p "${CACHE_DIR}"
mkdir -p "${PROJECT_DIR}/logs/slurm"

# ── Smoke test ────────────────────────────────────────────────────
echo ""
echo "Smoke test..."
python -c "
import torch, pyhealth, transformers
print(f'  torch      : {torch.__version__}')
print(f'  cuda avail : {torch.cuda.is_available()}')
print(f'  pyhealth   : {pyhealth.__version__}')
print(f'  transformers: {transformers.__version__}')
import pyhealth.models.ehrmamba, pyhealth.models.jamba_ehr
print('  EHRMamba   : OK')
print('  JambaEHR   : OK')
"

echo ""
echo "========================================================"
echo "  Setup complete. Activate with:"
echo "    module load miniconda3/24.9.2 && conda activate ${CONDA_ENV}"
echo "========================================================"
