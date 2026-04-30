#!/usr/bin/env bash
# download_data.sh — fetch an MSD dataset, extract N patients, convert to
# the RetinaUNetCTDataset on-disk layout.
#
# Usage:
#   scripts/download_data.sh hippocampus [N_PATIENTS]
#   scripts/download_data.sh spleen      [N_PATIENTS]
#
# Env overrides:
#   PYTHON        — python interpreter (default: ptorch2 env)
#   TAR_CACHE_DIR — where to cache the .tar (default: /tmp/msd_dl)
#
# Idempotent: existing tar is reused; existing per-patient .npy pairs
# are skipped. Requires ``nibabel`` in the target environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-}"
N_PATIENTS="${2:-10}"
PYTHON="python"
TAR_CACHE_DIR="${TAR_CACHE_DIR:-/tmp/msd_dl}"

case "$DATASET" in
  hippocampus)
    TAR_NAME="Task04_Hippocampus.tar"
    TAR_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/${TAR_NAME}"
    TAR_SUBDIR="Task04_Hippocampus"
    PATIENT_STEM="hippocampus"
    OUT_DIR="examples/data/hippocampus"
    TRANSPOSE="no"   # hippocampus volumes already have axis 0 as slice axis
    ;;
  spleen)
    TAR_NAME="Task09_Spleen.tar"
    TAR_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/${TAR_NAME}"
    TAR_SUBDIR="Task09_Spleen"
    PATIENT_STEM="spleen"
    OUT_DIR="examples/data/spleen"
    TRANSPOSE="yes"  # spleen volumes are (512, 512, D); move D to axis 0
    ;;
  *)
    echo "Usage: $0 {hippocampus|spleen} [N_PATIENTS]" >&2
    exit 1
    ;;
esac

# Env override for tests / custom install locations.
OUT_DIR="${DATA_OUT_DIR:-$OUT_DIR}"

TAR_PATH="${TAR_CACHE_DIR}/${TAR_NAME}"

# ---- 1) Download tar if missing ---------------------------------------------
mkdir -p "$TAR_CACHE_DIR"
if [[ ! -f "$TAR_PATH" ]]; then
  echo "[download] fetching $TAR_URL -> $TAR_PATH"
  curl -L --fail -o "$TAR_PATH" "$TAR_URL"
else
  echo "[download] cached tar: $TAR_PATH"
fi

# ---- 2) Extract and convert selected patients -------------------------------
echo "[convert] dataset=$DATASET  n_patients=$N_PATIENTS  out=$OUT_DIR"

"$PYTHON" - <<PY
import os, re, subprocess, pathlib, sys
import numpy as np
try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required. install with: pip install nibabel")

TAR         = "$TAR_PATH"
TAR_SUBDIR  = "$TAR_SUBDIR"
PATIENT_STEM= "$PATIENT_STEM"
OUT_DIR     = pathlib.Path("$OUT_DIR")
N_PATIENTS  = int("$N_PATIENTS")
TRANSPOSE   = "$TRANSPOSE" == "yes"

# List patient IDs in the tar, preserving zero-padding in the archive names.
result = subprocess.run(['tar', 'tf', TAR], capture_output=True, text=True, check=True)
pattern = re.compile(rf'imagesTr/{PATIENT_STEM}_(\d+)\.nii\.gz')
raw_ids = sorted(
    {m.group(1) for line in result.stdout.splitlines()
     if '._' not in line for m in [pattern.search(line)] if m},
    key=lambda s: int(s),  # numeric order, but keep the original padded string
)
pids = raw_ids[:N_PATIENTS]
print(f'[convert] selected patients: {pids}')

# Extract the NIfTIs we need to /tmp
members = [f'{TAR_SUBDIR}/imagesTr/{PATIENT_STEM}_{pid}.nii.gz' for pid in pids] + \
          [f'{TAR_SUBDIR}/labelsTr/{PATIENT_STEM}_{pid}.nii.gz' for pid in pids]
subprocess.run(['tar', 'xf', TAR, '-C', '/tmp'] + members, check=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
added = 0
for pid in pids:
    pdir = OUT_DIR / f'patient_{pid}'
    if (pdir / 'volume.npy').exists() and (pdir / 'mask.npy').exists():
        continue
    img_path = f'/tmp/{TAR_SUBDIR}/imagesTr/{PATIENT_STEM}_{pid}.nii.gz'
    lbl_path = f'/tmp/{TAR_SUBDIR}/labelsTr/{PATIENT_STEM}_{pid}.nii.gz'
    vol = nib.load(img_path).get_fdata().astype(np.float32)
    lbl = nib.load(lbl_path).get_fdata().astype(np.int32)

    if TRANSPOSE:
        vol = np.transpose(vol, (2, 0, 1))
        lbl = np.transpose(lbl, (2, 0, 1))
    else:
        # Hippocampus has the occasional outlier voxel (e.g. patient_204 spikes
        # to ~580000). Clip top 0.5% before saving so downstream windowing is
        # well-behaved.
        vol = np.clip(vol, 0, np.percentile(vol, 99.5))

    pdir.mkdir(exist_ok=True)
    np.save(pdir / 'volume.npy', vol)
    np.save(pdir / 'mask.npy', lbl.astype(np.int8))
    n_lesion = int((lbl.sum(axis=tuple(range(1, lbl.ndim))) > 0).sum())
    print(f'  patient_{pid}: vol {vol.shape} {vol.dtype} HU[{vol.min():.0f},{vol.max():.0f}]  lesion slices={n_lesion}/{vol.shape[0]}')
    added += 1

total = len([p for p in OUT_DIR.iterdir() if p.is_dir()])
print(f'[convert] added {added} patients. {OUT_DIR} now has {total} total.')
PY

echo "[done] $DATASET ready at $OUT_DIR"
