"""Preprocess raw TUAB/LEMON EEG recordings and save features to disk.

Reads raw EDF (TUAB) and BrainVision (LEMON) files, applies the EEG-GCNN
preprocessing pipeline, and writes the five pre-computed files required by
EEGGCNNDataset to ``precomputed_data/``:

    psd_features_data_X       joblib array  (N, 48)  PSD band powers
    labels_y                  joblib array  (N,)     "diseased" / "healthy"
    master_metadata_index.csv              (N rows)  patient_ID per window
    spec_coh_values.npy       numpy array  (N, 64)  spectral coherence
    standard_1010.tsv.txt                           electrode coordinates

Expected raw data layout::

    raw_data/
      tuab/train/normal/01_tcp_ar/*.edf
      tuab/train/abnormal/01_tcp_ar/*.edf   (optional)
      lemon/sub-<ID>/sub-<ID>.vhdr

Usage (from the examples/eeg_gcnn directory)::

    conda activate pyhealth

    # Full dataset
    python pre_compute.py --root raw_data --output precomputed_data

    # Sample dataset (3 TUAB + 3 LEMON subjects)
    python pre_compute.py --root sample_raw_data --output precomputed_data

    # Limit subjects for a faster run
    python pre_compute.py --root raw_data --max-tuab 10 --max-lemon 10

After this script completes, run the training pipeline::

    python training_pipeline_shallow_gcnn.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pyhealth.datasets.eeg_gcnn_raw import EEGGCNNRawDataset

DEFAULT_DATA_ROOT  = str(REPO_ROOT / "examples" / "eeg_gcnn" / "raw_data")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "examples" / "eeg_gcnn" / "precomputed_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing raw_data/tuab and raw_data/lemon.",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the five pre-computed output files.",
    )
    p.add_argument(
        "--subset",
        choices=["both", "tuab", "lemon"],
        default="both",
        help="Which data source(s) to process (default: both).",
    )
    p.add_argument(
        "--max-tuab",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N TUAB subjects (default: all).",
    )
    p.add_argument(
        "--max-lemon",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N LEMON subjects (default: all).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: data root not found: {args.root}")
        print("Set --root to a directory containing tuab/ and/or lemon/ subdirectories.")
        sys.exit(1)

    max_tuab  = args.max_tuab
    max_lemon = args.max_lemon

    print(f"\nEEG-GCNN Feature Precomputation")
    print(f"  raw data  : {args.root}")
    print(f"  output    : {args.output}")
    print(f"  subset    : {args.subset}")
    print(f"  max TUAB  : {max_tuab  if max_tuab  is not None else 'all'}")
    print(f"  max LEMON : {max_lemon if max_lemon is not None else 'all'}\n")

    dataset = EEGGCNNRawDataset(root=args.root, subset=args.subset)
    dataset.precompute_features(
        output_dir=args.output,
        max_tuab=max_tuab,
        max_lemon=max_lemon,
    )

    print(f"\nDone. Five pre-computed files written to: {args.output}")
    print("Next step: python training_pipeline_shallow_gcnn.py")


if __name__ == "__main__":
    main()
