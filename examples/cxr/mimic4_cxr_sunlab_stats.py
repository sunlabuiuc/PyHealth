"""Example script to load Sunlab MIMIC-CXR through MIMIC4Dataset
and print stats.

Usage:
  python examples/cxr/mimic4_cxr_sunlab_stats.py \
    --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \
        --note-root /shared/rsaas/physionet.org/files/mimic-note \
        --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \
        --cache-dir /shared/eng/pyhealth
"""

import argparse

from pyhealth.datasets import MIMIC4Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Sunlab MIMIC-CXR variant and run dataset.stats()."
    )
    parser.add_argument(
        "--ehr-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/mimiciv/2.2",
        help="Root directory for MIMIC-IV EHR data.",
    )
    parser.add_argument(
        "--note-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/mimic-note",
        help="Root directory for MIMIC-IV notes data.",
    )
    parser.add_argument(
        "--cxr-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/MIMIC-CXR",
        help="Root directory for the Sunlab MIMIC-CXR mirror.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/shared/eng/pyhealth",
        help="Optional cache directory for PyHealth dataset artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for dataset processing.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode (small subset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = MIMIC4Dataset(
        ehr_root=args.ehr_root,
        note_root=args.note_root,
        cxr_root=args.cxr_root,
        cxr_variant="sunlab",
        ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        note_tables=["discharge", "radiology"],
        cxr_tables=["metadata", "negbio", "chexpert", "split"],
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        dev=args.dev,
    )

    # Prints table/patient/event statistics from BaseDataset.
    dataset.stats()


if __name__ == "__main__":
    main()
