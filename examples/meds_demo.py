"""End-to-end example: loading a MEDS dataset with PyHealth.

This example uses the public *MIMIC-IV demo data in the Medical Event Data
Standard (MEDS)* (PhysioNet, v0.0.1, ODbL v1.0, ~100 subjects):
https://doi.org/10.13026/t2y8-ea41

Download it once (open access, ~a few MB):

    wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/

Then run:

    python examples/meds_demo.py \\
        --root physionet.org/files/mimic-iv-demo-meds/0.0.1

Any dataset following the MEDS layout (``data/**.parquet`` +
``metadata/subject_splits.parquet``) works the same way. See the MEDS
specification: https://github.com/Medical-Event-Data-Standard/meds
"""

import argparse

import polars as pl

from pyhealth.datasets import MEDSDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Root of the MEDS dataset (directory containing data/ and "
        "metadata/)",
    )
    parser.add_argument(
        "--subset",
        default="train",
        help="Split to load as a subset (default: train)",
    )
    args = parser.parse_args()

    # 1) Load the full dataset: every Parquet shard under data/ is read,
    #    including nested split directories (data/<split>/<shard>.parquet).
    dataset = MEDSDataset(root=args.root)
    dataset.stats()

    # 2) Peek at the canonical event frame (typed straight from Parquet:
    #    string patient ids, datetime64[ms] timestamps, float values).
    events = dataset.global_event_df
    print(events.head(5).collect())

    # 3) Load a split-restricted subset. Subjects are selected through the
    #    metadata/subject_splits.parquet assignment; each subset uses its
    #    own processing cache.
    subset = MEDSDataset(root=args.root, subset=args.subset)
    n_subset = len(subset.unique_patient_ids)
    n_total = len(dataset.unique_patient_ids)
    print(f"Subjects in subset '{args.subset}': {n_subset} / {n_total}")

    # 4) Static (null-time) MEDS events, e.g. demographics, are preserved.
    n_static = (
        events.filter(pl.col("timestamp").is_null())
        .select(pl.len())
        .collect()
        .item()
    )
    print(f"Static (null-time) events: {n_static}")

    # From here, the dataset behaves like any other PyHealth dataset: use
    # `dataset.set_task(...)` with an existing task to build samples.


if __name__ == "__main__":
    # BaseDataset spawns Dask worker processes; keep the main-module guard.
    main()
