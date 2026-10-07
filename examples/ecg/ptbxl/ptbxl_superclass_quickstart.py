"""PTB-XL 5-superclass multi-label quickstart.

Requires::

    pip install 'pyhealth[ptbxl]'

Download PTB-XL v1.0.3 from https://physionet.org/content/ptb-xl/1.0.3/
and point ``--root`` at the extracted version directory (contains
``ptbxl_database.csv``, ``scp_statements.csv``, ``records100/``).

Example::

    python examples/ecg/ptbxl/ptbxl_superclass_quickstart.py \\
        --root /data/ptb-xl/1.0.3 --dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyhealth.datasets import PTBXLDataset, split_by_strat_fold
from pyhealth.tasks import PTBXLSuperclassClassification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="PTB-XL version root (ptbxl_database.csv + records*/)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        choices=(100, 500),
        help="Waveform sampling rate (default: 100 Hz / filename_lr)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Limit patients via BaseDataset.dev mode",
    )
    args = parser.parse_args()

    dataset = PTBXLDataset(
        root=str(args.root),
        sampling_rate=args.sampling_rate,
        dev=args.dev,
    )
    task = PTBXLSuperclassClassification(
        scp_statements_path=str(args.root / "scp_statements.csv"),
    )
    samples = dataset.set_task(task)
    train, val, test = split_by_strat_fold(samples)
    print(
        f"samples={len(samples)} "
        f"train={len(train)} val={len(val)} test={len(test)}"
    )
    if len(samples):
        first = samples[0]
        print(
            "first sample:",
            {
                "patient_id": first["patient_id"],
                "record_id": first["record_id"],
                "labels": first["labels"],
                "strat_fold": first["strat_fold"],
                "signal_shape": tuple(first["signal"].shape),
            },
        )


if __name__ == "__main__":
    main()
