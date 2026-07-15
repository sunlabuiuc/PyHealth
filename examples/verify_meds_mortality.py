"""Standalone verification of the MEDS in-hospital mortality cohort.

Prints the cohort summary produced by ``InHospitalMortalityMEDS`` on a MEDS
dataset, so the positive rate (expected ~12/238 on the public MIMIC-IV demo)
can be confirmed through the actual task pipeline rather than only at the raw
Parquet level.

Usage:
    # Download the public demo once (open access, ODbL v1.0):
    #   wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
    python verify_meds_mortality.py \\
        --root physionet.org/files/mimic-iv-demo-meds/0.0.1

The task needs hadm_id; this script uses the bundled
``configs/meds_with_hadm.yaml`` automatically.
"""

import argparse
from pathlib import Path

import pyhealth.datasets.configs as meds_configs
from pyhealth.datasets import MEDSDataset
from pyhealth.tasks import InHospitalMortalityMEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Root of the MEDS dataset (contains data/ and metadata/).",
    )
    parser.add_argument(
        "--observation-window",
        default="full_stay",
        choices=["full_stay", "first_hours"],
    )
    parser.add_argument("--window-hours", type=float, default=48.0)
    args = parser.parse_args()

    cfg = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
    dataset = MEDSDataset(root=args.root, config_path=str(cfg))
    task = InHospitalMortalityMEDS(
        observation_window=args.observation_window,
        window_hours=args.window_hours,
    )

    # Apply per patient to keep raw (untokenized) samples, so the label
    # counts are directly inspectable. set_task would tokenize the codes.
    samples = []
    for patient_id in dataset.unique_patient_ids:
        samples.extend(task(dataset.get_patient(patient_id)))

    summary = InHospitalMortalityMEDS.summarize(samples)
    print(f"root                 : {args.root}")
    print(f"observation_window   : {args.observation_window}")
    print(f"n_samples (stays)    : {summary['n_samples']}")
    print(f"n_patients           : {summary['n_patients']}")
    print(f"n_positive (died)    : {summary['n_positive']}")
    print(f"positive_rate        : {summary['positive_rate']:.4f}")
    print(f"mean_sequence_length : {summary['mean_sequence_length']:.1f}")

    # Cross-check the intended user path returns the same sample count.
    sample_dataset = dataset.set_task(task)
    print(f"set_task sample count : {len(sample_dataset)} (should equal n_samples)")


if __name__ == "__main__":
    main()
