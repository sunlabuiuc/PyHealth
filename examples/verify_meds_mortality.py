"""Standalone verification of the MEDS in-hospital mortality cohort.

Applies ``InHospitalMortalityMEDS`` via ``set_task`` on a MEDS dataset and
prints basic cohort counts (expected ~12/238 positives on the public
MIMIC-IV demo). Prefer this path over a per-patient ``task(get_patient)``
loop: ``set_task`` uses the library's lazy loading and parallel processing.

Usage:
    # Download the public demo once (open access, ODbL v1.0):
    #   wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
    python examples/verify_meds_mortality.py \\
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

    samples = dataset.set_task(task)
    n = len(samples)
    n_positive = sum(int(samples[i]["mortality"]) for i in range(n))
    n_patients = len({samples[i]["patient_id"] for i in range(n)})

    print(f"root                 : {args.root}")
    print(f"observation_window   : {args.observation_window}")
    print(f"n_samples (stays)    : {n}")
    print(f"n_patients           : {n_patients}")
    print(f"n_positive (died)    : {n_positive}")
    print(f"positive_rate        : {((n_positive / n) if n else 0.0):.4f}")


if __name__ == "__main__":
    main()
