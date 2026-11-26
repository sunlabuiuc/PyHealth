"""ICU admission, readmission and prolonged stay tasks on MEDS data.

Runs ``ICUAdmissionMEDS``, ``HospitalReadmissionMEDS`` and
``ProlongedLengthOfStayMEDS`` through ``set_task``, prints each cohort, and
runs one training step of a small RNN on each task. These tasks only need a
code sequence, so any PyHealth sequence model works the same way.

With no arguments the script writes a small **synthetic** MEDS dataset to a
temporary folder. No data access is needed, and its numbers are a smoke test
with no clinical meaning:

    python examples/meds_outcome_tasks.py

To use a real MEDS dataset, such as the open-access MIMIC-IV demo in MEDS
(https://physionet.org/content/mimic-iv-demo-meds/0.0.1/):

    python examples/meds_outcome_tasks.py \\
        --root physionet.org/files/mimic-iv-demo-meds/0.0.1

For reference, Renc et al. (arXiv:2502.06124) report these AUROCs for ETHOS on
all of MIMIC-IV 2.2: ICU admission 0.932 and prolonged stay 0.853. This script
does not reproduce them.
"""

import argparse
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

import pyhealth.datasets.configs as meds_configs
from pyhealth.datasets import MEDSDataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import (
    HospitalReadmissionMEDS,
    ICUAdmissionMEDS,
    ProlongedLengthOfStayMEDS,
)

LAB_CODES = ["LAB//51221//%", "LAB//50912//mg/dL", "LAB//50971//mEq/L"]
DIAGNOSIS_CODES = ["DIAGNOSIS//ICD//10//I10", "DIAGNOSIS//ICD//10//E119"]


def write_synthetic_meds(root: Path, n_subjects: int = 30, seed: int = 0) -> None:
    """Writes a small synthetic MEDS dataset with one to three stays per subject.

    Args:
        root (Path): Output directory. ``data/0.parquet`` is created inside.
        n_subjects (int): Number of synthetic subjects.
        seed (int): Random seed.
    """
    rng = random.Random(seed)
    rows = []
    for subject_id in range(1, n_subjects + 1):
        start = datetime(2024, 1, 1, 8)  # noqa: DTZ001 (MEDS times are tz-naive)
        now = start + timedelta(days=rng.randint(0, 60))
        rows.append((subject_id, None, "GENDER//" + rng.choice("FM"), None))
        for stay in range(rng.randint(1, 3)):
            hadm_id = subject_id * 100 + stay
            length = timedelta(days=rng.uniform(1, 18))
            died = rng.random() < 0.1
            rows.append((subject_id, now, "HOSPITAL_ADMISSION//EW EMER.//ER", hadm_id))
            for _ in range(rng.randint(1, 6)):
                when = now + length * rng.random()
                rows.append((subject_id, when, rng.choice(LAB_CODES), hadm_id))
            if rng.random() < 0.3:
                when = now + length * 0.2
                rows.append((subject_id, when, "ICU_ADMISSION//MICU", hadm_id))
                when = now + length * 0.6
                rows.append((subject_id, when, "ICU_DISCHARGE//MICU", hadm_id))

            discharge = now + length
            location = "DIED" if died else "HOME"
            rows.append(
                (subject_id, discharge, f"HOSPITAL_DISCHARGE//{location}", hadm_id)
            )
            rows.append((subject_id, discharge, rng.choice(DIAGNOSIS_CODES), hadm_id))
            if died:
                rows.append(
                    (subject_id, discharge + timedelta(hours=2), "MEDS_DEATH", None)
                )
                break
            now = discharge + timedelta(days=rng.choice([5, 15, 25, 60, 120]))

    (root / "data").mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
            "code": pl.Series([r[2] for r in rows], dtype=pl.String),
            "numeric_value": pl.Series([None] * len(rows), dtype=pl.Float32),
            "hadm_id": pl.Series([r[3] for r in rows], dtype=pl.Int64),
        }
    ).write_parquet(root / "data" / "0.parquet")


def run_task(dataset, task, label_key: str) -> None:
    """Prints a task's cohort and runs one training step of a small RNN.

    Args:
        dataset (MEDSDataset): The dataset.
        task (BaseTask): The task to apply.
        label_key (str): The task's output key.
    """
    samples = dataset.set_task(task, num_workers=1)
    labels = [int(samples[i][label_key]) for i in range(len(samples))]
    subjects = {samples[i]["patient_id"] for i in range(len(samples))}
    rate = sum(labels) / len(labels) if labels else 0.0
    print(
        f"{task.task_name}: samples={len(labels)} subjects={len(subjects)} "
        f"positive={sum(labels)} rate={rate:.3f}"
    )
    if len(set(labels)) < 2:
        print("  skipping the RNN step: only one label value in this cohort")
        return

    model = RNN(dataset=samples, embedding_dim=16, hidden_dim=16)
    batch = next(iter(get_dataloader(samples, batch_size=16, shuffle=True)))
    output = model(**batch)
    output["loss"].backward()
    print(f"  one RNN step: loss={output['loss'].item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="MEDS dataset root. Omit to use generated synthetic data.",
    )
    args = parser.parse_args()

    config = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        if args.root is None:
            print("Using synthetic MEDS data: a smoke test, not a result.")
            root = temp_path / "synthetic_meds"
            write_synthetic_meds(root)
        else:
            root = Path(args.root)

        dataset = MEDSDataset(
            root=str(root),
            config_path=str(config),
            cache_dir=temp_path / "cache",
            num_workers=1,
        )
        tasks = [
            (ICUAdmissionMEDS(), "icu_admission"),
            (HospitalReadmissionMEDS(), "readmission"),
            (ProlongedLengthOfStayMEDS(), "prolonged_los"),
        ]
        for task, label_key in tasks:
            run_task(dataset, task, label_key)


if __name__ == "__main__":
    main()
