"""Train Delphi-2M on synthetic MEDS and MIMIC-IV disease trajectories.

The two tasks share one schema (``codes``, ``ages`` -> ``next_code``,
``time_to_next``), so one Delphi model trains on both. No credentialed
UK Biobank or PhysioNet download.

Paper: https://doi.org/10.1038/s41586-025-09529-3
Code: https://github.com/gerstung-lab/Delphi

Run from the PyHealth repository root:
    python examples/delphi_disease_trajectory.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Delphi
from pyhealth.tasks import DelphiTrajectoryMEDS, DelphiTrajectoryMIMIC4
from pyhealth.trainer import Trainer

torch.set_num_threads(1)


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _patient(patient_id, rows):
    frame = pl.from_dicts(rows, infer_schema_length=None)
    frame = frame.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(name).cast(pl.String) for name in frame.columns if "/" in name),
    )
    return Patient(patient_id=patient_id, data_source=frame)


def _meds_samples() -> list[dict]:
    """In-memory MEDS patients. ``MEDSDataset.set_task`` is the on-disk path."""
    birth = datetime(2020, 1, 1)
    task = DelphiTrajectoryMEDS()
    samples: list[dict] = []
    for subject, codes in (("s1", ("A", "B", "C")), ("s2", ("B", "C", "A"))):
        rows = [_event("meds", birth, code="MEDS_BIRTH")]
        for step, code in enumerate(codes, start=1):
            rows.append(_event("meds", birth + timedelta(days=30 * step), code=code))
        samples.extend(task(_patient(subject, rows)))
    print(f"MEDS samples: {len(samples)}")
    return samples


def _mimic_samples() -> list[dict]:
    """Two synthetic MIMIC-IV patients, including one in-hospital death."""
    start = datetime(2150, 6, 1, 8, 0, 0)
    died = _patient(
        "m1",
        [
            _event("patients", None, anchor_age=70, anchor_year=2150, dod=None),
            _event(
                "admissions",
                start,
                hadm_id=1,
                deathtime=(start + timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S"),
            ),
            _event("diagnoses_icd", start, hadm_id=1, icd_code="I10", seq_num=1),
            _event("diagnoses_icd", start, hadm_id=1, icd_code="E11", seq_num=2),
            _event(
                "admissions",
                start + timedelta(days=400),
                hadm_id=2,
                deathtime=None,
            ),
            _event(
                "diagnoses_icd",
                start + timedelta(days=400),
                hadm_id=2,
                icd_code="I50",
                seq_num=1,
            ),
        ],
    )
    alive = _patient(
        "m2",
        [
            _event("patients", None, anchor_age=55, anchor_year=2150, dod=None),
            _event("admissions", start, hadm_id=3, deathtime=None),
            _event("diagnoses_icd", start, hadm_id=3, icd_code="J18", seq_num=1),
            _event(
                "diagnoses_icd",
                start + timedelta(days=10),
                hadm_id=4,
                icd_code="N18",
                seq_num=1,
            ),
            _event("admissions", start + timedelta(days=10), hadm_id=4, deathtime=None),
        ],
    )
    task = DelphiTrajectoryMIMIC4()
    samples = task(died) + task(alive)
    shown = samples[-1]
    print(
        "MIMIC-IV trajectory:",
        shown["codes"],
        "->",
        shown["next_code"],
        f"in {shown['time_to_next']:.1f} days",
    )
    return samples


def main() -> None:
    samples = _meds_samples() + _mimic_samples()
    dataset = create_sample_dataset(
        samples,
        {"codes": "sequence", "ages": "tensor"},
        {"next_code": "multiclass", "time_to_next": "tensor"},
        dataset_name="delphi_example",
        task_name="DelphiTrajectory",
    )
    model = Delphi(
        dataset, n_embd=16, n_layer=1, n_head=2, dropout=0.0, mask_ties=True
    )
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    trainer = Trainer(
        model=model, metrics=["accuracy"], device="cpu", enable_logging=False
    )
    trainer.train(loader, epochs=1, monitor=None)
    scores = trainer.evaluate(loader)
    print(f"evaluation accuracy={scores['accuracy']:.3f} loss={scores['loss']:.3f}")
    batch = next(iter(loader))
    codes, ages = model.generate_trajectory(
        batch["codes"], batch["ages"], max_new_tokens=2, top_k=2
    )
    valid = codes[0] != 0
    print("simulated codes:", codes[0][valid].tolist())
    print("simulated ages:", [round(float(age), 2) for age in ages[0][valid]])


if __name__ == "__main__":
    main()
