"""Synthetic dataset and task tests for the Framingham pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from pyhealth.datasets import FraminghamDataset
from pyhealth.tasks import FraminghamCompetingRiskPrediction

FEATURES = [
    "sex",
    "current_smoker",
    "diabetes",
    "bp_meds",
    "education",
    "previous_chd",
    "previous_angina",
    "previous_mi",
    "previous_stroke",
    "previous_hypertension",
    "total_cholesterol",
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cigarettes_per_day",
    "bmi",
    "heart_rate",
    "glucose",
]


def _write_prepared(root: Path) -> None:
    rows = []
    for index, (duration, event) in enumerate([(100, 1), (200, 2), (300, 0)]):
        row = {"patient_id": f"p{index}"}
        row.update(
            {name: float(index + offset) for offset, name in enumerate(FEATURES)}
        )
        row.update({"duration_days": duration, "event": event})
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "framingham.csv", index=False)


def test_dataset_and_task_end_to_end(tmp_path):
    _write_prepared(tmp_path)
    dataset = FraminghamDataset(
        root=str(tmp_path),
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )
    samples = dataset.set_task(
        FraminghamCompetingRiskPrediction(standardize=False),
    )
    assert len(samples) == 3
    first = samples[0]
    assert first["features"].shape == (18,)
    assert first["duration"].item() == pytest.approx(100 / 365.25)
    assert first["event"].item() == 1


def test_administrative_censoring(tmp_path):
    _write_prepared(tmp_path)
    dataset = FraminghamDataset(
        root=str(tmp_path),
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )
    samples = dataset.set_task(
        FraminghamCompetingRiskPrediction(
            time_unit="days", horizon_days=150, standardize=False
        ),
    )
    by_patient = {samples[index]["patient_id"]: samples[index] for index in range(3)}
    assert by_patient["p0"]["event"].item() == 1
    assert by_patient["p1"]["event"].item() == 0
    assert by_patient["p1"]["duration"].item() == pytest.approx(150)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_unit": "months"},
        {"horizon_days": 0},
        {"horizon_days": float("nan")},
    ],
)
def test_invalid_task_configuration(kwargs):
    with pytest.raises(ValueError):
        FraminghamCompetingRiskPrediction(**kwargs)
