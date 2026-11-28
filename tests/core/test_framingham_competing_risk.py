"""Synthetic tests for FraminghamDataset and FraminghamCompetingRiskPrediction.

Every test uses two to three fabricated participants in a temporary directory.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyhealth.data import Patient
from pyhealth.datasets import FraminghamDataset
from pyhealth.datasets.framingham import (
    _SOURCE_COLUMNS,
    FRAMINGHAM_FEATURES,
    prepare_framingham,
)
from pyhealth.tasks import (
    CompetingRisksPreprocessor,
    FraminghamCompetingRiskPrediction,
)


def _write_prepared(root: Path) -> None:
    rows = []
    for index, (duration, event) in enumerate([(100, 1), (200, 2), (300, 0)]):
        row = {"patient_id": f"p{index}"}
        row.update(
            {
                name: float(index + offset)
                for offset, name in enumerate(FRAMINGHAM_FEATURES)
            }
        )
        row.update({"duration_days": duration, "event": event})
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "framingham.csv", index=False)


def _raw_rows() -> pd.DataFrame:
    """Longitudinal source: p1 has two exams, p2 has a missing glucose."""
    base = {column: 1.0 for column in _SOURCE_COLUMNS}
    rows = [
        # RANDID 1: CVD at day 500; second exam must be ignored.
        {
            **base,
            "RANDID": 1,
            "TIME": 0,
            "TIMECVD": 500,
            "TIMEDTH": 900,
            "CVD": 1,
            "DEATH": 1,
            "GLUCOSE": 80.0,
            "AGE": 40.0,
        },
        {
            **base,
            "RANDID": 1,
            "TIME": 300,
            "TIMECVD": 500,
            "TIMEDTH": 900,
            "CVD": 1,
            "DEATH": 1,
            "GLUCOSE": 999.0,
            "AGE": 41.0,
        },
        # RANDID 2: death without CVD at day 700; glucose missing.
        {
            **base,
            "RANDID": 2,
            "TIME": 0,
            "TIMECVD": 800,
            "TIMEDTH": 700,
            "CVD": 0,
            "DEATH": 1,
            "GLUCOSE": np.nan,
            "AGE": 50.0,
        },
        # RANDID 3: censored at day 1000.
        {
            **base,
            "RANDID": 3,
            "TIME": 0,
            "TIMECVD": 1000,
            "TIMEDTH": 1000,
            "CVD": 0,
            "DEATH": 0,
            "GLUCOSE": 100.0,
            "AGE": 60.0,
        },
    ]
    return pd.DataFrame(rows)


def _load(root: Path, **task_kwargs):
    """Load through the YAML config and apply the task to each patient.

    The table is read synchronously, as in the PBC tests, to keep the test
    fast without starting a multiprocessing ``set_task`` pipeline.
    """
    dataset = FraminghamDataset(root=str(root), cache_dir=root / "cache")
    frame = pl.from_pandas(
        dataset.load_table("framingham").compute(scheduler="synchronous")
    )
    task = FraminghamCompetingRiskPrediction(**task_kwargs)
    samples = []
    for key, partition in frame.partition_by("patient_id", as_dict=True).items():
        samples.extend(task(Patient(str(key[0]), partition)))
    return {sample["patient_id"]: sample for sample in samples}


def test_dataset_and_task_end_to_end(tmp_path):
    _write_prepared(tmp_path)
    samples = _load(tmp_path)
    assert set(samples) == {"p0", "p1", "p2"}
    first = samples["p0"]
    assert len(first["features"]) == len(FRAMINGHAM_FEATURES) == 18
    assert first["features"][0] == pytest.approx(0.0)
    assert first["time"] == pytest.approx(100 / 365.25)
    assert [samples[p]["event"] for p in ("p0", "p1", "p2")] == [1, 2, 0]


def test_samples_feed_competing_risks_preprocessor(tmp_path):
    _write_prepared(tmp_path)
    samples = list(_load(tmp_path, time_unit="days").values())
    processed = CompetingRisksPreprocessor().fit(samples).transform(samples)
    assert max(sample["time"] for sample in processed) == pytest.approx(1.0)
    assert all(sum(sample["missing_mask"]) == 0 for sample in processed)


def test_administrative_censoring(tmp_path):
    _write_prepared(tmp_path)
    samples = _load(tmp_path, time_unit="days", horizon_days=150)
    assert samples["p0"]["event"] == 1
    assert samples["p0"]["time"] == pytest.approx(100)
    assert samples["p1"]["event"] == 0
    assert samples["p1"]["time"] == pytest.approx(150)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_unit": "months"},
        {"horizon_days": 0},
        {"horizon_days": float("nan")},
        {"horizon_days": "100"},
    ],
)
def test_invalid_task_configuration(kwargs):
    with pytest.raises(ValueError):
        FraminghamCompetingRiskPrediction(**kwargs)


def test_prepare_from_longitudinal_source(tmp_path):
    _raw_rows().to_csv(tmp_path / "framingham_longitudinal.csv", index=False)
    prepared = pd.read_csv(prepare_framingham(tmp_path))
    assert prepared["patient_id"].astype(str).tolist() == ["1", "2", "3"]
    # First exam only; cause coding; duration from baseline plus one day.
    assert prepared["age"].tolist() == [40.0, 50.0, 60.0]
    assert prepared["event"].tolist() == [1, 2, 0]
    assert prepared["duration_days"].tolist() == [501.0, 701.0, 1001.0]
    # Missing glucose is mean-imputed from baseline rows only.
    assert prepared.loc[1, "glucose"] == pytest.approx(90.0)
    assert list(prepared.columns[1:19]) == list(FRAMINGHAM_FEATURES)


def test_prepare_is_idempotent(tmp_path):
    _write_prepared(tmp_path)
    before = (tmp_path / "framingham.csv").read_text()
    prepare_framingham(tmp_path)
    assert (tmp_path / "framingham.csv").read_text() == before


def test_missing_source_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_framingham(tmp_path, download=False)


def test_source_missing_columns(tmp_path):
    _raw_rows().drop(columns=["GLUCOSE"]).to_csv(
        tmp_path / "framingham_longitudinal.csv", index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        prepare_framingham(tmp_path)


def test_source_entirely_missing_feature(tmp_path):
    raw = _raw_rows()
    raw["BMI"] = np.nan
    raw.to_csv(tmp_path / "framingham_longitudinal.csv", index=False)
    with pytest.raises(ValueError, match="entirely missing"):
        prepare_framingham(tmp_path)


def test_source_nonpositive_duration(tmp_path):
    raw = _raw_rows()
    raw.loc[raw["RANDID"] == 3, "TIMEDTH"] = -5
    raw.to_csv(tmp_path / "framingham_longitudinal.csv", index=False)
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_framingham(tmp_path)


def _patient(*rows):
    return SimpleNamespace(
        patient_id="stub", get_events=lambda event_type=None: list(rows)
    )


def _row(**overrides):
    row = {name: 1.0 for name in FRAMINGHAM_FEATURES}
    row.update({"duration_days": 365.25, "event": 1})
    row.update(overrides)
    return row


def test_task_skips_unusable_patients():
    task = FraminghamCompetingRiskPrediction()
    assert task(_patient()) == []
    assert task(_patient(_row(glucose=None))) == []
    assert task(_patient(_row(bmi="not a number"))) == []
    assert task(_patient(_row(duration_days=0))) == []
    assert task(_patient(_row(event=3))) == []


def test_task_uses_first_row_and_warns(caplog):
    task = FraminghamCompetingRiskPrediction()
    with caplog.at_level(logging.WARNING):
        (sample,) = task(_patient(_row(event=2), _row(event=1)))
    assert sample["event"] == 2
    assert sample["time"] == pytest.approx(1.0)
    assert "using the first" in caplog.text
