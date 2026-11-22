# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Support2SurvivalTask on synthetic patients; needs only core dependencies.

The Support2Dataset.set_task path is exercised by
``examples/support2_survival_dcm.py``; a dataset build takes about 3 s, over
the 1 s per-test budget of the contribution guide.
"""

import math
from typing import Any, Optional

import polars as pl
import pytest

from pyhealth.data import Patient
from pyhealth.tasks import Support2SurvivalTask
from pyhealth.tasks.support2_survival import (
    SUPPORT2_DCM_CATEGORICAL,
    SUPPORT2_DCM_FEATURES,
    SUPPORT2_DCM_NUMERIC,
)

EXPECTED_FEATURES = {
    "sex", "dzgroup", "dzclass", "income", "race", "ca", "age", "num.co",
    "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea",
    "sod", "ph", "glucose", "bun", "urine", "adlp", "adls",
}  # fmt: skip


def make_row(**overrides: Any) -> dict[str, Any]:
    """Return one synthetic SUPPORT2 row with every DCM covariate filled."""
    values: dict[str, Any] = {key: f"{key}-value" for key in SUPPORT2_DCM_FEATURES}
    values.update({"d.time": 60.0, "death": 0.0, "surv2m": 0.4, "hospdead": 0.0})
    values.update(overrides)
    return values


def make_patient(rows: list[dict[str, Any]], event_type: str = "support2") -> Patient:
    """Wrap synthetic rows as one PyHealth patient's events."""
    events = [
        {
            "event_type": event_type,
            "timestamp": None,
            **{f"support2/{key}": value for key, value in row.items()},
        }
        for row in rows
    ]
    return Patient("synthetic-1", pl.DataFrame(events))


@pytest.fixture
def task() -> Support2SurvivalTask:
    """Provide a fresh task instance."""
    return Support2SurvivalTask()


def test_schema_and_feature_constants(task: Support2SurvivalTask) -> None:
    """The task exposes raw schemas and the 24 DCM covariates in two groups."""
    assert task.task_name == "support2_censored_survival"
    assert task.input_schema == {"covariates": "raw"}
    assert task.output_schema == {"time": "raw", "event": "raw"}
    assert len(SUPPORT2_DCM_FEATURES) == 24
    assert set(SUPPORT2_DCM_FEATURES) == EXPECTED_FEATURES
    assert SUPPORT2_DCM_FEATURES == SUPPORT2_DCM_CATEGORICAL + SUPPORT2_DCM_NUMERIC


def test_sample_has_all_features_and_labels(task: Support2SurvivalTask) -> None:
    """One row becomes one sample with patient id, 24 covariates and labels."""
    samples = task(make_patient([make_row(age=70.5)]))
    assert len(samples) == 1
    sample = samples[0]
    assert set(sample) == {"patient_id", "covariates", "time", "event"}
    assert sample["patient_id"] == "synthetic-1"
    assert set(sample["covariates"]) == EXPECTED_FEATURES
    assert sample["covariates"]["age"] == 70.5
    for key in SUPPORT2_DCM_CATEGORICAL:
        assert sample["covariates"][key] == f"{key}-value"
    assert sample["time"] == 60.0 and isinstance(sample["time"], float)
    assert sample["event"] == 0 and isinstance(sample["event"], int)


@pytest.mark.parametrize("death,event", [(0.0, 0), (1.0, 1), ("1", 1)])
def test_event_label(task: Support2SurvivalTask, death: Any, event: int) -> None:
    """``death`` maps to the event indicator, including numeric strings."""
    assert task(make_patient([make_row(death=death)]))[0]["event"] == event


def test_outcomes_and_derived_estimates_are_not_inputs(
    task: Support2SurvivalTask,
) -> None:
    """Labels, hospital death and model-derived survival never leak into inputs."""
    covariates = task(make_patient([make_row()]))[0]["covariates"]
    assert covariates.keys().isdisjoint({"death", "d.time", "surv2m", "hospdead"})


def test_missing_covariate_is_none(task: Support2SurvivalTask) -> None:
    """A null or absent covariate is returned as None, not imputed."""
    row = make_row(alb=None)
    del row["urine"]
    covariates = task(make_patient([row]))[0]["covariates"]
    assert covariates["alb"] is None
    assert covariates["urine"] is None
    assert len(covariates) == 24


def test_patient_without_support2_row(task: Support2SurvivalTask) -> None:
    """Patients without a SUPPORT2 event produce no sample."""
    assert task(make_patient([make_row()], event_type="other")) == []


def test_duplicate_rows_raise(task: Support2SurvivalTask) -> None:
    """Two baseline rows for one patient are rejected."""
    with pytest.raises(ValueError, match="exactly one"):
        task(make_patient([make_row(), make_row()]))


@pytest.mark.parametrize("bad_time", [0.0, -5.0, math.nan, math.inf, None, "abc"])
def test_invalid_follow_up_raises(
    task: Support2SurvivalTask, bad_time: Optional[Any]
) -> None:
    """Zero, negative, NaN, infinite, missing or non-numeric d.time raises."""
    row = make_row()
    row["d.time"] = bad_time
    if bad_time is None:
        del row["d.time"]
    with pytest.raises(ValueError, match="d.time"):
        task(make_patient([row]))


@pytest.mark.parametrize("bad_death", [2.0, -1.0, 0.5, None])
def test_invalid_death_raises(task: Support2SurvivalTask, bad_death: Any) -> None:
    """A death indicator outside {0, 1} or missing raises."""
    with pytest.raises(ValueError, match="death"):
        task(make_patient([make_row(death=bad_death)]))
