# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Tests for MIMIC-Extract CaliForest task
"""Tests for MIMICExtractCaliForestTask."""

import pytest

from pyhealth.tasks.mimic_extract_califorest import (
    MIMICExtractCaliForestTask,
)


class _FakePatient:
    """Minimal patient stub for testing."""

    def __init__(self, pid, features, mort_hosp, mort_icu, los_3, los_7):
        self.patient_id = pid
        self.features = features
        self.mort_hosp = mort_hosp
        self.mort_icu = mort_icu
        self.los_3 = los_3
        self.los_7 = los_7


class TestTaskInit:
    def test_valid_targets(self):
        for t in ("mort_hosp", "mort_icu", "los_3", "los_7"):
            task = MIMICExtractCaliForestTask(target=t)
            assert task.target == t
            assert t in task.task_name

    def test_invalid_target(self):
        with pytest.raises(ValueError):
            MIMICExtractCaliForestTask(target="invalid")

    def test_schema(self):
        task = MIMICExtractCaliForestTask()
        assert "features" in task.input_schema
        assert "label" in task.output_schema


class TestTaskCall:
    def test_produces_sample(self):
        task = MIMICExtractCaliForestTask(target="mort_hosp")
        patient = _FakePatient("p0", [0.1] * 10, 1, 0, 1, 0)
        samples = task(patient)
        assert len(samples) == 1
        assert samples[0]["label"] == 1
        assert samples[0]["features"] == [0.1] * 10

    def test_different_targets(self):
        patient = _FakePatient("p0", [0.5] * 5, 0, 1, 1, 0)
        assert MIMICExtractCaliForestTask("mort_hosp")(patient)[0]["label"] == 0
        assert MIMICExtractCaliForestTask("mort_icu")(patient)[0]["label"] == 1
        assert MIMICExtractCaliForestTask("los_3")(patient)[0]["label"] == 1
        assert MIMICExtractCaliForestTask("los_7")(patient)[0]["label"] == 0

    def test_missing_features(self):
        task = MIMICExtractCaliForestTask()
        patient = _FakePatient("p0", None, 1, 0, 0, 0)
        assert task(patient) == []

    def test_missing_label(self):
        task = MIMICExtractCaliForestTask(target="mort_hosp")
        patient = _FakePatient("p0", [1.0], None, 0, 0, 0)
        assert task(patient) == []
