"""
Unit tests for the Hourly ICU length-of-stay (LoS) task.

This module contains synthetic unit tests for validating the behavior of the
``HourlyLOSEICU`` task implementation. The tests verify correct construction
of hourly time-series features, target generation, and dataset-specific
handling for both eICU- and MIMIC-IV-style inputs.

Overview:
    The test suite checks:

    1. Formal task schema:
        - ``time_series`` is declared as a timeseries input
        - ``static`` is declared as a tensor input
        - ``target_los_hours`` is declared as a regression output

    2. Hourly time-series construction:
        - Latest observation within each hour is retained
        - Forward-filling of missing values
        - Correct decay feature computation (0.75 ** j)

    3. Pre-ICU handling:
        - Inclusion of pre-ICU observations during processing
        - Proper cropping of pre-ICU rows after feature construction

    4. Sample generation:
        - eICU-style patient processing (offset-based timestamps)
        - MIMIC-IV-style patient processing (datetime-based timestamps)
        - Presence and correctness of expected output fields

Implementation Notes:
    - Tests use lightweight synthetic data for speed and reproducibility.
    - No dependency on real eICU or MIMIC-IV datasets.
    - Designed to validate core preprocessing logic independent of model code.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pyhealth.tasks.hourly_los import HourlyLOSEICU


class DummyEvent:
    """Minimal event object used for task unit tests."""

    def __init__(self, attr_dict, timestamp=None):
        """Initialize the dummy event.

        Args:
            attr_dict: Event attribute dictionary.
            timestamp: Optional event timestamp.
        """
        self.attr_dict = attr_dict
        self.timestamp = timestamp


class DummyPatient:
    """Minimal patient object used for task unit tests."""

    def __init__(self, patient_id, tables):
        """Initialize the dummy patient.

        Args:
            patient_id: Patient identifier.
            tables: Mapping from table name to event list.
        """
        self.patient_id = patient_id
        self.tables = tables

    def get_events(self, table):
        """Return events for the requested table.

        Args:
            table: Table name.

        Returns:
            List of events for that table.
        """
        return self.tables.get(table, [])


def test_hourly_los_declares_expected_schema() -> None:
    """Test that the task declares the expected BaseModel-facing schema."""
    task = HourlyLOSEICU(
        time_series_tables=["lab"],
        time_series_features={"lab": ["creatinine"]},
    )

    assert task.input_schema == {
        "time_series": "tensor",
        "static": "tensor",
    }
    assert task.output_schema == {
        "target_los_hours": "regression",
    }


def test_make_hourly_tensor_keeps_latest_and_forward_fills() -> None:
    """Test that the latest within-hour measurement is kept and gaps are filled."""
    task = HourlyLOSEICU(
        time_series_tables=["lab"],
        time_series_features={"lab": ["creatinine"]},
        min_history_hours=1,
        max_hours=6,
    )

    observations = [
        (0, 0, 1.0, 0.1),
        (0, 0, 2.0, 0.9),  # later in same hour, should win
        (2, 0, 3.0, 2.2),
    ]

    time_series = task._make_hourly_tensor(
        observations=observations,
        usable_hours=4,
        num_features=1,
    )

    assert time_series[0][0] == 2.0
    assert time_series[0][1] == 1.0
    assert time_series[1][0] == 2.0
    assert time_series[1][1] == 0.0
    assert time_series[2][0] == 3.0
    assert time_series[2][1] == 1.0


def test_make_hourly_tensor_decay_behavior() -> None:
    """Test that decay follows the expected ``0.75 ** j`` rule."""
    task = HourlyLOSEICU(
        time_series_tables=["lab"],
        time_series_features={"lab": ["creatinine"]},
        min_history_hours=1,
        max_hours=6,
    )

    observations = [
        (0, 0, 5.0, 0.2),
    ]

    time_series = task._make_hourly_tensor(
        observations=observations,
        usable_hours=4,
        num_features=1,
    )

    assert time_series[0][2] == 1.0
    assert abs(time_series[1][2] - 0.75) < 1e-6
    assert abs(time_series[2][2] - (0.75 ** 2)) < 1e-6
    assert abs(time_series[3][2] - (0.75 ** 3)) < 1e-6


def test_cropped_hourly_tensor_removes_pre_icu_rows() -> None:
    """Test that pre-ICU rows are removed after extended-timeline fill."""
    task = HourlyLOSEICU(
        time_series_tables=["lab"],
        time_series_features={"lab": ["creatinine"]},
        min_history_hours=1,
        max_hours=6,
        pre_icu_hours=2,
    )

    observations = [
        (0, 0, 10.0, -2.0),
        (1, 0, 11.0, -1.0),
        (2, 0, 12.0, 0.0),
    ]

    time_series = task._make_cropped_hourly_tensor(
        observations=observations,
        total_hours=3.0,
        num_features=1,
    )

    assert len(time_series) == 3
    assert time_series[0][0] == 12.0


def test_eicu_patient_generates_samples() -> None:
    """Test eICU-style patient sample generation."""
    task = HourlyLOSEICU(
        time_series_tables=["lab"],
        time_series_features={"lab": ["creatinine"]},
        numeric_static_features=["age"],
        categorical_static_features=["gender"],
        min_history_hours=2,
        max_hours=6,
    )

    patient_event = DummyEvent(
        {
            "patientunitstayid": "stay1",
            "unitdischargeoffset": 240.0,
            "age": 65,
            "gender": "Male",
            "hospitaladmittime24": "08:00:00",
        }
    )

    lab_events = [
        DummyEvent(
            {
                "labname": "creatinine",
                "labresult": 1.2,
                "labresultoffset": 0.0,
            }
        ),
        DummyEvent(
            {
                "labname": "creatinine",
                "labresult": 1.4,
                "labresultoffset": 120.0,
            }
        ),
    ]

    patient = DummyPatient(
        patient_id="p1",
        tables={
            "patient": [patient_event],
            "lab": lab_events,
        },
    )

    samples = task(patient)

    assert len(samples) > 0
    sample = samples[0]

    assert "time_series" in sample
    assert "static" in sample
    assert "target_los_hours" in sample
    assert "target_los_sequence" in sample
    assert isinstance(sample["time_series"], list)
    assert isinstance(sample["static"], list)
    assert isinstance(sample["target_los_hours"], float)
    assert isinstance(sample["target_los_sequence"], list)
    assert len(sample["target_los_sequence"]) == sample["history_hours"]


def test_mimic_patient_generates_samples() -> None:
    """Test MIMIC-IV-style patient sample generation."""
    task = HourlyLOSEICU(
        time_series_tables=["labevents"],
        time_series_features={"labevents": ["creatinine"]},
        min_history_hours=2,
        max_hours=6,
        pre_icu_hours=2,
    )

    intime = datetime(2020, 1, 1, 10, 0, 0)
    outtime = intime + timedelta(hours=4)

    icu_event = DummyEvent(
        {
            "hadm_id": "hadm1",
            "stay_id": "stay1",
            "outtime": outtime.isoformat(),
        },
        timestamp=intime,
    )

    lab_event = DummyEvent(
        {
            "hadm_id": "hadm1",
            "label": "creatinine",
            "valuenum": 1.5,
        },
        timestamp=intime + timedelta(hours=1),
    )

    patient = DummyPatient(
        patient_id="p2",
        tables={
            "patients": [DummyEvent({})],
            "admissions": [DummyEvent({"hadm_id": "hadm1"})],
            "icustays": [icu_event],
            "labevents": [lab_event],
        },
    )

    samples = task(patient)

    assert len(samples) > 0
    sample = samples[0]
    assert "time_series" in sample
    assert "static" in sample
    assert "target_los_hours" in sample
    assert isinstance(sample["target_los_hours"], float)