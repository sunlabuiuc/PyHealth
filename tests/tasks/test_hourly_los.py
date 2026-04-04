from datetime import datetime, timedelta

import torch

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

    ts = task._make_hourly_tensor(observations, usable_hours=4, num_features=1)

    assert ts[0][0] == 2.0
    assert ts[0][1] == 1.0
    assert ts[1][0] == 2.0
    assert ts[1][1] == 0.0
    assert ts[2][0] == 3.0
    assert ts[2][1] == 1.0


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

    ts = task._make_hourly_tensor(observations, usable_hours=4, num_features=1)

    assert ts[0][2] == 1.0
    assert abs(ts[1][2] - 0.75) < 1e-6
    assert abs(ts[2][2] - (0.75 ** 2)) < 1e-6
    assert abs(ts[3][2] - (0.75 ** 3)) < 1e-6


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

    ts = task._make_cropped_hourly_tensor(
        observations=observations,
        total_hours=3.0,
        num_features=1,
    )

    assert len(ts) == 3
    assert ts[0][0] == 12.0


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
    assert isinstance(sample["target_los_sequence"], torch.Tensor)


def test_mimic_patient_generates_samples() -> None:
    """Test MIMIC-style patient sample generation."""
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
    assert "target_los_hours" in sample
