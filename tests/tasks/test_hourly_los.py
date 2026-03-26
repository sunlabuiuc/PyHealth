import torch
from pyhealth.tasks.hourly_los import HourlyLOSEICU


class DummyEvent:
    def __init__(self, attr_dict):
        self.attr_dict = attr_dict


class DummyPatient:
    def __init__(self, patient_id, patient_attr, lab_events):
        self.patient_id = patient_id
        self._events = {
            "patient": [DummyEvent(patient_attr)],
            "lab": [DummyEvent(e) for e in lab_events],
        }

    def get_events(self, table_name):
        return self._events.get(table_name, [])


def build_task():
    return HourlyLOSEICU(
        time_series_tables=["lab"],
        time_series_features={"lab": ["test_feature"]},
        numeric_static_features=[],
        categorical_static_features=[],
        min_history_hours=1,
        max_hours=10,
    )


def test_latest_measurement_within_hour():
    task = build_task()

    patient = DummyPatient(
        "p1",
        {"unitdischargeoffset": 600},  # 10 hours
        [
            {"labname": "test_feature", "labresult": 1.0, "labresultoffset": 30},
            {"labname": "test_feature", "labresult": 5.0, "labresultoffset": 50},
        ],
    )

    samples = task(patient)

    first_ts = samples[0]["time_series"][0]
    value = first_ts[0]

    assert value == 5.0


def test_forward_fill_behavior():
    task = build_task()

    patient = DummyPatient(
        "p2",
        {"unitdischargeoffset": 600},
        [
            {"labname": "test_feature", "labresult": 7.0, "labresultoffset": 30},
        ],
    )

    samples = task(patient)

    # use the sample with 2 hours of history
    ts = samples[1]["time_series"]

    # hour 0 observed
    assert ts[0][0] == 7.0
    assert ts[0][1] == 1.0

    # hour 1 forward-filled
    assert ts[1][0] == 7.0
    assert ts[1][1] == 0.0


def test_decay_increases():
    task = build_task()

    patient = DummyPatient(
        "p3",
        {"unitdischargeoffset": 600},
        [
            {"labname": "test_feature", "labresult": 3.0, "labresultoffset": 30},
        ],
    )

    samples = task(patient)

    # use the sample with 3 hours of history
    ts = samples[2]["time_series"]

    decay_values = [row[2] for row in ts]

    assert decay_values[0] == 0.0
    assert decay_values[1] == 1.0
    assert decay_values[2] == 2.0


def test_target_generation():
    task = build_task()

    patient = DummyPatient(
        "p4",
        {"unitdischargeoffset": 600},  # 10 hours
        [],
    )

    samples = task(patient)

    # first prediction at hour 1
    first = samples[0]

    expected_remaining = 10 - 1
    assert abs(first["target_los_hours"] - expected_remaining) < 1e-5


def test_tensor_shapes():
    task = build_task()

    patient = DummyPatient(
        "p5",
        {"unitdischargeoffset": 600},
        [],
    )

    samples = task(patient)

    ts = samples[0]["time_series"]

    # should be [T, 3] because 1 feature -> val/mask/decay
    assert len(ts[0]) == 3

def test_target_los_sequence_generation():
    task = build_task()

    patient = DummyPatient(
        "p6",
        {"unitdischargeoffset": 600},  # 10 hours
        [],
    )

    samples = task(patient)

    first = samples[0]  # history length = 1
    second = samples[1]  # history length = 2
    third = samples[2]  # history length = 3

    assert torch.allclose(first["target_los_sequence"], torch.tensor([9.0]))
    assert torch.allclose(second["target_los_sequence"], torch.tensor([9.0, 8.0]))
    assert torch.allclose(third["target_los_sequence"], torch.tensor([9.0, 8.0, 7.0]))
