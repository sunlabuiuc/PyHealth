from dataclasses import dataclass
from datetime import datetime

from pyhealth.tasks import CirculatoryFailurePredictionTask


@dataclass
class FakeEvent:
    event_type: str
    timestamp: datetime | None = None
    attr_dict: dict | None = None

    def __post_init__(self):
        if self.attr_dict is None:
            self.attr_dict = {}

    def __getattr__(self, name):
        if name in self.attr_dict:
            return self.attr_dict[name]
        raise AttributeError(name)


class FakePatient:
    patient_id = "1"

    def __init__(self):
        self.events = {
            "patients": [
                FakeEvent(
                    event_type="patients",
                    attr_dict={"gender": "F"},
                )
            ],
            "icustays": [
                FakeEvent(
                    event_type="icustays",
                    timestamp=datetime(2150, 1, 1, 0, 0, 0),
                    attr_dict={
                        "hadm_id": 100,
                        "icustay_id": 1001,
                        "intime": "2150-01-01 00:00:00",
                        "outtime": "2150-01-02 00:00:00",
                    },
                )
            ],
            "chartevents": [
                FakeEvent(
                    event_type="chartevents",
                    timestamp=datetime(2150, 1, 1, 0, 0, 0),
                    attr_dict={
                        "icustay_id": 1001,
                        "itemid": 220052,
                        "valuenum": 80.0,
                    },
                ),
                FakeEvent(
                    event_type="chartevents",
                    timestamp=datetime(2150, 1, 1, 1, 0, 0),
                    attr_dict={
                        "icustay_id": 1001,
                        "itemid": 220052,
                        "valuenum": 78.0,
                    },
                ),
                FakeEvent(
                    event_type="chartevents",
                    timestamp=datetime(2150, 1, 1, 10, 0, 0),
                    attr_dict={
                        "icustay_id": 1001,
                        "itemid": 220052,
                        "valuenum": 70.0,
                    },
                ),
                FakeEvent(
                    event_type="chartevents",
                    timestamp=datetime(2150, 1, 1, 11, 0, 0),
                    attr_dict={
                        "icustay_id": 1001,
                        "itemid": 220052,
                        "valuenum": 60.0,
                    },
                ),
            ],
        }

    def get_events(self, event_type=None, *args, **kwargs):
        if event_type is None:
            all_events = []
            for events in self.events.values():
                all_events.extend(events)
            return all_events
        return self.events.get(event_type, [])


def test_circulatory_failure_task_basic():
    task = CirculatoryFailurePredictionTask(prediction_window_hours=12)
    patient = FakePatient()

    samples = task(patient)

    assert len(samples) == 4
    assert samples[0]["label"] == 1
    assert samples[1]["label"] == 1
    assert samples[2]["label"] == 1
    assert samples[3]["label"] == 0
    assert samples[0]["map"] == 80.0
    assert samples[1]["map_diff"] == -2.0
    assert samples[0]["gender"] == "F"
    assert samples[0]["patient_id"] == "1"
    assert samples[0]["visit_id"] == "1001"