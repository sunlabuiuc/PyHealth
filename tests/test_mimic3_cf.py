from pyhealth.datasets import MIMIC3CirculatoryFailureDataset
from pyhealth.tasks import CirculatoryFailurePredictionTask


class DummyMIMIC3CFDataset(MIMIC3CirculatoryFailureDataset):
    """Small synthetic subclass for fast unit testing."""

    def __init__(self):
        # do not call super().__init__ because we don't want real files
        pass

    def load_cohort(self):
        return [
            {
                "patient_id": 1,
                "gender": "F",
                "hadm_id": 100,
                "icustay_id": 1001,
                "admittime": "2150-01-01 00:00:00",
                "intime": "2150-01-01 00:00:00",
                "outtime": "2150-01-02 00:00:00",
            }
        ]

    def get_patient_by_icustay_id(self, icustay_id: int):
        if icustay_id != 1001:
            return None

        return {
            "patient_id": 1,
            "icustay_id": 1001,
            "gender": "F",
            "intime": "2150-01-01 00:00:00",
            "outtime": "2150-01-02 00:00:00",
            "first_failure_time": "2150-01-01 11:00:00",
            "time_series": [
                {"charttime": "2150-01-01 00:00:00", "map": 80.0},
                {"charttime": "2150-01-01 01:00:00", "map": 78.0},
                {"charttime": "2150-01-01 10:00:00", "map": 70.0},
                {"charttime": "2150-01-01 11:00:00", "map": 60.0},
            ],
        }


def test_set_task_returns_samples():
    dataset = DummyMIMIC3CFDataset()
    task = CirculatoryFailurePredictionTask(prediction_window_hours=12)

    samples = dataset.set_task(task)

    assert isinstance(samples, list)
    assert len(samples) == 4
    assert samples[0]["patient_id"] == 1
    assert samples[0]["icustay_id"] == 1001
    assert samples[0]["features"]["map"] == 80.0
    assert samples[0]["label"] == 1
    assert samples[-1]["label"] == 0