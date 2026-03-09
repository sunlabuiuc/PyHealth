import tempfile
from pathlib import Path

import pandas as pd

from pyhealth.tasks.sleep_wake_classification import SleepWakeClassification


class FakeEvent:
    """A fake event class to simulate patient events."""
    def __init__(self, file_64hz):
        self.file_64hz = file_64hz

class FakePatient:
    """A fake patient class to simulate patient data and events."""
    def __init__(self, patient_id, file_64hz):
        self.patient_id = patient_id
        self._events = [FakeEvent(file_64hz)]

    """Returns the list of events for the patient."""
    def get_events(self):
        return self._events

def test_sleep_wake_classification_runs():
    """Test that the SleepWakeClassification task runs without errors and produces expected output format."""
    tmp = tempfile.mkdtemp()
    csv_path = Path(tmp) / "S001_whole_df.csv"

    df = pd.DataFrame(
        {
            "TIMESTAMP": [0, 1, 2, 3],
            "BVP": [0.1, 0.2, 0.3, 0.4],
            "EDA": [0.01, 0.02, 0.03, 0.04],
            "TEMP": [36.1, 36.1, 36.2, 36.2],
            "ACC_X": [1, 1, 2, 2],
            "ACC_Y": [0, 0, 1, 1],
            "ACC_Z": [0, 0, 1, 1],
            "HR": [60, 60, 61, 61],
            "Sleep Stage": ["Wake", "N2", "REM", "Wake"],
        }
    )
    df.to_csv(csv_path, index=False)

    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    patient = FakePatient("S001", str(csv_path))

    samples = task(patient)

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert "features" in samples[0]
    assert len(samples[0]["features"]) > 0
    assert samples[0]["label"] in [0, 1]