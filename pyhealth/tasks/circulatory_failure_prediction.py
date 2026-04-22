from pyhealth.tasks.base_task import BaseTask
from typing import List, Dict


class CirculatoryFailurePredictionTask(BaseTask):
    """Early-warning task for circulatory failure prediction.

    This task converts one ICU-stay patient record into multiple
    time-point prediction samples. At each timestamp t, the label is 1
    if the first circulatory failure event occurs within the next
    prediction window, and 0 otherwise.

    Circulatory failure is defined upstream using a proxy event based on
    MAP < 65 mmHg.

    Attributes:
        task_name: Unique task identifier used by PyHealth.
        input_schema: Expected input feature schema.
        output_schema: Expected output label schema.
        prediction_window_hours: Number of hours used for early-warning label
            generation.
    """
    task_name = "circulatory_failure_prediction"

    input_schema = {
        "map": float,
        "timestamp": "datetime",
        "gender": str,
    }

    output_schema = {
        "label": int,
    }

    def __init__(self, prediction_window_hours: int = 12):
        """Initializes the circulatory failure prediction task.

        Args:
            prediction_window_hours: Future prediction window in hours.
                A sample is labeled positive if the first failure event
                happens within this horizon.
        """
        super().__init__()
        self.prediction_window_hours = prediction_window_hours

    def __call__(self, patient) -> List[Dict]:
        """Converts one patient record into task samples.

        Args:
            patient: A task-ready patient dictionary containing ICU-stay
                metadata, time-series MAP measurements, and
                first_failure_time.

        Returns:
            A list of sample dictionaries. Each sample contains patient
            metadata, a timestamp, feature values, and a binary label.
            Returns an empty list if the patient has no usable
            time-series data or no failure time.
        """
        if not patient["time_series"]:
            return []

        import pandas as pd
        from datetime import timedelta

        first_failure_time = patient["first_failure_time"]
        if first_failure_time is None:
            return []

        first_failure_time = pd.to_datetime(first_failure_time)

        prediction_window = timedelta(hours=self.prediction_window_hours)

        samples = []

        for row in patient["time_series"]:
            t = pd.to_datetime(row["charttime"])
            map_value = row["map"]

            label = int(t < first_failure_time <= t + prediction_window)

            samples.append(
                {
                    "patient_id": patient["patient_id"],
                    "icustay_id": patient["icustay_id"],
                    "gender": patient["gender"],
                    "timestamp": t,
                    "features": {"map": map_value},
                    "label": label,
                }
            )

        return samples