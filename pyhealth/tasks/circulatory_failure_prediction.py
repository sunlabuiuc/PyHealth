from pyhealth.tasks.base_task import BaseTask
from typing import List, Dict


class CirculatoryFailurePredictionTask(BaseTask):

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
        super().__init__()
        self.prediction_window_hours = prediction_window_hours

    def __call__(self, patient) -> List[Dict]:
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