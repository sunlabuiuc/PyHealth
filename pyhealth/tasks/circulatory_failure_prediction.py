from datetime import timedelta
from typing import Dict, List, Optional


class CirculatoryFailurePredictionTask:
    """Early-warning task for circulatory failure prediction."""

    def __init__(
        self,
        prediction_window_hours: int = 12,
    ) -> None:
        self.prediction_window_hours = prediction_window_hours

    def _to_timestamp(self, value):
        """Converts a value to pandas.Timestamp lazily."""
        import pandas as pd

        if value is None:
            return None
        if pd.isna(value):
            return None
        return pd.to_datetime(value)

    def __call__(self, patient: Dict) -> Optional[List[Dict]]:
        """Converts one patient record into training samples."""
        time_series = patient.get("time_series", None)
        if not time_series:
            return None

        first_failure_time = self._to_timestamp(
            patient.get("first_failure_time", None)
        )
        prediction_window = timedelta(hours=self.prediction_window_hours)

        samples = []

        for point in time_series:
            charttime = self._to_timestamp(point["charttime"])
            map_value = point.get("map", None)

            if charttime is None or map_value is None:
                continue

            label = 0
            if first_failure_time is not None:
                label = int(
                    charttime < first_failure_time <= charttime + prediction_window
                )

            sample = {
                "patient_id": patient.get("patient_id"),
                "icustay_id": patient.get("icustay_id"),
                "gender": patient.get("gender"),
                "timestamp": charttime,
                "features": {
                    "map": float(map_value),
                },
                "label": label,
            }
            samples.append(sample)

        return samples if samples else None