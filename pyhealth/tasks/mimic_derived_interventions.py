from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from .base_task import BaseTask


def _to_datetime(value: Any) -> Optional[datetime]:
    """Best-effort conversion to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _duration_hours(start: Any, end: Any) -> float:
    """Compute non-negative duration in hours from start/end values."""
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end)
    if start_dt is None or end_dt is None:
        return 0.0
    return max((end_dt - start_dt).total_seconds() / 3600.0, 0.0)


class VasopressorDurationTask(BaseTask):
    """Structure vasopressor spans from MIMIC derived tables.

    Each ICU stay (patient_id equals stay_id in the derived configs) yields one
    sample with the raw vasopressor spans as a timeseries feature and the total
    vasopressor exposure time as a regression label.

    Works with both `mimic3_derived.yaml` (event_type `vasopressor_durations`)
    and `mimic4_derived.yaml` (event_type `vasopressin`) by overriding the
    `event_type` argument.
    """

    input_schema: Dict[str, str] = {"vasopressor": "timeseries"}
    output_schema: Dict[str, str] = {"vasopressor_hours": "regression"}

    def __init__(
        self,
        event_type: str = "vasopressin",
        duration_field: str = "duration_hours",
        endtime_field: str = "endtime",
    ):
        self.event_type = event_type
        self.duration_field = duration_field
        self.endtime_field = endtime_field
        self.task_name = f"VasopressorDuration/{event_type}"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events: pl.DataFrame = patient.get_events(
            event_type=self.event_type,
            return_df=True,
        )
        if events.height == 0:
            return []

        timestamps = events["timestamp"].to_list()
        duration_col = f"{self.event_type}/{self.duration_field}"
        end_col = f"{self.event_type}/{self.endtime_field}"

        if duration_col in events.columns:
            durations = (
                events.get_column(duration_col)
                .fill_null(0.0)
                .cast(pl.Float64)
                .to_list()
            )
        elif end_col in events.columns:
            end_times = events.get_column(end_col).to_list()
            durations = [
                _duration_hours(start, end) for start, end in zip(timestamps, end_times)
            ]
        else:
            durations = [0.0 for _ in timestamps]

        values = np.array(durations, dtype=float).reshape(-1, 1)
        total_hours = float(np.sum(values))

        sample = {
            "patient_id": patient.patient_id,
            "vasopressor": (timestamps, values),
            "vasopressor_hours": total_hours,
        }

        return [sample]


class VentilationDurationTask(BaseTask):
    """Structure ventilation spans from MIMIC derived tables.

    Each ICU stay is converted into a sample that contains a two-dimensional
    timeseries: duration (hours) and encoded ventilation status. Labels include
    total ventilation time (regression) and the set of encountered statuses
    (multilabel). Use `event_type='ventilation_durations'` for MIMIC-III
    derived tables and the default `ventilation` for MIMIC-IV derived tables.
    """

    input_schema: Dict[str, str] = {"ventilation": "timeseries"}
    output_schema: Dict[str, str] = {
        "ventilation_hours": "regression",
        "ventilation_statuses": "multilabel",
    }

    def __init__(
        self,
        event_type: str = "ventilation",
        status_field: str = "ventilation_status",
        duration_field: str = "duration_hours",
        endtime_field: str = "endtime",
    ):
        self.event_type = event_type
        self.status_field = status_field
        self.duration_field = duration_field
        self.endtime_field = endtime_field
        self.status_to_index: Dict[str, int] = {}
        self.task_name = f"VentilationDuration/{event_type}"

    def _status_index(self, status: str) -> int:
        status = status or "Unknown"
        if status not in self.status_to_index:
            self.status_to_index[status] = len(self.status_to_index)
        return self.status_to_index[status]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events: pl.DataFrame = patient.get_events(
            event_type=self.event_type, return_df=True
        )
        if events.height == 0:
            return []

        timestamps = events["timestamp"].to_list()
        duration_col = f"{self.event_type}/{self.duration_field}"
        end_col = f"{self.event_type}/{self.endtime_field}"
        status_col = f"{self.event_type}/{self.status_field}"

        if duration_col in events.columns:
            durations = (
                events.get_column(duration_col)
                .fill_null(0.0)
                .cast(pl.Float64)
                .to_list()
            )
        elif end_col in events.columns:
            end_times = events.get_column(end_col).to_list()
            durations = [
                _duration_hours(start, end) for start, end in zip(timestamps, end_times)
            ]
        else:
            durations = [0.0 for _ in timestamps]

        if status_col in events.columns:
            statuses = events.get_column(status_col).fill_null("Unknown").to_list()
        else:
            statuses = ["Unknown"] * len(timestamps)

        status_indices = [self._status_index(status) for status in statuses]
        values = np.column_stack(
            [
                np.array(durations, dtype=float),
                np.array(status_indices, dtype=float),
            ]
        )
        total_hours = float(np.sum(values[:, 0]))
        unique_statuses = sorted({status for status in statuses if status is not None})

        sample = {
            "patient_id": patient.patient_id,
            "ventilation": (timestamps, values),
            "ventilation_hours": total_hours,
            "ventilation_statuses": unique_statuses if unique_statuses else ["Unknown"],
        }
        return [sample]
