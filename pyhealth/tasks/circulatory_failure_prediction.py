"""
Circulatory Failure Prediction Task for PyHealth.

Dataset:
    MIMIC-III Clinical Database v1.4
    https://physionet.org/content/mimiciii/1.4/

Inspired by:
    Hoche, M., Mineeva, O., Burger, M., Blasimme, A., & Ratsch, G. (2024). 
    FAMEWS: A fairness auditing tool for medical early-warning systems. 
    Proceedings of the Fifth Conference on Health, Inference, and Learning, 248, 297–311. PMLR. 
    https://proceedings.mlr.press/v248/hoche24a.html

Description:
    Time-point prediction task for circulatory failure early warning.
    For each MAP measurement at time *t*, the sample is labelled positive
    if the first circulatory failure event occurs within the future
    prediction window.  Circulatory failure is approximated by a mean
    arterial pressure (MAP) value below 65 mmHg.

Authors:
    Kuang-Yu Wang (kuangyu4@illinois.edu)
    Ya Hsuan Yang (yhyang3@illinois.edu)
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask


MAP_ITEMID = 220052
MAP_FAILURE_THRESHOLD = 65.0


class CirculatoryFailurePredictionTask(BaseTask):
    """Early-warning task for circulatory failure prediction.

    This task converts a PyHealth Patient object into time-point prediction
    samples for circulatory failure early warning. For each MAP measurement at
    time t, the sample is labeled positive if the first circulatory failure
    event occurs within the future prediction window.
    Circulatory failure is approximated by a mean arterial pressure (MAP)
    value below 65 mmHg.

    Attributes:
        task_name: Unique task identifier.
        input_schema: Input feature schema for PyHealth processors.
        output_schema: Output label schema for PyHealth processors.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import CirculatoryFailurePredictionTask
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii",
        ...     tables=["chartevents"],
        ... )
        >>> task = CirculatoryFailurePredictionTask(prediction_window_hours=12)
        >>> sample_dataset = dataset.set_task(task)
        >>> sample_dataset[0]  # doctest: +SKIP
    """

    input_schema: Dict[str, str] = {
        "map": "tensor",
        "map_diff": "tensor"
    }

    output_schema: Dict[str, str] = {
        "label": "binary"
    }

    task_name: str = "circulatory_failure_prediction"

    def __init__(
        self,
        prediction_window_hours: int = 12,
        map_itemid: int = MAP_ITEMID,
        failure_threshold: float = MAP_FAILURE_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        """Initializes the circulatory failure prediction task.

        Args:
            prediction_window_hours: Future prediction window in hours.
            map_itemid: MIMIC-III ITEMID corresponding to MAP.
            failure_threshold: MAP threshold used to define circulatory failure.
            **kwargs: Additional keyword arguments passed to BaseTask.
        """
        super().__init__(**kwargs)
        self.prediction_window_hours = prediction_window_hours
        self.map_itemid = map_itemid
        self.failure_threshold = failure_threshold
        self.task_name = (
            f"circulatory_failure_prediction_{prediction_window_hours}h"
        )

    @staticmethod
    def _to_datetime(value: Any) -> Optional[pd.Timestamp]:
        """Converts a timestamp-like value to pandas Timestamp."""
        if value is None or pd.isna(value):
            return None
        return pd.to_datetime(value, errors="coerce")

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Converts a numeric-like value to float."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _event_attr(event: Any, attr: str, default: Any = None) -> Any:
        """Gets an event attribute from either object attr or attr_dict."""
        if hasattr(event, attr):
            return getattr(event, attr)

        attr_dict = getattr(event, "attr_dict", None)
        if isinstance(attr_dict, dict) and attr in attr_dict:
            return attr_dict[attr]

        return default

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Converts one Patient object into prediction samples.

        Args:
            patient: A PyHealth Patient object. The patient should contain
                `icustays`, `patients`, and `chartevents` events loaded from
                MIMIC-III.

        Returns:
            A list of sample dictionaries. Each sample contains patient/visit
            metadata, MAP-based features, and a binary early-warning label.
            Returning an empty list is valid when no usable MAP data or no
            failure event is found.
        """
        samples: List[Dict[str, Any]] = []
        prediction_window = timedelta(hours=self.prediction_window_hours)

        patient_events = patient.get_events(event_type="patients")
        gender = None
        if len(patient_events) > 0:
            gender = self._event_attr(patient_events[0], "gender")

        icu_stays = patient.get_events(event_type="icustays")
        chartevents = patient.get_events(event_type="chartevents")

        if len(icu_stays) == 0 or len(chartevents) == 0:
            return []

        for icu_stay in icu_stays:
            icustay_id = self._event_attr(icu_stay, "icustay_id")
            intime = self._to_datetime(self._event_attr(icu_stay, "intime"))
            outtime = self._to_datetime(self._event_attr(icu_stay, "outtime"))
            hadm_id = self._event_attr(icu_stay, "hadm_id")

            if icustay_id is None or intime is None or outtime is None:
                continue

            map_events = []
            for event in chartevents:
                event_icustay_id = self._event_attr(event, "icustay_id")
                itemid = self._event_attr(event, "itemid")
                charttime = self._to_datetime(event.timestamp)
                valuenum = self._to_float(self._event_attr(event, "valuenum"))

                if event_icustay_id != icustay_id:
                    continue
                if itemid != self.map_itemid:
                    continue
                if charttime is None or pd.isna(charttime):
                    continue
                if valuenum is None:
                    continue
                if not (intime <= charttime <= outtime):
                    continue

                map_events.append(
                    {
                        "charttime": charttime,
                        "map": valuenum,
                    }
                )

            if not map_events:
                continue

            map_events = sorted(map_events, key=lambda x: x["charttime"])

            failure_times = [
                row["charttime"]
                for row in map_events
                if row["map"] < self.failure_threshold
            ]

            if not failure_times:
                continue

            first_failure_time = min(failure_times)

            previous_map = None
            for row in map_events:
                timestamp = row["charttime"]
                map_value = row["map"]

                if previous_map is None:
                    map_diff = 0.0
                else:
                    map_diff = map_value - previous_map

                previous_map = map_value

                label = int(
                    timestamp < first_failure_time
                    <= timestamp + prediction_window
                )

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": str(icustay_id),
                        "hadm_id": hadm_id,
                        "icustay_id": icustay_id,
                        "gender": gender,
                        "timestamp": timestamp,
                        "map": map_value,
                        "map_diff": map_diff,
                        "label": label,
                    }
                )

        return samples