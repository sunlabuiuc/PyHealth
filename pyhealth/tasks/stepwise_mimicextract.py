from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from .base_task import BaseTask


class StepWiseMortalityPredictionMIMICExtract(BaseTask):
    """Build step-wise mortality samples from MIMIC-Extract style ICU events.

    The task collects ``vitals_labs`` events for each ICU stay, groups them by
    hour, and emits one fixed-order sequence of observed variable/value pairs per
    stay. Each step stores only the variables observed at that hour; the model is
    responsible for learning a dense embedding for those sparse step-wise inputs.

    Expected input events
    ---------------------
    ``vitals_labs`` events should expose:

    - ``visit_id``: admission / ICU stay identifier
    - ``hours_in``: hour offset from ICU admission
    - ``code``: representative variable identifier
    - ``mean``: observed numeric value

    Mortality labels are resolved from ``admissions`` events when available.
    Synthetic tests may also attach ``hospital_expire_flag`` directly to
    ``vitals_labs`` rows.

    Args:
        observation_window_hours: Maximum observation horizon to include.
        min_observed_steps: Minimum number of non-empty hourly steps required to
            emit a sample.
    """

    task_name: str = "StepWiseMortalityPredictionMIMICExtract"
    input_schema: Dict[str, str] = {
        "step_wise_inputs": "raw",
        "hours": "raw",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(
        self,
        observation_window_hours: int = 48,
        min_observed_steps: int = 2,
    ) -> None:
        self.observation_window_hours = observation_window_hours
        self.min_observed_steps = min_observed_steps

    def _resolve_mortality(
        self,
        patient: Any,
        visit_id: Any,
        fallback_events: Iterable[Any],
    ) -> int | None:
        """Resolve a binary mortality label for a specific stay."""
        admissions = patient.get_events(
            event_type="admissions",
            filters=[("visit_id", "==", visit_id)],
        )
        for event in admissions:
            if hasattr(event, "hospital_expire_flag"):
                value = getattr(event, "hospital_expire_flag")
                if value in [0, 1, "0", "1"]:
                    return int(value)
            if hasattr(event, "discharge_status"):
                value = str(getattr(event, "discharge_status")).lower()
                return 0 if value in {"0", "alive"} else 1

        for event in fallback_events:
            if hasattr(event, "hospital_expire_flag"):
                value = getattr(event, "hospital_expire_flag")
                if value in [0, 1, "0", "1"]:
                    return int(value)
        return None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generate stay-level mortality prediction samples."""
        vitals_events = patient.get_events(event_type="vitals_labs")
        if len(vitals_events) == 0:
            return []

        grouped_by_visit: dict[Any, list[Any]] = defaultdict(list)
        for event in vitals_events:
            visit_id = getattr(event, "visit_id", None)
            hours_in = getattr(event, "hours_in", None)
            code = getattr(event, "code", None)
            value = getattr(event, "mean", None)
            if visit_id is None or hours_in is None or code is None or value is None:
                continue
            if hours_in < 0 or hours_in >= self.observation_window_hours:
                continue
            grouped_by_visit[visit_id].append(event)

        samples: List[Dict[str, Any]] = []
        for visit_id, visit_events in grouped_by_visit.items():
            mortality = self._resolve_mortality(patient, visit_id, visit_events)
            if mortality is None:
                continue

            hour_buckets: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for event in visit_events:
                hour = int(getattr(event, "hours_in"))
                code = str(getattr(event, "code"))
                value = float(getattr(event, "mean"))
                hour_buckets[hour][code].append(value)

            ordered_hours = sorted(hour_buckets.keys())
            if len(ordered_hours) < self.min_observed_steps:
                continue

            steps: List[Dict[str, Any]] = []
            for hour in ordered_hours:
                code_map = hour_buckets[hour]
                codes = sorted(code_map.keys())
                values = [
                    sum(code_map[code]) / float(len(code_map[code])) for code in codes
                ]
                steps.append({"codes": codes, "values": values})

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(visit_id),
                    "hours": [float(hour) for hour in ordered_hours],
                    "step_wise_inputs": steps,
                    "mortality": mortality,
                }
            )

        return samples
