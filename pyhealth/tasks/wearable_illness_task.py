from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask


class WearableIllnessPrediction(BaseTask):
    """Predicts illness based on changes in wearable signals relative to a baseline window."""

    task_name: str = "WearableIllnessPrediction"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
    }

    output_schema: Dict[str, str] = {
        "label": "binary",
    }

    def __init__(self, baseline_window: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.baseline_window = baseline_window

    def __call__(self, patient: Patient) -> List[Dict]:
        # Get daily wearable events for this patient
        events: List[Event] = patient.get_events(event_type="wearable_daily")

        # Not enough history to compute a baseline
        if len(events) <= self.baseline_window:
            return []

        # Ensure events are in chronological order
        events = sorted(events, key=lambda x: x.timestamp)

        samples: List[Dict] = []

        # Slide a window across the timeline to build samples
        for i in range(self.baseline_window, len(events)):
            history = events[i - self.baseline_window : i]
            current = events[i]

            # Compute baseline values from recent history
            baseline_rhr = sum(float(e["resting_heart_rate"]) for e in history) / len(history)
            baseline_sleep = sum(float(e["sleep_duration"]) for e in history) / len(history)

            # Measure deviation from baseline
            delta_rhr = float(current["resting_heart_rate"]) - baseline_rhr
            delta_sleep = float(current["sleep_duration"]) - baseline_sleep

            # Convert numeric changes into simple tokens for modeling
            conditions = [
                f"rhr_{round(delta_rhr, 1)}",
                f"sleep_{round(delta_sleep, 1)}",
            ]

            samples.append(
                {
                    "visit_id": f"{patient.patient_id}_{int(current['day_index'])}",
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "label": int(current["is_ill"]),
                }
            )

        return samples