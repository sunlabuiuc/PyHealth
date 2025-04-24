from datetime import datetime, timedelta
from typing import Any, Dict, List

import polars as pl

from .base_task import BaseTask

class EEGIsSeizure(BaseTask):

    task_name: str = "EEGIsSeizure"
    input_schema: Dict[str, str] = {"eegs": "signal"}
    output_schema: Dict[str, str] = {"seizure": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="eeg_tusz")

        samples = []

        samples.append(
                {
                    "patient_id": patient.patient_id,
                    "session_id": admission.hadm_id,
                    "labs": (timestamps, lab_values),
                    "mortality": mortality,
                }
            )

        return []