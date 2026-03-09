from typing import Any, Dict, List

import pandas as pd

from pyhealth.tasks import BaseTask


class SleepWakeClassification(BaseTask):
    task_name: str = "SleepWakeClassification"
    input_schema: Dict[str, str] = {"features": "vector"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, epoch_seconds: int = 30):
        """Initializes the sleep-wake classification task.

        Args:
            epoch_seconds: Length of each epoch in seconds. Default is 30.
        """
        self.epoch_seconds = epoch_seconds
        super().__init__()

    def _map_sleep_label(self, label: str) -> int | None:
        """Maps DREAMT sleep stage labels to binary sleep/wake labels.

        Args:
            label: Original sleep stage label.

        Returns:
            1 for wake, 0 for sleep, or None if the label should be skipped.
        """
        if label is None or pd.isna(label):
            return None

        label = str(label).strip()

        if label.lower() == "wake":
            return 1

        if label.upper() in {"REM", "N1", "N2", "N3"}:
            return 0

        return None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        events = patient.get_events()
        if len(events) == 0:
            return samples

        # For DREAMT, each patient should typically have one wearable file event
        for event in events:
            if not hasattr(event, "file_64hz") or event.file_64hz is None:
                continue

            df = pd.read_csv(event.file_64hz)

            if "Sleep Stage" not in df.columns:
                continue

            unique_labels = df["Sleep Stage"].dropna().unique().tolist()

            for epoch_idx, raw_label in enumerate(unique_labels):
                label = self._map_sleep_label(raw_label)
                if label is None:
                    continue

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}-{epoch_idx}",
                        "epoch_index": epoch_idx,
                        "features": [],
                        "label": label,
                    }
                )

        return samples