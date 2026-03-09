import numpy as np
import pandas as pd
from typing import Any, Dict, List

import pandas as pd

from pyhealth.tasks import BaseTask


class SleepWakeClassification(BaseTask):
    task_name: str = "SleepWakeClassification"
    input_schema: Dict[str, str] = {"features": "vector"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, epoch_seconds: int = 30, sampling_rate: int = 64):
        """Initializes the sleep-wake classification task.

        Args:
            epoch_seconds: Length of each epoch in seconds. Default is 30.
            sampling_rate: Sampling rate of the wearable data in Hz. Default is 64.
        """
        self.epoch_seconds = epoch_seconds
        self.sampling_rate = sampling_rate
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
    
    def _extract_basic_features(self, epoch_df: pd.DataFrame) -> List[float]:
        """Extracts basic features (mean values) from the epoch data.
        
        Args:
            epoch_df: DataFrame containing the data for the current epoch.
        Returns:
            A list of basic features (mean values) for the epoch.
        """
        features = []

        for col in ["BVP", "HR", "TEMP", "EDA"]:
            if col in epoch_df.columns:
                values = pd.to_numeric(epoch_df[col], errors="coerce").dropna()
                features.append(float(values.mean()) if len(values) > 0 else 0.0)

        return features

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = []
        events = patient.get_events()
        if len(events) == 0:
            return samples

        epoch_size = self.epoch_seconds * self.sampling_rate

        for event in events:
            if not hasattr(event, "file_64hz") or event.file_64hz is None:
                continue

            df = pd.read_csv(event.file_64hz)

            if "Sleep Stage" not in df.columns:
                continue

            n_epochs = len(df) // epoch_size
            for epoch_idx in range(n_epochs):
                start = epoch_idx * epoch_size
                end = start + epoch_size
                epoch_df = df.iloc[start:end].copy()

                if len(epoch_df) == 0:
                    continue

                raw_label = epoch_df["Sleep Stage"].mode().iloc[0]
                label = self._map_sleep_label(raw_label)
                if label is None:
                    continue

                features = self._extract_basic_features(epoch_df)

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}-{epoch_idx}",
                        "epoch_index": epoch_idx,
                        "features": features,
                        "label": label,
                    }
                )

        return samples