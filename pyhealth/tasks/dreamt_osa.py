from typing import Any, Dict, List, Optional

import polars as pl

from .base_task import BaseTask


class DREAMTOSAClassification(BaseTask):
    """Patient-level OSA outcome tasks for the DREAMT dataset.

    This task assumes you are using `DREAMTDataset`, which exposes patient-level
    metadata such as:

        - age
        - gender
        - bmi
        - oahi
        - ahi
        - mean_sao2
        - arousal_index
        - medical_history
        - sleep_disorders

    We define OSA-related tasks using the AHI/OAHI values:

    Tasks
    -----
    - "ahi_severity_4class"
        Multi-class OSA severity from AHI:
            0: AHI < 5        (normal)
            1: 5 <= AHI < 15  (mild)
            2: 15 <= AHI < 30 (moderate)
            3: AHI >= 30      (severe)

    - "ahi_binary_15"
        Binary classification:
            0: AHI < 15       (no / mild OSA)
            1: AHI >= 15      (moderate / severe)

    - "oahi_binary_5"
        Binary classification:
            0: OAHI < 5       (no sleep apnea by OAHI)
            1: OAHI >= 5      (sleep apnea by OAHI)

    Features
    --------
    By default, the feature vector for each patient is a dictionary of
    clinical variables:

        ["age", "gender", "bmi", "mean_sao2", "arousal_index",
         "medical_history", "sleep_disorders"]

    You can override this with `feature_keys` when instantiating the task.

    Example
    -------
    >>> from pyhealth.datasets import DREAMTDataset
    >>> from pyhealth.tasks import DREAMTOSAClassification
    >>>
    >>> dataset = DREAMTDataset(root="/path/to/dreamt/version")
    >>> task = DREAMTOSAClassification(task="ahi_severity_4class")
    >>>
    >>> # later, in your dataloader construction:
    >>> # samples = task(patient)  # where `patient` is from DREAMTDataset
    """

    # Registry of supported tasks
    tasks = {
        "patient_level": [
            "ahi_severity_4class",
            "ahi_binary_15",
            "oahi_binary_5",
        ]
    }

    def __init__(
        self,
        task: str,
        feature_keys: Optional[List[str]] = None,
    ) -> None:
        if task not in self.tasks["patient_level"]:
            raise ValueError(
                f"Unsupported task '{task}'. "
                f"Choose from: {self.tasks['patient_level']}"
            )

        self.task = task
        self.task_name = f"DREAMTOSA/{task}"

        # Default clinical features to include
        self.feature_keys = feature_keys or [
            "age",
            "gender",
            "bmi",
            "mean_sao2",
            "arousal_index",
            "medical_history",
            "sleep_disorders",
        ]

        # Patient-level tabular input
        self.input_schema = {"feature": "tabular"}

        # Label type depends on which task is chosen
        if task == "ahi_severity_4class":
            self.output_schema = {"label": "multiclass"}  # classes 0–3
        elif task in {"ahi_binary_15", "oahi_binary_5"}:
            self.output_schema = {"label": "binary"}
        else:
            # Should not happen because of the check above
            raise ValueError(f"Unknown task: {task}")

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Optionally filter events before task construction.

        For patient-level OSA tasks, we typically just pass everything through.
        You can customize this to drop unrelated event types if needed.
        """
        return df

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Build one or more samples for a single patient.

        Each patient contributes exactly one sample for these tasks.
        """
        # Try to get split information if available, otherwise default to "train"
        split = "train"
        try:
            split_events = patient.get_events("splits")
            if len(split_events) == 1 and hasattr(split_events[0], "split"):
                split = split_events[0].split
        except Exception:
            # If there is no "splits" table, we just keep the default.
            pass

        # Extract label depending on task
        label = self._get_label_from_patient(patient)
        if label is None:
            # If we cannot compute a label (e.g., missing AHI/OAHI), skip this patient
            return []

        # Build feature dictionary from patient attributes
        features: Dict[str, Any] = {}
        for key in self.feature_keys:
            value = getattr(patient, key, None)
            features[key] = value

        sample = {
            "feature": features,
            "label": label,
            "split": split,
            "patient_id": getattr(patient, "patient_id", None),
        }
        return [sample]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _get_label_from_patient(self, patient: Any) -> Optional[int]:
        """Compute the task-specific label from patient metadata."""
        if self.task == "ahi_severity_4class":
            ahi = getattr(patient, "ahi", None)
            if ahi is None:
                return None
            try:
                ahi_val = float(ahi)
            except (TypeError, ValueError):
                return None

            if ahi_val < 5:
                return 0  # normal
            elif ahi_val < 15:
                return 1  # mild
            elif ahi_val < 30:
                return 2  # moderate
            else:
                return 3  # severe

        elif self.task == "ahi_binary_15":
            ahi = getattr(patient, "ahi", None)
            if ahi is None:
                return None
            try:
                ahi_val = float(ahi)
            except (TypeError, ValueError):
                return None
            return int(ahi_val >= 15.0)  # 1 = moderate/severe

        elif self.task == "oahi_binary_5":
            oahi = getattr(patient, "oahi", None)
            if oahi is None:
                return None
            try:
                oahi_val = float(oahi)
            except (TypeError, ValueError):
                return None
            return int(oahi_val >= 5.0)

        return None
