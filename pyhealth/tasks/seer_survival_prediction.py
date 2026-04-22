"""
Contributor: Adrianne Sun, Ruoyi Xie
NetID: ajsun2, ruoyix2
Paper Title: Reproducible Survival Prediction with SEER Cancer Data
Paper Link: https://proceedings.mlr.press/v85/hegselmann18a/hegselmann18a.pdf
Description: Implementation of the SEER Survival Prediction task.
"""
from typing import Dict, List
import numpy as np

from pyhealth.tasks import BaseTask


class SEERSurvivalPrediction(BaseTask):
    """Binary classification task for survival prediction on SEER dataset.

    Survival prediction aims at classifying whether a patient will survive past a
    defined time threshold (e.g., 60 months) based on pre-extracted clinical features. 
    The task is defined as a binary classification.

    Attributes:
        task_name (str): The name of the task, set to "SEERSurvivalPrediction".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains:
            - "label": "binary"
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "label": "binary"
    """

    task_name: str = "SEERSurvivalPrediction"
    input_schema = {"label": "binary"}
    output_schema = {"label": "binary"}

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: List[str] | None = None

    def __call__(self, patient) -> List[Dict]:
        """Processes a single patient for the SEER survival prediction task.

        Will be called automatically by `dataset.set_task()` to generate samples.

        Args:
            patient (Patient): A PyHealth Patient object containing SEER data.

        Returns:
            List[Dict]: A list of samples, where each sample is a dict containing 
                'patient_id', 'visit_id', 'features', and 'label'.
        
        Note that we define the task as a binary classification task.
        
        Examples:
            >>> from pyhealth.datasets import SEERDataset
            >>> seer_ds = SEERDataset(
            ...     root="/path/to/processed",
            ...     tables=["seer"]
            ... )
            >>> from pyhealth.tasks import SEERSurvivalPrediction
            >>> survival_ds = seer_ds.set_task(SEERSurvivalPrediction())
            >>> survival_ds.samples[0]
            {
                'patient_id': 'seer_0',
                'visit_id': 'seer_0_seer',
                'features': array([55., 2005., 1., 0., ...], dtype=float32),
                'label': 1
            }
        """
        samples = []

        # Access events through patient.get_events(...)
        events = patient.get_events(event_type="seer")
        if len(events) == 0:
            return samples

        # One event per patient in this custom SEER table
        event = events[0]
        data = event.attr_dict

        if "label" not in data:
            raise KeyError(f"Missing 'label' for patient {patient.patient_id}")

        raw_label = data["label"]
        try:
            label = int(float(raw_label))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Label must be numeric 0/1, got {raw_label!r} "
                f"for patient {patient.patient_id}"
            ) from e

        if label not in (0, 1):
            raise ValueError(
                f"Label must be binary 0/1, got {label} "
                f"for patient {patient.patient_id}"
            )

        # Deterministic feature order
        feature_cols = sorted(k for k in data.keys() if k != "label")

        # Save once so example scripts can do named ablations
        if self.feature_names is None:
            self.feature_names = feature_cols

        features = []
        for k in feature_cols:
            v = data[k]
            try:
                features.append(float(v))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Feature column {k!r} must be numeric, got value {v!r} "
                    f"for patient {patient.patient_id}"
                ) from e

        features = np.asarray(features, dtype=np.float32)

        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": f"{patient.patient_id}_seer",
                "features": features,
                "label": label,
            }
        )

        return samples