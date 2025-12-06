from typing import Any, Dict, List
import polars as pl

from .base_task import BaseTask


class HeartDiseasePrediction(BaseTask):
    """
    Task for predicting heart disease using the UCI Heart Disease dataset
    (Cleveland) fetched via ucimlrepo.

    Each patient is represented as a single event with all features as input,
    and the target is a binary label: 0 = no heart disease, 1 = heart disease.
    """

    task_name: str = "HeartDiseasePrediction"
    # All columns except patient_id are features
    input_schema: Dict[str, str] = {
        "age": "numeric",
        "sex": "numeric",
        "cp": "numeric",
        "trestbps": "numeric",
        "chol": "numeric",
        "fbs": "numeric",
        "restecg": "numeric",
        "thalach": "numeric",
        "exang": "numeric",
        "oldpeak": "numeric",
        "slope": "numeric",
        "ca": "numeric",
        "thal": "numeric",
    }
    output_schema: Dict[str, str] = {"heart_disease": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """
        Processes a single patient (row from Polars LazyFrame or pandas DataFrame)
        into a sample for prediction.
        """

        sample = {}
        for feature in self.input_schema.keys():
            if isinstance(patient, dict):
                sample[feature] = patient.get(feature, None)
            else:
                sample[feature] = getattr(patient, feature, None)

        if isinstance(patient, dict):
            sample["heart_disease"] = int(patient.get("target", 0))
        else:
            sample["heart_disease"] = int(getattr(patient, "target", 0))

        return [sample]
