from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask


class MortalityPredictionEICU(BaseTask):
    """
    Synthetic data evaluation and ICU mortality prediction on the eICU dataset.

    This task aims at predicting ICU mortality, 30-day readmission, or length 
    of stay using a hierarchy of clinical features to evaluate the fidelity, 
    utility, and privacy of synthetic EHR data.

    Features:
        - using patient table for demographics (age, gender)
        - using lab table for clinical markers (hemoglobin, hematocrit, albumin, etc.)
        - using admission/patient table for hospital stay info (hosp_los)

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import MortalityPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["patient", "lab"],
        ... )
        >>> task = MortalityPredictionEICU(task_type="mortality", num_features=10)
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name = "mortality_prediction_eicu"

    input_schema = {"features": Dict[str, float]}
    output_schema = {"label": int}

    def __init__(
        self,
        task_type: str = "mortality",
        num_features: int = 10,
        code_mapping: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        super().__init__(code_mapping=code_mapping)
        self.task_type = task_type
        self.num_features = num_features

        self.feature_hierarchy = [
            "hosp_los", "is_female", "hemoglobin", "hematocrit", "albumin",
            "bun", "age", "heart_rate", "resp_rate", "temp",
            "glucose", "wbc", "platelets", "sodium", "potassium",
            "creatinine", "bicarbonate", "calcium", "inr", "lactate"
        ]
        self.subset_keys = self.feature_hierarchy[:num_features]

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Processes a patient into samples based on task type."""
        samples = []
        is_female = 1 if getattr(patient, "gender", "").lower() == "female" else 0
        age = float(getattr(patient, "age", 0.0))

        for encounter in patient.encounters:
            if self.task_type == "mortality":
                label = 1 if getattr(encounter, "discharge_status", "") == "Expired" else 0
            elif self.task_type == "readmission":
                label = 1 if len(patient.encounters) > 1 else 0
            else:
                label = 1 if float(getattr(encounter, "los", 0.0)) > 3.0 else 0

            raw_features = {
                "age": age,
                "is_female": is_female,
                "hosp_los": float(getattr(encounter, "los", 0.0)),
            }
            features = {k: raw_features.get(k, 0.0) for k in self.subset_keys}

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id": encounter.visit_id,
                **features,
                "label": label,
            })
        return samples

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
        """Calculates fidelity metric D_KL(P || Q)."""
        p_hist, edges = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=edges, density=True)
        return float(entropy(p_hist + 1e-10, q_hist + 1e-10))

    @staticmethod
    def membership_advantage(m_scores: np.ndarray, nm_scores: np.ndarray) -> float:
        """Calculates max |P(s|member) - P(s|non-member)|."""
        all_s = np.sort(np.concatenate([m_scores, nm_scores]))
        adv = [abs(np.mean(m_scores <= s) - np.mean(nm_scores <= s)) for s in all_s]
        return float(np.max(adv))

    @staticmethod
    def empirical_risk(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculates R(h) using log loss."""
        return float(log_loss(y_true, y_prob))