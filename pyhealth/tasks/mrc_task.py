from typing import Any, Dict, List

from pyhealth.tasks import BaseTask


def mortality_prediction_mrc_fn(patient: Any) -> List[Dict[str, Any]]:
    """
    ICU mortality prediction on the Noisy MRC ICU dataset.

    Each ICU stay (row in mortality_alsocat / mortality_nocat) becomes one sample.
    The goal is to predict in-hospital mortality (y=0: survived, y=1: died) from all processed tabular features.
    """
    samples: List[Dict[str, Any]] = []

    # In this dataset, each patient has a single visit corresponding to one ICU stay.
    for idx, visit in enumerate(patient):
        # 1. Get the label
        if not hasattr(visit, "y"):
            # skip if no label (should not happen for this dataset)
            continue
        label = getattr(visit, "y")

        # 2. Build a feature dict from all attributes of this visit
        #    exclude IDs and the label column itself to be used in sample metadata
        feature_dict: Dict[str, Any] = {}
        for key, value in visit.__dict__.items():
            if key in {"patient_id", "visit_id", "encounter_id", "y"}:
                continue
            feature_dict[key] = value

        # 3. Assemble one sample
        samples.append(
            {
                # unique within patient
                "sample_id": f"{patient.patient_id}_{idx}",
                "patient_id": patient.patient_id,
                "features": feature_dict,
                "label": label,
            }
        )

    return samples


class MRCICUMortalityTask(BaseTask):
    """Task wrapper for ICU mortality prediction on the Noisy MRC dataset.

    Input:
        - features: dict of tabular feature_name -> value

    Output:
        - label: binary mortality label (0: survived, 1: died)
    """

    task_name: str = "MRCICUMortality"
    # single tabular input, feature dict is the values of the row (attributes in yaml)
    input_schema: Dict[str, str] = {"features": "tabular"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        return mortality_prediction_mrc_fn(patient)
