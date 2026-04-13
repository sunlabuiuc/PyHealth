# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Binary prediction task for mortality and length of stay
"""MIMIC-Extract flat-feature tasks for CaliForest.

Defines a single parameterised task that converts MIMIC-Extract
hourly time-series data into a flat feature vector suitable for
tree-based models such as CaliForest.  Four binary prediction
targets are supported:

* ``mort_hosp`` – in-hospital mortality
* ``mort_icu``  – in-ICU mortality
* ``los_3``     – ICU length-of-stay > 3 days
* ``los_7``     – ICU length-of-stay > 7 days

Paper: Y. Park and J. C. Ho. "CaliForest: Calibrated Random Forest
for Health Data." ACM CHIL, 2020.
"""

from typing import Any, Dict, List

from .base_task import BaseTask


class MIMICExtractCaliForestTask(BaseTask):
    """Flat-feature binary prediction task for MIMIC-Extract data.

    This task is designed for the ``MIMICExtractFlatDataset`` which
    already produces flat feature vectors and labels.  The task
    simply selects the requested prediction target.

    Args:
        target: One of ``"mort_hosp"``, ``"mort_icu"``,
            ``"los_3"``, or ``"los_7"``.  Default ``"mort_hosp"``.

    Examples:
        >>> task = MIMICExtractCaliForestTask(target="mort_hosp")
        >>> task.task_name
        'mimic_extract_califorest_mort_hosp'
        >>> task.output_schema
        {'label': 'binary'}
    """

    VALID_TARGETS = ("mort_hosp", "mort_icu", "los_3", "los_7")

    def __init__(self, target: str = "mort_hosp") -> None:
        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"target must be one of {self.VALID_TARGETS}, "
                f"got '{target}'"
            )
        self.target = target
        self.task_name = f"mimic_extract_califorest_{target}"
        self.input_schema: Dict[str, str] = {
            "features": "tensor",
        }
        self.output_schema: Dict[str, str] = {
            "label": "binary",
        }

    def __call__(self, patient: Any) -> List[Dict]:
        """Extracts a single sample from a patient record.

        Expects the patient object to carry ``features`` (a flat
        numeric vector) and the target label attribute.

        Args:
            patient: A ``Patient`` object from
                ``MIMICExtractFlatDataset``.

        Returns:
            A list containing one sample dictionary, or an empty
            list if the patient lacks the required data.
        """
        features = getattr(patient, "features", None)
        label = getattr(patient, self.target, None)
        if features is None or label is None:
            return []
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "label": int(label),
            }
        ]
