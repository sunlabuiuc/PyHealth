''' Name: Szymon Szymura
    NetID: szymura2
    Paper title: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
    Paper link: https://www.biorxiv.org/content/10.1101/2024.06.18.599483v2.full
    Description: This is the code for cancer survival task using custom dataset of bulk RNA-seq data from TCGA. For simplicity data is obtained from cBioPortal datasets website: 
    https://www.cbioportal.org/datasets instead of original TCGA portal. Survival task uses patient survival information as target variable and RNA-seq expression of top x genes 
    as features. Patient survival data is binned into bins and multiclass classification is performed.
    Files to review:
    - pyhealth/datasets/cbioportal_bulk_rna.py
    - pyhealth/tasks/bulk_rna_classification.py
    - pyhealth/tasks/bulk_rna_survival.py
    - tests/test_cbioportal_bulk_rna_dataset.py
    - tests/test_bulk_rna_tasks.py
    - examples/cbioportal_bulk_rna_showcase.ipynb
    - docs/api/datasets/pyhealth.datasets.cbioportal_bulk_rna.rst
    - docs/api/tasks/pyhealth.tasks.bulk_rna_cancer_classification.rst
    - docs/api/tasks/pyhealth.tasks.bulk_rna_survival_prediction.rst
'''


import json
from typing import Any, Dict, List, Optional

try:
    from pyhealth.tasks import BaseTask
except ImportError:
    from pyhealth.tasks.base_task import BaseTask


class BulkRNASurvivalPrediction(BaseTask):
    """Discretized overall survival prediction from bulk RNA-seq.
    This task generated one sample per patient using preprocessed gene 
    expression features and predicts a discretized overall survival label
    derived from clinical survival fields

    The task uses:
      - OS_MONTHS as survival time
      - OS_STATUS as event/censoring status

    Survival is converted into three bins:
      0: survival time < first bin edge
      1: first bin edge <= survival time < second bin edge
      2: survival time >= second bin edge

    This is not true time-to-event Cox survival modeling. Instead it is a
    simplified multiclass survival prediction task compatible with PyHealth's 
    standard supervised learning pipeline.

    Attributes:
        task_name (str): Name of the task
        input_schema (Dict[str, str]): Schema for input features.
        output_schema (Dict[str, str]): Schema for output labels.
        months_field (str): Field used for survival duration
        status_field (str): Field used for survival status.
        bin_edges (List[float]): Survival bin boundaries in months.
    
    Example:
        >>> task = BulkRNASurvivalPrediction()
    """

    task_name = "bulk_rna_survival_prediction"

    input_schema = {"x": "tensor"}
    output_schema = {"y": "multiclass"}

    def __init__(
        self,
        months_field: str = "os_months",
        status_field: str = "os_status",
        bin_edges: Optional[List[float]] = None,
    ) -> None:
        """Initialize survival prediction task.
        
        Args:
            months_field (str): Name of the field containing survival duration in months. Defaults to 'os_months'.
            status_field (str): Name of the field containing survival status. Defaults to 'os_status'.
            bin_edges (Optional[List[float]]): Survival bin boundaries in months. If not provided, defaults to '[12.0, 36.0]'
        """
 
        self.months_field = months_field
        self.status_field = status_field
        self.bin_edges = bin_edges or [12.0, 36.0]

    def __call__(self, patient) -> List[Dict[str, Any]]:
    
        row = patient.data_source.to_dicts()[0]

        expression_json = row.get("samples/expression_json")
        months = row.get(f"samples/{self.months_field}")
        status = row.get(f"samples/{self.status_field}")

        if expression_json is None or months is None or status is None:
            return []

    
        try:
            months = float(months)
        except (TypeError, ValueError):
            return []

        # cBioPortal often uses:
        #   "1:DECEASED", "0:LIVING"
        # or similar strings
        status_str = str(status).upper()

        # skip missing/unknown status

        if status_str.strip() == "" or status_str == "NAN":
            return []

        x = json.loads(expression_json)
        y = self._bin_survival(months)

        return [
            {
                "patient_id": patient.patient_id,
                "x": x,
                "y": y,
                "survival_months": months,
                "event": self._event_from_status(status_str),
            }
        ]

    def _bin_survival(self, months: float) -> int:

        if months < self.bin_edges[0]:
            return 0
        if months < self.bin_edges[1]:
            return 1
        return 2

    def _event_from_status(self, status_str: str) -> int:

        if "DECEASED" in status_str or status_str.startswith("1:"):
            return 1
        return 0