''' Name: Szymon Szymura
    NetID: szymura2
    Paper title: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
    Paper link: https://www.biorxiv.org/content/10.1101/2024.06.18.599483v2.full
    Description: This is the code for cancer classification task using custom dataset of bulk RNA-seq data from TCGA. For simplicity data is obtained from cBioPortal datasets website: 
    https://www.cbioportal.org/datasets instead of original TCGA portal. Classification task uses cancer type as target variable and RNA-seq expression of top x genes as features
    for multiclass classification.
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

import polars as pl

try:
    from pyhealth.tasks import BaseTask
except ImportError:
    from pyhealth.tasks.base_task import BaseTask


class BulkRNACancerClassification(BaseTask):
    """Classification task for bulk RNA-seq gene expression data.
    This task generated one sample per patient using preprocessed gene expression
    features and assigns a classification label based on a selected 
    clinical or cohort field. 
    By default, the task performs cross-cancer classification using the ``cancer_type``
    field derived from study folder names. It can also be configured to perform subtype
    classificatino within a single cancer cohort.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict[str, str]): Schema for input features.
        output_schema (Dict[str, str]): Schema for output labels.
        label_field (str): Field used as classification label.
    Example:
        >>> task = BulkRNACancerClassification(label_field="cancer_type")
    """

    task_name = "bulk_rna_cancer_classification"

    input_schema = {"x": "tensor"}
    output_schema = {"y": "multiclass"}

    def __init__(self, label_field: str = "cancer_type") -> None:
        """Initialize classification task.
        Args:
            label_field (str): Column name used as classification label. Supported values include:
            "cancer_type" (default): cross-cohort classification
            "subtype": within-cancer subtype classification
        """
        self.label_field = label_field

    def __call__(self, patient) -> List[Dict[str, Any]]:
        
        row = patient.data_source.to_dicts()[0]

        expression_json = row.get("samples/expression_json")
        label = row.get(f"samples/{self.label_field}")

        if expression_json is None or label is None:
            return []

        # skip empty subtype rows if user chooses subtype classification

        if str(label).strip() == "" or str(label).lower() == "nan":
            return []

        x = json.loads(expression_json)

        return [
            {
                "patient_id": patient.patient_id,
                "x": x,
                "y": str(label),
            }
        ]