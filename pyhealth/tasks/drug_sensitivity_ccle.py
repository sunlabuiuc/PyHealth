"""Drug sensitivity prediction task for the CCLE dataset.

Converts raw CCLE cell-line records into sample dicts compatible with
PyHealth's dataset pipeline and the CADRE model family.

This task is the recommended default when calling
``CCLEDataset.set_task()``.
"""

from typing import Dict, List

import numpy as np

from pyhealth.tasks.base_task import BaseTask


class DrugSensitivityPredictionCCLE(BaseTask):
    """Multi-label drug sensitivity prediction from CCLE gene expression profiles.

    Transforms a single CCLE *patient* record (one cancer cell line) into a
    sample dict containing the active-gene index sequence, binary drug
    sensitivity labels, and a tested-drug mask.

    Follows the PyHealth :class:`~pyhealth.tasks.BaseTask` interface:
    ``task_name``, ``input_schema``, ``output_schema``, and a callable
    ``__call__(patient) -> List[dict]``.

    **Input** patient dict (produced by
    :class:`~pyhealth.datasets.CCLEDataset`):

    .. code-block:: text

        patient_id       str               Cell-line identifier
        gene_expression  np.ndarray[int]   Binary indicator vector
        drug_sensitivity np.ndarray[float] Binary labels; NaN = untested
        drug_pathway_ids List[int]         Integer pathway ID per drug

    **Output** sample dict (one per cell line):

    .. code-block:: text

        patient_id       str        Cell-line identifier
        visit_id         str        Same as patient_id (one record per line)
        gene_indices     List[int]  1-indexed active gene positions
        labels           List[int]  Binary drug sensitivity labels
        mask             List[int]  1 = drug was tested, 0 = missing
        drug_pathway_ids List[int]  Integer pathway ID per drug

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks import DrugSensitivityPredictionCCLE
        >>> task = DrugSensitivityPredictionCCLE()
        >>> gene_expr = np.zeros(3000, dtype=int)
        >>> gene_expr[[10, 42]] = 1
        >>> drug_sens = np.array([1.0, np.nan, 0.0])
        >>> patient = {
        ...     "patient_id": "MCF7",
        ...     "gene_expression": gene_expr,
        ...     "drug_sensitivity": drug_sens,
        ...     "drug_pathway_ids": [0, 1, 2],
        ... }
        >>> samples = task(patient)
        >>> len(samples)
        1
        >>> samples[0]["gene_indices"]  # 1-indexed
        [11, 43]
        >>> samples[0]["mask"]
        [1, 0, 1]
    """

    task_name: str = "drug_sensitivity_prediction"

    input_schema: Dict = {
        "gene_indices": "sequence",
        "drug_pathway_ids": "sequence",
    }
    output_schema: Dict = {
        "labels": "raw",
        "mask": "raw",
    }

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, patient: Dict) -> List[Dict]:
        """Extract one sample dict from a CCLE cell-line patient record.

        Args:
            patient (dict): Must contain ``patient_id``, ``gene_expression``,
                ``drug_sensitivity``, and ``drug_pathway_ids``.

        Returns:
            List[dict]: Single-element list.
        """
        gene_vec = np.asarray(patient["gene_expression"])
        gene_indices = (np.where(gene_vec == 1)[0] + 1).tolist()

        sensitivity = np.asarray(patient["drug_sensitivity"], dtype=float)
        mask = (~np.isnan(sensitivity)).astype(int).tolist()
        labels = np.nan_to_num(sensitivity, nan=0.0).astype(int).tolist()

        return [
            {
                "patient_id": patient["patient_id"],
                "visit_id": patient["patient_id"],
                "gene_indices": gene_indices,
                "labels": labels,
                "mask": mask,
                "drug_pathway_ids": list(patient["drug_pathway_ids"]),
            }
        ]
