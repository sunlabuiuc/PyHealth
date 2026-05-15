"""Drug sensitivity prediction task for the GDSC dataset.

Converts raw GDSC cell-line records into sample dicts compatible with
PyHealth's dataset pipeline and the CADRE model family.

This task is the recommended default when calling
``GDSCDataset.set_task()``.
"""

from typing import Dict, List

import numpy as np

from pyhealth.tasks.base_task import BaseTask


class DrugSensitivityPredictionGDSC(BaseTask):
    """Multi-label drug sensitivity prediction from gene expression profiles.

    Transforms a single GDSC *patient* record (one cancer cell line) into a
    sample dict containing the active-gene index sequence, binary drug
    sensitivity labels, and a tested-drug mask.

    Follows the PyHealth :class:`~pyhealth.tasks.BaseTask` interface:
    ``task_name``, ``input_schema``, ``output_schema``, and a callable
    ``__call__(patient) -> List[dict]``.

    **Input** patient dict (produced by
    :class:`~pyhealth.datasets.GDSCDataset`):

    .. code-block:: text

        patient_id       str               COSMIC cell-line identifier
        gene_expression  np.ndarray[int]   Binary indicator vector, shape (3000,)
        drug_sensitivity np.ndarray[float] Binary labels; NaN = untested, shape (260,)
        drug_pathway_ids List[int]         Integer pathway ID per drug, length 260

    **Output** sample dict (one per cell line):

    .. code-block:: text

        patient_id       str        COSMIC cell-line identifier
        visit_id         str        Same as patient_id (one record per line)
        gene_indices     List[int]  1-indexed active gene positions (~1500 entries)
        labels           List[int]  Binary drug sensitivity labels (260,)
        mask             List[int]  1 = drug was tested, 0 = missing (260,)
        drug_pathway_ids List[int]  Integer pathway ID per drug (260,)

    The ``gene_indices`` field uses 1-based indexing so that index 0 is
    reserved as the padding token in the Gene2Vec embedding table.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks import DrugSensitivityPredictionGDSC
        >>> task = DrugSensitivityPredictionGDSC()
        >>> gene_expr = np.zeros(3000, dtype=int)
        >>> gene_expr[[1, 5, 99]] = 1          # 3 active genes
        >>> drug_sens = np.array([1.0, 0.0, np.nan])   # 2 tested, 1 missing
        >>> patient = {
        ...     "patient_id": "COSMIC.906826",
        ...     "gene_expression": gene_expr,
        ...     "drug_sensitivity": drug_sens,
        ...     "drug_pathway_ids": [0, 1, 2],
        ... }
        >>> samples = task(patient)
        >>> len(samples)
        1
        >>> samples[0]["gene_indices"]  # 1-indexed
        [2, 6, 100]
        >>> samples[0]["labels"]
        [1, 0, 0]
        >>> samples[0]["mask"]
        [1, 1, 0]
    """

    task_name: str = "drug_sensitivity_prediction"

    # Processor type strings tell PyHealth processors how to handle each field.
    # "sequence" = variable-length integer list; "multilabel" = fixed-length binary vector.
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
        """Extract one sample dict from a GDSC cell-line patient record.

        Args:
            patient (dict): Must contain ``patient_id``, ``gene_expression``
                (shape ``(3000,)``), ``drug_sensitivity`` (shape ``(260,)``),
                and ``drug_pathway_ids`` (length ``260``).

        Returns:
            List[dict]: Single-element list; the GDSC data model has one
            record per cell line.
        """
        gene_vec = np.asarray(patient["gene_expression"])
        # 1-indexed: embedding row 0 is reserved for padding
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
