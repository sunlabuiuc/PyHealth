"""Cancer type classification task for TCGA RNA-seq data.

This module provides a PyHealth task class for classifying cancer type from
bulk RNA-seq gene expression profiles using the 33 TCGA cohort labels.
"""

import json
from typing import Any, Dict, List

import torch

from .base_task import BaseTask


class TCGARNASeqCancerTypeClassification(BaseTask):
    """Task for classifying cancer type from bulk RNA-seq gene expression.

    Given a patient's RNA-seq profile stored as a pre-normalized gene expression
    vector, this task maps each sample to one of the 33 TCGA cancer cohort
    labels. A patient may have multiple RNA-seq samples; one output dict is
    returned per sample.

    Attributes:
        task_name (str): Unique identifier for this task.
        input_schema (Dict[str, str]): Schema declaring ``gene_expression``
            as a tensor input.
        output_schema (Dict[str, str]): Schema declaring ``label`` as a
            multiclass output (0–32).
        COHORT_TO_LABEL (Dict[str, int]): Mapping from TCGA cohort string to
            integer class label.

    Note:
        Events whose cohort is not present in ``COHORT_TO_LABEL`` are silently
        skipped so that downstream code never receives an out-of-range label.

    Examples:
        >>> from pyhealth.tasks import TCGARNASeqCancerTypeClassification
        >>> task = TCGARNASeqCancerTypeClassification()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "TCGARNASeqCancerTypeClassification"
    input_schema: Dict[str, str] = {"gene_expression": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    COHORT_TO_LABEL: Dict[str, int] = {
        "ACC": 0,
        "BLCA": 1,
        "BRCA": 2,
        "CESC": 3,
        "CHOL": 4,
        "COAD": 5,
        "DLBC": 6,
        "ESCA": 7,
        "GBM": 8,
        "HNSC": 9,
        "KICH": 10,
        "KIRC": 11,
        "KIRP": 12,
        "LAML": 13,
        "LGG": 14,
        "LIHC": 15,
        "LUAD": 16,
        "LUSC": 17,
        "MESO": 18,
        "OV": 19,
        "PAAD": 20,
        "PCPG": 21,
        "PRAD": 22,
        "READ": 23,
        "SARC": 24,
        "SKCM": 25,
        "STAD": 26,
        "TGCT": 27,
        "THCA": 28,
        "THYM": 29,
        "UCEC": 30,
        "UCS": 31,
        "UVM": 32,
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient's RNA-seq events into cancer type classification samples.

        Retrieves all ``rnaseq`` events for the patient. Each event contributes
        one sample dict containing the gene expression tensor and integer cohort
        label. Events with an unrecognized cohort are skipped.

        Args:
            patient: A PyHealth ``Patient`` object with RNA-seq events accessible
                via ``patient.get_events(event_type="rnaseq")``.

        Returns:
            List[Dict[str, Any]]: One dict per valid RNA-seq event, each with
            keys:

            - ``patient_id`` (str): The patient identifier.
            - ``gene_expression`` (torch.FloatTensor): Shape ``(num_genes,)``,
              log10(1+TPM) max-normalized expression values.
            - ``label`` (int): Integer class label in [0, 32] corresponding
              to the TCGA cancer cohort.

            Returns an empty list if the patient has no ``rnaseq`` events or
            all events have unrecognized cohort strings.

        Examples:
            >>> task = TCGARNASeqCancerTypeClassification()
            >>> samples = task(patient)
            >>> len(samples)
            1
            >>> samples[0]["label"]
            2
        """
        events = patient.get_events(event_type="rnaseq")
        if not events:
            return []

        samples: List[Dict[str, Any]] = []
        for event in events:
            cohort = getattr(event, "cohort", None)
            if cohort is None or str(cohort) == "nan":
                continue
            cohort = str(cohort)
            if cohort not in self.COHORT_TO_LABEL:
                continue

            raw_ge = getattr(event, "gene_expression", None)
            if raw_ge is None or str(raw_ge) == "nan":
                continue

            gene_expression = torch.FloatTensor(json.loads(str(raw_ge)))

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "gene_expression": gene_expression,
                    "label": self.COHORT_TO_LABEL[cohort],
                }
            )

        return samples
