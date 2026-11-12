"""TCGA survival prediction task for bulk RNA-seq.

Follows the PyHealth 2.1 survival pattern (see ``MetabricSurvivalPrediction``
and ``InHospitalSurvivalMIMIC4``): ``duration``/``event`` are ``"tensor"``
outputs, not label-processor fields, so the model computes its own Cox loss.
"""

from typing import Any, ClassVar

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

# The 33 TCGA cohorts, alphabetically. Used to encode the cohort label as int.
TCGA_COHORTS: tuple[str, ...] = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
COHORT_TO_INT: dict[str, int] = {c: i for i, c in enumerate(TCGA_COHORTS)}


class TCGASurvivalPrediction(BaseTask):
    """Survival prediction from TCGA bulk RNA-seq.

    Input: ``gene_expression`` — a fixed-length vector (n_genes or 256-d cached
    embedding, depending on the model mode).

    Outputs (all ``"tensor"``, so no ``LabelProcessor`` is created):
        - ``survival_time``: (1,) float tensor (days).
        - ``event``: (1,) float tensor (1.0 = death, 0.0 = censored).
        - ``cohort``: (1,) float tensor — int-encoded via ``COHORT_TO_INT``,
          for cohort-restricted Cox loss and per-cohort C-index.

    Args:
        cohort_to_int: Optional override of the cohort-to-int mapping.

    Example:
        >>> task = TCGASurvivalPrediction()
        >>> sample = task(dataset.get_patient("TCGA-XX-0001"))[0]
        >>> sample["cohort"]  # int-encoded
        [8.0]
    """

    task_name: str = "TCGASurvivalPrediction"
    input_schema: ClassVar[dict[str, str]] = {"gene_expression": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {
        "survival_time": "tensor",
        "event": "tensor",
        "cohort": "tensor",
    }

    def __init__(self, cohort_to_int: dict[str, int] | None = None) -> None:
        super().__init__()
        self.cohort_to_int = cohort_to_int or COHORT_TO_INT

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        clinical = patient.get_events(event_type="clinical")
        expression = patient.get_events(event_type="expression")
        if not clinical or not expression:
            return []

        clin = clinical[0]
        survival_time = float(clin["survival_time"])
        event = int(clin["event"])
        cohort_name = str(clin["cohort"])
        if cohort_name.startswith("TCGA-"):
            cohort_name = cohort_name[5:]
        if cohort_name not in self.cohort_to_int:
            raise ValueError(f"Unknown TCGA cohort: {cohort_name!r}")
        cohort_int = self.cohort_to_int[cohort_name]

        return [{
            "patient_id": patient.patient_id,
            "gene_expression": expression[0]["gene_expression"],
            "survival_time": [survival_time],
            "event": [float(event)],
            "cohort": [float(cohort_int)],
        }]