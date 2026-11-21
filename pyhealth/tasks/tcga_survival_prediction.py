"""TCGA survival prediction task for bulk RNA-seq.

Follows the PyHealth 2.1 survival pattern (see ``MetabricSurvivalPrediction``
and ``InHospitalSurvivalMIMIC4``): ``duration``/``event`` are ``"tensor"``
outputs, not label-processor fields, so the model computes its own Cox loss.
"""

import json
from typing import Any, ClassVar

import numpy as np

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

# The 33 TCGA cohorts, alphabetically. Used to encode the cohort label as int.
TCGA_COHORTS: tuple[str, ...] = (
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
)
COHORT_TO_INT: dict[str, int] = {c: i for i, c in enumerate(TCGA_COHORTS)}


class TCGASurvivalPrediction(BaseTask):
    """Survival prediction from TCGA bulk RNA-seq.

    Input: ``embedding`` — the cached 256-d BulkRNABert embedding of the
    patient's bulk RNA-seq profile. This task does not accept raw
    ~19,062-gene TPM vectors; embeddings are generated offline by the
    frozen BulkRNABert encoder (see the accompanying research repository).

    Outputs (all ``"tensor"``, so no ``LabelProcessor`` is created):
        - ``duration``: (1,) float tensor (days).
        - ``event``: (1,) float tensor (1.0 = death, 0.0 = censored).
        - ``cohort``: (1,) float tensor — int-encoded via ``COHORT_TO_INT``,
          for cohort-restricted Cox loss and per-cohort C-index.

    Args:
        cohort_to_int: Optional override of the cohort-to-int mapping.

    Note:
        Malformed rows (non-positive or non-finite duration, event not in
        {0, 1}, unknown cohort) raise ``ValueError``. The shipped
        ``tcga_rna_seq.csv`` is pre-validated upstream, so this only
        surfaces if a user supplies their own CSV.

    Example:
        >>> from pyhealth.datasets import TCGARNASeqDataset
        >>> from pyhealth.tasks import TCGASurvivalPrediction
        >>> ds = TCGARNASeqDataset.from_huggingface()  # doctest: +SKIP
        >>> task = TCGASurvivalPrediction()
        >>> sample = task(ds.get_patient("TCGA-BRCA-0001"))[0]  # doctest: +SKIP
        >>> sample["cohort"]  # int-encoded BRCA index  # doctest: +SKIP
        [2.0]
    """

    task_name: str = "TCGASurvivalPrediction"
    input_schema: ClassVar[dict[str, str]] = {"embedding": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {
        "duration": "tensor",
        "event": "tensor",
        "cohort": "tensor",
    }
    COHORT_TO_INT: ClassVar[dict[str, int]] = COHORT_TO_INT
    INT_TO_COHORT: ClassVar[dict[int, str]] = {i: c for c, i in COHORT_TO_INT.items()}

    def __init__(self, cohort_to_int: dict[str, int] | None = None) -> None:
        super().__init__()
        self.cohort_to_int = (
            dict(COHORT_TO_INT) if cohort_to_int is None else dict(cohort_to_int)
        )
        self.int_to_cohort = {idx: cohort for cohort, idx in self.cohort_to_int.items()}

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        events = patient.get_events(event_type="tcga_rna_seq")
        if not events:
            return []
        if len(events) > 1:
            raise ValueError(
                f"Expected exactly one tcga_rna_seq event for patient "
                f"{patient.patient_id}, got {len(events)}"
            )
        e = events[0]

        try:
            raw_dur = e["duration"]
            raw_ev = e["event"]
            raw_co = e["cohort"]
            raw_emb = e["embedding"]
        except KeyError as err:
            raise ValueError(
                f"Missing field {err} in event for patient {patient.patient_id}"
            ) from err

        try:
            duration = float(raw_dur)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid duration for patient {patient.patient_id}: {raw_dur!r}"
            ) from err
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"duration must be finite and positive, got {duration}")

        try:
            event_val = float(raw_ev)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid event indicator for patient {patient.patient_id}: {raw_ev!r}"
            ) from err
        if event_val not in (0.0, 1.0):
            raise ValueError(f"Event must be 0 or 1, got {event_val}")

        cohort_name = str(raw_co)
        cohort_name = cohort_name.removeprefix("TCGA-")
        if cohort_name not in self.cohort_to_int:
            raise ValueError(f"Unknown TCGA cohort: {cohort_name!r}")
        cohort_int = self.cohort_to_int[cohort_name]

        # The dataset stores the embedding as a JSON list string in the CSV.
        try:
            embedding = np.asarray(json.loads(raw_emb), dtype=np.float32)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid embedding for patient {patient.patient_id}: {err}"
            ) from err
        if embedding.shape != (256,):
            raise ValueError(f"Embedding must be shape (256,), got {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains non-finite values")

        return [
            {
                "patient_id": patient.patient_id,
                "embedding": embedding,
                "duration": [duration],
                "event": [event_val],
                "cohort": [float(cohort_int)],
            }
        ]
