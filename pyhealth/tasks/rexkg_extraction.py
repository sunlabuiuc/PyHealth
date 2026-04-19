"""
PyHealth task for radiology knowledge graph extraction using the CheXpert Plus
dataset and the ReXKG pipeline.

Dataset paper: (please cite if you use this dataset)
    Chambon, P., et al. "CheXpert Plus: Augmenting a Large Chest X-ray Dataset
    with Text Radiology Reports, Patient Demographics and Additional Image
    Format." arXiv:2405.19111 (2024).

ReXKG paper: (please also cite when using this task)
    Li, Z., et al. "ReXKG: A Structured Radiology Report Knowledge Graph for
    Chest X-ray Analysis." arXiv:2408.14397 (2024).

Authors:
    Aaron Miller (aaronm6@illinois.edu)
    Kathryn Thompson (kyt3@illinois.edu)
    Pushpendra Tiwari (pkt3@illinois.edu)
"""

import logging
from typing import Dict, List, Optional

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# NER labels used by the ReXKG PURE-based extractor (task: mimic01).
# These match the label set in src/ner/shared/const.py.
NER_LABELS: List[str] = [
    "Anatomy",
    "Observation",
    "Abnormality",
    "Device",
    "Descriptor",
    "Severity",
    "Size",
    "Uncertainty",
]

# Relation types produced by the ReXKG relation extractor.
RELATION_TYPES: List[str] = ["located_at", "modify", "measure"]


class RadiologyKGExtractionTask(BaseTask):
    """PyHealth task for extracting entities and relations from radiology
    reports to build a structured knowledge graph (ReXKG pipeline).

    Each sample contains the raw findings text for one radiology study. The
    model is expected to:

    1. Identify named entities (NER) from :attr:`NER_LABELS`.
    2. Classify pairwise relations between entity spans (:attr:`RELATION_TYPES`).
    3. (Optional) Link entities to UMLS concepts and build a KG.

    This task is intentionally kept label-free at the task level — the NER and
    relation labels are produced by the model at inference time, not by a
    supervised classification head over fixed classes. Consequently,
    ``output_schema`` uses ``"sequence"`` to store the raw predicted spans and
    relations as structured dicts.

    Attributes:
        task_name (str): Identifier for the task.
        input_schema (Dict[str, str]): Maps ``"text"`` to ``"str"``.
        output_schema (Dict[str, str]): Maps ``"entities"`` and
            ``"relations"`` to ``"sequence"``.

    Examples:
        >>> from pyhealth.datasets import CheXpertPlusDataset
        >>> from pyhealth.tasks import RadiologyKGExtractionTask
        >>> dataset = CheXpertPlusDataset(root="/path/to/chexpert_plus")
        >>> task = RadiologyKGExtractionTask()
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "RadiologyKGExtraction"
    input_schema: Dict[str, str] = {"text": "raw"}
    output_schema: Dict[str, str] = {
        "entities": "raw",
        "relations": "raw",
    }

    def __init__(
        self,
        findings_only: bool = True,
        min_text_length: int = 10,
    ) -> None:
        """Initializes the RadiologyKGExtractionTask.

        Args:
            findings_only (bool): If ``True`` (default), only the
                ``section_findings`` field is used as input text.  Set to
                ``False`` to concatenate ``section_impression`` as well.
            min_text_length (int): Minimum character length for a report to be
                included.  Reports shorter than this threshold are skipped.
                Defaults to 10.
        """
        super().__init__()
        self.findings_only = findings_only
        self.min_text_length = min_text_length

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generates extraction samples for a single patient (study).

        Args:
            patient (Patient): A patient object containing one or more
                ``chexpert_plus`` events.

        Returns:
            List[Dict]: One dict per event with keys:

                - ``"patient_id"`` (str): The study image path used as the
                  patient identifier.
                - ``"text"`` (str): The radiology findings text.
                - ``"entities"`` (list): Empty list at task-construction time;
                  populated by the model during inference.
                - ``"relations"`` (list): Empty list at task-construction time;
                  populated by the model during inference.

        Example::
            >>> samples = task(patient)
            >>> samples[0]["text"]
            'No acute cardiopulmonary process...'
        """
        events: List[Event] = patient.get_events(event_type="chexpert_plus")

        samples = []
        for event in events:
            findings = (event["section_findings"] if "section_findings" in event else None) or ""
            impression = (event["section_impression"] if "section_impression" in event else None) or ""

            if self.findings_only:
                text = findings.strip()
            else:
                parts = [s.strip() for s in [findings, impression] if s.strip()]
                text = " ".join(parts)

            if len(text) < self.min_text_length:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "text": text,
                    # Populated by ReXKGModel at inference time
                    "entities": [],
                    "relations": [],
                }
            )

        return samples
