"""PyHealth dataset for synthetic EHR-style clinical notes.

This dataset provides a compact, fully synthetic collection of clinical note
snippets labelled with a target diagnosis condition and a ground-truth
"is positive" flag. It is intended for reproducible, dependency-free
experimentation with evidence-retrieval workflows and is the primary
fall-back when MIMIC-III access is unavailable.

Dataset paper (implementation inspiration):
    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Dataset paper link:
    https://proceedings.mlr.press/v248/ahsan24a.html

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import csv
import logging
import os
from pathlib import Path
from typing import List, Optional

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


# A tiny, reproducible corpus of synthetic note snippets. Each row is shaped
# to approximate the structure of MIMIC-III discharge/radiology excerpts
# without containing any real patient information.
_SYNTHETIC_ROWS: List[dict] = [
    {
        "patient_id": "p001",
        "note_id": "n0001",
        "note_type": "discharge",
        "condition": "intracranial hemorrhage",
        "is_positive": 1,
        "text": (
            "Patient is a 62 year old male on warfarin following recent "
            "craniotomy. Mental status deteriorated acutely. CT head shows "
            "acute intraparenchymal hemorrhage in the right frontal lobe. "
            "Anticoagulation held and neurosurgery consulted."
        ),
    },
    {
        "patient_id": "p001",
        "note_id": "n0002",
        "note_type": "radiology",
        "condition": "intracranial hemorrhage",
        "is_positive": 1,
        "text": (
            "Non-contrast CT head: hyperdense focus in the right frontal "
            "lobe measuring 2.1 cm consistent with acute hemorrhage. No "
            "midline shift. Prior craniotomy changes are noted."
        ),
    },
    {
        "patient_id": "p002",
        "note_id": "n0003",
        "note_type": "discharge",
        "condition": "intracranial hemorrhage",
        "is_positive": 0,
        "text": (
            "Patient admitted for elective knee arthroplasty. Postoperative "
            "course uneventful. No neurologic symptoms. Discharged home on "
            "post-op day two with physical therapy follow-up."
        ),
    },
    {
        "patient_id": "p003",
        "note_id": "n0004",
        "note_type": "discharge",
        "condition": "stroke",
        "is_positive": 1,
        "text": (
            "History of atrial fibrillation off anticoagulation presenting "
            "with acute left sided weakness and dysarthria. MRI brain "
            "confirms right middle cerebral artery infarct. Started on "
            "aspirin and statin."
        ),
    },
    {
        "patient_id": "p003",
        "note_id": "n0005",
        "note_type": "radiology",
        "condition": "stroke",
        "is_positive": 1,
        "text": (
            "Diffusion weighted MRI demonstrates restricted diffusion in "
            "the right MCA territory compatible with acute ischemic stroke."
        ),
    },
    {
        "patient_id": "p004",
        "note_id": "n0006",
        "note_type": "discharge",
        "condition": "stroke",
        "is_positive": 0,
        "text": (
            "Patient evaluated in clinic for routine diabetes management. "
            "A1c improved. No focal neurologic deficits on exam. Plan to "
            "continue metformin and lifestyle modification."
        ),
    },
    {
        "patient_id": "p005",
        "note_id": "n0007",
        "note_type": "discharge",
        "condition": "pneumonia",
        "is_positive": 1,
        "text": (
            "Fever, productive cough, and right lower lobe crackles. Chest "
            "X-ray with right lower lobe consolidation. Started on "
            "ceftriaxone and azithromycin for community acquired pneumonia."
        ),
    },
    {
        "patient_id": "p005",
        "note_id": "n0008",
        "note_type": "radiology",
        "condition": "pneumonia",
        "is_positive": 0,
        "text": (
            "Chest radiograph: lungs clear bilaterally. No focal "
            "consolidation, pleural effusion, or pneumothorax."
        ),
    },
]

_CONDITIONS: List[str] = sorted({row["condition"] for row in _SYNTHETIC_ROWS})


class SyntheticEHRNotesDataset(BaseDataset):
    """Dataset class for a compact synthetic EHR-notes corpus.

    The corpus ships with the PyHealth repository and can be materialized
    on demand into a CSV that conforms to the
    ``synthetic_ehr_notes.yaml`` schema, which keeps the dataset fully
    self-contained and CI friendly.

    Attributes:
        root (str): Root directory where the synthetic CSV will be
            written (and later read) from.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the YAML configuration file.
        conditions (List[str]): Diagnosis conditions covered by the
            synthetic corpus.
    """

    conditions: List[str] = _CONDITIONS

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "synthetic_ehr_notes.yaml"
        ),
        generate: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the synthetic EHR notes dataset.

        Args:
            root (str): Root directory for the materialized synthetic
                CSV. Defaults to the current working directory.
            config_path (Optional[str]): Path to the YAML configuration
                file. Defaults to the packaged
                ``configs/synthetic_ehr_notes.yaml``.
            generate (bool): Whether to (re)generate the synthetic CSV
                on disk when it is missing. Defaults to ``True``.

        Raises:
            FileNotFoundError: If the dataset root does not exist after
                generation is disabled and no CSV is present.
            ValueError: If the materialized CSV is empty or malformed.

        Example:
            >>> dataset = SyntheticEHRNotesDataset(root="./synthetic_notes")
        """
        self._csv_path: str = os.path.join(root, "synthetic_notes.csv")

        if generate:
            self._generate(root)

        self._verify(root)

        super().__init__(
            root=root,
            tables=["notes"],
            dataset_name="SyntheticEHRNotes",
            config_path=config_path,
            **kwargs,
        )

    def _generate(self, root: str) -> None:
        """Materialize the synthetic corpus to ``<root>/synthetic_notes.csv``.

        Args:
            root (str): Root directory for the synthetic CSV.

        Raises:
            OSError: If the root directory cannot be created.
        """
        os.makedirs(root, exist_ok=True)
        columns = [
            "patient_id",
            "note_id",
            "note_type",
            "condition",
            "is_positive",
            "text",
        ]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in _SYNTHETIC_ROWS:
                writer.writerow(row)

    def _verify(self, root: str) -> None:
        """Verify that the synthetic corpus is present and non-empty.

        Args:
            root (str): Root directory for the synthetic CSV.

        Raises:
            FileNotFoundError: If the root directory or CSV is missing.
            ValueError: If the CSV exists but contains no data rows.
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        if not os.path.isfile(self._csv_path):
            raise FileNotFoundError(
                f"Synthetic notes CSV is missing: {self._csv_path}. "
                "Re-initialize the dataset with generate=True."
            )
        with open(self._csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError(
                "Synthetic notes CSV must contain at least one data row."
            )
