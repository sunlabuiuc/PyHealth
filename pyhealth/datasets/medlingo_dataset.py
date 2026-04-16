"""
MedLingo Dataset for PyHealth 2.0.

MedLingo is a medical abbreviation/acronym disambiguation dataset containing
clinical note abbreviations paired with their full expansions and contextual
fill-in-the-blank questions.

Dataset source:
    https://github.com/Flora-jia-jfr/diagnosing_our_datasets/tree/main/datasets/MedLingo

Contributor: [Your Name] ([your@email.edu])
Contribution Type: New Dataset
Description: Added support for the MedLingo medical abbreviation dataset.
Files to Review:
    - pyhealth/datasets/medlingo_dataset.py
    - pyhealth/datasets/configs/medlingo_dataset.yaml
    - pyhealth/tasks/medlingo_task.py
    - tests/test_medlingo_dataset.py
"""

import logging
import os
from typing import Optional

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MedLingoDataset(BaseDataset):
    """PyHealth 2.0 dataset class for the MedLingo medical abbreviation benchmark.

    MedLingo provides 100 clinical abbreviation entries, each containing:
      - An abbreviation (e.g., "PRN")
      - Its full expansion (e.g., "as needed")
      - A contextualised fill-in-the-blank question
      - The expected answer string

    In PyHealth 2.0, the dataset is entirely config-driven. The YAML config at
    ``configs/medlingo_dataset.yaml`` tells BaseDataset how to read
    ``questions.csv`` and map its columns to the standard
    ``patient_id / event_type / timestamp / attributes`` schema.

    Because MedLingo has no real patient structure, ``word1`` (the abbreviation)
    is used as ``patient_id`` so each abbreviation becomes one "patient" in the
    PyHealth data model. This follows the same pattern used by non-EHR datasets
    in PyHealth (e.g., COVID19CXRDataset).

    Dataset homepage:
        https://github.com/Flora-jia-jfr/diagnosing_our_datasets

    Args:
        root (str): Root directory that contains ``questions.csv``.
        dataset_name (Optional[str]): Name override. Defaults to
            ``"MedLingoDataset"``.
        config_path (Optional[str]): Path to a custom ``dataset.yaml``.
            Defaults to the bundled ``configs/medlingo_dataset.yaml``.
        cache_dir (Optional[str]): Directory for caching processed data.
        num_workers (int): Number of workers for parallel processing.
        dev (bool): If ``True``, limits to 1000 patients. Defaults to ``False``.

    Examples:
        >>> from pyhealth.datasets import MedLingoDataset
        >>> dataset = MedLingoDataset(root="test-resources/MedLingo")
        >>> dataset.stats()
        >>> for patient in dataset.iter_patients():
        ...     print(patient.patient_id)
    """

    # Default config bundled alongside this file
    _DEFAULT_CONFIG = os.path.join(
        os.path.dirname(__file__), "configs", "medlingo_dataset.yaml"
    )

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        # Validate that questions.csv exists before handing off to BaseDataset
        csv_path = os.path.join(root, "questions.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"MedLingoDataset: 'questions.csv' not found in '{root}'. "
                "Download it from https://github.com/Flora-jia-jfr/"
                "diagnosing_our_datasets/tree/main/datasets/MedLingo"
            )

        # Use bundled config if none provided
        resolved_config = config_path or self._DEFAULT_CONFIG

        super().__init__(
            root=root,
            tables=["questions"],
            dataset_name=dataset_name or "MedLingoDataset",
            config_path=resolved_config,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def get_abbreviation(self, abbrev: str):
        """Return the ``Patient`` object for a given abbreviation.

        Convenience wrapper around :meth:`get_patient` so callers can
        use domain-specific language.

        Args:
            abbrev: The medical abbreviation string, e.g. ``"PRN"``.

        Returns:
            Matching ``Patient`` or ``None`` if not found.
        """
        try:
            return self.get_patient(abbrev)
        except AssertionError:
            return None

    def default_task(self):
        """Return the default abbreviation-expansion task for this dataset."""
        from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo
        return AbbreviationExpansionMedLingo()