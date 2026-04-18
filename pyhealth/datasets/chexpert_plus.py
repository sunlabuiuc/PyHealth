"""
PyHealth dataset for the CheXpert Plus dataset.

Dataset link:
    https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1

Dataset paper: (please cite if you use this dataset)
    Chambon, P., et al. "CheXpert Plus: Augmenting a Large Chest X-ray Dataset with
    Text Radiology Reports, Patient Demographics and Additional Image Format."
    arXiv:2405.19111 (2024).

Dataset paper link:
    https://arxiv.org/abs/2405.19111

ReXKG paper: (please also cite if you use this module for KG extraction)
    Li, Z., et al. "ReXKG: A Structured Radiology Report Knowledge Graph for
    Chest X-ray Analysis." arXiv:2408.14397 (2024).

ReXKG paper link:
    https://arxiv.org/abs/2408.14397

Authors:
    Aaron Miller (aaronm6@illinois.edu)
    Kathryn Thompson (kyt3@illinois.edu)
    Pushpendra Tiwari (pkt3@illinois.edu)
"""

import logging
import os
from pathlib import Path
from typing import Optional

from pyhealth.datasets import BaseDataset
from pyhealth.tasks import RadiologyKGExtractionTask

logger = logging.getLogger(__name__)


class CheXpertPlusDataset(BaseDataset):
    """Dataset class for the CheXpert Plus chest X-ray report dataset.

    CheXpert Plus augments the original CheXpert dataset with structured
    free-text radiology reports. Each record corresponds to a single chest
    X-ray study identified by its image path. This class exposes the
    ``section_findings`` text for downstream NLP/NER/KG tasks.

    Attributes:
        root (str): Root directory containing ``df_chexpert_plus_240401.csv``.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the YAML configuration file.

    Example::
        >>> dataset = CheXpertPlusDataset(root="/path/to/chexpert_plus")
        >>> print(dataset.stats())
    """

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "chexpert_plus.yaml"
        ),
        **kwargs,
    ) -> None:
        """Initializes the CheXpert Plus dataset.

        Args:
            root (str): Root directory that contains
                ``df_chexpert_plus_240401.csv``. Defaults to the working
                directory.
            config_path (Optional[str]): Path to the YAML configuration file.
                Defaults to the bundled ``configs/chexpert_plus.yaml``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~pyhealth.datasets.BaseDataset`.

        Raises:
            FileNotFoundError: If the CSV file is not found under ``root``.

        Example::
            >>> dataset = CheXpertPlusDataset(root="./data")
        """
        self._csv_path = os.path.join(root, "df_chexpert_plus_240401.csv")
        self._verify_data(root)

        super().__init__(
            root=root,
            tables=["chexpert_plus"],
            dataset_name="CheXpertPlus",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self) -> RadiologyKGExtractionTask:
        """Returns the default KG extraction task for this dataset.

        Returns:
            RadiologyKGExtractionTask: Entity and relation extraction task.

        Example::
            >>> dataset = CheXpertPlusDataset(root="./data")
            >>> task = dataset.default_task
        """
        return RadiologyKGExtractionTask()

    def _verify_data(self, root: str) -> None:
        """Verifies the presence of the required CSV file.

        Args:
            root (str): Root directory to check.

        Raises:
            FileNotFoundError: If ``df_chexpert_plus_240401.csv`` is missing.
        """
        if not os.path.isfile(self._csv_path):
            raise FileNotFoundError(
                f"CheXpert Plus CSV not found at '{self._csv_path}'. "
                "Download the dataset from "
                "https://stanfordaimi.azurewebsites.net/datasets/"
                "5158c524-d3ab-4e02-96e9-6ee9efc110a1"
            )
