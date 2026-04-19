"""Synthea synthetic EHR dataset for PyHealth.

Author: Justin Xu

Paper: Raphael Poulain, Mehak Gupta, and Rahmatollah Beheshti.
    "CEHR-GAN-BERT: Incorporating Temporal Information from Structured EHR
    Data to Improve Prediction Tasks." MLHC 2022.
    https://proceedings.mlr.press/v182/poulain22a.html

Description: Wrapper for Synthea (https://synthetichealth.github.io/synthea/)
    synthetic patient records exported as CSV. Supports encounters,
    conditions, medications, and procedures tables.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class SyntheaDataset(BaseDataset):
    """A dataset class for handling Synthea synthetic EHR data.

    Synthea is an open-source synthetic patient generator that produces
    realistic patient records in a variety of formats. This class
    wraps the standard CSV export.

    The basic information is stored in the following tables:
        - patients: demographic info including birth/death dates,
            gender, race, and ethnicity.
        - encounters: clinical encounters with timestamps, class,
            codes, and reason information.

    We further support the following tables:
        - conditions: diagnoses recorded during encounters.
        - medications: medication orders linked to encounters.
        - procedures: procedures performed during encounters.

    Attributes:
        root (str): Root directory of the Synthea CSV export.
        tables (List[str]): Additional tables to include.
        dataset_name (Optional[str]): Name of the dataset.
        config_path (Optional[str]): Path to the YAML config file.

    Examples:
        >>> from pyhealth.datasets import SyntheaDataset
        >>> dataset = SyntheaDataset(
        ...     root="/path/to/synthea/csv",
        ...     tables=["conditions", "medications", "procedures"],
        ... )
        >>> dataset.stats()
        >>> patient = dataset.get_patient("patient_id")
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes SyntheaDataset with the given parameters.

        Args:
            root (str): Root directory of the Synthea CSV export.
            tables (List[str]): Additional tables to include.
            dataset_name (Optional[str]): Name of the dataset.
                Defaults to ``"synthea"``.
            config_path (Optional[str]): Path to the YAML config.
                If not provided, the bundled default config is used.
        """
        if config_path is None:
            logger.info(
                "No config path provided, using default config"
            )
            config_path = (
                Path(__file__).parent / "configs" / "synthea.yaml"
            )

        # Default tables loaded for every Synthea dataset
        default_tables = ["patients", "encounters"]
        tables = default_tables + tables

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "synthea",
            config_path=config_path,
            **kwargs,
        )
        return
