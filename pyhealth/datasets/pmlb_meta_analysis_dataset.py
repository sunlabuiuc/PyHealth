"""PMLB Meta-Analysis Dataset for PyHealth.

Dataset for conformal meta-analysis experiments using regression
datasets from the Penn Machine Learning Benchmark (PMLB).

PMLB is freely accessible at:
    https://github.com/EpistasisLab/pmlb
    Install: pip install pmlb

The three datasets used in Kaul & Gordon (2024) are:
    - 1196_BNG_pharynx
    - 1201_BNG_breastTumor
    - 1193_BNG_lowbwt

These regression datasets provide features X and target values Y
(used as true effects U in the meta-analysis simulations). Synthetic
prior means M and noise variances V are generated according to
controlled parameters (prior_error, effect_noise).

Reference:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    In Proceedings of Machine Learning Research, volume 259, 563-593.

    Olson, R. S., La Cava, W., Orzechowski, P., Urbanowicz, R. J.,
    and Moore, J. H. 2017. PMLB: a large benchmark suite for machine
    learning evaluation and comparison. BioData Mining, 10:1-13.

Examples:
    >>> from pyhealth.datasets import PMLBMetaAnalysisDataset
    >>> dataset = PMLBMetaAnalysisDataset(
    ...     root="/path/to/pmlb_data",
    ...     pmlb_dataset_name="1196_BNG_pharynx",
    ... )
    >>> dataset.stats()
    >>> samples = dataset.set_task()
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# The three PMLB datasets used in Kaul & Gordon (2024)
SUPPORTED_PMLB_DATASETS = [
    "1196_BNG_pharynx",
    "1201_BNG_breastTumor",
    "1193_BNG_lowbwt",
]


class PMLBMetaAnalysisDataset(BaseDataset):
    """PMLB regression dataset for conformal meta-analysis experiments.

    Each row in the PMLB dataset is treated as a simulated clinical
    trial. The mapping to PyHealth's Patient-Visit-Event structure is:

        - Patient = one data point (simulated trial)
        - Visit   = single observation for that trial
        - Event   = the features and target value

    The dataset fetches data from PMLB via the ``pmlb`` Python package
    on first use and caches a processed CSV locally.

    Args:
        root: Directory where the processed CSV will be stored.
        pmlb_dataset_name: Name of the PMLB dataset. Must be one of
            "1196_BNG_pharynx", "1201_BNG_breastTumor", or
            "1193_BNG_lowbwt".
        dataset_name: Optional name override. Defaults to
            "pmlb_{pmlb_dataset_name}".
        config_path: Optional path to config YAML. If None, uses
            the default config in the configs directory.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of parallel workers. Defaults to 1.
        dev: If True, loads only a small subset for development.

    Attributes:
        root: Root directory for data storage.
        pmlb_dataset_name: The PMLB dataset being used.
        feature_columns: List of feature column names.

    Examples:
        >>> dataset = PMLBMetaAnalysisDataset(
        ...     root="./data/pmlb",
        ...     pmlb_dataset_name="1196_BNG_pharynx",
        ... )
        >>> print(len(dataset.patients))
    """

    def __init__(
        self,
        root: str,
        pmlb_dataset_name: str = "1196_BNG_pharynx",
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if pmlb_dataset_name not in SUPPORTED_PMLB_DATASETS:
            raise ValueError(
                f"pmlb_dataset_name must be one of {SUPPORTED_PMLB_DATASETS}, "
                f"got '{pmlb_dataset_name}'"
            )

        self.pmlb_dataset_name = pmlb_dataset_name

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "pmlb_meta_analysis.yaml"
            )

        # Prepare the CSV if it doesn't exist yet
        csv_name = "pmlb_meta_analysis-metadata-pyhealth.csv"
        if not os.path.exists(os.path.join(root, csv_name)):
            self.prepare_metadata(root, pmlb_dataset_name)

        default_tables = ["pmlb_meta_analysis"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or f"pmlb_{pmlb_dataset_name}",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str, pmlb_dataset_name: str) -> None:
        """Fetch PMLB data and save as a PyHealth-compatible CSV.

        Downloads the dataset using the ``pmlb`` package, adds
        patient/visit identifiers, and saves to the root directory.

        Args:
            root: Directory to save the CSV file.
            pmlb_dataset_name: Name of the PMLB dataset to fetch.

        Raises:
            ImportError: If the ``pmlb`` package is not installed.
        """
        try:
            from pmlb import fetch_data
        except ImportError:
            raise ImportError(
                "The 'pmlb' package is required to fetch PMLB datasets. "
                "Install it with: pip install pmlb"
            )

        logger.info(f"Fetching PMLB dataset: {pmlb_dataset_name}")
        data = fetch_data(pmlb_dataset_name)

        df = pd.DataFrame(data.values, columns=data.columns)
        df.insert(0, "patient_id", [f"trial_{i}" for i in range(len(df))])
        df.insert(1, "visit_id", [f"visit_{i}" for i in range(len(df))])
        df = df.rename(columns={"target": "true_effect"})

        os.makedirs(root, exist_ok=True)
        csv_path = os.path.join(root, "pmlb_meta_analysis-metadata-pyhealth.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved PMLB metadata to {csv_path}")

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            ConformalMetaAnalysisTask: The default meta-analysis task.
        """
        # Import here to avoid circular imports
        from pyhealth.tasks.conformal_meta_analysis_task import (
            ConformalMetaAnalysisTask,
        )
        return ConformalMetaAnalysisTask()
