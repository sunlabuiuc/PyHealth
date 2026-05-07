"""
MIMIC-III Circulatory Failure Dataset for PyHealth.

Dataset:
    MIMIC-III Clinical Database v1.4
    https://physionet.org/content/mimiciii/1.4/

Inspired by:
    Hoche, M., Mineeva, O., Burger, M., Blasimme, A., & Ratsch, G. (2024). 
    FAMEWS: A fairness auditing tool for medical early-warning systems. 
    Proceedings of the Fifth Conference on Health, Inference, and Learning, 248, 297–311. PMLR. 
    https://proceedings.mlr.press/v248/hoche24a.html
    
Description:
    Configures the MIMIC-III tables required for a circulatory-failure
    early-warning task.  The dataset keeps data loading separate from
    task logic; sample generation is handled by
    ``CirculatoryFailurePredictionTask`` through the standard PyHealth
    ``dataset.set_task(task)`` pipeline.

Authors:
    Kuang-Yu Wang (kuangyu4@illinois.edu)
    Ya Hsuan Yang (yhyang3@illinois.edu)
"""

import logging
from pathlib import Path
from typing import List, Optional
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC3CirculatoryFailureDataset(BaseDataset):
    """MIMIC-III wrapper for circulatory failure early-warning prediction.

    This dataset configures the MIMIC-III tables required for a
    FAMEWS-inspired circulatory failure early-warning task. The dataset keeps
    data loading separate from task logic; sample generation is handled by
    ``CirculatoryFailurePredictionTask`` through the standard PyHealth
    ``dataset.set_task(task)`` pipeline.

    Args:
        root: Root directory of the MIMIC-III dataset.
        tables: Additional tables to load beyond the default tables.
        dataset_name: Name of the dataset instance.
        config_path: Path to the dataset config YAML file.
        **kwargs: Additional keyword arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import MIMIC3CirculatoryFailureDataset
        >>> from pyhealth.tasks import CirculatoryFailurePredictionTask
        >>> dataset = MIMIC3CirculatoryFailureDataset(
        ...     root="/path/to/mimic-iii",
        ... )
        >>> task = CirculatoryFailurePredictionTask(prediction_window_hours=12)
        >>> sample_dataset = dataset.set_task(task)
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the MIMIC-III circulatory failure dataset."""
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3_cf.yaml"

        default_tables = [
            "patients",
            "admissions",
            "icustays",
            "chartevents",
        ]

        if tables is None:
            tables = default_tables
        else:
            tables = list(dict.fromkeys(default_tables + tables))

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3_cf",
            config_path=str(config_path),
            **kwargs,
        )