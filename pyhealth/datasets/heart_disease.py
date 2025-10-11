import logging
from pathlib import Path
from typing import Optional

from ..tasks import HeartDiseasePrediction
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class HeartDiseaseDataset(BaseDataset):
    """Heart disease dataset from Kaggle
    Dataset is available at:
    https://www.kaggle.com/datasets/krishujeniya/heart-diseae
    Args:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset. Defaults to "heart_disease".
        config_path: Path to the configuration file. If None, uses default config.
    Attributes:
        root: Root directory of the raw data, stored as a csv file.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.
    Examples:
        >>> from pyhealth.datasets import HeartDiseaseDataset
        >>> dataset = HeartDiseaseDataset(
        ...     root="example_path",
        ... )
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "heart_disease.yaml"
            )
        default_tables = ["heart_disease"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "heart_disease",
            config_path=config_path,
        )
        return

    @property
    def default_task(self) -> HeartDiseasePrediction:
        """Returns the default task for this dataset."""
        return HeartDiseasePrediction()