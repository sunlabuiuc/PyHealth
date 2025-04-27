import logging
from pathlib import Path
from typing import Optional

from ..tasks import DiabetesPrediction
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class DiabetesDataset(BaseDataset):
    """Diabetes dataset from the Indian National Institute of Diabetes and Digestive and Kidney Diseases

    Dataset is available at:
    https://www.kaggle.com/datasets/mathchi/diabetes-data-set

    All patients are females of at least 21 years of age of Pima Indian heritage.

    Args:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset. Defaults to "diabetes".
        config_path: Path to the configuration file. If None, uses default config.

    Attributes:
        root: Root directory of the raw data (should contain many csv files).
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import DiabetesDataset
        >>> dataset = DiabetesDataset(
        ...     root="path/to/diabetes",
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
                Path(__file__).parent / "configs" / "diabetes.yaml"
            )
        default_tables = ["diabetes"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "diabetes",
            config_path=config_path,
        )
        return

    @property
    def default_task(self) -> DiabetesPrediction:
        """Returns the default task for this dataset."""
        return DiabetesPrediction()
