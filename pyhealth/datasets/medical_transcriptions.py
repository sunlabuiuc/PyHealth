import logging
from pathlib import Path
from typing import Optional

from ..tasks import MedicalTranscriptionsClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MedicalTranscriptionsDataset(BaseDataset):
    """Medical transcription data scraped from mtsamples.com.

    Dataset is available at:
    https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

    Args:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset. Defaults to "medical_transcriptions".
        config_path: Path to the configuration file. If None, uses default config.

    Attributes:
        root: Root directory of the raw data (should contain many csv files).
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MedicalTranscriptionsDataset
        >>> dataset = MedicalTranscriptionsDataset(
        ...     root="path/to/medical_transcriptions",
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
                Path(__file__).parent / "configs" / "medical_transcriptions.yaml"
            )
        default_tables = ["mtsamples"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "medical_transcriptions",
            config_path=config_path,
        )
        return

    @property
    def default_task(self) -> MedicalTranscriptionsClassification:
        """Returns the default task for this dataset."""
        return MedicalTranscriptionsClassification()
