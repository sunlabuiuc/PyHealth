import logging
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ExpGDSCDataset(BaseDataset):
    """
    Dataset is available at:
    https://github.com/yifengtao/CADRE/blob/master/data/input/exp_gdsc.csv

    Args:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset. Defaults to "exp_gdsc".
        config_path: Path to the configuration file. If None, uses default config.

    Attributes:
        root: Root directory of the raw data (should contain csv file).
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import ExpGDSCDataset
        >>> dataset = ExpGDSCDataset(
        ...     root="path/to/exp_gdsc",
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
                Path(__file__).parent / "configs" / "exp_gdsc.yaml"
            )
        default_tables = ["mtsamples"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "exp_gdsc",
            config_path=config_path,
        )
        return


