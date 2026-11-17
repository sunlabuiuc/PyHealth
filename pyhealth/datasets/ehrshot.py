import logging
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class EHRShotDataset(BaseDataset):
    """
    A dataset class for handling EHRShot data.

    This class is responsible for loading and managing the EHRShot dataset.

    Website: https://som-shahlab.github.io/ehrshot-website/

    Getting the data:
        - Source: https://redivis.com/datasets/53gc-8rhx41kgt (docs:
          https://som-shahlab.github.io/ehrshot-website/)
        - Access: restricted. Sign in to Redivis and request access, which
          requires signing the EHRSHOT research usage agreement.
        - Download: the ``EHRSHOT_ASSETS`` bundle from Redivis.
        - ``root``: the ``EHRSHOT_ASSETS`` folder, which holds ``data/``,
          ``splits/`` and ``benchmark/``.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import EHRShotDataset
        >>> # Load EHRShot dataset with benchmark tables
        >>> dataset = EHRShotDataset(
        ...     root="/path/to/EHRSHOT_ASSETS",
        ...     tables=["ehrshot", "chexpert", "guo_icu", "lab_anemia"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ehrshot.yaml"
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "ehrshot",
            config_path=config_path,
            **kwargs,
        )
        return
