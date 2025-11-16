import logging
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class Support2Dataset(BaseDataset):
    """
    A dataset class for handling SUPPORT2 (Study to Understand Prognoses and Preferences 
    for Outcomes and Risks of Treatments) data.

    The SUPPORT2 dataset contains data on seriously ill hospitalized adults, including
    patient demographics, diagnoses, clinical measurements, and outcomes.

    The dataset is available in R packages such as "rms" and "Hmisc".
    For more information, see: Knaus WA, Harrell FE, Lynn J, et al. The SUPPORT
    prognostic model: Objective estimates of survival for seriously ill hospitalized
    adults. Ann Intern Med. 1995;122(3):191-203.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "support2.yaml"
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "support2",
            config_path=config_path,
            **kwargs
        )
        return

