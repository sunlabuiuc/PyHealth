import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class MIMICSBDHDataset(BaseDataset):
    """
    Dataset for MIMIC-SBDH: Social and Behavioral Determinants of Health annotations for MIMIC-III discharge summaries.
    https://github.com/hibaahsan/MIMIC-SBDH

    Args:
        root: root directory containing the dataset files
        dataset_name: optional name of dataset, defaults to "mimic_sbdh"
        config_path: optional configuration file, defaults to "mimic_sbdh.yaml"
    """
    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic_sbdh.yaml"
        metadata_file = Path(root) / "MIMIC-SBDH.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"{metadata_file} does not exist")
        default_tables = ["mimic_sbdh"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "mimic_sbdh",
            config_path=config_path
        )

    def prepare_metadata(self, root: str) -> None:
        """
        Optionally implement to preprocess or merge tables as needed.
        """
        pass
