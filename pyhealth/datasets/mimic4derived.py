import logging
from pathlib import Path
from typing import Optional
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class Mimic4DerivedDataset(BaseDataset):
    """Derived Dataset for MIMIC-IV containing ventilation durations

    Dataset is available to be derived from the following link for the Metavision information system:
    https://physionet.org/content/mimiciv/3.1/
    Transformations derived from the following:
    https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_durations.sql
    

    Args:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset. Defaults to "mimic4derived".
        config_path: Path to the configuration file. If None, uses default config.

    Attributes:
        root: Root directory of the raw data (should contain many csv files).
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import Mimic4DerivedDataset
        >>> dataset = Mimic4DerivedDataset(
        ...     root="path/to/mimic4derived",
        ...     dataset_name="VentData"        
        ... )
        >>> dataset.stats()
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
                Path(__file__).parent / "configs" / "mimic4_derived.yaml"
            )
        default_tables = ["vasopressorduration", "ventduration", "ventclassification"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "mimic4_derived",
            config_path=config_path,
        )
        return
    
    def stats(self):
        df = self.collected_global_event_df
        if df.is_empty():
            logger.error("Data is not loaded")
        super().stats()
        


   
