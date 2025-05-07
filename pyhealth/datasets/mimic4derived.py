import logging
import polars as pl
from pathlib import Path
from typing import Optional
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC4DerivedDataset(BaseDataset):
    """Derived Dataset for MIMIC-IV containing ventilation and vasopressor durations

    Dataset is available to be derived from the following link for the Metavision information system:
    https://physionet.org/content/mimiciv/3.1/
    Transformations derived from the following and adapted for mimic-iv:
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
        self.vent_duration = self.filterTable("ventduration")
        self.vasopressor_duration = self.filterTable("vasopressorduration")
        return
    
    def filterTable(self, table_name):
        """
        Helper Function which filters out event_types according to the tableName provided

        Args:
        table_name - A string indicating the specific event_type to filter on. For now, we only support the default_tables

        Returns:
        A lazyframe with only columns corresponding to the event_type

        Used to extract a particular table from the dataset or in initialization to break up the dataset for later use
        """
        df = self.collected_global_event_df
        if table_name == "ventduration":
            cols = ["stay_id","ventnum","endtime","duration_hours"]
            cols = ["ventduration/" + s for s in cols]
            return df.filter(pl.col("event_type") == "ventduration").select(["patient_id", "event_type", "timestamp"] + cols)
        elif table_name == "ventclassification":
            cols = ["stay_id","MechVent","OxygenTherapy","Extubated","SelfExtubated"]
            cols = ["ventclassification/" + s for s in cols]
            return df.filter(pl.col("event_type") == "ventclassification").select(["patient_id", "event_type", "timestamp"] + cols)
        elif table_name == "vasopressorduration":
            cols = ["stay_id","vasonum","endtime","duration_hours"]
            cols = ["vasopressorduration/" + s for s in cols]
            return df.filter(pl.col("event_type") == "vasopressorduration").select(["patient_id", "event_type", "timestamp"] + cols)
        else:
            logger.error("Unknown table")

    def stats(self):
        df = self.collected_global_event_df
        if df.is_empty():
            logger.error("Data is not loaded")
            return
        print("---Vasopressor Duration Statistics---")
        vaso_col = self.vasopressor_duration.select(pl.col("vasopressorduration/duration_hours").cast(pl.Int64))
        vaso_mean = float(vaso_col.mean()[0,0])
        print(f"Mean duration (hrs): {vaso_mean}")
        vaso_median = int(vaso_col.median()[0,0])
        print(f"Median duration (hrs): {vaso_median}")
        vaso_max = int(vaso_col.max()[0,0])
        print(f"Max duration (hrs): {vaso_max}")
        print("---Ventilation Duration Statistics---")
        vent_col = self.vent_duration.select(pl.col("ventduration/duration_hours").cast(pl.Float64))
        vent_mean = float(vent_col.mean()[0,0])
        print(f"Mean duration (hrs): {vent_mean}")
        vent_median = int(vent_col.median()[0,0])
        print(f"Median duration (hrs): {vent_median}")
        vent_max = float(vent_col.max()[0,0])
        print(f"Max duration (hrs): {vent_max}")
        print(df.head())
        

