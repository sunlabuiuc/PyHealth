import logging
import os
import warnings
from typing import List, Optional

import polars as pl

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def log_memory_usage(tag=""):
    """Log current memory usage if psutil is available."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory usage {tag}: {mem_info.rss / (1024 * 1024):.1f} MB")
    else:
        logger.info(f"Memory tracking requested at {tag}, but psutil not available")


class MIMIC4EHRDataset(BaseDataset):
    """
    MIMIC-IV EHR dataset.

    This class is responsible for loading and managing the MIMIC-IV EHR dataset,
    which includes tables such as patients, admissions, and icustays.

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
            dataset_name: str = "mimic4_ehr",
            config_path: Optional[str] = None,
            **kwargs
    ):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_ehr.yaml")
            logger.info(f"Using default EHR config: {config_path}")
            default_tables = ["patients", "admissions", "icustays"]
            tables = tables + default_tables

        log_memory_usage(f"Before initializing {dataset_name}")

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )
        log_memory_usage(f"After initializing {dataset_name}")


class MIMIC4NoteDataset(BaseDataset):
    """
    MIMIC-IV Clinical Notes dataset.

    This class is responsible for loading and managing the MIMIC-IV Clinical Notes dataset,
    which includes tables such as discharge, discharge_detail, and radiology.

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
            dataset_name: str = "mimic4_note",
            config_path: Optional[str] = None,
            **kwargs
    ):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_note.yaml")
            logger.info(f"Using default note config: {config_path}")
        if "discharge" in tables:
            warnings.warn(
                "Events from discharge table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        if "discharge_detail" in tables:
            warnings.warn(
                "Events from discharge_detail table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        log_memory_usage(f"Before initializing {dataset_name}")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )
        log_memory_usage(f"After initializing {dataset_name}")


class MIMIC4CXRDataset(BaseDataset):
    """
    MIMIC-CXR Chest X-ray dataset.

    This class is responsible for loading and managing the MIMIC-CXR Chest X-ray dataset,
    which includes tables such as metadata, chexpert, and radiology.

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
            dataset_name: str = "mimic4_cxr",
            config_path: Optional[str] = None,
            **kwargs
    ):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_cxr.yaml")
            logger.info(f"Using default CXR config: {config_path}")
        log_memory_usage(f"Before initializing {dataset_name}")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )

        if "metadata" in tables:
            self._metadata = self._prepare_metadata(root)

        log_memory_usage(f"After initializing {dataset_name}")

    def _prepare_metadata(self, root: str) -> pl.LazyFrame:
        """
        Derive StudyTime and image_path columns for the metadata table.

        Constructs a new LazyFrame by loading the existing “metadata” table,
        formatting the timestamp into a zero-padded HHMMSS StudyTime string,
        and building the full image path for each record.

        Args:
            root (str): Filesystem root directory where the dataset is stored.

        Returns:
            pl.LazyFrame: The metadata frame augmented with “StudyTime” and “image_path” columns.
        """
        ldf = super().load_table("metadata")
        studytime = pl.col("timestamp").dt.strftime("%H%M%S").alias("StudyTime")
        base_path = pl.lit(f"{root}/files", dtype=pl.Utf8)
        image_path = (
                base_path
                + pl.lit("/")
                + pl.col("patient_id").str.slice(0, 3)
                + pl.lit("/")
                + pl.col("patient_id")
                + pl.lit("/s")
                + pl.col("metadata/study_id")
                + pl.lit("/")
                + pl.col("metadata/dicom_id")
                + pl.lit(".jpg")
        ).alias("image_path")
        return ldf.with_columns([studytime, image_path])

    def load_table(self, table_name: str) -> pl.LazyFrame:
        """
        Lazily load a table, returning cached metadata if applicable.

        Overrides the base loader to return an in-memory metadata table
        when “metadata” is requested and has been previously assigned.

        Args:
            table_name (str): Name of the table to load (e.g., “metadata”, “chexpert”).

        Returns:
            pl.LazyFrame: The requested table as a lazy frame.
        """
        if table_name.lower() == "metadata" and hasattr(self, "_metadata"):
            return self._metadata
        return super().load_table(table_name)


class MIMIC4Dataset(BaseDataset):
    """
    Unified MIMIC-IV dataset with support for EHR, clinical notes, and X-rays.

    This class combines data from multiple MIMIC-IV sources:
    - Core EHR data (demographics, admissions, diagnoses, etc.)
    - Clinical notes (discharge summaries, radiology reports)
    - Chest X-rays (images and metadata)

    Args:
        ehr_root: Root directory for MIMIC-IV EHR data
        note_root: Root directory for MIMIC-IV notes data
        cxr_root: Root directory for MIMIC-CXR data
        ehr_tables: List of EHR tables to include
        note_tables: List of clinical note tables to include
        cxr_tables: List of X-ray tables to include
        ehr_config_path: Path to the EHR config file
        note_config_path: Path to the note config file
        cxr_config_path: Path to the CXR config file
        dataset_name: Name of the dataset
        dev: Whether to enable dev mode (limit to 1000 patients)
    """

    def __init__(
            self,
            ehr_root: Optional[str] = None,
            note_root: Optional[str] = None,
            cxr_root: Optional[str] = None,
            ehr_tables: Optional[List[str]] = None,
            note_tables: Optional[List[str]] = None,
            cxr_tables: Optional[List[str]] = None,
            ehr_config_path: Optional[str] = None,
            note_config_path: Optional[str] = None,
            cxr_config_path: Optional[str] = None,
            dataset_name: str = "mimic4",
            dev: bool = False,  # Added dev parameter
    ):
        log_memory_usage("Starting MIMIC4Dataset init")

        # Initialize child datasets
        self.dataset_name = dataset_name
        self.sub_datasets = {}
        self.root = None
        self.tables = None
        self.config = None
        self.dev = dev  # Store dev mode flag

        # We need at least one root directory
        if not any([ehr_root, note_root, cxr_root]):
            raise ValueError("At least one root directory must be provided")

        # Initialize empty lists if None provided
        ehr_tables = ehr_tables or []
        note_tables = note_tables or []
        cxr_tables = cxr_tables or []

        # Initialize EHR dataset if root is provided
        if ehr_root:
            logger.info(f"Initializing MIMIC4EHRDataset with tables: {ehr_tables} (dev mode: {dev})")
            self.sub_datasets["ehr"] = MIMIC4EHRDataset(
                root=ehr_root,
                tables=ehr_tables,
                config_path=ehr_config_path,
                dev=dev  # Pass dev mode flag
            )
            log_memory_usage("After EHR dataset initialization")

        # Initialize Notes dataset if root is provided
        if note_root is not None and note_tables:
            logger.info(f"Initializing MIMIC4NoteDataset with tables: {note_tables} (dev mode: {dev})")
            self.sub_datasets["note"] = MIMIC4NoteDataset(
                root=note_root,
                tables=note_tables,
                config_path=note_config_path,
                dev=dev  # Pass dev mode flag
            )
            log_memory_usage("After Note dataset initialization")

        # Initialize CXR dataset if root is provided
        if cxr_root is not None:
            logger.info(f"Initializing MIMIC4CXRDataset with tables: {cxr_tables} (dev mode: {dev})")
            self.sub_datasets["cxr"] = MIMIC4CXRDataset(
                root=cxr_root,
                tables=cxr_tables,
                config_path=cxr_config_path,
                dev=dev  # Pass dev mode flag
            )
            log_memory_usage("After CXR dataset initialization")

        # Combine data from all sub-datasets
        log_memory_usage("Before combining data")
        self.global_event_df = self._combine_data()
        log_memory_usage("After combining data")

        # Cache attributes
        self._collected_global_event_df = None
        self._unique_patient_ids = None

        log_memory_usage("Completed MIMIC4Dataset init")

    def _combine_data(self) -> pl.LazyFrame:
        """
        Combines data from all initialized sub-datasets into a unified global event dataframe.

        Returns:
            pl.LazyFrame: Combined lazy frame from all data sources
        """
        frames = []

        # Collect global event dataframes from all sub-datasets
        for dataset_type, dataset in self.sub_datasets.items():
            logger.info(f"Combining data from {dataset_type} dataset")
            frames.append(dataset.global_event_df)

        # Concatenate all frames
        logger.info("Creating combined dataframe")
        if len(frames) == 1:
            return frames[0]
        else:
            return pl.concat(frames, how="diagonal")
