import logging
import os
import warnings
from typing import Dict, List, Optional

import pandas as pd
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
        **kwargs,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic4_ehr.yaml"
            )
            logger.info(f"Using default EHR config: {config_path}")

        log_memory_usage(f"Before initializing {dataset_name}")
        default_tables = ["patients", "admissions", "icustays"]
        tables = tables + default_tables
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs,
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
        **kwargs,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic4_note.yaml"
            )
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
            **kwargs,
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
        **kwargs,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic4_cxr.yaml"
            )
            logger.info(f"Using default CXR config: {config_path}")
        self.prepare_metadata(root)
        log_memory_usage(f"Before initializing {dataset_name}")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs,
        )
        log_memory_usage(f"After initializing {dataset_name}")

    def prepare_metadata(self, root: str) -> None:
        metadata = pd.read_csv(
            os.path.join(root, "mimic-cxr-2.0.0-metadata.csv.gz"), dtype=str
        )

        def process_studytime(x):
            # reformat studytime to be 6 digits (e.g. 123.002 -> 000123 which is 12:30:00)
            try:
                x = float(x)
                return f"{int(x):06d}"
            except Exception:
                return x

        metadata["StudyTime"] = metadata["StudyTime"].apply(process_studytime)

        def process_image_path(x):
            # files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
            subject_id = "p" + x["subject_id"]
            folder = subject_id[:3]
            study_id = "s" + x["study_id"]
            dicom_id = x["dicom_id"]
            return os.path.join(
                root, "files", folder, subject_id, study_id, f"{dicom_id}.jpg"
            )

        metadata["image_path"] = metadata.apply(process_image_path, axis=1)

        metadata.to_csv(
            os.path.join(root, "mimic-cxr-2.0.0-metadata-pyhealth.csv"), index=False
        )
        return


class MIMIC4Dataset(BaseDataset):
    """
    Unified MIMIC-IV dataset with support for EHR, clinical notes, and X-rays.

    This class combines data from multiple MIMIC-IV sources:
    - Core EHR data (demographics, admissions, diagnoses, etc.)
    - Clinical notes (discharge summaries, radiology reports)
    - Chest X-rays (images and metadata)

    You can use any combination of data sources. Patient IDs are determined
    by priority order:
    1. EHR dataset (if ehr_root provided) - primary source
    2. Note dataset (if note_root provided and no EHR)
    3. CXR dataset (if cxr_root provided and no EHR/Note)

    When using multiple data sources in streaming mode, all sub-datasets are
    automatically synchronized to use the same patient cohort from the primary
    source.

    Examples:
        # Use all three modalities (EHR determines patient cohort)
        dataset = MIMIC4Dataset(
            ehr_root="/path/to/ehr",
            note_root="/path/to/notes",
            cxr_root="/path/to/cxr",
            stream=True
        )

        # Use only chest X-rays (CXR determines patient cohort)
        dataset = MIMIC4Dataset(
            cxr_root="/path/to/cxr",
            cxr_tables=["metadata", "chexpert"],
            stream=True
        )

        # Use only clinical notes (Note determines patient cohort)
        dataset = MIMIC4Dataset(
            note_root="/path/to/notes",
            note_tables=["discharge"],
            stream=True
        )

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
        dev: Whether to enable dev mode (limit patients)
        dev_max_patients: Maximum number of patients in dev mode (default 1000)
        stream: Whether to enable streaming mode for memory efficiency
        cache_dir: Directory for streaming cache
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
        dev: bool = False,
        dev_max_patients: int = 1000,
        stream: bool = False,
        cache_dir: Optional[str] = None,
    ):
        log_memory_usage("Starting MIMIC4Dataset init")

        # We need at least one root directory
        if not any([ehr_root, note_root, cxr_root]):
            raise ValueError("At least one root directory must be provided")

        # Initialize base class attributes for streaming mode
        # MIMIC4Dataset doesn't follow the normal BaseDataset pattern
        # (no single root/tables/config), so we initialize the attributes
        # that BaseDataset would normally set
        self.dataset_name = dataset_name
        self.dev = dev
        self.dev_max_patients = dev_max_patients
        self.stream = stream

        # Handle cache_dir (convert to Path if needed)
        if cache_dir is None:
            from pathlib import Path

            cache_dir = Path.home() / ".cache" / "pyhealth" / dataset_name
        elif isinstance(cache_dir, str):
            from pathlib import Path

            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

        # Initialize streaming attributes (normally done in BaseDataset.__init__)
        self._collected_global_event_df = None
        self._unique_patient_ids = None
        self._patient_cache_path = None
        self._patient_index_path = None
        self._patient_index = None

        # Setup streaming cache if enabled
        if self.stream:
            from .processing.streaming import setup_streaming_cache

            setup_streaming_cache(self)

        # MIMIC4-specific attributes
        self.sub_datasets = {}
        self.root = None  # Composite dataset has no single root
        self.tables = None  # No single tables list
        self.config = None  # No single config

        # Initialize empty lists if None provided
        ehr_tables = ehr_tables or []
        note_tables = note_tables or []
        cxr_tables = cxr_tables or []

        # Initialize EHR dataset if root is provided
        if ehr_root:
            logger.info(
                f"Initializing MIMIC4EHRDataset with tables: {ehr_tables} (dev mode: {dev})"
            )
            self.sub_datasets["ehr"] = MIMIC4EHRDataset(
                root=ehr_root,
                tables=ehr_tables,
                config_path=ehr_config_path,
                dev=dev,
                dev_max_patients=dev_max_patients,
                stream=stream,
                cache_dir=cache_dir,
            )
            log_memory_usage("After EHR dataset initialization")

        # Initialize Notes dataset if root is provided
        if note_root is not None and note_tables:
            logger.info(
                f"Initializing MIMIC4NoteDataset with tables: {note_tables} (dev mode: {dev})"
            )
            self.sub_datasets["note"] = MIMIC4NoteDataset(
                root=note_root,
                tables=note_tables,
                config_path=note_config_path,
                dev=dev,
                dev_max_patients=dev_max_patients,
                stream=stream,
                cache_dir=cache_dir,
            )
            log_memory_usage("After Note dataset initialization")

        # Initialize CXR dataset if root is provided
        if cxr_root is not None:
            logger.info(
                f"Initializing MIMIC4CXRDataset with tables: {cxr_tables} (dev mode: {dev})"
            )
            self.sub_datasets["cxr"] = MIMIC4CXRDataset(
                root=cxr_root,
                tables=cxr_tables,
                config_path=cxr_config_path,
                dev=dev,
                dev_max_patients=dev_max_patients,
                stream=stream,
                cache_dir=cache_dir,
            )
            log_memory_usage("After CXR dataset initialization")

        # Combine data from all sub-datasets
        log_memory_usage("Before combining data")
        self.global_event_df = self._combine_data()
        log_memory_usage("After combining data")

        # CRITICAL: Trigger patient synchronization befosre cache building
        # This ensures sub-datasets have synchronized patient IDs from the parent
        # BEFORE their patient caches are built in set_task_streaming().
        # Without this, sub-dataset caches are built with ALL patients (unsynchronized),
        # causing a mismatch with the parent's patient_ids.
        #
        # In streaming mode, we MUST trigger this early to ensure:
        # 1. parent._unique_patient_ids is set
        # 2. sub-dataset._unique_patient_ids is synchronized to parent
        # 3. When set_task() calls build_patient_cache(), it uses the synchronized IDs
        if self.stream:
            logger.info("Pre-computing patient IDs for streaming mode...")
            # Access unique_patient_ids to trigger synchronization
            _ = self.unique_patient_ids
            logger.info(
                f"Initialized with {len(self._unique_patient_ids)} patients "
                f"from {len(self.sub_datasets)} sub-dataset(s)"
            )

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

    @property
    def unique_patient_ids(self) -> List[str]:
        """
        Get unique patient IDs from the dataset.

        Patient ID determination logic:
        1. If EHR is present: Use EHR patient IDs (EHR is the linking key)
        2. If only Notes + CXR: Use intersection (only patients in both)
        3. If single source: Use all patients from that source

        When EHR is present, it takes precedence because it contains the
        core patient demographics and links to all other modalities.

        Returns:
            List[str]: List of unique patient IDs
        """
        # Cache the patient IDs if not already computed
        if self._unique_patient_ids is None:
            if len(self.sub_datasets) == 0:
                raise ValueError(
                    "MIMIC4Dataset has no sub-datasets. At least one of "
                    "ehr_root, note_root, or cxr_root must be provided."
                )

            # Get patient IDs from all sub-datasets
            all_patient_id_sets = {}
            for dataset_name, dataset in self.sub_datasets.items():
                patient_ids = dataset.unique_patient_ids
                all_patient_id_sets[dataset_name] = set(patient_ids)
                logger.info(f"{dataset_name} dataset has {len(patient_ids)} patients")

            # Strategy 1: EHR takes precedence (it's the linking key)
            if "ehr" in self.sub_datasets:
                self._unique_patient_ids = list(all_patient_id_sets["ehr"])
                logger.info(
                    f"Using {len(self._unique_patient_ids)} patients "
                    f"from EHR dataset (EHR is primary source)"
                )

                # In streaming mode, sync other datasets to use EHR patients
                # by directly setting their _unique_patient_ids
                if self.stream and len(self.sub_datasets) > 1:
                    logger.info("Synchronizing Notes/CXR to use EHR patient set")
                    for dataset_name, dataset in self.sub_datasets.items():
                        if dataset_name != "ehr":
                            # Directly set patient IDs (no separate variable)
                            dataset._unique_patient_ids = self._unique_patient_ids
                            logger.debug(f"Synchronized {dataset_name} with EHR")

            # Strategy 2: No EHR, single source - use all patients
            elif len(self.sub_datasets) == 1:
                self._unique_patient_ids = list(list(all_patient_id_sets.values())[0])
                logger.info(
                    f"Using all {len(self._unique_patient_ids)} patients "
                    f"from single data source"
                )

            # Strategy 3: No EHR, multiple sources - use intersection
            else:
                common_patients = set.intersection(*all_patient_id_sets.values())
                self._unique_patient_ids = sorted(list(common_patients))

                logger.info(
                    f"No EHR dataset - using intersection of "
                    f"{len(self.sub_datasets)} sources: "
                    f"{len(self._unique_patient_ids)} patients"
                )

                # In streaming mode, sync all datasets to common set
                if self.stream:
                    for dataset_name, dataset in self.sub_datasets.items():
                        dataset._unique_patient_ids = self._unique_patient_ids

        return self._unique_patient_ids
