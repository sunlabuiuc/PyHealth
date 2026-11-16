import logging
import os
import warnings
from typing import List, Optional

import pandas as pd
import dask.dataframe as dd

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
        process = psutil.Process(os.getpid())  # type: ignore
        mem_info = process.memory_info()
        logger.info(f"Memory usage {tag}: {mem_info.rss / (1024 * 1024):.1f} MB")
    else:
        logger.info(f"Memory tracking requested at {tag}, but psutil not available")


class MIMIC4EHRDataset(BaseDataset):
    """
    MIMIC-IV EHR dataset.

    This class is responsible for loading and managing the MIMIC-IV EHR dataset,
    which includes tables such as patients, admissions, and icustays.

    Getting the data:
        - Source: https://physionet.org/content/mimiciv/ (latest is 3.1)
        - Access: credentialed. Create a PhysioNet account, complete
          credentialing (https://physionet.org/settings/credentialing/) and
          the CITI "Data or Specimens Only Research" course, then sign the
          data use agreement on the dataset page.
        - Download: ``wget -r -N -c -np --user <username> --ask-password
          https://physionet.org/files/mimiciv/3.1/``
        - ``root``: the version folder holding ``hosp/`` and ``icu/``.
        - No access yet? The open 100-patient demo works with this class:
          https://physionet.org/content/mimic-iv-demo/2.2/

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> # Load MIMIC-IV EHR dataset with clinical tables
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["diagnoses_icd", "procedures_icd", "labevents"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: str = "mimic4_ehr",
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
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
            cache_dir=cache_dir,
            **kwargs,
        )
        log_memory_usage(f"After initializing {dataset_name}")


class MIMIC4NoteDataset(BaseDataset):
    """
    MIMIC-IV Clinical Notes dataset.

    This class is responsible for loading and managing the MIMIC-IV Clinical Notes dataset,
    which includes tables such as discharge, discharge_detail, and radiology.

    Getting the data:
        - Source: https://physionet.org/content/mimic-iv-note/2.2/
        - Access: credentialed, with the same PhysioNet credentialing and
          CITI training as MIMIC-IV, plus this dataset's own data use
          agreement.
        - Download: ``wget -r -N -c -np --user <username> --ask-password
          https://physionet.org/files/mimic-iv-note/2.2/``
        - ``root``: the version folder holding ``note/``.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MIMIC4NoteDataset
        >>> # Load MIMIC-IV clinical notes dataset
        >>> dataset = MIMIC4NoteDataset(
        ...     root="/path/to/mimic-iv-note/2.2",
        ...     tables=["discharge", "radiology"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: str = "mimic4_note",
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
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
            cache_dir=cache_dir,
            **kwargs,
        )
        log_memory_usage(f"After initializing {dataset_name}")


class MIMIC4CXRDataset(BaseDataset):
    """
    MIMIC-CXR Chest X-ray dataset.

    This class is responsible for loading and managing the MIMIC-CXR Chest X-ray dataset,
    which includes tables such as metadata, chexpert, and radiology.

    Getting the data:
        - Source: MIMIC-CXR-JPG, https://physionet.org/content/mimic-cxr-jpg/2.1.0/
          (this class reads the JPG release, not the DICOM release
          ``mimic-cxr``).
        - Access: credentialed, with the same PhysioNet credentialing and
          CITI training as MIMIC-IV, plus this dataset's own data use
          agreement.
        - Download: ``wget -r -N -c -np --user <username> --ask-password
          https://physionet.org/files/mimic-cxr-jpg/2.1.0/`` (about 570 GB).
        - ``root``: the folder holding ``files/`` and
          ``mimic-cxr-2.0.0-metadata.csv.gz``.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MIMIC4CXRDataset
        >>> # Load MIMIC-CXR dataset with chest X-ray images and labels
        >>> dataset = MIMIC4CXRDataset(
        ...     root="/path/to/mimic-cxr/2.0.0",
        ...     tables=["metadata", "chexpert"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: str = "mimic4_cxr",
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
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
            cache_dir=cache_dir,
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


class MIMIC4EDDataset(BaseDataset):
    """MIMIC-IV-ED, the emergency department module of MIMIC-IV.

    Loads the ED tables of `MIMIC-IV-ED <https://physionet.org/content/mimic-iv-ed/>`_
    (``edstays``, ``triage``, ``vitalsign``, ``medrecon``, ``pyxis`` and
    ``diagnosis``) as patient events keyed by ``subject_id``. ``edstays`` is
    always loaded. Every event carries its ``stay_id``; ``triage`` is stamped
    with the stay's arrival time and ``diagnosis`` with its departure time,
    because ED diagnoses are coded at the end of the stay. The open
    `MIMIC-IV-ED demo <https://physionet.org/content/mimic-iv-ed-demo/>`_ has
    the same layout.

    To link ED stays with hospital outcomes and demographics (``patients``,
    ``admissions``, ``icustays``), use :class:`MIMIC4Dataset` with both
    ``ehr_root`` and ``ed_root``.

    Getting the data:
        - Source: https://physionet.org/content/mimic-iv-ed/ (latest is 2.2)
        - Access: credentialed, with the same PhysioNet credentialing and
          CITI training as MIMIC-IV, plus this dataset's own data use
          agreement.
        - Download: ``wget -r -N -c -np --user <username> --ask-password
          https://physionet.org/files/mimic-iv-ed/2.2/``
        - ``root``: the version folder holding ``ed/``.
        - No access yet? The open demo has the same layout and works with
          this class: https://physionet.org/content/mimic-iv-ed-demo/

    Args:
        root: The folder that holds ``ed/``.
        tables: ED tables to load besides ``edstays``.
        dataset_name: Name of the dataset.
        config_path: Path to a YAML config; defaults to ``mimic4_ed.yaml``.
        cache_dir: Directory for the processed cache.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`, e.g.
            ``dev`` or ``num_workers``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EDDataset
        >>> dataset = MIMIC4EDDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv-ed/2.2",
        ...     tables=["triage", "vitalsign", "pyxis", "medrecon", "diagnosis"],
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str = "mimic4_ed",
        config_path: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic4_ed.yaml"
            )
            logger.info(f"Using default ED config: {config_path}")
        tables = list(tables or [])
        if "edstays" not in tables:
            tables.append("edstays")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            cache_dir=cache_dir,
            **kwargs,
        )


class MIMIC4Dataset(BaseDataset):
    """
    Unified MIMIC-IV dataset with support for EHR, clinical notes, and X-rays.

    This class combines data from multiple MIMIC-IV sources:
    - Core EHR data (demographics, admissions, diagnoses, etc.)
    - Clinical notes (discharge summaries, radiology reports)
    - Chest X-rays (images and metadata)
    - Emergency department stays (MIMIC-IV-ED)

    Getting the data:
        All four sources are credentialed PhysioNet datasets. One PhysioNet
        credentialing plus the CITI course covers them, but you sign each
        dataset's data use agreement separately. See the matching class for
        each source's download command and expected layout.

        - ``ehr_root``: MIMIC-IV, https://physionet.org/content/mimiciv/
          (see :class:`MIMIC4EHRDataset`). Open demo:
          https://physionet.org/content/mimic-iv-demo/2.2/
        - ``note_root``: MIMIC-IV-Note,
          https://physionet.org/content/mimic-iv-note/2.2/
          (see :class:`MIMIC4NoteDataset`)
        - ``cxr_root``: MIMIC-CXR-JPG,
          https://physionet.org/content/mimic-cxr-jpg/2.1.0/
          (see :class:`MIMIC4CXRDataset`)
        - ``ed_root``: MIMIC-IV-ED, https://physionet.org/content/mimic-iv-ed/
          (see :class:`MIMIC4EDDataset`). Open demo:
          https://physionet.org/content/mimic-iv-ed-demo/

    Args:
        ehr_root: Root directory for MIMIC-IV EHR data
        note_root: Root directory for MIMIC-IV notes data
        cxr_root: Root directory for MIMIC-CXR data
        ed_root: Root directory for MIMIC-IV-ED data (the folder holding
            ``ed/``); ``edstays`` is always loaded with it
        ehr_tables: List of EHR tables to include
        note_tables: List of clinical note tables to include
        cxr_tables: List of X-ray tables to include
        ed_tables: List of MIMIC-IV-ED tables to include
        ehr_config_path: Path to the EHR config file
        note_config_path: Path to the note config file
        cxr_config_path: Path to the CXR config file
        ed_config_path: Path to the ED config file
        dataset_name: Name of the dataset
        dev: Whether to enable dev mode (limit to 1000 patients)

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> # Load unified MIMIC-IV dataset with EHR, notes, and CXR data
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     note_root="/path/to/mimic-iv-note/2.2",
        ...     cxr_root="/path/to/mimic-cxr/2.0.0",
        ...     ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        ...     note_tables=["discharge", "radiology"],
        ...     cxr_tables=["metadata", "chexpert"],
        ... )
        >>> dataset.stats()
        >>>
        >>> # Load with only EHR and notes (without CXR)
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     note_root="/path/to/mimic-iv-note/2.2",
        ...     ehr_tables=["diagnoses_icd", "labevents"],
        ...     note_tables=["discharge"],
        ... )
        >>> dataset.stats()
        >>>
        >>> # EHR plus the emergency department module (MIMIC-IV-ED)
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     ed_root="/path/to/mimic-iv-ed/2.2",
        ...     ed_tables=["triage", "vitalsign", "pyxis", "medrecon"],
        ... )
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
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        ed_root: str | None = None,
        ed_tables: list[str] | None = None,
        ed_config_path: str | None = None,
    ):
        log_memory_usage("Starting MIMIC4Dataset init")

        # We need at least one root directory
        if not any([ehr_root, note_root, cxr_root, ed_root]):
            raise ValueError("At least one root directory must be provided")

        # Initialize empty lists if None provided
        ehr_tables = ehr_tables or []
        note_tables = note_tables or []
        cxr_tables = cxr_tables or []
        # without ed_root the ED arguments are ignored, so the tables (and the
        # cache key) are the same as before ED support
        ed_tables = list(ed_tables or []) if ed_root else []
        if ed_root and "edstays" not in ed_tables:
            ed_tables.append("edstays")

        root = f"{ehr_root}|{note_root}|{cxr_root}"
        if ed_root:
            # appended only when given, so existing cache keys are unchanged
            root += f"|{ed_root}"
        super().__init__(
            root=root,
            tables=ehr_tables + note_tables + cxr_tables + ed_tables,
            dataset_name=dataset_name,
            config_path=None,
            dev=dev,
            cache_dir=cache_dir,
            num_workers=num_workers,
        )

        # Initialize child datasets
        self.sub_datasets: dict[str, BaseDataset] = {}

        # Initialize EHR dataset if root is provided
        if ehr_root:
            logger.info(
                f"Initializing MIMIC4EHRDataset with tables: {ehr_tables} (dev mode: {dev})"
            )
            self.sub_datasets["ehr"] = MIMIC4EHRDataset(
                root=ehr_root,
                tables=ehr_tables,
                config_path=ehr_config_path,
                cache_dir=str(self.cache_dir),
                dev=dev,
                num_workers=num_workers,
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
                cache_dir=str(self.cache_dir),
                dev=dev,
                num_workers=num_workers,
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
                cache_dir=str(self.cache_dir),
                dev=dev,
                num_workers=num_workers,
            )
            log_memory_usage("After CXR dataset initialization")

        if ed_root:
            logger.info(
                f"Initializing MIMIC4EDDataset with tables: {ed_tables} (dev mode: {dev})"
            )
            self.sub_datasets["ed"] = MIMIC4EDDataset(
                root=ed_root,
                tables=ed_tables,
                config_path=ed_config_path,
                cache_dir=str(self.cache_dir),
                dev=dev,
                num_workers=num_workers,
            )
            log_memory_usage("After ED dataset initialization")

        log_memory_usage("Completed MIMIC4Dataset init")

    def load_data(self) -> dd.DataFrame:
        """
        Combines data from all initialized sub-datasets into a unified global event dataframe.

        Returns:
            pl.LazyFrame: Combined lazy frame from all data sources
        """
        frames = []

        # Collect global event dataframes from all sub-datasets
        for dataset_type, dataset in self.sub_datasets.items():
            logger.info(f"Combining data from {dataset_type} dataset")
            frames.append(dataset.load_data())

        # Concatenate all frames
        logger.info("Creating combined dataframe")
        if len(frames) == 1:
            return frames[0]
        else:
            return dd.concat(frames, axis=0, join="outer")
