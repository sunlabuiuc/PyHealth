import logging
import os
import gc
from pathlib import Path
from typing import List, Optional, Dict

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


class MIMIC4_EHR(BaseDataset):
    """MIMIC-IV Electronic Health Records dataset."""
    
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
        
        log_memory_usage(f"Before initializing {dataset_name}")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )
        log_memory_usage(f"After initializing {dataset_name}")


class MIMIC4_Note(BaseDataset):
    """MIMIC-IV Clinical Notes dataset."""
    
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
        
        log_memory_usage(f"Before initializing {dataset_name}")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )
        log_memory_usage(f"After initializing {dataset_name}")


class MIMIC4_CXR(BaseDataset):
    """MIMIC-CXR Chest X-ray dataset."""
    
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
        log_memory_usage(f"After initializing {dataset_name}")
        
    class MIMIC4_CXR(BaseDataset):
        """MIMIC-CXR Chest X-ray dataset with improved image path handling."""
        
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
            
            super().__init__(
                root=root,
                tables=tables,
                dataset_name=dataset_name,
                config_path=config_path,
                **kwargs
            )
        
        def get_image_path(self, patient_id: str, study_id: str, dicom_id: str) -> str:
            """
            Constructs the correct path to a MIMIC-CXR image file.
            
            Args:
                patient_id: The patient ID (e.g., "p10000032")
                study_id: The study ID (e.g., "s50414267")
                dicom_id: The DICOM ID (e.g., "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014")
            
            Returns:
                The absolute path to the image file
            """
            # Extract the first 3 characters of the patient_id as the parent folder
            parent_folder = patient_id[:3]
            return os.path.join(self.root, "files", parent_folder, patient_id, study_id, f"{dicom_id}.jpg")
        
        def get_study_images(self, patient_id: str, study_id: str) -> List[str]:
            """
            Gets all image paths for a specific study.
            
            Args:
                patient_id: The patient ID (e.g., "p10000032")
                study_id: The study ID (e.g., "s50414267")
            
            Returns:
                A list of absolute paths to all images in the study
            """
            # Get the metadata for this patient and study
            df = self.collected_global_event_df
            study_metadata = df.filter(
                (pl.col("event_type") == "xrays_metadata") &
                (pl.col("patient_id") == patient_id) &
                (pl.col("xrays_metadata/study_id") == study_id)
            )
            
            # Construct paths for all dicom_ids in this study
            image_paths = []
            for row in study_metadata.iter_rows(named=True):
                dicom_id = row["xrays_metadata/dicom_id"]
                image_paths.append(self.get_image_path(patient_id, study_id, dicom_id))
            
            return image_paths
        
        def get_patient_images(self, patient_id: str) -> Dict[str, List[str]]:
            """
            Gets all image paths for a specific patient, organized by study.
            
            Args:
                patient_id: The patient ID (e.g., "p10000032")
            
            Returns:
                A dictionary mapping study_id to a list of image paths
            """
            # Get the metadata for this patient
            df = self.collected_global_event_df
            patient_metadata = df.filter(
                (pl.col("event_type") == "xrays_metadata") &
                (pl.col("patient_id") == patient_id)
            )
            
            # Group by study_id
            studies = {}
            for row in patient_metadata.iter_rows(named=True):
                study_id = row["xrays_metadata/study_id"]
                dicom_id = row["xrays_metadata/dicom_id"]
                
                if study_id not in studies:
                    studies[study_id] = []
                
                studies[study_id].append(self.get_image_path(patient_id, study_id, dicom_id))
            
            return studies

class MIMIC4Dataset(BaseDataset):
    """
    Unified MIMIC-IV dataset with support for EHR, clinical notes, and X-rays.
    
    This class combines data from multiple MIMIC-IV sources:
    - Core EHR data (demographics, admissions, diagnoses, etc.)
    - Clinical notes (discharge summaries, radiology reports)
    - Chest X-rays (images and metadata)
    
    Args:
        ehr_root: Root directory for MIMIC-IV EHR data
        notes_root: Root directory for MIMIC-IV notes data
        cxr_root: Root directory for MIMIC-CXR data
        ehr_tables: List of EHR tables to include
        note_tables: List of clinical note tables to include
        cxr_tables: List of X-ray tables to include
        dataset_name: Name of the dataset
        dev: Whether to enable dev mode (limit to 1000 patients)
    """
    
    def __init__(
        self,
        ehr_root: Optional[str] = None,
        notes_root: Optional[str] = None,
        cxr_root: Optional[str] = None,
        ehr_tables: Optional[List[str]] = None,
        note_tables: Optional[List[str]] = None,
        cxr_tables: Optional[List[str]] = None,
        dataset_name: str = "mimic4",
        dev: bool = False,  # Added dev parameter
    ):
        log_memory_usage("Starting MIMIC4Dataset init")
        
        # Initialize child datasets
        self.dataset_name = dataset_name
        self.sub_datasets = {}
        self.root = ehr_root  # Default root for parent class
        self.dev = dev  # Store dev mode flag
        
        # We need at least one root directory
        if not any([ehr_root, notes_root, cxr_root]):
            raise ValueError("At least one root directory must be provided")
        
        # Initialize empty lists if None provided
        ehr_tables = ehr_tables or []
        note_tables = note_tables or []
        cxr_tables = cxr_tables or []
        
        # Initialize EHR dataset if root is provided and tables specified
        if ehr_root is not None and ehr_tables:
            logger.info(f"Initializing MIMIC4_EHR with tables: {ehr_tables} (dev mode: {dev})")
            self.sub_datasets["ehr"] = MIMIC4_EHR(
                root=ehr_root,
                tables=ehr_tables,
                config_path=os.path.join(os.path.dirname(__file__), "configs", "mimic4_ehr.yaml"),
                dev=dev  # Pass dev mode flag
            )
            log_memory_usage("After EHR dataset initialization")
        
        # Initialize Notes dataset if root is provided and tables specified
        if notes_root is not None and note_tables:
            logger.info(f"Initializing MIMIC4_Note with tables: {note_tables} (dev mode: {dev})")
            self.sub_datasets["note"] = MIMIC4_Note(
                root=notes_root,
                tables=note_tables,
                config_path=os.path.join(os.path.dirname(__file__), "configs", "mimic4_note.yaml"),
                dev=dev  # Pass dev mode flag
            )
            log_memory_usage("After Note dataset initialization")
        
        # Initialize CXR dataset if root is provided and tables specified
        if cxr_root is not None and cxr_tables:
            logger.info(f"Initializing MIMIC4_CXR with tables: {cxr_tables} (dev mode: {dev})")
            self.sub_datasets["cxr"] = MIMIC4_CXR(
                root=cxr_root,
                tables=cxr_tables,
                config_path=os.path.join(os.path.dirname(__file__), "configs", "mimic4_cxr.yaml"),
                dev=dev  # Pass dev mode flag
            )
            log_memory_usage("After CXR dataset initialization")
        
        # Handle case where no subdatasets were created (no matching tables)
        if not self.sub_datasets:
            logger.warning("No tables specified for any provided roots. Creating minimal dataset.")
            # Create a minimal dataset with the first available root
            if ehr_root:
                self.sub_datasets["ehr"] = MIMIC4_EHR(
                    root=ehr_root,
                    tables=["patients"],  # Minimal table
                    config_path=os.path.join(os.path.dirname(__file__), "configs", "mimic4_ehr.yaml"),
                    dev=dev  # Pass dev mode flag
                )
            elif notes_root:
                self.sub_datasets["note"] = MIMIC4_Note(
                    root=notes_root,
                    tables=["discharge"],  # Minimal table
                    config_path=os.path.join(os.path.dirname(__file__), "configs", "mimic4_note.yaml"),
                    dev=dev  # Pass dev mode flag
                )
            elif cxr_root:
                self.sub_datasets["cxr"] = MIMIC4_CXR(
                    root=cxr_root,
                    tables=["xrays_metadata"],  # Minimal table
                    config_path=os.path.join(os.path.dirname(__file__), "configs", "mimic4_cxr.yaml"),
                    dev=dev  # Pass dev mode flag
                )
        
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

    # CXR
    def get_cxr_image_path(self, patient_id: str, study_id: str, dicom_id: str) -> str:
        """
        Constructs the correct path to a MIMIC-CXR image file.
        
        Args:
            patient_id: The patient ID (e.g., "p10000032")
            study_id: The study ID (e.g., "s50414267")
            dicom_id: The DICOM ID (e.g., "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014")
        
        Returns:
            The absolute path to the image file
        
        Raises:
            ValueError: If CXR data is not loaded or cxr_root is not provided
        """
        if "cxr" not in self.sub_datasets:
            raise ValueError("CXR data is not loaded. Please initialize the dataset with cxr_root and cxr_tables.")
        
        # Use the get_image_path method from the CXR dataset
        return self.sub_datasets["cxr"].get_image_path(patient_id, study_id, dicom_id)
    
    def get_cxr_study_images(self, patient_id: str, study_id: str) -> List[str]:
        """
        Gets all image paths for a specific study.
        
        Args:
            patient_id: The patient ID (e.g., "p10000032")
            study_id: The study ID (e.g., "s50414267")
        
        Returns:
            A list of absolute paths to all images in the study
        
        Raises:
            ValueError: If CXR data is not loaded
        """
        if "cxr" not in self.sub_datasets:
            raise ValueError("CXR data is not loaded. Please initialize the dataset with cxr_root and cxr_tables.")
        
        return self.sub_datasets["cxr"].get_study_images(patient_id, study_id)
    
    def get_cxr_patient_images(self, patient_id: str) -> Dict[str, List[str]]:
        """
        Gets all image paths for a specific patient, organized by study.
        
        Args:
            patient_id: The patient ID (e.g., "p10000032")
        
        Returns:
            A dictionary mapping study_id to a list of image paths
        
        Raises:
            ValueError: If CXR data is not loaded
        """
        if "cxr" not in self.sub_datasets:
            raise ValueError("CXR data is not loaded. Please initialize the dataset with cxr_root and cxr_tables.")
        
        return self.sub_datasets["cxr"].get_patient_images(patient_id)

def test_mimic4_dataset():
    """Test function for the MIMIC4Dataset class."""
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Use original paths
    mimic_iv_root = "/srv/local/data/jw3/physionet.org/files/MIMIC-IV/2.0"
    mimic_note_root = "/srv/local/data/jw3/physionet.org/files/mimic-iv-note/2.2/note"
    mimic_cxr_root = "/srv/local/data/jw3/physionet.org/files/MIMIC-CXR"
    
    log_memory_usage("Before any tests")
    
    try:
        # Test 1: EHR only
        logger.info("=== Test 1: MIMIC-IV EHR only ===")
        dataset_ehr = MIMIC4Dataset(
            ehr_root=mimic_iv_root,
            ehr_tables=["patients", "admissions", "diagnoses_icd"]
        )
        
        logger.info("--- Statistics for Test 1 ---")
        dataset_ehr.stats()
        
        # Free memory before next test
        del dataset_ehr
        gc.collect()
        log_memory_usage("After Test 1")
        
        # Test 2: EHR + Notes
        logger.info("=== Test 2: MIMIC-IV EHR + Notes ===")
        dataset_ehr_notes = MIMIC4Dataset(
            ehr_root=mimic_iv_root,
            notes_root=mimic_note_root,
            ehr_tables=["patients", "admissions", "diagnoses_icd"],
            note_tables=["discharge"]
        )
        
        logger.info("--- Statistics for Test 2 ---")
        dataset_ehr_notes.stats()
        
        # Free memory before next test
        del dataset_ehr_notes
        gc.collect()
        log_memory_usage("After Test 2")
        
        # Test 3: Complete dataset
        logger.info("=== Test 3: Complete MIMIC-IV dataset ===")
        logger.info("Warning: This test may require significant memory")
        
        # Use more memory-efficient approach for Test 3
        dataset_complete = MIMIC4Dataset(
            ehr_root=mimic_iv_root,
            notes_root=mimic_note_root,
            cxr_root=mimic_cxr_root,
            ehr_tables=["patients"],  # Minimal EHR table to save memory
            note_tables=["discharge"],  # Single note table
            cxr_tables=["xrays_metadata"]  # Metadata only
        )
        
        logger.info("--- Statistics for Test 3 ---")
        dataset_complete.stats()
        
        # Free memory
        del dataset_complete
        gc.collect()
        log_memory_usage("After Test 3")
        
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    test_mimic4_dataset()