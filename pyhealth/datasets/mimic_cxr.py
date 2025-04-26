import logging
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import pydicom

from ..tasks import BaseTask

class XrayImageClassification(BaseTask):
    """Task class for X-ray image classification based on view positions."""
    
    def __init__(
        self,
        label_col="view_position",
        mode="multiclass",
        sample_weight=None,
        **kwargs
    ):
        super(XrayImageClassification, self).__init__(
            label_col=label_col,
            mode=mode,
            sample_weight=sample_weight,
            **kwargs
        )
    
    def __call__(self, dataset, **kwargs):
        """Process the dataset for the X-ray image classification task.

        Args:
            dataset: Dataset object containing the X-ray images.
            **kwargs: Additional arguments to pass to the task.

        Returns:
            List of samples, each containing image path and label.
        """
        samples = []
        df = dataset.get_dataframe("mimic_cxr")
        
        # Get unique labels for multiclass classification
        if self.mode == "multiclass":
            unique_labels = df[self.label_col].unique().tolist()
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for _, row in df.iterrows():
            sample = {"patient_id": row["subject_id"]}
            sample["visit_id"] = row["study_id"]
            sample["image_id"] = row["dicom_id"]
            sample["path"] = row["path"]
            sample["view_position"] = row["view_position"]
            
            # Process label based on mode
            if self.mode == "binary":
                sample["label"] = int(row[self.label_col])
            elif self.mode == "multiclass":
                sample["label"] = label_to_idx[row[self.label_col]]
                sample["label_mapping"] = label_to_idx
            elif self.mode == "multilabel":
                # For multilabel, assuming label_col contains comma-separated labels
                labels = row[self.label_col].split(",") if isinstance(row[self.label_col], str) else [row[self.label_col]]
                sample["label"] = labels
            
            if self.sample_weight is not None:
                sample["sample_weight"] = float(row[self.sample_weight])
            
            samples.append(sample)
        
        return samples

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class MIMICCXRDataset(BaseDataset):
    """Dataset class for MIMIC-CXR Dataset with view filtering capability.

    MIMIC-CXR is a large publicly available dataset of chest radiographs
    in DICOM format with free-text radiology reports.

    Dataset is available at:
    https://physionet.org/content/mimic-cxr/2.0.0/

    Args:
        root: Root directory of the raw data containing the dataset files.
        views: Optional list of view positions to include (e.g., ["AP", "PA", "LATERAL"]).
            If None, includes all views.
        dataset_name: Optional name of the dataset. Defaults to "mimic_cxr".
        config_path: Optional path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MIMICCXRDataset
        >>> # Load all views
        >>> dataset = MIMICCXRDataset(
        ...     root="/path/to/mimic-cxr"
        ... )
        >>> dataset.stats()
        >>> 
        >>> # Load only AP and PA views
        >>> dataset = MIMICCXRDataset(
        ...     root="/path/to/mimic-cxr",
        ...     views=["AP", "PA"]
        ... )
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        views: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.views = views
        
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "mimic_cxr.yaml"
            )
        
        metadata_filename = "mimic_cxr-metadata-pyhealth.csv"
        if not os.path.exists(os.path.join(root, metadata_filename)):
            self.prepare_metadata(root)
        
        # If views are specified, filter the metadata
        if views is not None:
            filtered_metadata_filename = f"mimic_cxr-metadata-pyhealth-{'-'.join(views)}.csv"
            if not os.path.exists(os.path.join(root, filtered_metadata_filename)):
                self.filter_metadata_by_views(root, views, metadata_filename, filtered_metadata_filename)
            metadata_filename = filtered_metadata_filename
        
        default_tables = ["mimic_cxr"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "mimic_cxr",
            config_path=config_path,
        )
        return

    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the MIMIC-CXR dataset.

        Args:
            root: Root directory containing the dataset files.
        """
        logger.info("Preparing MIMIC-CXR metadata")
        
        # Load CSV files
        record_list_path = os.path.join(root, 'cxr-record-list.csv')
        study_list_path = os.path.join(root, 'cxr-study-list.csv')
        
        if not os.path.exists(record_list_path) or not os.path.exists(study_list_path):
            raise FileNotFoundError(
                f"Required CSV files not found at {record_list_path} or {study_list_path}. "
                "Please ensure the MIMIC-CXR dataset is properly set up."
            )
        
        df_studies = pd.read_csv(study_list_path)
        df_records = pd.read_csv(record_list_path)
        
        logger.info(f"Original studies: {df_studies.shape[0]}")
        logger.info(f"Original records: {df_records.shape[0]}")
        
        # Merge dataframes
        df = pd.merge(
            df_studies[['subject_id', 'study_id']],
            df_records[['subject_id', 'study_id', 'dicom_id']],
            on=['subject_id', 'study_id']
        )
        
        logger.info(f"Original merged dataframe: {df.shape[0]} rows")
        
        # Extract view position for each DICOM file
        metadata = []
        for _, row in df.iterrows():
            subject_id = row['subject_id']
            study_id = row['study_id']
            dicom_id = row['dicom_id']
            
            # Generate paths
            subject_prefix = f"p{str(subject_id)[:2]}"
            subject_dir = f"p{subject_id}"
            study_dir = f"s{study_id}"
            dicom_file = f"{dicom_id}.dcm"
            dicom_path = os.path.join(root, 'files', subject_prefix, subject_dir, study_dir, dicom_file)
            
            if os.path.exists(dicom_path):
                try:
                    # Read DICOM metadata (without loading pixel data)
                    dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                    view_position = dcm.ViewPosition if hasattr(dcm, 'ViewPosition') else "Unknown"
                    
                    metadata.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'view_position': view_position,
                        'path': dicom_path
                    })
                except Exception as e:
                    logger.warning(f"Error processing {dicom_path}: {e}")
        
        # Create DataFrame
        metadata_df = pd.DataFrame(metadata)
        
        # Save metadata
        metadata_df.to_csv(
            os.path.join(root, "mimic_cxr-metadata-pyhealth.csv"),
            index=False
        )
        
        logger.info(f"Metadata created with {len(metadata_df)} records")
        logger.info(f"View distribution: {metadata_df['view_position'].value_counts().to_dict()}")
        
        return
    
    def filter_metadata_by_views(self, root: str, views: List[str], 
                                input_filename: str, output_filename: str) -> None:
        """Filter metadata by specified views.

        Args:
            root: Root directory containing the dataset files.
            views: List of view positions to include.
            input_filename: Input metadata filename.
            output_filename: Output filtered metadata filename.
        """
        metadata_path = os.path.join(root, input_filename)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        
        # Filter by view positions
        filtered_df = metadata_df[metadata_df['view_position'].isin(views)]
        
        # Save filtered metadata
        filtered_df.to_csv(
            os.path.join(root, output_filename),
            index=False
        )
        
        logger.info(f"Filtered metadata by views {views}")
        logger.info(f"Original records: {len(metadata_df)}, Filtered records: {len(filtered_df)}")
        
        return

    @property
    def default_task(self):
        """Returns the default task for this dataset."""
        return XrayImageClassification()