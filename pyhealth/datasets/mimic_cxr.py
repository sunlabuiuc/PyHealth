import os
import logging
import pandas as pd
from typing import List, Optional
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMICCXRDataset(BaseDataset):
    """
    MIMIC-CXR dataset implementation.

    This class handles loading and processing of the MIMIC-CXR dataset including:
    - Provider information
    - Record-level metadata (DICOM images)
    - Study-level information (reports)
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: str = "mimic_cxr",
        config_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the MIMIC-CXR dataset.

        Args:
            root: Root directory containing the dataset files
            tables: List of tables to load (provider, record, study)
            dataset_name: Name for this dataset instance
            config_path: Path to custom config file
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic_cxr.yaml"
            )
            logger.info(f"Using default CXR config: {config_path}")

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs,
        )

        # Rename path columns to be more explicit
        self.rename_path_columns()

    def rename_path_columns(self) -> None:
        """Rename path columns to be more explicit about their purpose."""
        if "record" in self.tables:
            if "path" in self.tables["record"].columns:
                self.tables["record"] = self.tables["record"].rename(
                    columns={"path": "image_path"}
                )
                # Add existence check
                self.tables["record"]["image_exists"] = self.tables["record"][
                    "image_path"
                ].apply(lambda x: os.path.exists(os.path.join(self.root, x)))

        if "study" in self.tables:
            if "path" in self.tables["study"].columns:
                self.tables["study"] = self.tables["study"].rename(
                    columns={"path": "report_path"}
                )
                # Add existence check
                self.tables["study"]["report_exists"] = self.tables["study"][
                    "report_path"
                ].apply(lambda x: os.path.exists(os.path.join(self.root, x)))

    def get_image_table(self) -> pd.DataFrame:
        """Get the image table with existence verification."""
        if "record" not in self.tables:
            raise ValueError("Record table not loaded")

        return self.tables["record"][
            ["subject_id", "study_id", "dicom_id", "image_path", "image_exists"]
        ].copy()

    def get_report_table(self) -> pd.DataFrame:
        """Get the report table with existence verification."""
        if "study" not in self.tables:
            raise ValueError("Study table not loaded")

        return self.tables["study"][
            ["subject_id", "study_id", "report_path", "report_exists"]
        ].copy()
