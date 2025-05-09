import os
import logging
from typing import Dict, List
from dataclasses import field

import pandas as pd
import polars as pl
from pathlib import Path
from pyhealth.datasets import MIMICCXRDataset
from pyhealth.data.data import Patient
from .base_task import BaseTask


logger = logging.getLogger(__name__)


class MIMICCXRImageReportPairs(BaseTask):
    """Task to pair MIMIC-CXR images with their corresponding reports.

    This task creates pairs of chest X-ray images and their corresponding radiology reports.

    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """

    task_name: str = "mimic_cxr_image_report_pairs"
    input_schema: Dict[str, str] = field(
        default_factory=lambda: {"subject_id": "str", "study_id": "str"}
    )
    output_schema: Dict[str, str] = field(
        default_factory=lambda: {"image_path": "str", "report_path": "str"}
    )

    def __init__(self, root_path: str, tables: list = None, **kwargs):
        """
        Initialize the MIMIC-CXR image-report pairing task.

        Args:
            root_path: Root directory containing MIMIC-CXR data
            tables: List of tables to load from the dataset
        """
        super().__init__(**kwargs)
        self.root_path = root_path
        self.tables = tables or ["provider", "record", "study"]

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter patients who have both images and reports."""
        filtered_df = df.filter(
            pl.col("patient_id").is_in(
                df.filter(pl.col("image_exists") == True)
                .select("patient_id")
                .unique()
                .to_series()
            )
            & pl.col("patient_id").is_in(
                df.filter(pl.col("report_exists") == True)
                .select("patient_id")
                .unique()
                .to_series()
            )
        )
        return filtered_df

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract valid image-report path pairs.

        Args:
            patient: Patient object containing events

        Returns:
            List of samples, each containing image and report paths
        """
        samples = []

        # Initialize dataset for this patient (could be optimized)
        dataset = MIMICCXRDataset(
            root=self.root_path,
            tables=self.tables,
            dataset_name="mimic_cxr",
        )

        # Get merged image-report pairs
        merged_df = pd.merge(
            dataset.tables["record"],
            dataset.tables["study"],
            on=["subject_id", "study_id"],
            how="inner",
        )

        # Filter for existing images and reports
        valid_pairs = merged_df[
            (merged_df["image_exists"] == True) & (merged_df["report_exists"] == True)
        ]

        # Create samples for each valid pair
        for _, row in valid_pairs.iterrows():
            samples.append(
                {
                    "subject_id": row["subject_id"],
                    "study_id": row["study_id"],
                    "image_path": os.path.join(self.root_path, row["image_path"]),
                    "report_path": os.path.join(self.root_path, row["report_path"]),
                }
            )

        return samples


def main():
    """Comprehensive test of the MIMIC-CXR image-report pairing task."""

    # Test configuration
    test_config = {
        "root_path": "/srv/local/data/physionet.org/files/mimic-cxr/2.1.0",
    }

    # Create a test logger
    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    test_logger.addHandler(handler)

    def run_test_case(test_name, test_func):
        test_logger.info(f"Running test case: {test_name}")
        try:
            test_func()
            test_logger.info("PASSED\n")
        except Exception as e:
            test_logger.error(f"FAILED: {str(e)}\n")
            raise

    # Test Case 1: Basic functionality test
    def test_basic_functionality():
        task = MIMICCXRImageReportPairs(root_path=test_config["root_path"])
        dataset = MIMICCXRDataset(
            root=test_config["root_path"],
            tables=["provider", "record", "study"],
            dataset_name="mimic_cxr",
        )

        samples = dataset.set_task(task)
        assert len(samples) > 0, "No samples generated"

        sample = samples[0]
        assert Path(sample["image_path"]).exists(), "Image path does not exist"
        assert Path(sample["report_path"]).exists(), "Report path does not exist"
        assert isinstance(sample["subject_id"], str), "Subject ID should be string"
        assert isinstance(sample["study_id"], str), "Study ID should be string"

    # Test Case 2: Missing tables test
    def test_missing_tables():
        task = MIMICCXRImageReportPairs(
            root_path=test_config["root_path"],
            tables=["record"],  # Only load record table
        )
        dataset = MIMICCXRDataset(
            root=test_config["root_path"],
            tables=["record"],
            dataset_name="mimic_cxr",
        )

        samples = dataset.set_task(task)
        assert len(samples) == 0, "Should return empty list when missing study table"

    # Run all test cases
    test_logger.info("Starting MIMICCXRImageReportPairs test suite...\n")

    test_cases = [
        ("Basic Functionality", test_basic_functionality),
        ("Missing Tables", test_missing_tables),
    ]

    for test_name, test_func in test_cases:
        run_test_case(test_name, test_func)

    test_logger.info("All test cases completed successfully!")


if __name__ == "__main__":
    # Configure basic logging for the main execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
