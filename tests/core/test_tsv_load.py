import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
import yaml

from pyhealth.datasets.base_dataset import BaseDataset


class TestTSVLoad(unittest.TestCase):
    """Test TSV loading functionality with BaseDataset."""

    def setUp(self):
        """Set up temporary directory and create pseudo dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_pseudo_dataset()
        self._create_config_file()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_pseudo_dataset(self):
        """Create pseudo TSV dataset files with random data."""
        # Create patients table
        patients_data = {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "gender": ["M", "F", "M", "F", "M"],
            "age": [45, 32, 67, 28, 53],
            "admission_date": [
                "2023-01-15",
                "2023-02-20",
                "2023-03-10",
                "2023-01-25",
                "2023-04-05",
            ],
        }
        patients_df = pl.DataFrame(patients_data)
        patients_path = Path(self.temp_dir) / "patients.tsv"
        patients_df.write_csv(patients_path, separator="\t")

        # Create diagnoses table
        diagnoses_data = {
            "patient_id": ["P001", "P001", "P002", "P003", "P004", "P005"],
            "diagnosis_code": ["A01.1", "B15.9", "C78.0", "D50.0", "E11.9", "F32.9"],
            "diagnosis_desc": [
                "Typhoid fever",
                "Hepatitis A",
                "Lung cancer",
                "Iron deficiency",
                "Type 2 diabetes",
                "Depression",
            ],
            "timestamp": [
                "2023-01-15 10:00",
                "2023-01-16 14:30",
                "2023-02-20 09:15",
                "2023-03-10 11:45",
                "2023-01-25 16:20",
                "2023-04-05 08:30",
            ],
        }
        diagnoses_df = pl.DataFrame(diagnoses_data)
        diagnoses_path = Path(self.temp_dir) / "diagnoses.tsv"
        diagnoses_df.write_csv(diagnoses_path, separator="\t")

        # Create procedures table
        procedures_data = {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "procedure_code": ["99213", "99214", "99215", "99213", "99214"],
            "procedure_desc": [
                "Office visit",
                "Extended visit",
                "Complex visit",
                "Office visit",
                "Extended visit",
            ],
            "timestamp": [
                "2023-01-15 11:00",
                "2023-02-20 10:30",
                "2023-03-10 12:00",
                "2023-01-25 17:00",
                "2023-04-05 09:00",
            ],
        }
        procedures_df = pl.DataFrame(procedures_data)
        procedures_path = Path(self.temp_dir) / "procedures.tsv"
        procedures_df.write_csv(procedures_path, separator="\t")

        self.patients_file = str(patients_path)
        self.diagnoses_file = str(diagnoses_path)
        self.procedures_file = str(procedures_path)

    def _create_config_file(self):
        """Create YAML configuration file for the pseudo dataset."""
        config_data = {
            "version": "1.0",
            "tables": {
                "patients": {
                    "file_path": "patients.tsv",
                    "patient_id": "patient_id",
                    "timestamp": None,
                    "attributes": ["gender", "age", "admission_date"],
                },
                "diagnoses": {
                    "file_path": "diagnoses.tsv",
                    "patient_id": "patient_id",
                    "timestamp": "timestamp",
                    "timestamp_format": "%Y-%m-%d %H:%M",
                    "attributes": ["diagnosis_code", "diagnosis_desc"],
                },
                "procedures": {
                    "file_path": "procedures.tsv",
                    "patient_id": "patient_id",
                    "timestamp": "timestamp",
                    "timestamp_format": "%Y-%m-%d %H:%M",
                    "attributes": ["procedure_code", "procedure_desc"],
                },
            },
        }

        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def test_tsv_load(self):
        """Test loading TSV dataset with BaseDataset and using stats() function."""
        # Test loading the dataset with different table combinations
        tables_to_test = [
            ["patients"],
            ["diagnoses"],
            ["procedures"],
            ["patients", "diagnoses"],
            ["diagnoses", "procedures"],
            ["patients", "diagnoses", "procedures"],
        ]

        for tables in tables_to_test:
            with self.subTest(tables=tables):
                # Create BaseDataset instance
                dataset = BaseDataset(
                    root=self.temp_dir,
                    tables=tables,
                    dataset_name="TestTSVDataset",
                    config_path=str(self.config_path),
                    dev=False,
                )

                # Verify the dataset was loaded
                self.assertIsNotNone(dataset.global_event_df)
                self.assertIsNotNone(dataset.config)

                # Test that we can collect the dataframe
                collected_df = dataset.global_event_df.collect()
                self.assertIsInstance(collected_df, pl.DataFrame)
                self.assertGreater(
                    collected_df.height, 0, "Dataset should have at least one row"
                )

                # Verify patient_id column exists
                self.assertIn("patient_id", collected_df.columns)

                # Test stats() function
                try:
                    dataset.stats()
                except Exception as e:
                    self.fail(f"dataset.stats() failed with tables {tables}: {e}")

    def test_tsv_load_dev_mode(self):
        """Test loading TSV dataset in dev mode."""
        # Create dataset in dev mode
        dataset = BaseDataset(
            root=self.temp_dir,
            tables=["patients", "diagnoses", "procedures"],
            dataset_name="TestTSVDatasetDev",
            config_path=str(self.config_path),
            dev=True,
        )

        # Verify dev mode is enabled
        self.assertTrue(dataset.dev)

        # Test stats() function in dev mode
        try:
            dataset.stats()
        except Exception as e:
            self.fail(f"dataset.stats() failed in dev mode: {e}")

    def test_tsv_file_detection(self):
        """Test that TSV files are correctly detected and loaded."""
        dataset = BaseDataset(
            root=self.temp_dir,
            tables=["patients"],
            dataset_name="TestTSVDetection",
            config_path=str(self.config_path),
            dev=False,
        )

        collected_df = dataset.global_event_df.collect()

        # Verify we have the expected number of patients
        self.assertEqual(collected_df["patient_id"].n_unique(), 5)

        # Verify we have the expected columns from the patients table
        # Note: attribute columns are prefixed with table name (e.g., "patients/gender")
        expected_base_columns = ["patient_id", "event_type", "timestamp"]
        expected_patient_columns = [
            "patients/gender",
            "patients/age",
            "patients/admission_date",
        ]

        for col in expected_base_columns:
            self.assertIn(col, collected_df.columns)

        for col in expected_patient_columns:
            self.assertIn(col, collected_df.columns)

    def test_multiple_tsv_tables(self):
        """Test loading and joining multiple TSV tables."""
        dataset = BaseDataset(
            root=self.temp_dir,
            tables=["diagnoses", "procedures"],
            dataset_name="TestMultipleTSV",
            config_path=str(self.config_path),
            dev=False,
        )

        collected_df = dataset.global_event_df.collect()

        # Should have data from both tables
        self.assertGreater(collected_df.height, 5)  # More than just patients table

        # Should have timestamp column since both diagnoses and procedures have timestamps
        self.assertIn("timestamp", collected_df.columns)

        # Should have both diagnosis and procedure data
        # Note: columns from different tables are prefixed with table names
        all_columns = set(collected_df.columns)

        # Check for diagnosis-specific columns (prefixed with table name)
        diagnosis_columns = {"diagnoses/diagnosis_code", "diagnoses/diagnosis_desc"}
        procedure_columns = {"procedures/procedure_code", "procedures/procedure_desc"}

        # At least some of these should be present in the concatenated result
        self.assertTrue(
            len(diagnosis_columns.intersection(all_columns)) > 0
            or len(procedure_columns.intersection(all_columns)) > 0,
            f"Expected some diagnosis or procedure columns in {all_columns}",
        )


if __name__ == "__main__":
    unittest.main()
