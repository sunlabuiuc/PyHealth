import unittest
import tempfile
import shutil
import subprocess
import os
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.mortality_prediction import (
    MortalityPredictionMIMIC3,
    MultimodalMortalityPredictionMIMIC3,
)


class TestMIMIC3MortalityPrediction(unittest.TestCase):
    """Test MIMIC-3 mortality prediction tasks with demo data downloaded from PhysioNet."""

    def setUp(self):
        """Download and set up demo dataset for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self._download_demo_dataset()
        self._load_dataset()

    def tearDown(self):
        """Clean up downloaded dataset after each test."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _download_demo_dataset(self):
        """Download MIMIC-III demo dataset using wget."""
        download_url = "https://physionet.org/files/mimiciii-demo/1.4/"

        # Use wget to download the demo dataset recursively
        cmd = [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            "--directory-prefix",
            self.temp_dir,
            download_url,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise unittest.SkipTest(f"Failed to download MIMIC-III demo dataset: {e}")
        except FileNotFoundError:
            raise unittest.SkipTest("wget not available - skipping download test")

        # Find the downloaded dataset path
        physionet_dir = (
            Path(self.temp_dir) / "physionet.org" / "files" / "mimiciii-demo" / "1.4"
        )
        if physionet_dir.exists():
            self.demo_dataset_path = str(physionet_dir)
        else:
            raise unittest.SkipTest("Downloaded dataset not found in expected location")

    def _load_dataset(self):
        """Load the dataset for testing."""
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions", "noteevents"]
        self.dataset = MIMIC3Dataset(root=self.demo_dataset_path, tables=tables)

    def test_dataset_stats(self):
        """Test that the dataset loads correctly and stats() works."""
        try:
            self.dataset.stats()
        except Exception as e:
            self.fail(f"dataset.stats() failed: {e}")

    def test_mortality_prediction_mimic3_set_task(self):
        """Test MortalityPredictionMIMIC3 task with set_task() method."""
        task = MortalityPredictionMIMIC3()

        # Test that task is properly initialized
        self.assertEqual(task.task_name, "MortalityPredictionMIMIC3")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("mortality", task.output_schema)

        # Test using set_task method
        try:
            sample_dataset = self.dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            self.assertTrue(
                hasattr(sample_dataset, "samples"), "Sample dataset should have samples"
            )

            # Verify we got some samples
            self.assertGreater(
                len(sample_dataset.samples), 0, "Should generate at least one sample"
            )

            # Test sample structure
            if len(sample_dataset.samples) > 0:
                sample = sample_dataset.samples[0]
                required_keys = [
                    "hadm_id",
                    "patient_id",
                    "conditions",
                    "procedures",
                    "drugs",
                    "mortality",
                ]
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")

                # Verify mortality label is binary (0 or 1)
                self.assertIn(
                    sample["mortality"], [0, 1], "Mortality label should be 0 or 1"
                )

                print(f"Generated {len(sample_dataset.samples)} mortality samples")
                print(f"Sample keys: {list(sample.keys())}")

        except Exception as e:
            self.fail(f"Failed to use set_task with MortalityPredictionMIMIC3: {e}")

    def test_multimodal_mortality_prediction_mimic3_set_task(self):
        """Test MultimodalMortalityPredictionMIMIC3 task with set_task() method."""
        task = MultimodalMortalityPredictionMIMIC3()

        # Test that task is properly initialized
        self.assertEqual(task.task_name, "MultimodalMortalityPredictionMIMIC3")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("clinical_notes", task.input_schema)
        self.assertIn("mortality", task.output_schema)

        # Test using set_task method
        try:
            sample_dataset = self.dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            self.assertTrue(
                hasattr(sample_dataset, "samples"), "Sample dataset should have samples"
            )

            # Verify we got some samples
            self.assertGreater(
                len(sample_dataset.samples), 0, "Should generate at least one sample"
            )

            # Test sample structure
            if len(sample_dataset.samples) > 0:
                sample = sample_dataset.samples[0]
                required_keys = [
                    "hadm_id",
                    "patient_id",
                    "conditions",
                    "procedures",
                    "drugs",
                    "clinical_notes",
                    "mortality",
                ]
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")

                # Verify data types
                self.assertIsInstance(
                    sample["clinical_notes"], str, "clinical_notes should be a string"
                )
                self.assertIn(
                    sample["mortality"], [0, 1], "Mortality label should be 0 or 1"
                )

                print(f"Generated {len(sample_dataset.samples)} multimodal samples")
                print(f"Clinical notes length: {len(sample['clinical_notes'])}")

        except Exception as e:
            self.fail(
                f"Failed to use set_task with MultimodalMortalityPredictionMIMIC3: {e}"
            )


if __name__ == "__main__":
    unittest.main()
