import unittest
import tempfile
import shutil
import subprocess
import os
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset


class TestMIMIC3Demo(unittest.TestCase):
    """Test MIMIC3 dataset with demo data downloaded from PhysioNet."""

    # Class-level variables for shared dataset
    temp_dir = None
    demo_dataset_path = None
    dataset = None

    @classmethod
    def setUpClass(cls):
        """Download demo dataset once for all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls._download_demo_dataset()
        cls._load_dataset()

    @classmethod
    def tearDownClass(cls):
        """Clean up downloaded dataset after all tests."""
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @classmethod
    def _download_demo_dataset(cls):
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
            cls.temp_dir,
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
            Path(cls.temp_dir) / "physionet.org" / "files" / "mimiciii-demo" / "1.4"
        )
        if physionet_dir.exists():
            cls.demo_dataset_path = str(physionet_dir)
        else:
            raise unittest.SkipTest("Downloaded dataset not found in expected location")

    @classmethod
    def _load_dataset(cls):
        """Load the dataset once for all tests."""
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions", "noteevents"]

        cls.dataset = MIMIC3Dataset(root=cls.demo_dataset_path, tables=tables)

    def test_stats(self):
        """Test .stats() method execution."""
        try:
            self.dataset.stats()
        except Exception as e:
            self.fail(f"dataset.stats() failed: {e}")

    def test_get_events(self):
        """Test get_patient and get_events methods with patient 10006."""
        # Test get_patient method
        patient = self.dataset.get_patient("10006")
        self.assertIsNotNone(patient, msg="Patient 10006 should exist in demo dataset")

        # Test get_events method
        events = patient.get_events()
        self.assertIsNotNone(events, msg="get_events() should not return None")
        self.assertIsInstance(events, list, msg="get_events() should return a list")
        self.assertGreater(
            len(events), 0, msg="get_events() should not return an empty list"
        )


if __name__ == "__main__":
    unittest.main()
