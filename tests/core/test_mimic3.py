import unittest
import os
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset


class TestMIMIC3Demo(unittest.TestCase):
    """Test MIMIC3 dataset with demo data from local test resources."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        self._setup_dataset_path()
        self._load_dataset()

    def _setup_dataset_path(self):
        """Get path to local MIMIC-III demo dataset in test resources."""
        # Get the path to the test-resources/core/mimic3demo directory
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "mimic3demo")

        print(f"\n{'='*60}")
        print(f"Setting up MIMIC-III demo dataset")
        print(f"Dataset path: {self.demo_dataset_path}")

        # List files in the dataset directory
        files = os.listdir(self.demo_dataset_path)
        print(f"Found {len(files)} files in dataset directory:")
        for f in sorted(files):
            file_path = os.path.join(self.demo_dataset_path, f)
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")
        print(f"{'='*60}\n")

    def _load_dataset(self):
        """Load the dataset for testing."""
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions"]
        print(f"Loading MIMIC3Dataset with tables: {tables}")
        self.dataset = MIMIC3Dataset(root=self.demo_dataset_path, tables=tables)
        print(f"✓ Dataset loaded successfully")
        print()

    def test_stats(self):
        """Test .stats() method execution."""
        print(f"\n{'='*60}")
        print("TEST: test_stats()")
        print(f"{'='*60}")
        try:
            print("Calling dataset.stats()...")
            self.dataset.stats()
            print("✓ dataset.stats() executed successfully")
        except Exception as e:
            print(f"✗ dataset.stats() failed with error: {e}")
            self.fail(f"dataset.stats() failed: {e}")

    def test_get_events(self):
        """Test get_patient and get_events methods with patient 10006."""
        print(f"\n{'='*60}")
        print("TEST: test_get_events()")
        print(f"{'='*60}")

        # Test get_patient method
        print("Getting patient 10006...")
        patient = self.dataset.get_patient("10006")
        self.assertIsNotNone(patient, msg="Patient 10006 should exist in demo dataset")
        print(f"✓ Patient 10006 found: {patient}")

        # Test get_events method
        print("Getting events for patient 10006...")
        events = patient.get_events()
        self.assertIsNotNone(events, msg="get_events() should not return None")
        self.assertIsInstance(events, list, msg="get_events() should return a list")
        self.assertGreater(
            len(events), 0, msg="get_events() should not return an empty list"
        )
        print(f"✓ Retrieved {len(events)} events")
        print(f"  Event types: {set(e.event_type for e in events)}")

        # Show sample events
        print(f"\nSample events (first 3):")
        for i, event in enumerate(events[:3]):
            print(f"  {i+1}. Type: {event.event_type}, Time: {event.timestamp}, Data: {event.attr_dict}")

        print(f"✓ test_get_events() passed successfully")


if __name__ == "__main__":
    unittest.main()
