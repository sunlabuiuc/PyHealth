import unittest
import os
from pathlib import Path

from pyhealth.datasets import eICUDataset


class TesteICUDemo(unittest.TestCase):
    """Test eICU dataset with demo data from local test resources."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        self._setup_dataset_path()
        self._load_dataset()

    def _setup_dataset_path(self):
        """Get path to local eICU demo dataset in test resources."""
        # Get the path to the test-resources/core/eicudemo directory
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "eicudemo")

        print(f"\n{'='*60}")
        print(f"Setting up eICU demo dataset")
        print(f"Dataset path: {self.demo_dataset_path}")

        # List files in the dataset directory
        files = os.listdir(self.demo_dataset_path)
        csv_files = [f for f in sorted(files) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files in dataset directory:")
        for f in csv_files:
            file_path = os.path.join(self.demo_dataset_path, f)
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")
        print(f"{'='*60}\n")

    def _load_dataset(self):
        """Load the dataset for testing."""
        tables = ["diagnosis", "medication", "physicalexam"]
        print(f"Loading eICUDataset with tables: {tables}")
        self.dataset = eICUDataset(root=self.demo_dataset_path, tables=tables)
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

    def test_unique_patient_ids(self):
        """Test that we can get unique patient IDs."""
        print(f"\n{'='*60}")
        print("TEST: test_unique_patient_ids()")
        print(f"{'='*60}")
        
        patient_ids = self.dataset.unique_patient_ids
        self.assertIsInstance(patient_ids, list)
        self.assertGreater(len(patient_ids), 0, "Should have at least one patient")
        print(f"✓ Found {len(patient_ids)} unique patients")

    def test_get_patient(self):
        """Test get_patient method."""
        print(f"\n{'='*60}")
        print("TEST: test_get_patient()")
        print(f"{'='*60}")

        patient_ids = self.dataset.unique_patient_ids
        patient_id = patient_ids[0]
        
        print(f"Getting patient {patient_id}...")
        patient = self.dataset.get_patient(patient_id)
        self.assertIsNotNone(patient, f"Patient {patient_id} should exist")
        print(f"✓ Patient {patient_id} found: {patient}")

    def test_get_events(self):
        """Test get_patient and get_events methods."""
        print(f"\n{'='*60}")
        print("TEST: test_get_events()")
        print(f"{'='*60}")

        patient_ids = self.dataset.unique_patient_ids
        patient = self.dataset.get_patient(patient_ids[0])
        
        # Test get_events for patient table
        print("Getting patient stay events...")
        patient_events = patient.get_events(event_type="patient")
        self.assertIsNotNone(patient_events)
        self.assertIsInstance(patient_events, list)
        print(f"✓ Found {len(patient_events)} patient stay events")
        
        # Test get_events for diagnosis
        print("Getting diagnosis events...")
        diagnosis_events = patient.get_events(event_type="diagnosis")
        self.assertIsNotNone(diagnosis_events)
        print(f"✓ Found {len(diagnosis_events)} diagnosis events")

        # Show sample event
        if patient_events:
            print(f"\nSample patient event:")
            event = patient_events[0]
            print(f"  Event type: {event.event_type}")
            print(f"  Attributes: {list(event.attr_dict.keys())[:10]}...")

        print(f"✓ test_get_events() passed successfully")


class TesteICUDatasetWithAllTables(unittest.TestCase):
    """Test eICU dataset with all supported tables."""

    def test_load_all_tables(self):
        """Test loading dataset with all supported clinical tables."""
        print(f"\n{'='*60}")
        print("TEST: test_load_all_tables()")
        print(f"{'='*60}")

        test_dir = Path(__file__).parent.parent.parent
        demo_path = str(test_dir / "test-resources" / "core" / "eicudemo")

        tables = ["diagnosis", "medication", "treatment", "lab", "physicalexam", "admissiondx"]
        print(f"Loading eICUDataset with all tables: {tables}")
        
        dataset = eICUDataset(root=demo_path, tables=tables)
        self.assertIsNotNone(dataset)
        
        # Verify stats works
        dataset.stats()
        
        # Get a patient and verify we can access different event types
        patient_ids = dataset.unique_patient_ids
        patient = dataset.get_patient(patient_ids[0])
        
        event_types = ["patient", "diagnosis", "medication", "treatment", "lab", "physicalexam", "admissiondx"]
        for event_type in event_types:
            events = patient.get_events(event_type=event_type)
            print(f"  {event_type}: {len(events)} events")
        
        print("✓ All tables loaded and accessible")


if __name__ == "__main__":
    unittest.main()



