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
        # Get the path to the test-resources/core/mimic3demo directory
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "eicudemo")
        
        print(f"\n{'='*60}")
        print(f"Setting up eICU demo dataset")
        print(f"Dataset path: {self.demo_dataset_path}")
        
        # Verify the dataset exists
        if not os.path.exists(self.demo_dataset_path):
            raise unittest.SkipTest(
                f"eICU demo dataset not found at {self.demo_dataset_path}"
            )
        
        # List files in the dataset directory
        files = os.listdir(self.demo_dataset_path)
        print(f"Found {len(files)} files in dataset directory:")
        for f in sorted(files):
            file_path = os.path.join(self.demo_dataset_path, f)
            size = os.path.getsize(file_path) / 1024  # 
            print(f"  - {f} ({size:.1f} KB)")
        print(f"{'='*60}\n")

    def _load_dataset(self):
        """Load the dataset for testing."""
        tables = ["hospital", "admissiondx", "diagnosis", "medication", "lab", "treatment", "physicalExam", "admissiondrug", "allergy", "apacheapsvar", 
                  "apachepatientresult", "apachepredvar", "careplancareprovider", "careplaneol", "careplangeneral","careplangoal",
                  "careplaninfectiousdisease","customlab","infusiondrug", "intakeoutput","microlab","note",
                  "nurseassessment","nursecare","nursecharting", "pasthistory",
                  "respiratorycare", "respiratorycharting",
                  "vitalPeriodic","vitalAperiodic"]
        print(f"Loading eICUDataset with tables: {tables}")
        self.dataset = eICUDataset(root=self.demo_dataset_path, tables=tables)
        print(f"✓ Dataset loaded successfully")
        print(self.dataset)
        #print(f"  Total patients: {len(self.dataset.patient)}")
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
        """Test get_patient and get_events methods with patient 141764."""
        print(f"\n{'='*60}")
        print("TEST: test_get_events()")
        print(f"{'='*60}")
        
        # Test get_patient method
        print("Getting patient 141764...")
        patient = self.dataset.get_patient("141764")
        self.assertIsNotNone(patient, msg="Patient 141764 should exist in demo dataset")
        print(f"✓ Patient 141764 found: {patient}")

        # Test get_events method
        print("Getting events for patient 141764...")
        events = patient.get_events()
        self.assertIsNotNone(events, msg="get_events() should not return None")
        self.assertIsInstance(events, list, msg="get_events() should return a list")
        self.assertGreater(
            len(events), 0, msg="get_events() should not return an empty list"
         )
        print(f"✓ Retrieved {len(events)} events")
      #  print(f"  Event types: {set(e[0] for e in events)}")
        
        # Show sample events
        print(f"\nSample events (first 3):")
        for i, event in enumerate(events[:3]):
            print(f"  {i+1}. Type: {event['event_type']}, Time: {event['timestamp']}")
#            print(f"  {i+1}. Type: {event[0]}, Time: {event[1]}, Data: {event[2]}")
        
        print(f"✓ test_get_events() passed successfully")


if __name__ == "__main__":
    unittest.main()
