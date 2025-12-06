import os
import unittest
import tempfile
import shutil
import io
import sys
from pathlib import Path
import pandas as pd
import torch

from pyhealth.datasets import ISIC2018BiasDataset

class TestISIC2018BiasDataset(unittest.TestCase):
    """Test ISIC2018BiasDataset with synthetic test data"""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.isic_dataset_path = Path(__file__).parent.parent.parent / "test-resources" / "core" / "isic2018"


    def test_patient_count(self):
        """Load the dataset for testing."""
        self.dataset = ISIC2018BiasDataset(root=str(self.isic_dataset_path))
        dataset_size = len(self.dataset.unique_patient_ids)
        print(f"Total Unique patients: {dataset_size}")
        self.assertEqual(dataset_size, 10, "Invalid number of patients")

    def test_stats(self):
        """Test .stats() method execution."""
        try:
            self.dataset = ISIC2018BiasDataset(root=str(self.isic_dataset_path))
            self.dataset.stats()
            print("dataset.stats() executed successfully")
        except Exception as e:
            print(f"✗ dataset.stats() failed with error: {e}")
            self.fail(f"dataset.stats() failed: {e}")

    def test_get_patient_events_by_id(self):
        """Test get_patient and get_events methods with sample 10006."""
        self.dataset = ISIC2018BiasDataset(root=str(self.isic_dataset_path))
        patient = self.dataset.get_patient("9")
        self.assertIsNotNone(patient, msg="Sample ISIC_10 should exist in the dataset")
        print(f"Sample ISIC_9 found: {patient}")

        print("Getting events for patient ISIC_9...")
        events = patient.get_events()
        self.assertEqual(
            len(events), 1, msg="get_events() one sample"
        )
        print(f"Retrieved {len(events)} events")
        self.assertEqual(events[0].event_type, "isic2018_artifacts")
        self.assertEqual(events[0].__getitem__("dark_corner"), "1")
        self.assertEqual(events[0].__getitem__("hair"), "1")
        self.assertEqual(events[0].__getitem__("gel_border"), "0")
        self.assertEqual(events[0].__getitem__("gel_bubble"), "1")
        self.assertEqual(events[0].__getitem__("ruler"), "0")
        self.assertEqual(events[0].__getitem__("ink"), "1")
        self.assertEqual(events[0].__getitem__("patches"), "0")
        self.assertEqual(events[0].__getitem__("label"), "0")
        self.assertEqual(events[0].__getitem__("label_string"), "malignant")




