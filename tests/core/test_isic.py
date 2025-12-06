import os
import unittest
import tempfile
import shutil
import io
import sys
from pathlib import Path
import pandas as pd
import torch

from pyhealth.datasets import ISICBiasDataset

class TestISICBiasDataset(unittest.TestCase):
    """Test ISIC2018BiasDataset with synthetic test data"""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.isic_dataset_path = Path(__file__).parent.parent.parent / "test-resources" / "core" / "isic_artifacts"
        cls.raw_isic_dataset_path = cls.isic_dataset_path / "isic_artifacts_raw"
        cls.clean_isic_dataset_path = cls.isic_dataset_path / "isic_artifacts_cleaned"

    def test_raw_artifact_csv(self):
        # Ensure normalized CSV is now comma-separated
        csv_file = self.raw_isic_dataset_path / "isic_bias.csv"
        with open(csv_file, "r") as f:
            line = f.readline()
            self.assertIn(",", line)  # Should now be comma-separated
            self.assertNotIn(";", line)

    def test_clean_artifact_csv(self):
        # Ensure file remains comma-delimited
        csv_file = self.clean_isic_dataset_path / "isic_bias.csv"
        with open(csv_file, "r") as f:
            line = f.readline()
            self.assertIn(",", line)

    def test_patient_count(self):
        """Ensure both raw and cleaned versions load correctly."""

        test_roots = [
            self.raw_isic_dataset_path,  # folder with semicolon CSV
            self.clean_isic_dataset_path,  # folder with comma CSV
        ]

        for root in test_roots:
            with self.subTest(root=root):
                dataset = ISICBiasDataset(root=str(root))
                dataset_size = len(dataset.unique_patient_ids)

                print(f"[{root}] Total Unique patients: {dataset_size}")
                self.assertEqual(dataset_size, 10, "Invalid number of patients")

    def test_stats(self):
        """Test .stats() method execution."""
        test_roots = [
            self.raw_isic_dataset_path,  # folder with semicolon CSV
            self.clean_isic_dataset_path,  # folder with comma CSV
        ]
        for root in test_roots:
            try:
                self.dataset = ISICBiasDataset(root=str(root))
                self.dataset.stats()
                print("dataset.stats() executed successfully")
            except Exception as e:
                print(f"✗ dataset.stats() failed with error: {e}")
                self.fail(f"dataset.stats() failed: {e}")

    def test_get_patient_events_by_id(self):
        test_roots = [
            self.raw_isic_dataset_path,  # folder with semicolon CSV
            self.clean_isic_dataset_path,  # folder with comma CSV
        ]
        for root in test_roots:
            """Test get_patient and get_events methods for image 9."""
            self.dataset = ISICBiasDataset(root=str(root))
            patient = self.dataset.get_patient("9")
            self.assertIsNotNone(patient, msg="ISIC_10 should exist in the dataset")
            print(f"ISIC_10 found: {patient}")

            print("Getting events for patient ISIC_10...")
            events = patient.get_events()
            self.assertEqual(
                len(events), 1, msg="get_events() one sample"
            )
            print(f"Retrieved {len(events)} events")
            self.assertEqual(events[0].event_type, "isic_artifacts_raw")
            self.assertEqual(events[0].__getitem__("dark_corner"), "1")
            self.assertEqual(events[0].__getitem__("hair"), "1")
            self.assertEqual(events[0].__getitem__("gel_border"), "0")
            self.assertEqual(events[0].__getitem__("gel_bubble"), "1")
            self.assertEqual(events[0].__getitem__("ruler"), "0")
            self.assertEqual(events[0].__getitem__("ink"), "1")
            self.assertEqual(events[0].__getitem__("patches"), "0")
            self.assertEqual(events[0].__getitem__("label"), "0")
            self.assertEqual(events[0].__getitem__("label_string"), "malignant")




