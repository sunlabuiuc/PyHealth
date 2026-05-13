"""
Unit tests for the MIMICCXRLongitudinalDataset, MIMICCXRLongitudinalClassification, and HISTAID classes.

Author:
    Joey Stack (jkstack2@illinois.edu)
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import sys

# This adds the parent directory (Pyhealth/) to the system path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() # Fallback for Jupyter Notebooks
    
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, root_dir)

from pyhealth.datasets.mimic_cxr_longitudinal import MIMICCXRLongitudinalDataset
from pyhealth.tasks.mimic_cxr_longitudinal_classification import MIMICCXRLongitudinalClassificationTask
from pyhealth.models.hist_aid import HistAID

class TestMIMICCXRLongitudinalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Setup Temporary Environment
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmpdir.name)
        
        # 2. Generate Synthetic Data Files
        cls.generate_fake_data()
        
        # 3. Initialize Dataset
        cls.dataset = MIMICCXRLongitudinalDataset(
            root=str(cls.root), 
            refresh_cache=True
        )

        # 4. Initialize Task Samples (K=2 for testing windowing/padding)
        cls.K = 2
        cls.task = MIMICCXRLongitudinalClassificationTask(K=cls.K)
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.tmpdir.cleanup()

    @classmethod
    def generate_fake_data(cls):
        # Create Metadata CSV (Gzipped)
        # Patient 100 has 3 studies (longitudinal), Patient 200 has 1 study (static)
        meta_data = {
            "subject_id": [100, 100, 100, 200],
            "study_id": [501, 502, 503, 601],
            "dicom_id": ["d1", "d2", "d3", "d4"]
        }
        meta_df = pd.DataFrame(meta_data)
        meta_df.to_csv(cls.root / "mimic-cxr-2.0.0-metadata.csv.gz", compression='gzip', index=False)
        
        # Create CheXpert Labels CSV (Gzipped) - 14 standard categories
        label_cols = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        label_data = {col: np.random.choice([0, 1], 4) for col in label_cols}
        label_data["subject_id"] = [100, 100, 100, 200]
        label_data["study_id"] = [501, 502, 503, 601]
        
        labels_df = pd.DataFrame(label_data)
        labels_df.to_csv(cls.root / "mimic-cxr-2.0.0-chexpert.csv.gz", compression='gzip', index=False)

    # --- Dataset Tests ---
    
    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        # We expect 2 unique subjects (100 and 200)
        self.assertEqual(len(self.dataset.patients), 2)

    def test_patient_event_parsing(self):
        # Check if Subject 100 has 3 chronologically ordered visits
        patient_100 = self.dataset.patients[100]
        self.assertEqual(len(patient_100), 3)
        self.assertEqual(patient_100[0]['study_id'], 501)
        self.assertEqual(patient_100[2]['study_id'], 503)

    # --- Task Tests ---

    def test_task_samples_count(self):
        # 3 visits for P100 + 1 visit for P200 = 4 samples total
        self.assertEqual(len(self.samples), 4)

    def test_longitudinal_padding(self):
        # The very first visit of any patient should have K empty strings for history
        first_sample = self.samples[0] 
        self.assertEqual(len(first_sample["history_text"]), self.K)
        self.assertTrue(all(text == "" for text in first_sample["history_text"]))

    def test_longitudinal_windowing(self):
        # The 3rd visit for P100 should have 2 historical reports (since K=2)
        third_sample = self.samples[2]
        self.assertEqual(len(third_sample["history_text"]), self.K)
        # It should contain the text from the first two visits
        self.assertIsInstance(third_sample["history_text"][0], str)

    def test_label_integrity(self):
        # Verify the label vector is 14-dimensional
        sample = self.samples[0]
        self.assertEqual(len(sample["label"]), 14)

    # --- Model Tests ---

    def test_hist_aid_forward_pass(self):
        # Initialize model
        feature_size = 768
        model = HistAID(
            feature_size=feature_size,
            num_history=self.K,
            num_labels=14
        )
        
        # Simulate batch of 2
        batch_size = 2
        dummy_image = torch.randn(batch_size, feature_size)
        dummy_history = torch.randn(batch_size, self.K, feature_size)
        
        output = model(image_features=dummy_image, history_features=dummy_history)
        
        # Verify output shape matches num_labels
        self.assertEqual(output["logits"].shape, (batch_size, 14))
        
        # Test gradient flow
        loss = output["logits"].sum()
        loss.backward()
        self.assertIsNotNone(model.linear.weight.grad)

if __name__ == "__main__":
    unittest.main()