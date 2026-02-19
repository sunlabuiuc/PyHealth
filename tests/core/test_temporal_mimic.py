"""
Unit tests for the TemporalMIMICDataset and TemporalMIMICMultilabelClassification classes.
"""
import os
import shutil
import unittest
import datetime

import numpy as np
from PIL import Image
import pandas as pd

from pyhealth.datasets.temporal_mimic import TemporalMIMICDataset
from pyhealth.tasks.temporal_mimic import TemporalMIMICMultilabelClassification

class TestTemporalMIMICDataset(unittest.TestCase):
    """Sets up the mock data and TemporalMIMICDataset dataset for testing"""
    def setUp(self):
        if os.path.exists("test_temporal"):
            shutil.rmtree("test_temporal")

        # Create mock image
        os.makedirs("test_temporal/images/files/p10/p10000032/s50414267")
        img_path = "test_temporal/images/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png"
        img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8))
        img.save(img_path)

        # Create mock temporal_mimic.csv with one row
        data = {
            "subject_id": ["00000001"],
            "study_id": ["00000001"],
            "split": ["train"],
            "cxrtime": ["2025-12-04 00:00:00"],
            "img_folders": ["['files/p10/p10000032/s50414267']"],
            "img_filenames": ["['02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png']"],
            "img_deltacharttimes": [0],
            "text_deltacharttimes": [0],
            "radnotes": ["This is a test report."],
            "Atelectasis": [0],
            "Cardiomegaly": [1],
            "Consolidation": [0],
            "Edema": [1],
            "Enlarged Cardiomediastinum": [0],
            "Fracture": [0],
            "Lung Lesion": [0],
            "Lung Opacity": [1],
            "No Finding": [0],
            "Pleural Effusion": [0],
            "Pleural Other": [0],
            "Pneumonia": [1],
            "Pneumothorax": [0],
            "Support Devices": [0],
            "impression": ["Impression text."],
            "findings": ["Findings text."],
            "last_paragraph": ["Last paragraph text."],
            "comparison": ["Comparison text."],
            "history": ["History text."],
            "indication": ["Indication text."],
        }

        df = pd.DataFrame(data)
        df.to_csv("test_temporal/temporal_mimic.csv", index=False)

        self.dataset = TemporalMIMICDataset(
            root="test_temporal",
            tables=["temporal_mimic"],
        )

    def tearDown(self):
        shutil.rmtree("test_temporal")

    def test_get_patient(self):
        """Test that the temporal_mimic CSV is properly loaded into the PyHealth Framework"""
        patient = self.dataset.get_patient("00000001")
        events = patient.get_events("temporal_mimic")

        self.assertEqual(len(events), 1)

        event = events[0]

        self.assertEqual(event["study_id"], "00000001")
        self.assertEqual(event["split"], "train")
        self.assertEqual(event["timestamp"], datetime.datetime(2025, 12, 4, 0, 0))
        self.assertEqual(event["img_folders"], "['files/p10/p10000032/s50414267']")
        self.assertEqual(event["img_filenames"], "['02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png']")
        self.assertEqual(event["img_deltacharttimes"], '0')
        self.assertEqual(event["text_deltacharttimes"], '0')
        self.assertEqual(event["radnotes"], "This is a test report.")
        self.assertEqual(event["Atelectasis"], '0')
        self.assertEqual(event["Cardiomegaly"], '1')
        self.assertEqual(event["Consolidation"], '0')
        self.assertEqual(event["Edema"], '1')
        self.assertEqual(event["Enlarged Cardiomediastinum"], '0')
        self.assertEqual(event["Fracture"], '0')
        self.assertEqual(event["Lung Lesion"], '0')
        self.assertEqual(event["Lung Opacity"], '1')
        self.assertEqual(event["No Finding"], '0')
        self.assertEqual(event["Pleural Effusion"], '0')
        self.assertEqual(event["Pleural Other"], '0')
        self.assertEqual(event["Pneumonia"], '1')
        self.assertEqual(event["Pneumothorax"], '0')
        self.assertEqual(event["Support Devices"], '0')
        self.assertEqual(event["impression"], "Impression text.")
        self.assertEqual(event["findings"], "Findings text.")
        self.assertEqual(event["last_paragraph"], "Last paragraph text.")
        self.assertEqual(event["comparison"], "Comparison text.")
        self.assertEqual(event["history"], "History text.")
        self.assertEqual(event["indication"], "Indication text.")

    def test_set_task(self):
        """Tests the default task TemporalMIMICMultilabelClassification"""
        task = TemporalMIMICMultilabelClassification(
            image_root="test_temporal/images"
        )
        sample_dataset = self.dataset.set_task(task)
        self.assertEqual(len(sample_dataset), 1)

        sample = sample_dataset.samples[0]
        self.assertIn("radnotes", sample)
        self.assertIn("image_path", sample)
        self.assertIn("labels", sample)

        predicted_labels = {
            name for name, idx in sample_dataset.output_processors["labels"].label_vocab.items()
            if sample["labels"][idx].item() == 1.0
        }

        expected_labels = {"Cardiomegaly", "Edema", "Lung Opacity", "Pneumonia"}

        self.assertEqual(predicted_labels, expected_labels)

if __name__ == "__main__":
    unittest.main()
