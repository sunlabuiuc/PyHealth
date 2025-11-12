"""
Unit tests for the ChestXray14Dataset, ChestXray14BinaryClassification, and ChestXray14MultilabelClassification classes.

Author:
    Eric Schrock (ejs9@illinois.edu)
"""
import os
from pathlib import Path
import shutil
import unittest

import numpy as np
from PIL import Image

from pyhealth.datasets import ChestXray14Dataset
from pyhealth.tasks import ChestXray14BinaryClassification
from pyhealth.tasks import ChestXray14MultilabelClassification

class TestChestXray14Dataset(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test"):
            shutil.rmtree("test")
        os.makedirs("test/images")

        # Source: https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468
        lines = [
            "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Sex,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],",
            "00000001_000.png,Cardiomegaly,0,1,57,M,PA,2682,2749,0.14300000000000002,0.14300000000000002,",
            "00000001_001.png,Cardiomegaly|Emphysema,1,1,58,M,PA,2894,2729,0.14300000000000002,0.14300000000000002,",
            "00000001_002.png,Cardiomegaly|Effusion,2,1,58,M,PA,2500,2048,0.168,0.168,",
            "00000002_000.png,No Finding,0,2,80,M,PA,2500,2048,0.171,0.171,",
            "00000003_001.png,Hernia,0,3,74,F,PA,2500,2048,0.168,0.168,",
            "00000003_002.png,Hernia,1,3,75,F,PA,2048,2500,0.168,0.168,",
            "00000003_003.png,Hernia|Infiltration,2,3,76,F,PA,2698,2991,0.14300000000000002,0.14300000000000002,",
            "00000003_004.png,Hernia,3,3,77,F,PA,2500,2048,0.168,0.168,",
            "00000003_005.png,Hernia,4,3,78,F,PA,2686,2991,0.14300000000000002,0.14300000000000002,",
            "00000003_006.png,Hernia,5,3,79,F,PA,2992,2991,0.14300000000000002,0.14300000000000002,",
        ]

        # Create mock images to test image loading
        for line in lines[1:]: # Skip header row
            name = line.split(',')[0]
            img = Image.fromarray(np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8), mode="RGBA")
            img.save(os.path.join("test/images", name))

        # Save image labels to file
        with open("test/Data_Entry_2017_v2020.csv", 'w') as f:
            f.write("\n".join(lines))

        self.dataset = ChestXray14Dataset(root="./test")

    def tearDown(self):
        if os.path.exists("test"):
            shutil.rmtree("test")

    def test_stats(self):
        self.dataset.stats()

    def test_len(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 10)

    def test_get_patient(self):
        events = self.dataset.get_patient(self.dataset.unique_patient_ids[0]).get_events()
        self.assertEqual(len(events), 1)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ChestXray14MultilabelClassification)

    def test_task_given_invalid_disease(self):
        with self.assertRaises(ValueError):
            _ = ChestXray14BinaryClassification(disease="toothache")

    def test_task_classify_cardiomegaly(self):
        task = ChestXray14BinaryClassification(disease="cardiomegaly")
        samples = self.dataset.set_task(task)
        self.assertEqual(len(samples), 10)
        self.assertEqual(sum(sample["label"] for sample in samples), 3)

    def test_task_classify_hernia(self):
        task = ChestXray14BinaryClassification(disease="hernia")
        samples = self.dataset.set_task(task)
        self.assertEqual(len(samples), 10)
        self.assertEqual(sum(sample["label"] for sample in samples), 6)

    def test_task_classify_all(self):
        samples = self.dataset.set_task()
        self.assertEqual(len(samples), 10)

        actual_labels = [sample["labels"].tolist() for sample in samples]

        expected_labels = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]

        self.assertCountEqual(actual_labels, expected_labels)

if __name__ == "__main__":
    unittest.main()
