"""
Unit tests for the ChestXray14Dataset and ChestXray14BinaryClassification classes.

Author:
    Eric Schrock (ejs9@illinois.edu)
"""
import os
from pathlib import Path
import requests
import shutil
import unittest

import numpy as np
from PIL import Image

from ...datasets.chestxray14 import ChestXray14Dataset
from ...tasks.chestxray14_binary_classification import ChestXray14BinaryClassification

class TestChestXray14Dataset(unittest.TestCase):
    def setUp(self):
        os.mkdir("test")
        os.mkdir("test/images")

        images = [
            "00000001_000.png",
            "00000001_001.png",
            "00000001_002.png",
            "00000002_000.png",
            "00000003_001.png",
            "00000003_002.png",
            "00000003_003.png",
            "00000003_004.png",
            "00000003_005.png",
            "00000003_006.png"
        ]

        for name in images:
            img = Image.fromarray(np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8), mode="RGB")
            img.save(os.path.join("test/images", name))

        # https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468 (mirrored to Google Drive)
        # I couldn't figure out a way to download this file directly from box.com
        response = requests.get('https://drive.google.com/uc?export=download&id=1mkOZNfYt-Px52b8CJZJANNbM3ULUVO3f')
        with open("test/Data_Entry_2017_v2020.csv", "wb") as file:
            file.write(response.content)

        with open("test/Data_Entry_2017_v2020.csv", 'r') as f:
            lines = f.readlines()

        with open("test/Data_Entry_2017_v2020.csv", 'w') as f:
            f.writelines(lines[:11])

        self.dataset = ChestXray14Dataset(root="./test", config_path=str(Path(__file__).parent.parent.parent / "datasets" / "configs" / "chestxray14.yaml"), download=False, partial=True)

    def tearDown(self):
        shutil.rmtree("test")

    def test_len(self):
        self.assertEqual(len(self.dataset), 10)

    def test_get_no_findings(self):
        data = self.dataset[3]
        self.assertEqual(data['path'], str(os.path.join(self.dataset._image_path, '00000002_000.png')))
        self.assertEqual(data['patient_age'], 80)
        self.assertEqual(data['patient_sex'], 'M')
        self.assertEqual(data['atelectasis'], 0)
        self.assertEqual(data['cardiomegaly'], 0)
        self.assertEqual(data['consolidation'], 0)
        self.assertEqual(data['edema'], 0)
        self.assertEqual(data['effusion'], 0)
        self.assertEqual(data['emphysema'], 0)
        self.assertEqual(data['fibrosis'], 0)
        self.assertEqual(data['hernia'], 0)
        self.assertEqual(data['infiltration'], 0)
        self.assertEqual(data['mass'], 0)
        self.assertEqual(data['nodule'], 0)
        self.assertEqual(data['pleural_thickening'], 0)
        self.assertEqual(data['pneumonia'], 0)
        self.assertEqual(data['pneumothorax'], 0)

    def test_get_one_finding(self):
        data = self.dataset[0]
        self.assertEqual(data['path'], str(os.path.join(self.dataset._image_path, '00000001_000.png')))
        self.assertEqual(data['patient_age'], 57)
        self.assertEqual(data['patient_sex'], 'M')
        self.assertEqual(data['atelectasis'], 0)
        self.assertEqual(data['cardiomegaly'], 1)
        self.assertEqual(data['consolidation'], 0)
        self.assertEqual(data['edema'], 0)
        self.assertEqual(data['effusion'], 0)
        self.assertEqual(data['emphysema'], 0)
        self.assertEqual(data['fibrosis'], 0)
        self.assertEqual(data['hernia'], 0)
        self.assertEqual(data['infiltration'], 0)
        self.assertEqual(data['mass'], 0)
        self.assertEqual(data['nodule'], 0)
        self.assertEqual(data['pleural_thickening'], 0)
        self.assertEqual(data['pneumonia'], 0)
        self.assertEqual(data['pneumothorax'], 0)

    def test_get_multiple_findings(self):
        data = self.dataset[6]
        self.assertEqual(data['path'], str(os.path.join(self.dataset._image_path, '00000003_003.png')))
        self.assertEqual(data['patient_age'], 76)
        self.assertEqual(data['patient_sex'], 'F')
        self.assertEqual(data['atelectasis'], 0)
        self.assertEqual(data['cardiomegaly'], 0)
        self.assertEqual(data['consolidation'], 0)
        self.assertEqual(data['edema'], 0)
        self.assertEqual(data['effusion'], 0)
        self.assertEqual(data['emphysema'], 0)
        self.assertEqual(data['fibrosis'], 0)
        self.assertEqual(data['hernia'], 1)
        self.assertEqual(data['infiltration'], 1)
        self.assertEqual(data['mass'], 0)
        self.assertEqual(data['nodule'], 0)
        self.assertEqual(data['pleural_thickening'], 0)
        self.assertEqual(data['pneumonia'], 0)
        self.assertEqual(data['pneumothorax'], 0)

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

    def test_dataset_and_task_classes_match(self):
        # Must define list of classes in both the dataset and task to avoid circular import in this file
        self.assertEqual(ChestXray14Dataset.classes, ChestXray14BinaryClassification.classes)

if __name__ == "__main__":
    unittest.main(verbosity=2)
