import os
from pathlib import Path
import shutil
import unittest

from PIL import Image
import torch
import torchvision.transforms as transforms

from pyhealth.datasets.chestxray14 import ChestXray14Dataset

class TestChestXray14Dataset(unittest.TestCase):
    dataset = ChestXray14Dataset(partial=True)

    def test_len(self):
        self.assertEqual(len(self.dataset), 14999)

    def test_patient_age(self):
        _, meta = self.dataset[0]
        self.assertEqual(meta['patient_age'], 57)

    def test_patient_sex(self):
        _, meta = self.dataset[0]
        self.assertEqual(meta['patient_sex'], 'M')

        _, meta = self.dataset[4]
        self.assertEqual(meta['patient_sex'], 'F')

    def test_get_no_findings(self):
        _, meta = self.dataset[3]
        self.assertEquals(sum(meta['labels'].values()), 0)

    def test_get_one_finding(self):
        _, meta = self.dataset[0]
        self.assertTrue(meta['labels']['cardiomegaly'])
        self.assertEquals(sum(meta['labels'].values()), 1)

    def test_get_multiple_findings(self):
        _, meta = self.dataset[1]
        self.assertTrue(meta['labels']['cardiomegaly'])
        self.assertTrue(meta['labels']['emphysema'])
        self.assertEquals(sum(meta['labels'].values()), 2)

    def test_local_dataset(self):
        ds = ChestXray14Dataset(download=False)
        self.assertEqual(len(ds), 14999)

    def test_path(self):
        with self.assertRaises(FileNotFoundError):
            ds = ChestXray14Dataset(download=False, root="dataset")

        os.makedirs("dataset")
        shutil.move("images", "dataset")
        shutil.move("Data_Entry_2017_v2020.csv", "dataset")

        ds = ChestXray14Dataset(download=False, root="dataset")

        shutil.move("dataset/images", ".")
        shutil.move("dataset/Data_Entry_2017_v2020.csv", ".")
        os.rmdir("dataset")

    def test_transform(self):
        image, _ = self.dataset[0]
        self.assertIsInstance(image, Image.Image)

        ds = ChestXray14Dataset(download=False, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
        image, _ = ds[0]
        self.assertIsInstance(image, torch.Tensor)

    def test_dataset_verification(self):
        os.rename("Data_Entry_2017_v2020.csv", "temp.csv")
        with self.assertRaises(FileNotFoundError):
            ds = ChestXray14Dataset(download=False)
        os.rename("temp.csv", "Data_Entry_2017_v2020.csv")
        ds = ChestXray14Dataset(download=False)

        os.rename("images", "data")
        with self.assertRaises(FileNotFoundError):
            ds = ChestXray14Dataset(download=False)

        os.mkdir("images")
        with self.assertRaises(ValueError):
            ds = ChestXray14Dataset(download=False)
        os.rmdir("images")
        os.rename("data", "images")
        ds = ChestXray14Dataset(download=False)

if __name__ == "__main__":
    unittest.main(verbosity=2)
