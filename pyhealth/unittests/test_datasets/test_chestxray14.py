import os
import shutil
import unittest

from pyhealth.datasets.chestxray14 import ChestXray14Dataset

class TestChestXray14Dataset(unittest.TestCase):
    dataset = ChestXray14Dataset(partial=True)

    def test_len(self):
        self.assertEqual(len(self.dataset), 14999)

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

    def test_local_dataset(self):
        ds = ChestXray14Dataset(download=False)
        self.assertEqual(len(ds), 14999)

    def test_root(self):
        with self.assertRaises(FileNotFoundError):
            _ = ChestXray14Dataset(download=False, root="dataset")

        os.makedirs("dataset")
        shutil.move("images", "dataset")
        shutil.move("Data_Entry_2017_v2020.csv", "dataset")

        d_ = ChestXray14Dataset(download=False, root="dataset")

        shutil.move("dataset/images", ".")
        shutil.move("dataset/Data_Entry_2017_v2020.csv", ".")
        os.rmdir("dataset")

    def test_dataset_verification(self):
        os.rename("Data_Entry_2017_v2020.csv", "temp.csv")
        with self.assertRaises(FileNotFoundError):
            _ = ChestXray14Dataset(download=False)
        os.rename("temp.csv", "Data_Entry_2017_v2020.csv")
        _ = ChestXray14Dataset(download=False)

        os.rename("images", "data")
        with self.assertRaises(FileNotFoundError):
            _ = ChestXray14Dataset(download=False)

        os.mkdir("images")
        with self.assertRaises(ValueError):
            _ = ChestXray14Dataset(download=False)
        os.rmdir("images")
        os.rename("data", "images")
        _ = ChestXray14Dataset(download=False)

if __name__ == "__main__":
    unittest.main(verbosity=2)
