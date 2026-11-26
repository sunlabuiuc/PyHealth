"""Tests for the Brain Tumor MRI dataset loader on a synthetic image tree."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets import BrainTumorMRIDataset
from tests.core.brain_tumor_mri_test_utils import build_brain_tumor_root


class TestBrainTumorMRIDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = build_brain_tumor_root(self.tmp / "btm")
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_drop_duplicates_keeps_testing_copy(self):
        dataset = BrainTumorMRIDataset(root=str(self.root), cache_dir=str(self.cache))
        meta = pd.read_csv(dataset.metadata_path).set_index("record_id")
        self.assertEqual(len(meta), 8)
        self.assertIn("Testing/glioma_tumor/t_g.jpg", meta.index)
        self.assertNotIn("Training/glioma_tumor/g_copy.jpg", meta.index)
        self.assertIn("Training/no_tumor/n1.jpg", meta.index)
        self.assertNotIn("Training/no_tumor/n_copy.jpg", meta.index)
        self.assertEqual(meta.loc["Testing/glioma_tumor/t_g.jpg", "cross_split_duplicate"], 1)
        self.assertEqual((meta["split"] == "test").sum(), 4)
        self.assertTrue((meta.loc[meta["split"] == "test", "fold"] == 10).all())
        self.assertTrue(meta.loc[meta["split"] == "train", "fold"].between(1, 9).all())
        self.assertFalse(any(self.root.rglob("*pyhealth*")))

    def test_keep_duplicates_shares_patient_id(self):
        dataset = BrainTumorMRIDataset(
            root=str(self.root), cache_dir=str(self.cache), drop_duplicates=False
        )
        meta = pd.read_csv(dataset.metadata_path).set_index("record_id")
        self.assertEqual(len(meta), 10)
        self.assertEqual(
            meta.loc["Training/glioma_tumor/g_copy.jpg", "patient_id"],
            meta.loc["Testing/glioma_tumor/t_g.jpg", "patient_id"],
        )
        self.assertEqual(meta.loc["Training/no_tumor/n_copy.jpg", "duplicate_count"], 2)
        self.assertEqual(len(dataset.unique_patient_ids), 8)

    def test_switching_drop_duplicates_rebuilds(self):
        BrainTumorMRIDataset(root=str(self.root), cache_dir=str(self.cache))
        dataset = BrainTumorMRIDataset(
            root=str(self.root), cache_dir=str(self.cache), drop_duplicates=False
        )
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 10)

    def test_missing_split_folder_raises(self):
        shutil.rmtree(self.root / "Testing")
        with self.assertRaises(FileNotFoundError):
            BrainTumorMRIDataset(root=str(self.root), cache_dir=str(self.cache))

    def test_default_task(self):
        dataset = BrainTumorMRIDataset(root=str(self.root), cache_dir=str(self.cache))
        self.assertEqual(dataset.default_task.task_name, "BrainTumorMRIClassification")


if __name__ == "__main__":
    unittest.main()
