"""Tests for the Brain Tumor MRI dataset loader on a synthetic image tree."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pyhealth.datasets import BrainTumorMRIDataset


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (16, 16), dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def build_brain_tumor_root(root: Path) -> Path:
    """Ten images, two of them copies of others.

    - ``Training/glioma_tumor/g_copy.jpg`` is a copy of the Testing image
      ``Testing/glioma_tumor/t_g.jpg`` (a cross-split duplicate).
    - ``Training/no_tumor/n_copy.jpg`` copies ``Training/no_tumor/n1.jpg``.
    """
    images = {
        "Training/no_tumor/n1.jpg": 1,
        "Training/no_tumor/n_copy.jpg": 1,
        "Training/glioma_tumor/g1.jpg": 2,
        "Training/glioma_tumor/g_copy.jpg": 3,
        "Training/meningioma_tumor/m1.jpg": 4,
        "Training/pituitary_tumor/p1.jpg": 5,
        "Testing/no_tumor/t_n.jpg": 6,
        "Testing/glioma_tumor/t_g.jpg": 3,
        "Testing/meningioma_tumor/t_m.jpg": 7,
        "Testing/pituitary_tumor/t_p.png": 8,
    }
    for relative, seed in images.items():
        target = root / relative
        source = next(
            (root / other for other, s in images.items()
             if s == seed and (root / other).exists()),
            None,
        )
        if source is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        else:
            _write_image(target, seed)
    (root / "Training" / "no_tumor" / "notes.txt").write_text("not an image")
    return root


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
