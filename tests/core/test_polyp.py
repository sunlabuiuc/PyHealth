"""Tests for PolypDataset and PolypDetection (a synthetic DATA/POLYP folder)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pyhealth.datasets import POLYP_TEST_SETS, PolypDataset
from pyhealth.tasks import PolypDetection


def _write_set(root, subset, images):
    """images: list of (file_name, boxes)."""
    if subset == "train":
        ann_path, image_dir = root / "annotations" / "Polyp_train.json", root / "train" / "images"
    else:
        ann_path = root / "annotations" / f"{subset}_val.json"
        image_dir = root / "val" / subset / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "polyp"}]}
    for i, (name, boxes) in enumerate(images):
        Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(image_dir / name)
        ann["images"].append({"id": i, "file_name": name, "width": 30, "height": 20})
        for b in boxes:
            ann["annotations"].append({"id": len(ann["annotations"]), "image_id": i,
                                       "category_id": 1, "bbox": b, "iscrowd": 0})
    ann_path.write_text(json.dumps(ann))


class TestPolypDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "POLYP"
        _write_set(self.root, "CVC-300", [("1.png", [[1, 2, 5, 6]]), ("2.png", [])])
        _write_set(self.root, "Kvasir", [("a.png", [[0, 0, 3, 3], [10, 5, 4, 4]])])
        _write_set(self.root, "train", [("t.png", [[2, 2, 2, 2]])])
        self.cache = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def test_samples(self):
        ds = PolypDataset(root=str(self.root), subsets=["Kvasir", "CVC-300"],
                          cache_dir=str(self.cache))
        samples = ds.set_task(PolypDetection())
        by_id = {samples[i]["patient_id"]: samples[i] for i in range(len(samples))}
        self.assertEqual(sorted(by_id), ["CVC-300/1", "CVC-300/2", "Kvasir/a"])
        s = by_id["Kvasir/a"]
        self.assertEqual(s["boxes"], [[0.0, 0.0, 3.0, 3.0], [10.0, 5.0, 4.0, 4.0]])
        self.assertEqual((s["subset"], s["width"], s["height"]), ("Kvasir", 30, 20))
        self.assertTrue(s["image"].endswith("val/Kvasir/images/a.png"))
        self.assertEqual(by_id["CVC-300/2"]["boxes"], [])
        self.assertEqual(type(ds.default_task).__name__, "PolypDetection")

    def test_train_subset_and_refresh(self):
        ds = PolypDataset(root=str(self.root), subsets="train", cache_dir=str(self.cache))
        samples = ds.set_task()
        self.assertEqual(samples[0]["patient_id"], "train/t")
        # A different subset selection rebuilds the metadata.
        ds = PolypDataset(root=str(self.root), subsets=["CVC-300"], cache_dir=str(self.cache))
        self.assertEqual(len(ds.unique_patient_ids), 2)

    def test_errors(self):
        with self.assertRaises(ValueError):
            PolypDataset(root=str(self.root), subsets=["Kvasir-SEG"])
        with self.assertRaises(ValueError):
            PolypDataset(root=str(self.root), subsets=[])
        with self.assertRaisesRegex(FileNotFoundError, "annotations"):
            PolypDataset(root=str(self.root), subsets=["ETIS-LaribPolypDB"],
                         cache_dir=str(self.cache))
        (self.root / "val" / "CVC-300" / "images" / "2.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "TestDataset"):
            PolypDataset(root=str(self.root), subsets=["CVC-300"], cache_dir=str(self.cache))

    def test_test_sets(self):
        self.assertEqual(len(POLYP_TEST_SETS), 5)


if __name__ == "__main__":
    unittest.main()
