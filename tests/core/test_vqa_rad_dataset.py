import json
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image

from pyhealth.datasets import VQARADDataset


class TestVQARADDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(self.images_dir / "img1.png")
        Image.new("RGB", (8, 8), color=(0, 255, 0)).save(self.images_dir / "img2.png")

        annotations = [
            {
                "image": "img1.png",
                "question": "Is there a fracture?",
                "answer": "no",
                "split": "train",
                "question_id": "q1",
                "image_id": "img1",
            },
            {
                "image": "img2.png",
                "question": "Is this a CT scan?",
                "answer": "yes",
                "split": "test",
                "question_id": "q2",
                "image_id": "img2",
            },
            {
                "image": "img1.png",
                "question": "Is this duplicate leakage?",
                "answer": "maybe",
                "split": "test",
                "question_id": "q3",
                "image_id": "img3",
            },
        ]
        with open(self.root / "annotations.json", "w", encoding="utf-8") as file:
            json.dump(annotations, file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_metadata_creation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = VQARADDataset(root=str(self.root), images_dir="images")

        metadata_path = self.root / "vqa_rad-metadata-pyhealth.csv"
        self.assertTrue(metadata_path.exists())

        metadata = pd.read_csv(metadata_path)
        self.assertIn("path", metadata.columns)
        self.assertIn("question", metadata.columns)
        self.assertIn("answer", metadata.columns)
        self.assertIn("split", metadata.columns)
        self.assertIn("question_id", metadata.columns)
        self.assertIn("image_id", metadata.columns)

    def test_split_filtering(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = VQARADDataset(root=str(self.root), images_dir="images", split="test")

        metadata = pd.read_csv(self.root / "vqa_rad-metadata-pyhealth.csv")
        self.assertTrue((metadata["split"] == "test").all())

    def test_clean_split_override(self):
        clean_split = [
            {"question_id": "q1", "split": "test"},
            {"question_id": "q2", "split": "train"},
        ]
        with open(self.root / "clean_split.json", "w", encoding="utf-8") as file:
            json.dump(clean_split, file)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = VQARADDataset(
                root=str(self.root),
                images_dir="images",
                clean_split_path="clean_split.json",
                refresh_metadata=True,
            )

        metadata = pd.read_csv(self.root / "vqa_rad-metadata-pyhealth.csv")
        q1_split = metadata.loc[metadata["question_id"] == "q1", "split"].iloc[0]
        q2_split = metadata.loc[metadata["question_id"] == "q2", "split"].iloc[0]
        self.assertEqual(q1_split, "test")
        self.assertEqual(q2_split, "train")

    def test_dedup_hook_no_crash_auto(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = VQARADDataset(
                root=str(self.root),
                images_dir="images",
                enable_dedup=True,
                dedup_method="auto",
                refresh_metadata=True,
            )

        self.assertTrue((self.root / "vqa_rad-metadata-pyhealth.csv").exists())
        self.assertTrue(any("dedup" in str(item.message).lower() for item in caught))


if __name__ == "__main__":
    unittest.main()
