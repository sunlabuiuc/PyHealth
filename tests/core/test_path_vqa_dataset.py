import json
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image

from pyhealth.datasets import PathVQADataset


class TestPathVQADataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (8, 8), color=(50, 50, 50)).save(self.images_dir / "slide1.png")
        Image.new("RGB", (8, 8), color=(80, 80, 80)).save(self.images_dir / "slide2.png")
        Image.new("RGB", (8, 8), color=(120, 120, 120)).save(self.images_dir / "slide3.png")

        annotations = [
            {
                "image": "slide1.png",
                "question": "Is this malignant?",
                "answer": "no",
                "split": "train",
                "question_id": "pq1",
                "image_id": "pimg1",
            },
            {
                "image": "slide2.png",
                "question": "Is this benign?",
                "answer": "yes",
                "split": "validation",
                "question_id": "pq2",
                "image_id": "pimg2",
            },
            {
                "image": "slide3.png",
                "question": "Is this necrotic?",
                "answer": "no",
                "split": "test",
                "question_id": "pq3",
                "image_id": "pimg3",
            },
        ]
        with open(self.root / "annotations.json", "w", encoding="utf-8") as file:
            json.dump(annotations, file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_metadata_creation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = PathVQADataset(root=str(self.root), images_dir="images")

        metadata_path = self.root / "path_vqa-metadata-pyhealth.csv"
        self.assertTrue(metadata_path.exists())

        metadata = pd.read_csv(metadata_path)
        self.assertIn("path", metadata.columns)
        self.assertIn("question", metadata.columns)
        self.assertIn("answer", metadata.columns)
        self.assertIn("split", metadata.columns)

    def test_split_filtering_validation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = PathVQADataset(root=str(self.root), images_dir="images", split="validation")

        metadata = pd.read_csv(self.root / "path_vqa-metadata-pyhealth.csv")
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata["split"].iloc[0], "validation")

    def test_clean_split_override(self):
        clean_split = {
            "pq1": "test",
            "pq2": "train",
            "pq3": "validation",
        }
        with open(self.root / "clean_split.json", "w", encoding="utf-8") as file:
            json.dump(clean_split, file)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = PathVQADataset(
                root=str(self.root),
                images_dir="images",
                clean_split_path="clean_split.json",
                refresh_metadata=True,
            )

        metadata = pd.read_csv(self.root / "path_vqa-metadata-pyhealth.csv")
        self.assertEqual(
            metadata.loc[metadata["question_id"] == "pq1", "split"].iloc[0],
            "test",
        )
        self.assertEqual(
            metadata.loc[metadata["question_id"] == "pq2", "split"].iloc[0],
            "train",
        )
        self.assertEqual(
            metadata.loc[metadata["question_id"] == "pq3", "split"].iloc[0],
            "validation",
        )

    def test_dedup_hook_no_crash(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = PathVQADataset(
                root=str(self.root),
                images_dir="images",
                enable_dedup=True,
                dedup_method="auto",
                refresh_metadata=True,
            )

        self.assertTrue((self.root / "path_vqa-metadata-pyhealth.csv").exists())
        self.assertTrue(any("dedup" in str(item.message).lower() for item in caught))


if __name__ == "__main__":
    unittest.main()
