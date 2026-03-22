"""Tests for Daily and Sports Activities (DSA) dataset."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyhealth.datasets import DSADataset


def _write_segment(path: Path, n_rows: int = 125, n_cols: int = 45) -> None:
    line = ",".join(["0.0"] * n_cols)
    path.write_text("\n".join([line] * n_rows) + "\n", encoding="utf-8")


class TestDSADataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root_path = cls._tmpdir.name
        # Minimal tree: one activity, one subject, one segment
        seg_dir = Path(cls.root_path) / "a01" / "p1"
        seg_dir.mkdir(parents=True)
        _write_segment(seg_dir / "s01.txt")

        cls.dataset = DSADataset(root=cls.root_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_dataset_initialization(self):
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.dataset_name, "dsa")

    def test_get_subject_ids(self):
        subject_ids = self.dataset.get_subject_ids()
        self.assertIsInstance(subject_ids, list)
        self.assertEqual(subject_ids, ["p1"])

    def test_get_activity_labels(self):
        activity_labels = self.dataset.get_activity_labels()
        self.assertIsInstance(activity_labels, dict)
        self.assertEqual(len(activity_labels), 19)
        self.assertEqual(activity_labels.get("sitting"), 0)

    def test_subject_data_loading(self):
        subject_ids = self.dataset.get_subject_ids()
        self.assertTrue(subject_ids)
        subject_id = subject_ids[0]
        subject_data = self.dataset.get_subject_data(subject_id)

        self.assertIsInstance(subject_data, dict)
        self.assertEqual(subject_data["id"], subject_id)
        self.assertIn("activities", subject_data)
        self.assertIn("sitting", subject_data["activities"])

        activity_data = subject_data["activities"]["sitting"]
        self.assertIsInstance(activity_data["segments"], list)
        self.assertTrue(activity_data["segments"])

        segment = activity_data["segments"][0]
        self.assertIsInstance(segment["data"], np.ndarray)
        self.assertEqual(segment["sampling_rate"], 25)
        self.assertEqual(segment["data"].shape, (125, 45))

    def test_data_consistency(self):
        self.dataset.get_subject_ids()
        self.assertIsNotNone(self.dataset._metadata)
        for _subject_id, subject_info in self.dataset._metadata["subjects"].items():
            for _activity_name, activity_info in subject_info["activities"].items():
                for segment_file in activity_info["segments"]:
                    file_path = os.path.join(activity_info["path"], segment_file)
                    with open(file_path, encoding="utf-8") as f:
                        line = f.readline()
                    self.assertEqual(len(line.strip().split(",")), 45)


if __name__ == "__main__":
    unittest.main()
