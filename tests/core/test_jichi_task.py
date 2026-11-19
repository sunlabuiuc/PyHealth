"""Tests for JichiDavisGrading (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import JichiDRDataset, PatientSplit
from pyhealth.tasks import JichiDavisGrading

from tests.core.test_jichi_dataset import write_fixture


class TestJichiDavisGrading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        write_fixture(cls.tmp / "dmr")
        cls.dataset = JichiDRDataset(
            root=str(cls.tmp / "dmr"), cache_dir=str(cls.tmp / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_unknown_column_raises(self):
        with self.assertRaises(ValueError):
            JichiDavisGrading(label_column="grade")

    def test_ordinal_labels(self):
        task = JichiDavisGrading()
        labels = {
            s["image_id"]: s["label"]
            for pid in ("1", "2", "3")
            for s in task(self.dataset.get_patient(pid))
        }
        self.assertEqual(
            labels,
            {
                "1_1_R.jpg": 0,
                "1_2_L.jpg": 1,
                "2_1_R.jpg": 2,
                "3_1_L.jpg": 3,
                "3_2_R.jpg": 0,
            },
        )

    def test_concatenated_column(self):
        task = JichiDavisGrading(label_column="davis_concatenated")
        labels = sorted(s["label"] for s in task(self.dataset.get_patient("3")))
        self.assertEqual(labels, [1, 3])

    def test_set_task_and_patient_split(self):
        task = JichiDavisGrading(image_size=16)
        parts = self.dataset.set_task(
            task, split=PatientSplit((0.34, 0.33, 0.33), seed=0)
        )
        try:
            self.assertEqual(sum(len(p) for p in parts), 5)
            self.assertEqual(parts[0].output_processors["label"].size(), 4)
            first = next(p for p in parts if len(p))
            self.assertEqual(tuple(first[0]["image"].shape), (3, 16, 16))
            seen = [{p[i]["patient_id"] for i in range(len(p))} for p in parts]
            self.assertFalse(
                seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]
            )
        finally:
            for part in parts:
                part.close()


if __name__ == "__main__":
    unittest.main()
