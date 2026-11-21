"""Tests for the COVID-BLUES frame classification task."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import COVIDBLUESDataset, PatientSplit, get_dataloader
from pyhealth.tasks import COVIDBLUESFrameClassification

try:
    from tests.core.test_covid_blues_dataset import build_covid_blues_root
except ImportError:  # pragma: no cover - run from tests/core
    from test_covid_blues_dataset import build_covid_blues_root


class TestCOVIDBLUESFrameClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root, frames = build_covid_blues_root(cls.temp_dir)
        cls.dataset = COVIDBLUESDataset(
            root=str(root), frames_dir=str(frames), cache_dir=str(cls.temp_dir / "c")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_frame_samples(self):
        samples = self.dataset.set_task(COVIDBLUESFrameClassification(), num_workers=1)
        self.assertEqual(len(samples), 14)
        labels = {}
        for sample in samples:
            labels.setdefault(sample["patient_id"], set()).add(int(sample["covid"].item()))
        self.assertEqual(labels, {"1": {1}, "2": {0}, "3": {1}, "4": {0}})
        self.assertEqual(tuple(samples[0]["image"].shape), (1, 224, 224))
        batch = next(iter(get_dataloader(samples, batch_size=4)))
        self.assertEqual(tuple(batch["image"].shape), (4, 1, 224, 224))
        self.assertEqual(len(batch["video_id"]), 4)

    def test_patient_split(self):
        train, test = self.dataset.set_task(
            COVIDBLUESFrameClassification(),
            split=PatientSplit(ratios=(0.5, 0.5), seed=1),  # both labels in train
            num_workers=1,
        )
        train_ids = {s["patient_id"] for s in train}
        test_ids = {s["patient_id"] for s in test}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train) + len(test), 14)


if __name__ == "__main__":
    unittest.main()
