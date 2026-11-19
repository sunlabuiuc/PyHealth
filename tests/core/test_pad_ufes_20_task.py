"""Tests for PADUFES20Classification (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import PADUFES20Dataset, PatientSplit
from pyhealth.tasks import PADUFES20Classification
from tests.core.test_pad_ufes_20_dataset import write_fixture


class TestPADUFES20Classification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "pad"
        root.mkdir()
        write_fixture(root)
        cls.dataset = PADUFES20Dataset(root=str(root), cache_dir=str(cls.tmp / "cache"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_unknown_scheme_raises(self):
        with self.assertRaises(ValueError):
            PADUFES20Classification(label_scheme="isic8")

    def test_call_pad6(self):
        task = PADUFES20Classification()
        samples = task(self.dataset.get_patient("PAT_1"))
        self.assertEqual(sorted(s["label"] for s in samples), ["BCC", "SCC"])
        self.assertTrue(all(s["patient_id"] == "PAT_1" for s in samples))

    def test_call_climb5(self):
        task = PADUFES20Classification(label_scheme="climb5")
        labels = sorted(
            s["label"]
            for pid in ("PAT_1", "PAT_3")
            for s in task(self.dataset.get_patient(pid))
        )
        self.assertEqual(labels, ["AKIEC", "BCC", "OTHER"])

    def test_set_task_rgb_and_fixed_classes(self):
        samples = self.dataset.set_task(PADUFES20Classification(image_size=32))
        try:
            self.assertEqual(len(samples), 4)
            self.assertEqual(tuple(samples[0]["image"].shape), (3, 32, 32))
            self.assertEqual(samples.output_processors["label"].size(), 6)
        finally:
            samples.close()

    def test_patient_split_keeps_patients_together(self):
        parts = self.dataset.set_task(
            PADUFES20Classification(image_size=16),
            split=PatientSplit((0.34, 0.33, 0.33), seed=0),
        )
        try:
            seen = [{part[i]["patient_id"] for i in range(len(part))} for part in parts]
            self.assertEqual(sum(len(p) for p in parts), 4)
            for a in range(len(seen)):
                for b in range(a + 1, len(seen)):
                    self.assertFalse(seen[a] & seen[b])
        finally:
            for part in parts:
                part.close()


if __name__ == "__main__":
    unittest.main()
