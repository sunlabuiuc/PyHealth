"""Tests for the CMMD malignancy classification task."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import CMMDDataset, PatientSplit, get_dataloader
from pyhealth.tasks import CMMDMalignancyClassification

try:
    from tests.core.test_cmmd_dataset import build_cmmd_root, pydicom
except ImportError:  # pragma: no cover - run from tests/core
    from test_cmmd_dataset import build_cmmd_root, pydicom


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCMMDMalignancyClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root = cls.temp_dir / "cmmd"
        root.mkdir()
        build_cmmd_root(root)
        cls.dataset = CMMDDataset(root=str(root), cache_dir=str(cls.temp_dir / "c"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_samples_per_breast_side(self):
        samples = self.dataset.set_task(CMMDMalignancyClassification(), num_workers=1)
        # 8 images, 2 on D2-0001's unlabelled right breast are skipped.
        self.assertEqual(len(samples), 6)
        labels = {
            (s["patient_id"], s["laterality"], s["view"]): int(s["malignant"].item())
            for s in samples
        }
        self.assertEqual(
            labels,
            {
                ("D1-0001", "R", "CC"): 0,
                ("D1-0001", "R", "MLO"): 0,
                ("D1-0002", "L", "CC"): 1,
                ("D1-0002", "R", "CC"): 0,
                ("D2-0001", "L", "CC"): 1,
                ("D2-0001", "L", "MLO"): 1,
            },
        )
        image = samples.input_processors["image"].materialize(samples[0]["image"])
        self.assertEqual(tuple(image.shape), (1, 512, 512))
        batch = next(iter(get_dataloader(samples, batch_size=3)))
        self.assertEqual(tuple(batch["image"].shape), (3, 1, 512, 512))

    def test_patient_split_keeps_patients_together(self):
        train, test = self.dataset.set_task(
            CMMDMalignancyClassification(),
            split=PatientSplit(ratios=(0.67, 0.33), seed=1),
            num_workers=1,
        )
        train_ids = {s["patient_id"] for s in train}
        test_ids = {s["patient_id"] for s in test}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train) + len(test), 6)


if __name__ == "__main__":
    unittest.main()
