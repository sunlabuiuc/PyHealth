"""Tests for BrainTumorMRIClassification on a synthetic image tree."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import BrainTumorMRIDataset, split_by_strat_fold
from pyhealth.tasks import BrainTumorMRIClassification
from tests.core.brain_tumor_mri_test_utils import build_brain_tumor_root


class TestBrainTumorMRIClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = build_brain_tumor_root(cls.tmp / "btm")
        cls.dataset = BrainTumorMRIDataset(root=str(root), cache_dir=str(cls.tmp / "cache"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_samples_labels_and_shape(self):
        samples = self.dataset.set_task(
            BrainTumorMRIClassification(image_size=32), num_workers=1
        )
        try:
            self.assertEqual(len(samples), 8)
            processor = samples.output_processors["label"]
            self.assertEqual(processor.size(), 4)
            self.assertEqual(processor.label_vocab["no_tumor"], 0)
            self.assertEqual(processor.label_vocab["pituitary_tumor"], 3)
            self.assertEqual(tuple(samples[0]["image"].shape), (1, 32, 32))
        finally:
            samples.close()

    def test_official_split_by_fold(self):
        samples = self.dataset.set_task(
            BrainTumorMRIClassification(image_size=16), num_workers=1
        )
        try:
            train, val, test = split_by_strat_fold(
                samples,
                train_folds=tuple(range(1, 10)),
                val_folds=(),
                test_folds=(10,),
                fold_field="fold",
                check_patient_disjoint=True,
            )
            self.assertEqual(len(test), 4)
            self.assertEqual(len(train) + len(val), 4)
        finally:
            samples.close()


if __name__ == "__main__":
    unittest.main()
