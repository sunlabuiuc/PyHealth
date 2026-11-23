"""Tests for BCSSTissueSegmentation (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import BCSSDataset, get_dataloader
from pyhealth.datasets.bcss import BCSS_CLASS_CODES
from pyhealth.models import UNet2D
from pyhealth.tasks import BCSSTissueSegmentation
from pyhealth.tasks.bcss_tissue_segmentation import BCSS_IGNORE_INDEX
from pyhealth.trainer import Trainer
from tests.core.test_bcss_dataset import write_fixture


class TestBCSSTissueSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        write_fixture(cls.tmp / "bcss")
        # Whole 8x8 ROIs as single tiles; every class may label a tile.
        cls.dataset = BCSSDataset(
            root=str(cls.tmp / "bcss"),
            tile_size=8,
            class_codes=BCSS_CLASS_CODES,
            min_purity=0.2,
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            BCSSTissueSegmentation(image_size=0)
        with self.assertRaises(ValueError):
            BCSSTissueSegmentation(min_labelled=1.5)
        with self.assertRaises(ValueError):
            BCSSTissueSegmentation(groups={1: 7}, num_classes=5)

    def test_grouping_and_ignore(self):
        task = BCSSTissueSegmentation(image_size=8, min_labelled=0.0)
        # TCGA-A1-AAAA quadrants: tumor (1), stroma (2), outside ROI (0), fat (9).
        (sample,) = task(self.dataset.get_patient("TCGA-A1-AAAA"))
        mask = sample["mask"]
        self.assertEqual(mask.shape, (1, 8, 8))
        self.assertEqual(int(mask[0, 0, 0]), 1)  # tumor
        self.assertEqual(int(mask[0, 0, 7]), 2)  # stroma
        self.assertEqual(int(mask[0, 7, 0]), BCSS_IGNORE_INDEX)  # outside ROI
        self.assertEqual(int(mask[0, 7, 7]), 0)  # fat -> other
        # A quarter of the tile is unlabelled, so a 0.8 threshold drops it.
        strict = BCSSTissueSegmentation(image_size=8, min_labelled=0.8)
        self.assertEqual(strict(self.dataset.get_patient("TCGA-A1-AAAA")), [])

    def test_resize_uses_nearest(self):
        task = BCSSTissueSegmentation(image_size=4, min_labelled=0.0)
        (sample,) = task(self.dataset.get_patient("TCGA-A2-BBBB"))
        self.assertEqual(sample["mask"].shape, (1, 4, 4))
        self.assertTrue(set(sample["mask"].ravel().tolist()) <= {1, 3, 4})

    def test_set_task_and_unet(self):
        train, test = self.dataset.set_task(
            BCSSTissueSegmentation(image_size=8, min_labelled=0.0),
            split=self.dataset.official_split(),
        )
        try:
            self.assertEqual((len(train), len(test)), (2, 1))
            self.assertEqual(tuple(train[0]["image"].shape), (3, 8, 8))
            self.assertEqual(train[0]["mask"].dtype, torch.long)
            model = UNet2D(train, num_classes=5, base_channels=4, depth=2)
            self.assertEqual(model.mode, "segmentation_multiclass")
            trainer = Trainer(model=model, enable_logging=False)
            trainer.train(
                train_dataloader=get_dataloader(train, batch_size=2), epochs=1
            )
            scores = trainer.evaluate(get_dataloader(test, batch_size=1))
            self.assertIn("dice", scores)
        finally:
            train.close()
            test.close()


if __name__ == "__main__":
    unittest.main()
