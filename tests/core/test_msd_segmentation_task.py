"""Tests for the MSD segmentation task."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.datasets import MSDHippocampusDataset, get_dataloader


@unittest.skipUnless(nib is not None, "nibabel is not installed")
class TestMSDSegmentationTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir) / "msd"
        for directory in ("imagesTr", "labelsTr", "imagesTs"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        image = np.zeros((4, 5, 6), dtype=np.float32)
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1:3, 2:4, 2:5] = 1
        nib.save(
            nib.Nifti1Image(image, np.eye(4)),
            str(self.root / "imagesTr/case_001.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(mask, np.eye(4)),
            str(self.root / "labelsTr/case_001.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(image, np.eye(4)),
            str(self.root / "imagesTs/case_002.nii.gz"),
        )
        specification = {
            "tensorImageSize": "3D",
            "modality": {"0": "MRI"},
            "labels": {"0": "background", "1": "target"},
            "training": [
                {
                    "image": "./imagesTr/case_001.nii.gz",
                    "label": "./labelsTr/case_001.nii.gz",
                }
            ],
            "test": ["./imagesTs/case_002.nii.gz"],
        }
        with open(self.root / "dataset.json", "w") as handle:
            json.dump(specification, handle)
        self.dataset = MSDHippocampusDataset(
            root=str(self.root),
            cache_dir=str(Path(self.temp_dir) / "cache"),
            metadata_dir=str(Path(self.temp_dir) / "metadata"),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_task(self):
        self.assertEqual(
            self.dataset.default_task.task_name, "MSDSegmentation"
        )

    def test_set_task(self):
        samples = self.dataset.set_task(num_workers=1)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "case_001")
        image = samples.input_processors["image"].materialize(sample["image"])
        mask = samples.output_processors["mask"].materialize(sample["mask"])
        self.assertEqual(tuple(image.shape), (1, 6, 5, 4))
        self.assertEqual(tuple(mask.shape), (1, 6, 5, 4))
        self.assertEqual(str(mask.dtype), "torch.int64")
        self.assertEqual(set(mask.unique().tolist()), {0, 1})
        self.assertEqual(set(samples.input_schema), {"image"})
        self.assertEqual(set(samples.output_schema), {"mask"})

        loader = get_dataloader(samples, batch_size=1)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["image"].shape), (1, 1, 6, 5, 4))
        self.assertEqual(tuple(batch["mask"].shape), (1, 1, 6, 5, 4))


if __name__ == "__main__":
    unittest.main()
