"""Tests that lazy image references keep the processed cache small."""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

try:
    import litdata
except ImportError:  # pragma: no cover
    litdata = None

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.processors.image_io.types import ImageRef


@unittest.skipUnless(
    nib is not None and litdata is not None,
    "nibabel and litdata are required",
)
class TestLazyImageCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.image_path = self.root / "scan.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6),
                np.eye(4),
            ),
            str(self.image_path),
        )
        self.samples = [
            {"patient_id": "p1", "image": str(self.image_path), "label": 1},
            {"patient_id": "p2", "image": str(self.image_path), "label": 0},
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_disk_cache_stores_reference(self):
        dataset = create_sample_dataset(
            self.samples,
            {"image": ("nifti", {"lazy": True})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            self.assertFalse(dataset.input_processors["image"].stores_tensor)
            record = litdata.StreamingDataset(dataset.path)[0]
            payload = pickle.loads(record["_pyhealth_objects"])
            self.assertIsInstance(payload["values"]["image"], ImageRef)
        finally:
            dataset.close()

    def test_getitem_materializes_reference(self):
        dataset = create_sample_dataset(
            self.samples,
            {"image": ("nifti", {"lazy": True})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (1, 6, 5, 4))
            raw = SampleDataset(dataset.path, resolve_refs=False)
            self.assertIsInstance(raw[0]["image"], ImageRef)
        finally:
            dataset.close()

    def test_in_memory_iteration_materializes_reference(self):
        dataset = create_sample_dataset(
            self.samples,
            {"image": ("nifti", {"lazy": True})},
            {"label": "binary"},
        )
        shapes = [tuple(sample["image"].shape) for sample in dataset]
        self.assertEqual(shapes, [(1, 6, 5, 4), (1, 6, 5, 4)])

    def test_dataloader_collates_materialized_volumes(self):
        dataset = create_sample_dataset(
            self.samples,
            {"image": ("nifti", {"lazy": True})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            loader = get_dataloader(dataset, batch_size=2)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 6, 5, 4))
        finally:
            dataset.close()


if __name__ == "__main__":
    unittest.main()
