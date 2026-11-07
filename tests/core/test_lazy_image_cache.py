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

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.processors.image_io.types import ImageRef


@unittest.skipUnless(
    nib is not None and litdata is not None,
    "nibabel and litdata are required",
)
class TestLazyImageCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        image_path = Path(self.temp_dir) / "scan.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6),
                np.eye(4),
            ),
            str(image_path),
        )
        self.samples = [
            {"patient_id": "p1", "image": str(image_path), "label": 1},
            {"patient_id": "p2", "image": str(image_path), "label": 0},
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dataset(self, in_memory: bool = False):
        return create_sample_dataset(
            self.samples,
            {"image": ("nifti", {"lazy": True})},
            {"label": "binary"},
            in_memory=in_memory,
        )

    def test_disk_cache_stores_reference(self):
        dataset = self._dataset()
        try:
            self.assertFalse(dataset.input_processors["image"].stores_tensor)
            record = litdata.StreamingDataset(dataset.path)[0]
            payload = pickle.loads(record["_pyhealth_objects"])
            self.assertIsInstance(payload["values"]["image"], ImageRef)
        finally:
            dataset.close()

    def test_direct_getitem_returns_reference(self):
        dataset = self._dataset()
        try:
            reference = dataset[0]["image"]
            self.assertIsInstance(reference, ImageRef)
            tensor = dataset.input_processors["image"].materialize(reference)
            self.assertEqual(tuple(tensor.shape), (1, 6, 5, 4))
        finally:
            dataset.close()

    def test_dataloader_materializes_reference(self):
        dataset = self._dataset()
        try:
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 6, 5, 4))
        finally:
            dataset.close()

    def test_in_memory_dataloader_materializes_reference(self):
        dataset = self._dataset(True)
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 6, 5, 4))
