"""Tests for PatchDataset."""

import multiprocessing
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.datasets import PatchDataset, create_sample_dataset, get_dataloader
from pyhealth.sampler import PatchSampler


def _make_dataset(shape=(8, 8, 8), mask_value=1, channels=1, in_memory=True):
    image = torch.zeros((channels, *shape), dtype=torch.float32)
    mask = torch.zeros((1, *shape), dtype=torch.long)
    center = tuple(size // 2 for size in shape)
    image[(slice(None), *center)] = 5.0
    mask[(0, *center)] = mask_value
    samples = [
        {"patient_id": "p0", "image": image, "mask": mask},
        {"patient_id": "p1", "image": image + 1, "mask": mask},
    ]
    return create_sample_dataset(
        samples, {"image": "raw"}, {"mask": "raw"}, in_memory=in_memory
    )


class TestPatchDataset(unittest.TestCase):
    def test_center_patch_shape_and_alignment(self):
        dataset = _make_dataset()
        patches = PatchDataset(
            dataset, PatchSampler((4, 4, 4)), mode="center"
        )
        sample = patches[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 4, 4, 4))
        self.assertEqual(tuple(sample["mask"].shape), (1, 4, 4, 4))
        self.assertGreater(float(sample["image"].max()), 0.0)
        self.assertGreater(int(sample["mask"].sum()), 0)
        self.assertEqual(sample["patch_origin"], [2, 2, 2])

    def test_foreground_patch_contains_label(self):
        dataset = _make_dataset()
        patches = PatchDataset(
            dataset,
            PatchSampler((2, 2, 2), foreground_prob=1.0),
            mode="train",
        )
        for index in range(len(patches)):
            sample = patches[index]
            self.assertGreater(int(sample["mask"].sum()), 0)

    def test_2d_patch_from_3d_volume(self):
        dataset = _make_dataset()
        patches = PatchDataset(
            dataset, PatchSampler((4, 4)), mode="center"
        )
        sample = patches[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 4, 4))
        self.assertEqual(tuple(sample["mask"].shape), (1, 4, 4))
        self.assertIsNotNone(sample["patch_slice"])

    def test_padding_uses_ignore_index(self):
        dataset = _make_dataset(shape=(3, 3, 3))
        patches = PatchDataset(
            dataset,
            PatchSampler((4, 4, 4)),
            mode="center",
            ignore_index=255,
        )
        sample = patches[0]
        self.assertEqual(tuple(sample["mask"].shape), (1, 4, 4, 4))
        self.assertIn(255, sample["mask"].unique().tolist())
        self.assertGreater(int(sample["mask"].sum()), 0)

    def test_num_patches(self):
        dataset = _make_dataset()
        patches = PatchDataset(
            dataset,
            PatchSampler((4, 4, 4), foreground_prob=1.0),
            mode="train",
            num_patches=3,
        )
        self.assertEqual(len(patches), 6)

    @unittest.skipUnless(nib is not None, "nibabel is not installed")
    def test_lazy_nifti_dataset_materialises(self):
        temp_dir = tempfile.mkdtemp()
        try:
            image_path = Path(temp_dir) / "image.nii.gz"
            mask_path = Path(temp_dir) / "mask.nii.gz"
            image = np.zeros((8, 8, 8), dtype=np.float32)
            image[4, 4, 4] = 5.0
            mask = np.zeros((8, 8, 8), dtype=np.int16)
            mask[4, 4, 4] = 1
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
            dataset = create_sample_dataset(
                [
                    {
                        "patient_id": "p0",
                        "image": str(image_path),
                        "mask": str(mask_path),
                    }
                ],
                {"image": ("nifti", {"lazy": True})},
                {"mask": ("nifti", {"lazy": True, "is_mask": True})},
            )
            try:
                patches = PatchDataset(
                    dataset, PatchSampler((4, 4, 4)), mode="center"
                )
                sample = patches[0]
                self.assertEqual(tuple(sample["image"].shape), (1, 4, 4, 4))
                self.assertEqual(tuple(sample["mask"].shape), (1, 4, 4, 4))
                self.assertGreater(float(sample["image"].max()), 0.0)
            finally:
                dataset.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dataloader_collates_3d_and_2d_patches(self):
        for patch_size, expected in (
            ((4, 4, 4), (2, 1, 4, 4, 4)),
            ((4, 4), (2, 1, 4, 4)),
        ):
            patches = PatchDataset(
                _make_dataset(), PatchSampler(patch_size), mode="center"
            )
            batch = next(iter(get_dataloader(patches, batch_size=2)))
            self.assertEqual(tuple(batch["image"].shape), expected)
            self.assertEqual(tuple(batch["mask"].shape), expected)
            if len(patch_size) == 3:
                self.assertNotIn("patch_slice", batch)
            else:
                self.assertEqual(len(batch["patch_slice"]), 2)

    def test_forked_loader_reads_correct_samples(self):
        """Forked patch workers read the samples they asked for.

        ``litdata>=0.2.71`` uses mmap for map-style readers, so workers no
        longer inherit a shared chunk handle from the parent.
        """
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("this regression needs the fork start method")
        image = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        mask = torch.zeros((1, 8, 8, 8), dtype=torch.long)
        mask[(0, 4, 4, 4)] = 1
        samples = [
            {
                "patient_id": f"p{i}",
                "payload": f"{i:04d}-" + "x" * 200,
                "image": image,
                "mask": mask,
            }
            for i in range(400)
        ]
        dataset = create_sample_dataset(
            samples,
            {"image": "raw", "payload": "raw"},
            {"mask": "raw"},
            in_memory=False,
        )
        try:
            patches = PatchDataset(
                dataset,
                PatchSampler((4, 4, 4)),
                mode="center",
                num_patches=2,
            )
            # Open the parent's cache before the workers fork.
            self.assertEqual(patches[0]["patient_id"], "p0")

            loader = get_dataloader(
                patches,
                batch_size=16,
                num_workers=2,
                shuffle=True,
                multiprocessing_context="fork",
            )
            seen = set()
            for _ in range(3):
                for batch in loader:
                    for patient_id, payload in zip(
                        batch["patient_id"], batch["payload"]
                    ):
                        expected = f"{int(patient_id[1:]):04d}-"
                        self.assertTrue(
                            payload.startswith(expected),
                            f"{patient_id} read bytes from {payload[:8]!r}",
                        )
                        seen.add(patient_id)
            self.assertEqual(seen, {f"p{i}" for i in range(400)})
        finally:
            dataset.close()


if __name__ == "__main__":
    unittest.main()
