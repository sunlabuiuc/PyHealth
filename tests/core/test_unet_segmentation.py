"""Smoke tests for U-Net models, Trainer integration and sliding windows."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.datasets import PatchDataset, create_sample_dataset, get_dataloader
from pyhealth.models import (
    UNet,
    UNet2D,
    UNet3D,
    evaluate_segmentation,
    sliding_window_inference,
)
from pyhealth.sampler import PatchSampler
from pyhealth.trainer import Trainer


def _make_3d_dataset():
    samples = []
    for index in range(3):
        image = torch.randn(1, 16, 16, 16)
        mask = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        mask[:, 4:8, 5:9, 6:10] = 1
        samples.append(
            {"patient_id": f"p{index}", "image": image, "mask": mask}
        )
    return create_sample_dataset(samples, {"image": "raw"}, {"mask": "raw"})


def _make_2d_dataset():
    samples = []
    for index in range(2):
        image = torch.randn(1, 16, 16)
        mask = torch.zeros(1, 16, 16, dtype=torch.long)
        mask[:, 4:8, 5:9] = 1
        samples.append(
            {"patient_id": f"p{index}", "image": image, "mask": mask}
        )
    return create_sample_dataset(samples, {"image": "raw"}, {"mask": "raw"})


class TestUNet(unittest.TestCase):
    def test_3d_forward_and_loss(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        self.assertEqual(model.mode, "segmentation_multiclass")
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        output = model(**batch)
        self.assertEqual(tuple(output["logit"].shape), (2, 2, 16, 16, 16))
        self.assertEqual(tuple(output["y_prob"].shape), (2, 2, 16, 16, 16))
        output["loss"].backward()

    def test_2d_forward(self):
        dataset = _make_2d_dataset()
        model = UNet2D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        output = model(**batch)
        self.assertEqual(tuple(output["logit"].shape), (2, 2, 16, 16))

    def test_partial_class_mask_batch(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=5, base_channels=4, depth=2
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        batch["class_mask"] = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]
        output = model(**batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()

    def test_binary_mode(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=1, base_channels=4, depth=2
        )
        self.assertEqual(model.mode, "segmentation_binary")
        batch = next(iter(get_dataloader(dataset, batch_size=1)))
        output = model(**batch)
        self.assertEqual(tuple(output["logit"].shape), (1, 1, 16, 16, 16))

    def test_small_input_is_padded(self):
        samples = [
            {
                "patient_id": "p0",
                "image": torch.zeros(1, 1, 1),
                "mask": torch.zeros(1, 1, 1, dtype=torch.long),
            }
        ]
        dataset = create_sample_dataset(
            samples, {"image": "raw"}, {"mask": "raw"}
        )
        model = UNet2D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        output = model(image=torch.zeros(1, 1, 1, 1))
        self.assertEqual(tuple(output["logit"].shape), (1, 2, 1, 1))

    def test_sliding_window_inference(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        probability = sliding_window_inference(
            model, dataset[0]["image"], patch_size=(8, 8, 8), overlap=0.5
        )
        self.assertEqual(tuple(probability.shape), (2, 16, 16, 16))

    def test_evaluate_segmentation(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        scores = evaluate_segmentation(model, dataset, patch_size=None)
        self.assertIn("dice", scores)
        self.assertIn("iou", scores)

    def test_evaluate_segmentation_sliding_window(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        scores = evaluate_segmentation(
            model, dataset, patch_size=(8, 8, 8)
        )
        self.assertIn("dice", scores)
        self.assertIn("iou", scores)

    def test_evaluate_segmentation_with_class_mask(self):
        probability = torch.zeros(1, 3, 8, 8)
        probability[0, 0] = 0.5
        probability[0, 1, 2, 2] = 1.0
        probability[0, 1, 3, 3] = 1.0
        probability[0, 1, 4, 4] = 1.0
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[0, 2, 2] = 1

        class FixedModel(torch.nn.Module):
            feature_key = "image"
            label_key = "mask"
            num_classes = 3
            segmentation_ignore_index = 255
            mode = "segmentation_multiclass"
            device = torch.device("cpu")

            def __init__(self, y_prob):
                super().__init__()
                self.y_prob = y_prob

            def forward(self, **kwargs):
                return {"y_prob": self.y_prob}

        dataset = [
            {
                "image": torch.zeros(1, 8, 8),
                "mask": target,
                "class_mask": [True, True, False],
            }
        ]
        scores = evaluate_segmentation(
            FixedModel(probability), dataset, patch_size=None
        )
        # One true positive and two extra predicted pixels:
        # Dice = 2 * 1 / (1 + 3) = 0.5, IoU = 1 / 3.
        self.assertAlmostEqual(scores["dice"], 0.5, places=5)
        self.assertAlmostEqual(scores["iou"], 1 / 3, places=5)

    def test_constructor_validation_errors(self):
        dataset = _make_3d_dataset()
        base = {
            "in_channels": 1,
            "num_classes": 2,
            "base_channels": 4,
            "depth": 2,
        }
        cases = [
            ({"num_classes": -1}, "num_classes"),
            ({"depth": 0}, "depth"),
            ({"base_channels": 0}, "base_channels"),
            (
                {"depth": 2, "bilinear": False, "base_channels": 1},
                "base_channels",
            ),
            ({"norm": "layer"}, "norm"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, message
            ):
                UNet3D(dataset, **{**base, **overrides})
        with self.assertRaisesRegex(ValueError, "spatial dimension"):
            UNet(
                dataset,
                spatial_dim=4,
                in_channels=1,
                num_classes=2,
            )

    def test_infers_shapes_and_classes_from_raw_dataset(self):
        dataset = _make_3d_dataset()
        model = UNet(dataset, base_channels=4, depth=2)
        self.assertEqual(model.spatial_dim, 3)
        self.assertEqual(model.in_channels, 1)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.mode, "segmentation_multiclass")

    def test_infer_spatial_dim_rejects_unknown_rank(self):
        samples = [
            {
                "patient_id": "p0",
                "image": torch.zeros(16, 16),
                "mask": torch.zeros(16, 16, dtype=torch.long),
            }
        ]
        dataset = create_sample_dataset(
            samples, {"image": "raw"}, {"mask": "raw"}
        )
        with self.assertRaisesRegex(ValueError, "Unable to infer spatial_dim"):
            UNet(dataset, base_channels=4, depth=2)

    def test_infer_num_classes_ignores_padding_only_mask(self):
        samples = [
            {
                "patient_id": "p0",
                "image": torch.zeros(1, 8, 8, 8),
                "mask": torch.full(
                    (1, 8, 8, 8), 255, dtype=torch.long
                ),
            }
        ]
        dataset = create_sample_dataset(
            samples, {"image": "raw"}, {"mask": "raw"}
        )
        model = UNet3D(dataset, in_channels=1, base_channels=4, depth=2)
        self.assertEqual(model.num_classes, 1)
        self.assertEqual(model.mode, "segmentation_binary")

    def test_bilinear_and_norm_variants(self):
        dataset = _make_3d_dataset()
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        for bilinear, norm in (
            (True, "batch"),
            (True, "instance"),
            (False, "batch"),
            (False, "instance"),
        ):
            with self.subTest(bilinear=bilinear, norm=norm):
                model = UNet3D(
                    dataset,
                    in_channels=1,
                    num_classes=2,
                    base_channels=4,
                    depth=2,
                    bilinear=bilinear,
                    norm=norm,
                )
                output = model(**batch)
                self.assertTrue(torch.isfinite(output["loss"]))
                output["loss"].backward()
        model = UNet3D(
            dataset,
            in_channels=1,
            num_classes=2,
            base_channels=1,
            depth=2,
            bilinear=True,
        )
        self.assertEqual(model.base_channels, 1)

    def test_forward_rejects_wrong_rank(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        with self.assertRaisesRegex(ValueError, "Expected a 5-D image"):
            model(image=torch.zeros(1, 8, 8))

    def test_sliding_window_validation_errors(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        image = torch.zeros(1, 16, 16, 16)
        with self.assertRaisesRegex(ValueError, "overlap"):
            sliding_window_inference(
                model, image, patch_size=(8, 8, 8), overlap=1.0
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            sliding_window_inference(model, image, patch_size=(8, 8))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            sliding_window_inference(model, image, patch_size=(0, 8, 8))
        with self.assertRaisesRegex(
            ValueError, "Expected a 2D or 3D image"
        ):
            sliding_window_inference(
                model, torch.zeros(8, 8), patch_size=(8, 8, 8)
            )

    def test_nondivisible_input_is_cropped_not_resized(self):
        size = 15
        samples = [
            {
                "patient_id": "p0",
                "image": torch.randn(1, size, size, size),
                "mask": torch.zeros(
                    1, size, size, size, dtype=torch.long
                ),
            }
        ]
        dataset = create_sample_dataset(
            samples, {"image": "raw"}, {"mask": "raw"}
        )
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=3
        )
        image = dataset[0]["image"].unsqueeze(0)
        factor = 2 ** (model.depth - 1)
        padded_shape = tuple(
            max(factor, ((dim + factor - 1) // factor) * factor)
            for dim in image.shape[2:]
        )
        pads = []
        for dim, padded in zip(
            reversed(image.shape[2:]), reversed(padded_shape)
        ):
            pads.extend([0, padded - dim])
        padded_image = F.pad(image, pads, mode="replicate")
        with torch.no_grad():
            raw = model.network(padded_image)
            output = model(image=image)
        expected = raw[..., :size, :size, :size]
        torch.testing.assert_close(output["logit"], expected)
        self.assertEqual(
            tuple(output["logit"].shape), (1, 2, size, size, size)
        )

    @unittest.skipUnless(nib is not None, "nibabel is not installed")
    def test_evaluate_segmentation_with_lazy_nifti(self):
        temp_dir = tempfile.mkdtemp()
        try:
            image_path = Path(temp_dir) / "image.nii.gz"
            mask_path = Path(temp_dir) / "mask.nii.gz"
            image = np.zeros((16, 16, 16), dtype=np.float32)
            image[4:8, 4:8, 4:8] = 1.0
            mask = np.zeros((16, 16, 16), dtype=np.int16)
            mask[4:8, 4:8, 4:8] = 1
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
                model = UNet3D(
                    dataset,
                    in_channels=1,
                    num_classes=2,
                    base_channels=4,
                    depth=2,
                )
                scores = evaluate_segmentation(
                    model, dataset, patch_size=None
                )
                self.assertIn("dice", scores)
                self.assertIn("iou", scores)
            finally:
                dataset.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evaluate_segmentation_restores_training_mode(self):
        dataset = _make_3d_dataset()
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        model.train()
        evaluate_segmentation(model, dataset, patch_size=None)
        self.assertTrue(model.training)

    @unittest.skipUnless(nib is not None, "nibabel is not installed")
    def test_infers_in_channels_and_classes_from_lazy_nifti(self):
        temp_dir = tempfile.mkdtemp()
        try:
            image_path = Path(temp_dir) / "image.nii.gz"
            mask_path = Path(temp_dir) / "mask.nii.gz"
            image = np.zeros((8, 8, 8), dtype=np.float32)
            mask = np.zeros((8, 8, 8), dtype=np.int16)
            mask[2:4, 2:4, 2:4] = 1
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
                model = UNet3D(dataset, base_channels=4, depth=2)
                self.assertEqual(model.in_channels, 1)
                self.assertEqual(model.num_classes, 2)
            finally:
                dataset.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_trainer_evaluation_ignores_patch_padding(self):
        samples = []
        for index in range(2):
            image = torch.randn(1, 6, 6, 6)
            mask = torch.zeros(1, 6, 6, 6, dtype=torch.long)
            mask[:, 2:4, 2:4, 2:4] = 1
            samples.append(
                {"patient_id": f"p{index}", "image": image, "mask": mask}
            )
        dataset = create_sample_dataset(
            samples, {"image": "raw"}, {"mask": "raw"}
        )
        # 6^3 volumes are padded to the 8^3 patch with 255.
        patches = PatchDataset(
            dataset, PatchSampler((8, 8, 8)), mode="center"
        )
        loader = get_dataloader(patches, batch_size=2)
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        trainer = Trainer(
            model,
            device="cpu",
            enable_logging=False,
            metrics=["dice", "iou"],
        )
        scores = trainer.evaluate(loader)
        self.assertIn("dice", scores)
        self.assertIn("iou", scores)
        self.assertTrue(np.isfinite(scores["dice"]))
        self.assertTrue(np.isfinite(scores["iou"]))

    def test_trainer_one_epoch(self):
        dataset = _make_3d_dataset()
        sampler = PatchSampler((8, 8, 8), foreground_prob=1.0)
        patches = PatchDataset(dataset, sampler, mode="train")
        loader = get_dataloader(patches, batch_size=2, shuffle=True)
        model = UNet3D(
            dataset, in_channels=1, num_classes=2, base_channels=4, depth=2
        )
        trainer = Trainer(
            model,
            device="cpu",
            enable_logging=False,
            metrics=["dice", "iou"],
        )
        trainer.train(loader, epochs=1)
        scores = trainer.evaluate(loader)
        self.assertIn("dice", scores)
        self.assertIn("iou", scores)
        self.assertIn("loss", scores)


if __name__ == "__main__":
    unittest.main()
