"""Smoke tests for U-Net models, Trainer integration and sliding windows."""

import unittest

import torch

from pyhealth.datasets import PatchDataset, create_sample_dataset, get_dataloader
from pyhealth.models import (
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
        samples = []
        for index in range(3):
            image = torch.randn(1, 16, 16, 16)
            mask = torch.zeros(1, 16, 16, 16, dtype=torch.long)
            mask[:, 2:5, 2:5, 2:5] = 1
            mask[:, 8:11, 8:11, 8:11] = 2
            samples.append(
                {"patient_id": f"p{index}", "image": image, "mask": mask}
            )
        dataset = create_sample_dataset(
            samples, {"image": "raw"}, {"mask": "raw"}
        )
        model = UNet3D(
            dataset, in_channels=1, num_classes=3, base_channels=4, depth=2
        )

        class MaskedView:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, index):
                sample = dict(dataset[index])
                sample["class_mask"] = torch.tensor([True, True, False])
                return sample

        scores = evaluate_segmentation(
            model, MaskedView(), patch_size=None
        )
        self.assertIn("dice", scores)
        self.assertIn("iou", scores)

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
