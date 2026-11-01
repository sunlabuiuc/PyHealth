"""Tests for PyHealth 2.0 image and sequence VAEs."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pyhealth.datasets import (
    COVID19CXRDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import VAE
from pyhealth.tasks import COVID19CXRImageGeneration
from pyhealth.trainer import Trainer


def _write_images(root: Path, n: int, size: int, seed: int = 0) -> list[str]:
    """Write small synthetic grayscale images and return their paths."""
    rng = np.random.default_rng(seed)
    paths = []
    for index in range(n):
        pixels = rng.integers(0, 255, (size, size), dtype=np.uint8)
        path = root / f"image-{seed}-{index}.png"
        Image.fromarray(pixels, mode="L").save(path)
        paths.append(str(path))
    return paths


class TestVAE(unittest.TestCase):
    """Exercise image/sequence VAE reconstruction and Trainer integration."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        samples = []
        for index in range(4):
            image_path = os.path.join(cls.temp_dir, f"image-{index}.png")
            pixels = np.full((32, 32), index * 32, dtype=np.uint8)
            Image.fromarray(pixels, mode="L").save(image_path)
            samples.append(
                {
                    "patient_id": f"patient-{index}",
                    "image": image_path,
                    "label": "normal" if index % 2 == 0 else "disease",
                }
            )
        cls.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
            output_schema={"label": "multiclass"},
            dataset_name="test_vae",
        )
        sequence_samples = [
            {"patient_id": "p0", "conditions": ["A", "B"], "label": 0},
            {"patient_id": "p1", "conditions": ["B", "C", "A"], "label": 1},
            {"patient_id": "p2", "conditions": ["C"], "label": 0},
        ]
        cls.sequence_dataset = create_sample_dataset(
            samples=sequence_samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_vae_sequence",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_conditional_reconstruction_and_generation(self):
        model = VAE(
            self.dataset, hidden_dim=8, channel_dim=4, condition_key="label"
        )
        batch = next(iter(get_dataloader(self.dataset, batch_size=2)))

        result = model(**batch)
        self.assertEqual(result["reconstruction"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()

        generated = model.generate(conditions=batch["label"])
        self.assertEqual(generated.shape, (2, 1, 32, 32))

    def test_trainer_evaluates_reconstruction_loss(self):
        model = VAE(self.dataset, hidden_dim=8, channel_dim=4)
        trainer = Trainer(model=model, device="cpu")
        scores = trainer.evaluate(get_dataloader(self.dataset, batch_size=2))
        self.assertIn("loss", scores)
        self.assertTrue(np.isfinite(scores["loss"]))

    def test_sequence_reconstruction_generation_and_trainer(self):
        model = VAE(
            self.sequence_dataset,
            feature_key="conditions",
            hidden_dim=8,
            sequence_hidden_dim=8,
        )
        batch = next(iter(get_dataloader(self.sequence_dataset, batch_size=2)))

        result = model(**batch)
        self.assertEqual(
            result["reconstruction_logits"].shape[:2], batch["conditions"].shape
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()

        generated = model.generate(num_samples=2, sequence_length=4)
        self.assertEqual(generated.shape, (2, 4))
        self.assertTrue(bool((generated > 0).all()))
        self.assertTrue(bool((generated < model.vocab_size).all()))
        self.assertFalse(bool((generated == model.pad_token).any()))
        if model.unk_token is not None:
            self.assertFalse(bool((generated == model.unk_token).any()))

        trainer = Trainer(model=model, device="cpu")
        scores = trainer.evaluate(get_dataloader(self.sequence_dataset, batch_size=2))
        self.assertTrue(np.isfinite(scores["loss"]))

    def test_unconditional_image_reconstruction_and_generation(self):
        samples = [
            {
                "patient_id": f"unconditional-{index}",
                "image": os.path.join(self.temp_dir, f"image-{index}.png"),
                "label": "normal" if index % 2 == 0 else "disease",
                "extra_label": index % 2,
            }
            for index in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
            output_schema={"label": "multiclass", "extra_label": "binary"},
            dataset_name="test_vae_unconditional",
        )
        model = VAE(dataset, hidden_dim=8, channel_dim=4)
        self.assertIsNone(model.condition_key)

        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        result = model(**batch)
        self.assertTrue(torch.isfinite(result["loss"]))
        generated = model.generate(num_samples=3)
        self.assertEqual(generated.shape, (3, 1, 32, 32))

    def test_image_reconstruction_keeps_trainer_compatibility(self):
        model = VAE(self.dataset, hidden_dim=8, channel_dim=4)
        batch = next(iter(get_dataloader(self.dataset, batch_size=2)))
        model.eval()
        with torch.no_grad():
            first = model(**batch)
            second = model(**batch)
        self.assertEqual(model.mode, "regression")
        self.assertEqual(model.image_shape, (1, 32, 32))
        self.assertEqual(first["y_prob"].shape, first["y_true"].shape)
        torch.testing.assert_close(first["y_prob"], second["y_prob"])

        trainer = Trainer(
            model=model,
            device="cpu",
            metrics=["mse", "mae"],
            enable_logging=False,
        )
        scores = trainer.evaluate(get_dataloader(self.dataset, batch_size=2))
        self.assertTrue(np.isfinite(scores["mse"]))
        self.assertTrue(np.isfinite(scores["mae"]))

    def test_normalized_images_use_a_finite_reconstruction_loss(self):
        samples = [
            {
                "patient_id": f"normalized-{index}",
                "image": os.path.join(self.temp_dir, f"image-{index}.png"),
            }
            for index in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "image": (
                    "image",
                    {
                        "image_size": 32,
                        "mode": "L",
                        "normalize": True,
                        "mean": [0.5],
                        "std": [0.25],
                    },
                )
            },
            output_schema={},
            dataset_name="test_vae_normalized",
        )
        model = VAE(dataset, hidden_dim=8, channel_dim=4)
        result = model(**next(iter(get_dataloader(dataset, batch_size=2))))
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_image_processor_requires_explicit_mode(self):
        samples = [
            {
                "patient_id": f"mode-{index}",
                "image": os.path.join(self.temp_dir, f"image-{index}.png"),
            }
            for index in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"image": ("image", {"image_size": 32})},
            output_schema={},
            dataset_name="test_vae_missing_image_mode",
        )
        with self.assertRaisesRegex(
            ValueError, 'mode must be set to "L", "RGB" or "RGBA"'
        ):
            VAE(dataset)

    def test_legacy_hidden_dim_and_generate_device_are_supported(self):
        model = VAE(self.dataset, 8, channel_dim=4)
        self.assertEqual(model.latent_dim, 8)
        generated = model.generate(2, "cpu")
        self.assertEqual(generated.shape, (2, 1, 32, 32))
        self.assertFalse(generated.requires_grad)

    def test_conditioning_is_opt_in_and_rejects_non_categorical_labels(self):
        samples = [
            {
                "patient_id": f"regression-{index}",
                "image": os.path.join(self.temp_dir, f"image-{index}.png"),
                "score": index / 4,
            }
            for index in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
            output_schema={"score": "regression"},
            dataset_name="test_vae_regression_label",
        )
        model = VAE(dataset, hidden_dim=8, channel_dim=4)
        self.assertIsNone(model.condition_key)
        self.assertEqual(model.generate(num_samples=2).shape, (2, 1, 32, 32))
        with self.assertRaisesRegex(ValueError, "binary or multiclass"):
            VAE(dataset, hidden_dim=8, channel_dim=4, condition_key="score")

    def test_unsupported_image_mode_raises_value_error(self):
        dataset = create_sample_dataset(
            samples=[
                {
                    "patient_id": f"mode-{index}",
                    "image": os.path.join(self.temp_dir, f"image-{index}.png"),
                }
                for index in range(4)
            ],
            input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
            output_schema={},
            dataset_name="test_vae_unsupported_image_mode",
        )
        dataset.input_processors["image"].mode = "P"
        with self.assertRaisesRegex(ValueError, "got 'P'"):
            VAE(dataset)

    def test_trainer_train_reduces_reconstruction_loss(self):
        torch.manual_seed(7)
        model = VAE(self.dataset, hidden_dim=8, channel_dim=2)
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        trainer = Trainer(model=model, device="cpu", enable_logging=False)
        before = trainer.evaluate(loader)["loss"]
        trainer.train(
            train_dataloader=loader,
            epochs=1,
            optimizer_params={"lr": 1e-2},
        )
        after = trainer.evaluate(loader)["loss"]
        self.assertLess(after, before)


class TestCOVID19CXRImageGeneration(unittest.TestCase):
    """Test the unlabeled COVID CXR task and its compatibility with the VAE."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        rows = []
        for label in ("COVID", "Normal"):
            images = cls.root / label / "images"
            images.mkdir(parents=True)
            for path in _write_images(images, n=2, size=64, seed=len(rows)):
                rows.append({"path": path, "url": "", "label": label})
        pd.DataFrame(rows).to_csv(
            cls.root / "covid19_cxr-metadata-pyhealth.csv", index=False
        )
        cls.cache = tempfile.mkdtemp()
        dataset = COVID19CXRDataset(root=str(cls.root), cache_dir=cls.cache)
        cls.samples = dataset.set_task(COVID19CXRImageGeneration())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.cache, ignore_errors=True)

    def test_one_unlabeled_grayscale_sample_per_image(self):
        self.assertEqual(len(self.samples), 4)
        self.assertEqual(self.samples.output_schema, {})
        self.assertEqual(tuple(self.samples[0]["image"].shape), (1, 128, 128))

    def test_vae_reconstructs_the_generation_task_samples(self):
        model = VAE(dataset=self.samples, hidden_dim=8, channel_dim=4)
        output = model(**next(iter(get_dataloader(self.samples, batch_size=2))))
        self.assertEqual(output["y_prob"].shape, (2, 1, 128, 128))
        output["loss"].backward()


if __name__ == "__main__":
    unittest.main()
