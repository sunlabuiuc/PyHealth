"""Tests for the PyHealth 2.0 conditional image VAE."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import VAE
from pyhealth.trainer import Trainer


class TestVAE(unittest.TestCase):
    """Exercise VAE reconstruction, conditioning, and Trainer integration."""

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
        model = VAE(self.dataset, latent_dim=8, hidden_dim=4)
        batch = next(iter(get_dataloader(self.dataset, batch_size=2)))

        result = model(**batch)
        self.assertEqual(result["reconstruction"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()

        generated = model.generate(conditions=batch["label"])
        self.assertEqual(generated.shape, (2, 1, 32, 32))

    def test_trainer_evaluates_reconstruction_loss(self):
        model = VAE(self.dataset, latent_dim=8, hidden_dim=4)
        trainer = Trainer(model=model, device="cpu")
        scores = trainer.evaluate(get_dataloader(self.dataset, batch_size=2))
        self.assertIn("loss", scores)
        self.assertTrue(np.isfinite(scores["loss"]))

    def test_sequence_reconstruction_generation_and_trainer(self):
        model = VAE(
            self.sequence_dataset,
            feature_key="conditions",
            latent_dim=8,
            hidden_dim=8,
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
        model = VAE(dataset, latent_dim=8, hidden_dim=4)
        self.assertIsNone(model.condition_key)

        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        result = model(**batch)
        self.assertTrue(torch.isfinite(result["loss"]))
        generated = model.generate(num_samples=3)
        self.assertEqual(generated.shape, (3, 1, 32, 32))

    def test_image_reconstruction_keeps_trainer_compatibility(self):
        model = VAE(self.dataset, latent_dim=8, hidden_dim=4)
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
        model = VAE(dataset, latent_dim=8, hidden_dim=4)
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


if __name__ == "__main__":
    unittest.main()
