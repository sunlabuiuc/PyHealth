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


if __name__ == "__main__":
    unittest.main()
