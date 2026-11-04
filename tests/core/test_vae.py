import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

from pyhealth.datasets import COVID19CXRDataset, create_sample_dataset, get_dataloader
from pyhealth.models import VAE
from pyhealth.tasks import COVID19CXRImageGeneration
from pyhealth.trainer import Trainer


def _write_images(root: Path, n: int, size: int, mode: str, seed: int = 0):
    """Writes n random images and returns their paths."""
    rng = np.random.default_rng(seed)
    shape = (size, size) if mode == "L" else (size, size, 3)
    paths = []
    for i in range(n):
        path = root / f"{mode}_{size}_{seed}_{i}.png"
        Image.fromarray(rng.integers(0, 255, shape, dtype=np.uint8), mode).save(path)
        paths.append(str(path))
    return paths


def _image_dataset(paths, image_size: int, mode: str = "L", **processor_kwargs):
    samples = [{"patient_id": f"p{i}", "image": p} for i, p in enumerate(paths)]
    schema = {"image_size": image_size, "mode": mode, **processor_kwargs}
    return create_sample_dataset(
        samples=samples,
        input_schema={"image": ("image", schema)},
        output_schema={},
        dataset_name="test_vae",
    )


class TestVAE(unittest.TestCase):
    """VAE on small synthetic grayscale and RGB images."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.gray_paths = _write_images(cls.root, n=4, size=40, mode="L")
        cls.dataset = _image_dataset(cls.gray_paths, image_size=32)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        torch.manual_seed(42)
        self.model = VAE(dataset=self.dataset, hidden_dim=8)
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=2)))

    def test_shape_and_mode_come_from_the_dataset(self):
        self.assertEqual(self.model.feature_key, "image")
        self.assertEqual(self.model.image_shape, (1, 32, 32))
        self.assertEqual(self.model.label_keys, [])
        self.assertEqual(self.model.mode, "regression")

    def test_forward_reconstructs_the_input_shape(self):
        output = self.model(**self.batch)
        self.assertEqual(output["y_prob"].shape, (2, 1, 32, 32))
        self.assertEqual(output["y_prob"].shape, output["y_true"].shape)
        self.assertEqual(output["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertGreaterEqual(output["y_prob"].min().item(), 0.0)
        self.assertLessEqual(output["y_prob"].max().item(), 1.0)

    def test_backward_reaches_encoder_and_decoder(self):
        self.model(**self.batch)["loss"].backward()
        first = self.model.encoder[0].conv1.weight.grad
        last = self.model.decoder[-1].weight.grad
        self.assertIsNotNone(first)
        self.assertIsNotNone(last)
        self.assertGreater(first.abs().sum().item(), 0.0)
        self.assertGreater(last.abs().sum().item(), 0.0)

    def test_eval_reconstruction_is_deterministic(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(**self.batch)["y_prob"]
            second = self.model(**self.batch)["y_prob"]
        torch.testing.assert_close(first, second)

    def test_handles_any_size_and_channel_count(self):
        rgb = _image_dataset(
            _write_images(self.root, n=2, size=60, mode="RGB", seed=1),
            image_size=50,
            mode="RGB",
        )
        model = VAE(dataset=rgb, hidden_dim=8)
        output = model(**next(iter(get_dataloader(rgb, batch_size=2))))
        self.assertEqual(model.image_shape, (3, 50, 50))
        self.assertEqual(output["y_prob"].shape, (2, 3, 50, 50))

    def test_normalized_images_use_an_unbounded_output(self):
        normalized = _image_dataset(
            self.gray_paths, image_size=32, normalize=True, mean=[0.5], std=[0.25]
        )
        model = VAE(dataset=normalized, hidden_dim=8)
        self.assertIsInstance(model.output_activation, nn.Identity)
        output = model(**next(iter(get_dataloader(normalized, batch_size=2))))
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_labels_in_the_dataset_are_ignored(self):
        samples = [
            {"patient_id": f"p{i}", "image": p, "disease": i % 2}
            for i, p in enumerate(self.gray_paths)
        ]
        labeled = create_sample_dataset(
            samples=samples,
            input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
            output_schema={"disease": "multiclass"},
            dataset_name="test_vae_labeled",
        )
        model = VAE(dataset=labeled, hidden_dim=8)
        self.assertEqual(model.mode, "regression")
        output = model(**next(iter(get_dataloader(labeled, batch_size=2))))
        self.assertEqual(output["y_prob"].shape, (2, 1, 32, 32))

    def test_rejects_a_feature_that_is_not_an_image(self):
        sequences = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "codes": ["a", "b"]},
                {"patient_id": "p1", "codes": ["b", "c"]},
            ],
            input_schema={"codes": "sequence"},
            output_schema={},
            dataset_name="test_vae_sequence",
        )
        with self.assertRaisesRegex(TypeError, "only supports image inputs"):
            VAE(dataset=sequences)

    def test_rejects_more_than_one_feature(self):
        samples = [
            {"patient_id": f"p{i}", "image": p, "codes": ["a"]}
            for i, p in enumerate(self.gray_paths)
        ]
        mixed = create_sample_dataset(
            samples=samples,
            input_schema={"image": ("image", {"image_size": 32}), "codes": "sequence"},
            output_schema={},
            dataset_name="test_vae_mixed",
        )
        with self.assertRaisesRegex(ValueError, "exactly one input feature"):
            VAE(dataset=mixed)

    def test_rejects_images_smaller_than_eight_pixels(self):
        tiny = _image_dataset(self.gray_paths, image_size=4)
        with self.assertRaisesRegex(ValueError, "at least 8 pixels"):
            VAE(dataset=tiny)

    def test_generate_returns_images_and_keeps_the_training_mode(self):
        self.model.train()
        images = self.model.generate(num_samples=3)
        self.assertEqual(images.shape, (3, 1, 32, 32))
        self.assertGreaterEqual(images.min().item(), 0.0)
        self.assertLessEqual(images.max().item(), 1.0)
        self.assertTrue(self.model.training)

    def test_state_dict_round_trip(self):
        self.model.eval()
        restored = VAE(dataset=self.dataset, hidden_dim=8)
        restored.load_state_dict(self.model.state_dict())
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                self.model(**self.batch)["y_prob"], restored(**self.batch)["y_prob"]
            )

    def test_trains_and_evaluates_with_the_trainer(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        trainer = Trainer(
            model=self.model, metrics=["mse", "mae"], enable_logging=False
        )
        trainer.train(train_dataloader=loader, val_dataloader=loader, epochs=1)
        scores = trainer.evaluate(loader)
        self.assertTrue(np.isfinite(scores["mse"]))
        self.assertTrue(np.isfinite(scores["mae"]))

    def test_learns_to_reconstruct_structured_images(self):
        """Training on discs must at least halve the reconstruction error.

        Over 20 seeds this setup ends at 0.25 to 0.30 of the untrained error,
        so the 0.5 bound leaves room for other platforms and BLAS builds.
        """
        rng = np.random.default_rng(3)
        paths = []
        for i in range(16):
            image = Image.new("L", (16, 16), 10)
            r = int(rng.integers(3, 5))
            cx, cy = rng.integers(r, 16 - r, 2)
            ImageDraw.Draw(image).ellipse([cx - r, cy - r, cx + r, cy + r], fill=230)
            path = self.root / f"disc_{i}.png"
            image.save(path)
            paths.append(str(path))
        discs = _image_dataset(paths, image_size=16)
        model = VAE(dataset=discs, hidden_dim=8)
        loader = get_dataloader(discs, batch_size=8, shuffle=True)

        def reconstruction_error():
            model.eval()
            with torch.no_grad():
                errors = [
                    ((out["y_prob"] - out["y_true"]) ** 2).mean().item()
                    for out in (model(**b) for b in get_dataloader(discs, batch_size=8))
                ]
            model.train()
            return float(np.mean(errors))

        before = reconstruction_error()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(60):
            for batch in loader:
                optimizer.zero_grad()
                model(**batch)["loss"].backward()
                optimizer.step()
        self.assertLess(reconstruction_error(), 0.5 * before)


class TestCOVID19CXRImageGeneration(unittest.TestCase):
    """The generation task on a tiny synthetic copy of the dataset layout."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        rows = []
        for label, folder in [("COVID", "COVID"), ("Normal", "Normal")]:
            images = cls.root / folder / "images"
            images.mkdir(parents=True)
            for path in _write_images(images, n=2, size=64, mode="L", seed=len(rows)):
                rows.append({"path": path, "url": "", "label": label})
        # With the merged metadata present, the dataset skips the Excel files.
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

    def test_vae_trains_on_the_task(self):
        model = VAE(dataset=self.samples, hidden_dim=8)
        output = model(**next(iter(get_dataloader(self.samples, batch_size=2))))
        self.assertEqual(output["y_prob"].shape, (2, 1, 128, 128))
        output["loss"].backward()


if __name__ == "__main__":
    unittest.main()
