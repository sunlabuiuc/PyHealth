import os
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GAN


class TestGAN(unittest.TestCase):
    """Test cases for the GAN model."""

    def setUp(self):
        """Bag-of-codes generative dataset (no labels) and a tiny model."""
        torch.manual_seed(0)
        self.samples = [
            {"patient_id": "patient-0", "visits": ["A05B", "A05C", "A11D", "C129"]},
            {"patient_id": "patient-1", "visits": ["A05B", "A04A", "B035"]},
            {"patient_id": "patient-2", "visits": ["C129", "A11D", "A05C", "A04A"]},
            {"patient_id": "patient-3", "visits": ["B035", "A05B", "C129"]},
        ]
        self.input_schema = {"visits": "multi_hot"}
        self.output_schema = {}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_gan",
        )

        self.model = GAN(dataset=self.dataset, hidden_dim=8)

    def test_model_initialization(self):
        """Code-set mode sizes both networks from the code vocabulary."""
        self.assertIsInstance(self.model, GAN)
        self.assertEqual(self.model.feature_key, "visits")
        self.assertEqual(self.model.label_keys, [])
        self.assertEqual(self.model.hidden_dim, 8)

        vocab_size = self.dataset.input_processors["visits"].size()
        self.assertEqual(self.model.input_dim, vocab_size)
        self.assertEqual(self.model.generator[-2].out_features, vocab_size)
        self.assertEqual(self.model.discriminator[0].in_features, vocab_size)

    def test_model_forward(self):
        """A forward pass scores a real batch against as many fakes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        for key in ("d_loss", "g_loss", "real_prob", "fake_prob", "fake"):
            self.assertIn(key, ret)
        self.assertEqual(ret["d_loss"].dim(), 0)
        self.assertEqual(ret["g_loss"].dim(), 0)
        self.assertEqual(ret["real_prob"].shape, (2, 1))
        self.assertEqual(ret["fake_prob"].shape, (2, 1))
        self.assertEqual(ret["fake"].shape, (2, self.model.input_dim))

        # The discriminator and generator both end in a sigmoid.
        for key in ("real_prob", "fake_prob", "fake"):
            self.assertTrue(((ret[key] >= 0) & (ret[key] <= 1)).all())

    def test_model_backward(self):
        """d_loss reaches only the discriminator; g_loss reaches the generator."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["d_loss"].backward()
        for param in self.model.generator.parameters():
            self.assertIsNone(param.grad)
        self.assertTrue(
            any(p.grad is not None for p in self.model.discriminator.parameters())
        )

        ret["g_loss"].backward()
        self.assertTrue(
            all(p.grad is not None for p in self.model.generator.parameters())
        )

    def test_train_model_runs(self):
        """train_model returns one mean loss per epoch for each network."""
        history = self.model.train_model(
            self.dataset, epochs=2, batch_size=2, device="cpu"
        )
        self.assertEqual(len(history["d_loss"]), 2)
        self.assertEqual(len(history["g_loss"]), 2)
        for loss in history["d_loss"] + history["g_loss"]:
            self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_train_model_updates_weights(self):
        """A training epoch changes both networks' weights."""
        gen_before = [p.clone() for p in self.model.generator.parameters()]
        disc_before = [p.clone() for p in self.model.discriminator.parameters()]

        self.model.train_model(self.dataset, epochs=1, batch_size=2, device="cpu")

        gen_after = list(self.model.generator.parameters())
        disc_after = list(self.model.discriminator.parameters())
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(gen_before, gen_after))
        )
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(disc_before, disc_after))
        )

    def test_generate(self):
        """generate() decodes synthetic patients into vocabulary codes."""
        vocab = {code for sample in self.samples for code in sample["visits"]}

        synthetic = self.model.generate(num_samples=4)
        self.assertEqual(len(synthetic), 4)
        for i, patient in enumerate(synthetic):
            self.assertEqual(patient["patient_id"], f"synthetic_{i}")
            # One aggregate visit per patient, matching MedGAN's format.
            self.assertEqual(len(patient["visits"]), 1)
            for code in patient["visits"][0]:
                self.assertIn(code, vocab)

    def test_generate_random_sampling(self):
        """random_sampling=True still produces well-formed patients."""
        synthetic = self.model.generate(num_samples=3, random_sampling=True)
        self.assertEqual(len(synthetic), 3)
        for patient in synthetic:
            self.assertIn("patient_id", patient)
            self.assertEqual(len(patient["visits"]), 1)

    def test_custom_hyperparameters(self):
        """hidden_dim sets the noise size and the hidden layer widths."""
        model = GAN(dataset=self.dataset, hidden_dim=32)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.sampling(5, "cpu").shape, (5, 32))
        self.assertEqual(model.generator[0].in_features, 32)
        self.assertEqual(model.discriminator[0].out_features, 32)

    def test_non_multi_hot_feature_raises(self):
        """Code-set mode rejects a feature that is not multi-hot."""
        bad = create_sample_dataset(
            samples=[
                {"patient_id": "p1", "visits": ["A", "B"]},
                {"patient_id": "p2", "visits": ["B"]},
            ],
            input_schema={"visits": "sequence"},
            output_schema={},
        )
        with self.assertRaises(TypeError):
            GAN(dataset=bad)

    def test_multiple_features_raise(self):
        """Code-set mode generates exactly one feature."""
        bad = create_sample_dataset(
            samples=[
                {"patient_id": "p1", "visits": ["A"], "procedures": ["X"]},
                {"patient_id": "p2", "visits": ["B"], "procedures": ["Y"]},
            ],
            input_schema={"visits": "multi_hot", "procedures": "multi_hot"},
            output_schema={},
        )
        with self.assertRaises(ValueError):
            GAN(dataset=bad)


class TestGANImageMode(unittest.TestCase):
    """An image feature puts the GAN in image mode."""

    def setUp(self):
        torch.manual_seed(0)
        image_path = os.path.join(
            os.path.dirname(__file__), "../../test-resources/core/cameraman.tif"
        )
        self.samples = [
            {"patient_id": f"patient-{i}", "image": image_path} for i in range(4)
        ]
        self.dataset = self._image_dataset({"image_size": 32, "mode": "L"})
        self.model = GAN(dataset=self.dataset, hidden_dim=8)

    def _image_dataset(self, processor_kwargs):
        return create_sample_dataset(
            samples=self.samples,
            input_schema={"image": ("image", processor_kwargs)},
            output_schema={},
            dataset_name="test_gan_image",
        )

    def test_model_initialization(self):
        """Image mode sizes the CNNs from the processor and the images."""
        self.assertEqual(self.model.gan_mode, "image")
        self.assertEqual(self.model.feature_key, "image")
        self.assertEqual(self.model.discriminator[0].conv1.in_channels, 1)

    def test_model_forward(self):
        """A forward pass scores real images against fakes of the same shape."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertEqual(ret["real_prob"].shape, (2, 1))
        self.assertEqual(ret["fake"].shape, (2, 1, 32, 32))

    def test_train_model_runs(self):
        """train_model takes an image SampleDataset."""
        history = self.model.train_model(
            self.dataset, epochs=1, batch_size=2, device="cpu"
        )
        self.assertEqual(len(history["d_loss"]), 1)
        self.assertTrue(torch.isfinite(torch.tensor(history["g_loss"][0])))

    def test_generate(self):
        """generate() returns a tensor of images in [0, 1]."""
        images = self.model.generate(num_samples=3)
        self.assertEqual(images.shape, (3, 1, 32, 32))
        self.assertTrue(((images >= 0) & (images <= 1)).all())
        with self.assertRaises(ValueError):
            self.model.generate(num_samples=1, random_sampling=True)

    def test_unsupported_image_size_raises(self):
        """Image sizes other than 32, 64 and 128 are rejected up front."""
        with self.assertRaises(ValueError):
            GAN(dataset=self._image_dataset({"image_size": 48}))

    def test_normalized_images_raise(self):
        """The generator outputs [0, 1], so normalized images are rejected."""
        dataset = self._image_dataset(
            {"image_size": 32, "normalize": True, "mean": [0.5], "std": [0.5]}
        )
        with self.assertRaises(ValueError):
            GAN(dataset=dataset)

    def test_pre_2_0_arguments_raise(self):
        """The pre-2.0 constructor explains the dataset form."""
        with self.assertRaisesRegex(TypeError, "deprecated"):
            GAN(1, 32)
        with self.assertRaisesRegex(TypeError, "deprecated"):
            GAN(input_channel=1, input_size=32)
        with self.assertRaisesRegex(TypeError, "deprecated"):
            GAN()


if __name__ == "__main__":
    unittest.main()
