import unittest

import torch

from pyhealth.models import GAN


class TestGAN32(unittest.TestCase):
    """Test GAN model with 32x32 input size."""

    def setUp(self):
        """Set up GAN model with 32x32 single-channel input."""
        torch.manual_seed(42)
        self.model = GAN(input_channel=1, input_size=32, hidden_dim=16)

    def test_initialization(self):
        """Test that the GAN model initializes correctly."""
        self.assertIsInstance(self.model, GAN)
        self.assertEqual(self.model.hidden_dim, 16)
        self.assertIsNotNone(self.model.discriminator)
        self.assertIsNotNone(self.model.generator)

    def test_discriminator_output_shape(self):
        """Test that the discriminator produces correct output shape."""
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = self.model.discriminate(x)
        self.assertEqual(out.shape, (2, 1))

    def test_discriminator_output_range(self):
        """Test that discriminator outputs are in [0, 1] range."""
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = self.model.discriminate(x)
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_generate_fake_shape(self):
        """Test that the generator produces correct output shape."""
        with torch.no_grad():
            fake = self.model.generate_fake(n_samples=3, device="cpu")
        self.assertEqual(fake.shape[0], 3)
        self.assertEqual(fake.shape[1], 1)

    def test_generate_fake_pixel_range(self):
        """Test that generated pixels are in [0, 1] range."""
        with torch.no_grad():
            fake = self.model.generate_fake(n_samples=2, device="cpu")
        self.assertTrue(torch.all(fake >= 0))
        self.assertTrue(torch.all(fake <= 1))

    def test_sampling_shape(self):
        """Test that latent sampling produces correct shape."""
        eps = self.model.sampling(n_samples=4, device="cpu")
        self.assertEqual(eps.shape, (4, 16, 1, 1))

    def test_discriminator_backward(self):
        """Test that gradients flow through the discriminator."""
        x = torch.randn(2, 1, 32, 32)
        out = self.model.discriminate(x)
        loss = out.mean()
        loss.backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.discriminator.parameters()
        )
        self.assertTrue(has_gradient)

    def test_generator_backward(self):
        """Test that gradients flow through the generator."""
        fake = self.model.generate_fake(n_samples=2, device="cpu")
        loss = fake.mean()
        loss.backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.generator.parameters()
        )
        self.assertTrue(has_gradient)


class TestGAN64(unittest.TestCase):
    """Test GAN model with 64x64 input size."""

    def setUp(self):
        """Set up GAN model with 64x64 single-channel input."""
        torch.manual_seed(42)
        self.model = GAN(input_channel=1, input_size=64, hidden_dim=16)

    def test_discriminator_output_shape(self):
        """Test that the discriminator produces correct output shape."""
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out = self.model.discriminate(x)
        self.assertEqual(out.shape, (2, 1))

    def test_generate_fake_shape(self):
        """Test that the generator produces correct output shape."""
        with torch.no_grad():
            fake = self.model.generate_fake(n_samples=2, device="cpu")
        self.assertEqual(fake.shape[0], 2)
        self.assertEqual(fake.shape[1], 1)

    def test_end_to_end(self):
        """Test the full generate-then-discriminate pipeline."""
        with torch.no_grad():
            fake = self.model.generate_fake(n_samples=2, device="cpu")
            score = self.model.discriminate(fake)
        self.assertEqual(score.shape, (2, 1))


class TestGAN128(unittest.TestCase):
    """Test GAN model with 128x128 input size."""

    def setUp(self):
        """Set up GAN model with 128x128 three-channel input."""
        torch.manual_seed(42)
        self.model = GAN(input_channel=3, input_size=128, hidden_dim=16)

    def test_discriminator_output_shape(self):
        """Test that the discriminator produces correct output shape."""
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = self.model.discriminate(x)
        self.assertEqual(out.shape, (2, 1))

    def test_generate_fake_shape(self):
        """Test that the generator produces correct output shape."""
        with torch.no_grad():
            fake = self.model.generate_fake(n_samples=2, device="cpu")
        self.assertEqual(fake.shape[0], 2)
        self.assertEqual(fake.shape[1], 3)

    def test_multichannel_end_to_end(self):
        """Test the full pipeline with multi-channel images."""
        with torch.no_grad():
            fake = self.model.generate_fake(n_samples=2, device="cpu")
            score = self.model.discriminate(fake)
        self.assertEqual(score.shape, (2, 1))
        self.assertTrue(torch.all(score >= 0))
        self.assertTrue(torch.all(score <= 1))


if __name__ == "__main__":
    unittest.main()
