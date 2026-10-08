import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import CorGAN


class TestCorGAN(unittest.TestCase):
    """Test cases for the CorGAN synthetic-EHR generator."""

    def setUp(self):
        """Bag-of-codes generative dataset (no labels) and a tiny model."""
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
            dataset_name="test_corgan",
        )

        # Small vocab -> linear autoencoder is auto-selected.
        self.model = CorGAN(
            dataset=self.dataset,
            latent_dim=8,
            hidden_dim=8,
            discriminator_hidden_dim=16,
            batch_size=2,
            ae_epochs=1,
            gan_epochs=1,
            n_iter_D=1,
        )

    def test_model_initialization(self):
        """Vocab is derived from MultiHotProcessor; generation is unconditional."""
        self.assertIsInstance(self.model, CorGAN)
        self.assertEqual(self.model.feature_keys, ["visits"])
        self.assertEqual(self.model.label_keys, [])

        proc_vocab = self.dataset.input_processors["visits"].size()
        self.assertEqual(self.model.input_dim, proc_vocab)

    def test_small_vocab_falls_back_to_linear(self):
        """For tiny vocabs the 6-layer CNN can't compress -- expect linear AE."""
        self.assertEqual(self.model.autoencoder_type, "linear")

    def test_components_present(self):
        """Autoencoder, generator, and critic are all registered submodules."""
        self.assertTrue(hasattr(self.model, "autoencoder"))
        self.assertTrue(hasattr(self.model, "generator"))
        self.assertTrue(hasattr(self.model, "critic"))

        # Critic is a Wasserstein critic -> unbounded scalar (no sigmoid).
        x = torch.zeros(2, self.model.input_dim)
        with torch.no_grad():
            c_out = self.model.critic(x)
        self.assertEqual(c_out.shape, (2, 1))

    def test_forward_raises(self):
        """CorGAN's BaseModel forward intentionally errors out."""
        with self.assertRaises(NotImplementedError):
            self.model.forward()

    def test_train_model_runs(self):
        """train_model completes a tiny two-phase loop on CPU and returns history."""
        with tempfile.TemporaryDirectory() as tmp:
            model = CorGAN(
                dataset=self.dataset,
                latent_dim=8,
                hidden_dim=8,
                discriminator_hidden_dim=16,
                batch_size=2,
                ae_epochs=1,
                gan_epochs=1,
                n_iter_D=1,
                save_dir=tmp,
            )
            history = model.train_model(self.dataset, device="cpu")

            self.assertIn("autoencoder_loss", history)
            self.assertIn("critic_loss", history)
            self.assertIn("generator_loss", history)
            self.assertEqual(len(history["autoencoder_loss"]), 1)
            self.assertEqual(len(history["critic_loss"]), 1)
            self.assertEqual(len(history["generator_loss"]), 1)
            self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_generate(self):
        """generate() returns the requested number of decoded synthetic patients.

        CorGAN is a bag-of-codes model, so each patient gets a single visit
        containing the aggregate set of codes; the outer ``visits`` list
        wraps that single visit to match HALO's nested format.
        """
        synthetic = self.model.generate(num_samples=4, device="cpu")
        self.assertEqual(len(synthetic), 4)
        for i, patient in enumerate(synthetic):
            self.assertEqual(patient["patient_id"], f"synthetic_{i}")
            self.assertIsInstance(patient["visits"], list)
            # Exactly one aggregate visit per patient.
            self.assertEqual(len(patient["visits"]), 1)
            visit = patient["visits"][0]
            self.assertIsInstance(visit, list)
            for code in visit:
                self.assertIsInstance(code, str)
                self.assertNotIn(code, ("<pad>", "<unk>"))

    def test_generate_random_sampling(self):
        """random_sampling=True still produces well-formed patients."""
        synthetic = self.model.generate(
            num_samples=3, random_sampling=True, device="cpu"
        )
        self.assertEqual(len(synthetic), 3)
        for patient in synthetic:
            self.assertIn("patient_id", patient)
            self.assertIn("visits", patient)

    def test_save_and_load_roundtrip(self):
        """save_model + load_model preserves weights and vocabulary."""
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/corgan.pt"
            self.model.save_model(path)

            other = CorGAN(
                dataset=self.dataset,
                latent_dim=8,
                hidden_dim=8,
                discriminator_hidden_dim=16,
                batch_size=2,
                ae_epochs=1,
                gan_epochs=1,
                n_iter_D=1,
                save_dir=tmp,
            )
            other.load_model(path)

            for p1, p2 in zip(
                self.model.generator.parameters(),
                other.generator.parameters(),
            ):
                self.assertTrue(torch.allclose(p1, p2))
            self.assertEqual(other._idx_to_code, self.model._idx_to_code)

    def test_missing_visits_processor_raises(self):
        """A dataset without a 'visits' feature should be rejected."""
        bad = create_sample_dataset(
            samples=[
                {"patient_id": "p1", "codes": ["A", "B"]},
                {"patient_id": "p2", "codes": ["B"]},
            ],
            input_schema={"codes": "multi_hot"},
            output_schema={},
        )
        with self.assertRaises(ValueError):
            CorGAN(bad, latent_dim=8, hidden_dim=8)

    def test_unknown_autoencoder_type_raises(self):
        """An unknown autoencoder_type is rejected."""
        with self.assertRaises(ValueError):
            CorGAN(self.dataset, autoencoder_type="rnn")

    def test_cnn_path_with_large_vocab(self):
        """A vocabulary big enough survives the 6-layer CNN chain end-to-end."""
        all_codes = [f"C{i:04d}" for i in range(1200)]
        samples = [
            {
                "patient_id": f"p{i}",
                "visits": all_codes[i * 200 : (i + 1) * 200]
                + all_codes[1100:1200],
            }
            for i in range(6)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"visits": "multi_hot"},
            output_schema={},
            dataset_name="cnn_corgan",
        )
        with tempfile.TemporaryDirectory() as tmp:
            model = CorGAN(
                dataset=dataset,
                batch_size=2,
                ae_epochs=1,
                gan_epochs=1,
                n_iter_D=1,
                save_dir=tmp,
            )
            self.assertEqual(model.autoencoder_type, "cnn")
            # CNN bottleneck is fixed at 128 -- generator should track it.
            self.assertEqual(model.latent_dim, 128)
            self.assertEqual(model.hidden_dim, 128)
            model.train_model(dataset, device="cpu")
            out = model.generate(num_samples=2, device="cpu")
            self.assertEqual(len(out), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_train_and_generate_on_cuda(self):
        """When CUDA is available, the device arg moves training to GPU."""
        with tempfile.TemporaryDirectory() as tmp:
            model = CorGAN(
                dataset=self.dataset,
                latent_dim=8,
                hidden_dim=8,
                discriminator_hidden_dim=16,
                batch_size=2,
                ae_epochs=1,
                gan_epochs=1,
                n_iter_D=1,
                save_dir=tmp,
            )
            model.train_model(self.dataset, device="cuda")
            self.assertTrue(next(model.parameters()).is_cuda)

            synthetic = model.generate(num_samples=2, device="cuda")
            self.assertEqual(len(synthetic), 2)


if __name__ == "__main__":
    unittest.main()
