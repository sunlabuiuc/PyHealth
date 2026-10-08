import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GPT2


class TestGPT2(unittest.TestCase):
    """Test cases for the GPT-2 baseline synthetic-EHR generator."""

    def setUp(self):
        """Set up a synthetic generative dataset (no labels) and a tiny model."""
        self.samples = [
            {"patient_id": "patient-0", "visits": [["A05B", "A05C"], ["A11D"], ["C129"]]},
            {"patient_id": "patient-1", "visits": [["A05B"], ["A04A", "B035"]]},
            {"patient_id": "patient-2", "visits": [["C129", "A11D"], ["A05C"], ["A04A"]]},
            {"patient_id": "patient-3", "visits": [["B035"], ["A05B", "C129"]]},
        ]

        # Generative task: one nested-sequence input feature, no output labels.
        self.input_schema = {"visits": "nested_sequence"}
        self.output_schema = {}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_gpt2",
        )

        # Small model; embed_dim must be divisible by n_heads.
        self.model = GPT2(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            max_len=64,
            batch_size=2,
            epochs=1,
        )

    def test_model_initialization(self):
        """Vocab/special-token ids are derived from the processor."""
        self.assertIsInstance(self.model, GPT2)
        self.assertEqual(self.model.feature_keys, ["visits"])
        self.assertEqual(self.model.label_keys, [])

        proc_vocab = self.dataset.input_processors["visits"].vocab_size()
        self.assertEqual(self.model.code_vocab_size, proc_vocab)
        self.assertEqual(self.model.bos_id, proc_vocab)
        self.assertEqual(self.model.eos_id, proc_vocab + 1)
        self.assertEqual(self.model.delim_id, proc_vocab + 2)
        self.assertEqual(self.model.gpt2.config.vocab_size, proc_vocab + 3)

    def test_forward_input_format(self):
        """The standard dataloader pads the visit dimension into a 3D tensor."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertIsInstance(batch["visits"], torch.Tensor)
        self.assertEqual(batch["visits"].dim(), 3)  # (B, max_visits, max_codes)

    def test_model_forward(self):
        """Forward returns a finite scalar loss and a probability tensor."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(ret["loss"]).all())
        # y_prob: (B, L, vocab_size)
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_prob"].shape[2], self.model.gpt2.config.vocab_size)

    def test_model_backward(self):
        """Backward populates gradients on model parameters."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward")

    def test_generate(self):
        """generate() returns the requested number of decoded synthetic patients."""
        synthetic = self.model.generate(num_samples=4)

        self.assertEqual(len(synthetic), 4)
        for i, patient in enumerate(synthetic):
            self.assertEqual(patient["patient_id"], f"synthetic_{i}")
            self.assertIsInstance(patient["visits"], list)
            for visit in patient["visits"]:
                self.assertIsInstance(visit, list)
                for code in visit:
                    self.assertIsInstance(code, str)
                    self.assertNotIn(code, ("<pad>", "<unk>"))

    def _build_model(self, save_dir):
        return GPT2(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            max_len=64,
            batch_size=2,
            epochs=1,
            save_dir=save_dir,
        )

    def test_train_and_generate_accept_device_arg(self):
        """train_model/generate accept an explicit device arg (CPU always works)."""
        with tempfile.TemporaryDirectory() as tmp:
            model = self._build_model(tmp)
            model.train_model(self.dataset, device="cpu")
            self.assertEqual(next(model.parameters()).device.type, "cpu")

            synthetic = model.generate(num_samples=2, device="cpu")
            self.assertEqual(len(synthetic), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_train_and_generate_on_cuda(self):
        """When CUDA is available, the device arg moves training/generation to GPU."""
        with tempfile.TemporaryDirectory() as tmp:
            model = self._build_model(tmp)
            model.train_model(self.dataset, device="cuda")
            self.assertTrue(next(model.parameters()).is_cuda)

            # forward should now run on CUDA without an explicit move.
            loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            with torch.no_grad():
                ret = model(**batch)
            self.assertTrue(ret["y_prob"].is_cuda)
            self.assertTrue(torch.isfinite(ret["loss"]).all())

            synthetic = model.generate(num_samples=2, device="cuda")
            self.assertEqual(len(synthetic), 2)


if __name__ == "__main__":
    unittest.main()
