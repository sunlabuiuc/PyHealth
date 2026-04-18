import shutil
import tempfile
import unittest
import torch
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import Wav2Sleep


class TestWav2Sleep(unittest.TestCase):
    def setUp(self):
        """
        Set up a tiny synthetic dataset for fast unit testing.
        """
        self.test_dir = tempfile.mkdtemp()
        # 1 patient, 5 sleep epochs (30s each)
        # ECG @ 100Hz = 3000 points, Resp @ 25Hz = 750 points
        self.samples = [{
            "patient_id": "p1",
            "ecg": torch.randn(5, 3000).tolist(),
            "resp": torch.randn(5, 750).tolist(),
            "label": [0, 1, 2, 1, 0],
        }]

        # Use input_schema to pass labels as tensors directly,
        # bypassing the problematic LabelProcessor for sequences.
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={
                "ecg": "tensor",
                "resp": "tensor",
                "label": "tensor"
            },
            output_schema={},
        )

        self.modalities = {"ecg": 3000, "resp": 750}
        self.embedding_dim = 64  # Reduced dim for faster CPU testing

    def tearDown(self):
        """Clean up any temporary resources."""
        shutil.rmtree(self.test_dir)
        del self.samples
        del self.dataset

    def test_instantiation(self):
        """Tests model initialization and configuration."""
        model = Wav2Sleep(
            dataset=self.dataset,
            modalities=self.modalities,
            label_key="label",
            mode="multiclass",
            num_classes=5,
            embedding_dim=self.embedding_dim
        )
        self.assertEqual(model.mode, "multiclass")
        self.assertEqual(model.num_classes, 5)
        # Check if encoders are properly created for each modality
        self.assertTrue(hasattr(model, "encoders"))
        self.assertEqual(len(model.encoders), 2)

    def test_forward_and_output_shapes(self):
        """Tests the forward pass and validates output tensor dimensions."""
        model = Wav2Sleep(
            dataset=self.dataset,
            modalities=self.modalities,
            label_key="label",
            mode="multiclass",
            num_classes=5,
            embedding_dim=self.embedding_dim
        )

        from pyhealth.datasets import get_dataloader
        loader = get_dataloader(self.dataset, batch_size=1)
        batch = next(iter(loader))

        model.eval()
        with torch.no_grad():
            output = model(**batch)

        # Expected shape: [Batch=1, Epochs=5, Classes=5]
        self.assertEqual(output["y_prob"].shape, (1, 5, 5))
        self.assertIn("loss", output)
        self.assertIn("logit", output)

    def test_gradient_computation(self):
        """
        Verifies that gradients flow through the entire architecture.
        """
        model = Wav2Sleep(
            dataset=self.dataset,
            modalities=self.modalities,
            label_key="label",
            mode="multiclass",
            num_classes=5,
            embedding_dim=self.embedding_dim
        )

        from pyhealth.datasets import get_dataloader
        loader = get_dataloader(self.dataset, batch_size=1)
        batch = next(iter(loader))

        model.train()
        output = model(**batch)
        loss = output["loss"]
        loss.backward()

        # Verify that all trainable parameters (except dummy) have gradients
        for name, param in model.named_parameters():
            # Skip the pyhealth internal dummy parameter
            if "_dummy_param" in name:
                continue

            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter {name} is missing gradients!"
                )

        self.assertGreater(loss.item(), 0)

    def test_missing_modality_robustness(self):
        """
        Tests if the model handles cases where some modalities are missing.
        This mirrors the 'Stochastic Masking' logic from the paper.
        """
        model = Wav2Sleep(
            dataset=self.dataset,
            modalities=self.modalities,
            label_key="label",
            mode="multiclass",
            num_classes=5
        )
        # Mock a batch containing only ECG but missing Resp
        # Shape: (Batch=1, 1, Total_Points=15000)
        batch = {
            "ecg": torch.randn(1, 1, 15000),
            "label": torch.randint(0, 5, (1, 5))
        }

        model.eval()
        with torch.no_grad():
            output = model(**batch)

        # Should still produce predictions for all 5 epochs
        self.assertEqual(output["y_prob"].shape, (1, 5, 5))


if __name__ == "__main__":
    unittest.main()
