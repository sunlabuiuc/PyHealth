"""Unit tests for TissueAwareSimCLR."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TissueAwareSimCLR

class TestTissueAwareSimCLR(unittest.TestCase):
    """Synthetic tests for TissueAwareSimCLR."""

    def _build_sample_dataset(self):
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "tile_bag": torch.randn(2, 3, 4, 4).tolist(),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "tile_bag": torch.randn(2, 3, 4, 4).tolist(),
                "label": 0,
            },
        ]
        return create_sample_dataset(
            samples=samples,
            input_schema={"tile_bag": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="tcga_crck_test",
        )

    def test_model_instantiation(self):
        """Test that the model initializes correctly."""
        dataset = self._build_sample_dataset()
        model = TissueAwareSimCLR(
            dataset=dataset,
            hidden_dim=16,
            dropout=0.1,
            freeze_encoder=True,
            pooling="attention",
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.feature_key, "tile_bag")
        self.assertEqual(model.label_key, "label")

    def test_forward_pass_shapes(self):
        """Test forward pass output keys and tensor shapes."""
        dataset = self._build_sample_dataset()
        model = TissueAwareSimCLR(dataset=dataset, hidden_dim=16)

        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertIn("attention_weights", ret)

        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["y_prob"].shape, (2, 1))
        self.assertEqual(ret["y_true"].shape, (2, 1))
        self.assertEqual(ret["attention_weights"].shape[0], 2)

    def test_backward_pass(self):
        """Test that gradients can be computed."""
        dataset = self._build_sample_dataset()
        model = TissueAwareSimCLR(dataset=dataset, hidden_dim=16)

        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        ret = model(**batch)

        self.assertIsNotNone(ret["loss"])
        ret["loss"].backward()
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_mean_pooling_variant(self):
        """Test the model with mean MIL pooling."""
        dataset = self._build_sample_dataset()
        model = TissueAwareSimCLR(
            dataset=dataset,
            hidden_dim=8,
            pooling="mean",
        )

        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        ret = model(**batch)

        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["attention_weights"].shape, (2, 2))

    def test_checkpoint_loading(self):
        """Test loading a prefixed encoder checkpoint."""
        dataset = self._build_sample_dataset()
        model = TissueAwareSimCLR(dataset=dataset, hidden_dim=8)

        prefixed_state = {
            f"encoder.{k}": v.cpu() for k, v in model.encoder.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "simclr_encoder.pt"
            torch.save({"state_dict": prefixed_state}, ckpt_path)

            loaded_model = TissueAwareSimCLR(
                dataset=dataset,
                checkpoint_path=str(ckpt_path),
                hidden_dim=8,
            )
            self.assertIsNotNone(loaded_model)


if __name__ == "__main__":
    unittest.main()