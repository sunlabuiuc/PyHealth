import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCLModel


class TestEBCLModel(unittest.TestCase):
    """Test cases for the EBCLModel."""

    def setUp(self):
        """Set up small synthetic dataset, temp directory, and test models."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "pre": [
                    [0.1, 1.0, 0.5],
                    [0.2, 2.0, 1.5],
                    [0.0, 0.0, 0.0],
                ],
                "post": [
                    [0.3, 3.0, -0.2],
                    [0.4, 4.0, 0.8],
                    [0.0, 0.0, 0.0],
                ],
                "pre_mask": [1, 1, 0],
                "post_mask": [1, 1, 0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "pre": [
                    [0.2, 5.0, 0.7],
                    [0.3, 6.0, 1.2],
                    [0.0, 0.0, 0.0],
                ],
                "post": [
                    [0.5, 7.0, -0.4],
                    [0.6, 8.0, 0.9],
                    [0.0, 0.0, 0.0],
                ],
                "pre_mask": [1, 1, 0],
                "post_mask": [1, 1, 0],
                "label": 0,
            },
        ]

        self.input_schema = {
            "pre": "tensor",
            "post": "tensor",
            "pre_mask": "tensor",
            "post_mask": "tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_ebcl",
        )

        self.pretrain_model = EBCLModel(
            dataset=self.dataset,
            num_features=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            projection_dim=8,
            hidden_dim=16,
            dropout=0.1,
            mode="pretrain",
        )

        self.finetune_model = EBCLModel(
            dataset=self.dataset,
            num_features=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            projection_dim=8,
            hidden_dim=16,
            dropout=0.1,
            mode="finetune",
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _get_batch(self):
        """Gets one batch of synthetic data."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["pre_mask"] = batch["pre_mask"].bool()
        batch["post_mask"] = batch["post_mask"].bool()
        return batch

    def test_model_initialization(self):
        """Test that EBCL models initialize correctly."""
        self.assertIsInstance(self.pretrain_model, EBCLModel)
        self.assertIsInstance(self.finetune_model, EBCLModel)
        self.assertEqual(self.pretrain_model.mode, "pretrain")
        self.assertEqual(self.finetune_model.mode, "finetune")

    def test_pretrain_forward(self):
        """Test pretraining forward pass and output shapes."""
        batch = self._get_batch()

        with torch.no_grad():
            ret = self.pretrain_model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("z_pre", ret)
        self.assertIn("z_post", ret)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["z_pre"].shape, (2, 8))
        self.assertEqual(ret["z_post"].shape, (2, 8))

    def test_finetune_forward(self):
        """Test finetuning forward pass and output shapes."""
        batch = self._get_batch()

        with torch.no_grad():
            ret = self.finetune_model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("logit", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["y_prob"].shape, (2, 1))
        self.assertEqual(ret["y_true"].shape, (2, 1))

    def test_pretrain_backward(self):
        """Test gradient computation for pretraining mode."""
        batch = self._get_batch()
        ret = self.pretrain_model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.pretrain_model.parameters()
        )
        self.assertTrue(has_gradient)

    def test_finetune_backward(self):
        """Test gradient computation for finetuning mode."""
        batch = self._get_batch()
        ret = self.finetune_model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.finetune_model.parameters()
        )
        self.assertTrue(has_gradient)


if __name__ == "__main__":
    unittest.main()