import tempfile
import unittest
from typing import Dict

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.event_contrastive import EBCLModel


class TestEBCLModel(unittest.TestCase):
    """Unit tests for the EBCLModel using small synthetic data."""

    def setUp(self) -> None:
        """Creates a tiny synthetic dataset and model instances."""
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
                "visit_id": "visit-1",
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

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={
                "pre": "tensor",
                "post": "tensor",
                "pre_mask": "tensor",
                "post_mask": "tensor",
            },
            output_schema={"label": "binary"},
            dataset_name="test_ebcl_model",
        )

        self.pretrain_model = EBCLModel(
            dataset=self.dataset,
            num_features=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            ff_hidden_dim=32,
            projection_dim=8,
            dropout=0.1,
            stage="pretrain",
            task="binary",
        )

        self.finetune_model = EBCLModel(
            dataset=self.dataset,
            num_features=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            ff_hidden_dim=32,
            projection_dim=8,
            dropout=0.1,
            stage="finetune",
            task="binary",
        )

    def tearDown(self) -> None:
        """Cleans up temporary resources after each test."""
        self.temp_dir.cleanup()

    def _get_batch(self) -> Dict:
        """Returns one small batch from the synthetic dataset."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["pre_mask"] = batch["pre_mask"].bool()
        batch["post_mask"] = batch["post_mask"].bool()
        return batch

    def test_model_initialization(self) -> None:
        """Tests that both model stages initialize correctly."""
        self.assertIsInstance(self.pretrain_model, EBCLModel)
        self.assertIsInstance(self.finetune_model, EBCLModel)

        self.assertEqual(self.pretrain_model.stage, "pretrain")
        self.assertEqual(self.finetune_model.stage, "finetune")

        # PyHealth compatibility: mode should represent prediction type.
        self.assertEqual(self.pretrain_model.mode, "binary")
        self.assertEqual(self.finetune_model.mode, "binary")

    def test_pretrain_forward_pass(self) -> None:
        """Tests pretraining forward pass outputs and shapes."""
        batch = self._get_batch()

        with torch.no_grad():
            output = self.pretrain_model(
                pre=batch["pre"],
                post=batch["post"],
                pre_mask=batch["pre_mask"],
                post_mask=batch["post_mask"],
            )

        self.assertIn("loss", output)
        self.assertIn("z_pre", output)
        self.assertIn("z_post", output)

        self.assertEqual(output["loss"].dim(), 0)
        self.assertEqual(output["z_pre"].shape, (2, 8))
        self.assertEqual(output["z_post"].shape, (2, 8))

    def test_finetune_forward_pass(self) -> None:
        """Tests finetuning forward pass outputs and shapes."""
        batch = self._get_batch()

        with torch.no_grad():
            output = self.finetune_model(
                pre=batch["pre"],
                pre_mask=batch["pre_mask"],
                label=batch["label"],
            )

        self.assertIn("loss", output)
        self.assertIn("logit", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)

        self.assertEqual(output["loss"].dim(), 0)
        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertEqual(output["y_prob"].shape, (2, 1))
        self.assertEqual(output["y_true"].shape, (2, 1))

    def test_finetune_forward_without_label(self) -> None:
        """Tests finetuning forward pass when no label is provided."""
        batch = self._get_batch()

        with torch.no_grad():
            output = self.finetune_model(
                pre=batch["pre"],
                pre_mask=batch["pre_mask"],
            )

        self.assertIn("logit", output)
        self.assertIn("y_prob", output)
        self.assertNotIn("loss", output)
        self.assertNotIn("y_true", output)

        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertEqual(output["y_prob"].shape, (2, 1))

    def test_encode_output_shape(self) -> None:
        """Tests that the shared encoder returns the expected shape."""
        batch = self._get_batch()

        with torch.no_grad():
            encoded = self.finetune_model.encode(
                batch["pre"],
                batch["pre_mask"],
            )

        self.assertEqual(encoded.shape, (2, 16))

    def test_pretrain_backward_pass(self) -> None:
        """Tests gradient computation in pretraining mode."""
        batch = self._get_batch()

        output = self.pretrain_model(
            pre=batch["pre"],
            post=batch["post"],
            pre_mask=batch["pre_mask"],
            post_mask=batch["post_mask"],
        )
        output["loss"].backward()

        has_gradient = any(
            parameter.requires_grad and parameter.grad is not None
            for parameter in self.pretrain_model.parameters()
        )
        self.assertTrue(has_gradient)

    def test_finetune_backward_pass(self) -> None:
        """Tests gradient computation in finetuning mode."""
        batch = self._get_batch()

        output = self.finetune_model(
            pre=batch["pre"],
            pre_mask=batch["pre_mask"],
            label=batch["label"],
        )
        output["loss"].backward()

        has_gradient = any(
            parameter.requires_grad and parameter.grad is not None
            for parameter in self.finetune_model.parameters()
        )
        self.assertTrue(has_gradient)

    def test_encoder_state_transfer(self) -> None:
        """Tests exporting and loading shared encoder weights."""
        encoder_state = self.pretrain_model.get_encoder_state_dict()

        transferred_model = EBCLModel(
            dataset=self.dataset,
            num_features=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            ff_hidden_dim=32,
            projection_dim=8,
            dropout=0.1,
            stage="finetune",
            task="binary",
        )
        transferred_model.load_encoder_state_dict(encoder_state)

        batch = self._get_batch()

        with torch.no_grad():
            encoded = transferred_model.encode(
                batch["pre"],
                batch["pre_mask"],
            )

        self.assertEqual(encoded.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()