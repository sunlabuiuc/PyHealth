import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import AttentionLSTM


def _make_samples():
    """Two synthetic patients with sequence-type conditions."""
    return [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": ["A", "B", "C", "D"],
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": ["B", "C"],
            "label": 0,
        },
    ]


class TestAttentionLSTM(unittest.TestCase):
    """Unit tests for the AttentionLSTM model."""

    def setUp(self) -> None:
        """Build a tiny synthetic dataset and model."""
        torch.manual_seed(42)
        np.random.seed(42)

        samples = _make_samples()
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_attention_lstm",
        )
        self.model = AttentionLSTM(
            dataset=self.dataset,
            embedding_dim=8,
            hidden_dim=4,
            dropout=0.0,
        )
        self.loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        self.batch = next(iter(self.loader))

    def test_instantiation(self) -> None:
        """Model initializes without error."""
        self.assertIsInstance(self.model, AttentionLSTM)
        self.assertEqual(self.model.embedding_dim, 8)
        self.assertEqual(self.model.hidden_dim, 4)
        self.assertIn("conditions", self.model.rnn)
        self.assertIn("conditions", self.model.attention)

    def test_forward_output_keys(self) -> None:
        """Forward pass returns all required output keys."""
        with torch.no_grad():
            out = self.model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit", "attention_weights"):
            self.assertIn(key, out, f"Missing key: {key}")

    def test_forward_shapes(self) -> None:
        """y_prob and logit have correct batch dimension."""
        with torch.no_grad():
            out = self.model(**self.batch)
        B = 2
        self.assertEqual(out["y_prob"].shape[0], B)
        self.assertEqual(out["logit"].shape[0], B)
        self.assertEqual(out["y_true"].shape[0], B)
        self.assertEqual(out["loss"].dim(), 0)

    def test_attention_weights_shape(self) -> None:
        """Attention weights have shape (B, T) for each feature key."""
        with torch.no_grad():
            out = self.model(**self.batch)
        attn = out["attention_weights"]
        self.assertIn("conditions", attn)
        self.assertEqual(attn["conditions"].dim(), 2)
        self.assertEqual(attn["conditions"].shape[0], 2)

    def test_backward_gradients(self) -> None:
        """Backward pass computes gradients for model parameters."""
        self.model.zero_grad()
        out = self.model(**self.batch)
        out["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients after backward pass")

    def test_state_dict_roundtrip(self) -> None:
        """State dict save/load produces identical outputs."""
        with torch.no_grad():
            out_before = self.model(**self.batch)

        state = self.model.state_dict()
        model2 = AttentionLSTM(
            dataset=self.dataset,
            embedding_dim=8,
            hidden_dim=4,
            dropout=0.0,
        )
        model2.load_state_dict(state)

        with torch.no_grad():
            out_after = model2(**self.batch)

        self.assertTrue(
            torch.allclose(out_before["logit"], out_after["logit"]),
            "Logits differ after state dict roundtrip",
        )

    def test_deletion_mask(self) -> None:
        """Forward with deletion_mask produces different y_prob than clean pass."""
        with torch.no_grad():
            clean = self.model(**self.batch)

        # Build a mask that zeros out the first time step for all samples
        T = clean["attention_weights"]["conditions"].shape[1]
        dm = torch.zeros(2, T, dtype=torch.bool)
        dm[:, 0] = True

        with torch.no_grad():
            masked = self.model(deletion_mask={"conditions": dm}, **self.batch)

        self.assertFalse(
            torch.allclose(clean["y_prob"], masked["y_prob"]),
            "deletion_mask had no effect on y_prob",
        )


if __name__ == "__main__":
    unittest.main()
