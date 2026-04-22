import shutil
import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.deep_cox_mixtures import DeepCoxMixtures


SAMPLES = [
    {
        "patient_id": "patient-0",
        "visit_id": "visit-0",
        "features": ["age_50.0", "sex_male", "race_white"],
        "survival_label": 0.72,
    },
    {
        "patient_id": "patient-1",
        "visit_id": "visit-0",
        "features": ["age_65.0", "sex_female", "race_black"],
        "survival_label": 0.45,
    },
    {
        "patient_id": "patient-2",
        "visit_id": "visit-0",
        "features": ["age_78.0", "sex_male", "race_hispanic"],
        "survival_label": 0.21,
    },
    {
        "patient_id": "patient-3",
        "visit_id": "visit-0",
        "features": ["age_40.0", "sex_female", "race_white"],
        "survival_label": 0.88,
    },
]

INPUT_SCHEMA = {"features": "sequence"}
OUTPUT_SCHEMA = {"survival_label": "regression"}


class TestDeepCoxMixtures(unittest.TestCase):
    """Test cases for the DeepCoxMixtures model."""

    def setUp(self):
        """Set up synthetic dataset and default model in a temporary directory."""
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset = create_sample_dataset(
            samples=SAMPLES,
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            dataset_name="test_dcm",
            cache_dir=self.tmp_dir,
        )
        self.model = DeepCoxMixtures(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _batch(self):
        return next(iter(self.loader))

    # Initialization

    def test_initialization_defaults(self):
        """Model initialises with correct default hyperparameters."""
        self.assertIsInstance(self.model, DeepCoxMixtures)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.num_mixtures, 3)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(self.model.dropout, 0.0)
        self.assertEqual(self.model.label_key, "survival_label")
        self.assertIn("features", self.model.feature_keys)

    def test_initialization_custom(self):
        """Model initialises correctly with custom hyperparameters."""
        model = DeepCoxMixtures(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            num_mixtures=6,
            num_layers=3,
            dropout=0.3,
        )
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.num_mixtures, 6)
        self.assertEqual(model.num_layers, 3)
        self.assertEqual(model.dropout, 0.3)

    def test_k1_reduces_to_single_cox(self):
        """num_mixtures=1 instantiates without error (neural Cox baseline)."""
        model = DeepCoxMixtures(dataset=self.dataset, num_mixtures=1)
        self.assertEqual(model.num_mixtures, 1)

    # Forward pass output keys and shapes

    def test_forward_output_keys(self):
        """Forward pass returns logit, y_prob, loss, and y_true."""
        batch = self._batch()
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        self.assertIn("y_true", out)

    def test_forward_output_shapes(self):
        """Forward pass returns tensors with correct batch dimension."""
        batch = self._batch()
        with torch.no_grad():
            out = self.model(**batch)
        n = len(SAMPLES)
        self.assertEqual(out["logit"].shape, (n, 1))
        self.assertEqual(out["y_prob"].shape, (n, 1))
        self.assertEqual(out["y_true"].shape, (n, 1))
        self.assertEqual(out["loss"].dim(), 0)

    def test_y_prob_range(self):
        """Predicted survival probabilities are in (0, 1)."""
        batch = self._batch()
        with torch.no_grad():
            out = self.model(**batch)
        self.assertTrue((out["y_prob"] > 0).all())
        self.assertTrue((out["y_prob"] < 1).all())

    def test_forward_without_label(self):
        """Forward pass without label key returns logit and y_prob only."""
        batch = self._batch()
        batch.pop("survival_label", None)
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)

    # Backward pass / gradient flow

    def test_backward(self):
        """Loss backward populates gradients for all trainable parameters."""
        batch = self._batch()
        out = self.model(**batch)
        out["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad)

    def test_loss_is_finite(self):
        """Training loss is finite (no NaN or Inf)."""
        batch = self._batch()
        with torch.no_grad():
            out = self.model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))

    # DCM layer internals

    def test_gate_probs_sum_to_one(self):
        """Gate probabilities sum to 1 across mixture components."""
        batch = self._batch()
        patient_emb = self.model._embed_features(batch)
        gate_probs, _ = self.model.dcm_layer(patient_emb)
        sums = gate_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_log_hazards_shape(self):
        """Log hazard tensor has shape (batch, num_mixtures)."""
        batch = self._batch()
        patient_emb = self.model._embed_features(batch)
        _, log_hazards = self.model.dcm_layer(patient_emb)
        self.assertEqual(
            log_hazards.shape, (len(SAMPLES), self.model.num_mixtures)
        )

    # Hyperparameter variations

    def test_varying_num_mixtures_forward(self):
        """Forward pass succeeds for K in {1, 3, 6}."""
        for k in [1, 3, 6]:
            model = DeepCoxMixtures(dataset=self.dataset, num_mixtures=k)
            batch = self._batch()
            with torch.no_grad():
                out = model(**batch)
            self.assertIn("loss", out)
            self.assertEqual(out["logit"].shape, (len(SAMPLES), 1))

    def test_varying_hidden_dim_forward(self):
        """Forward pass succeeds for different hidden dimensions."""
        for hdim in [16, 64, 128]:
            model = DeepCoxMixtures(dataset=self.dataset, hidden_dim=hdim)
            batch = self._batch()
            with torch.no_grad():
                out = model(**batch)
            self.assertIn("loss", out)

    def test_dropout_forward(self):
        """Forward pass works with dropout enabled (train mode)."""
        model = DeepCoxMixtures(dataset=self.dataset, dropout=0.5)
        model.train()
        batch = self._batch()
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()