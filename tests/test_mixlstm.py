import unittest
import tempfile
import shutil
import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MixLSTM


class TestMixLSTMRegression(unittest.TestCase):
    """Test MixLSTM in per-timestep regression mode (MLHC2019 synthetic task)."""

    def setUp(self):
        """Set up small synthetic regression dataset and model."""
        self.tmp_dir = tempfile.mkdtemp()

        T = 10
        input_dim = 2
        prev_used = 3
        n = 20

        rng = np.random.RandomState(42)
        x = np.zeros((n, T, input_dim))
        nz = int(n * T * input_dim * 0.1)
        idx = rng.choice(n * T * input_dim, size=nz, replace=False)
        x.flat[idx] = rng.uniform(size=nz) * 10
        y = np.zeros((n, T))
        for t in range(prev_used, T):
            y[:, t] = x[:, t - prev_used:t, :].sum(axis=(1, 2))

        self.samples = [
            {
                "patient_id": f"p-{i}",
                "visit_id": "v-0",
                "series": x[i].tolist(),
                "y": y[i].tolist(),
            }
            for i in range(n)
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"series": "tensor"},
            output_schema={"y": "tensor"},
            dataset_name="test_mixlstm_reg",
        )

        self.model = MixLSTM(
            dataset=self.dataset,
            num_experts=2,
            hidden_size=16,
            prev_used_timestamps=prev_used,
        )
        self.batch_size = 4

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_instantiation(self):
        """Test that model initializes with correct attributes."""
        self.assertIsInstance(self.model, MixLSTM)
        self.assertTrue(self.model._per_timestep)
        self.assertEqual(self.model.input_size, 2)
        self.assertEqual(self.model.time_steps, 10)
        self.assertEqual(self.model.hidden_size, 16)
        self.assertEqual(self.model.prev_used_timestamps, 3)

    def test_forward_output_keys(self):
        """Test that forward returns expected keys for regression."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("logit", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

    def test_forward_output_shapes(self):
        """Test output tensor shapes for per-timestep regression."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        bs = ret["logit"].shape[0]
        # logit: (batch, T, 1)
        self.assertEqual(ret["logit"].shape, (bs, 10, 1))
        # y_prob same as logit for regression
        self.assertEqual(ret["y_prob"].shape, (bs, 10, 1))
        # y_true: (batch, T, 1)
        self.assertEqual(ret["y_true"].shape, (bs, 10, 1))
        # loss is scalar
        self.assertEqual(ret["loss"].dim(), 0)

    def test_forward_no_labels(self):
        """Test forward without labels returns logit/y_prob but no loss."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))
        # Remove the label key
        del batch["y"]

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("logit", ret)
        self.assertIn("y_prob", ret)
        self.assertNotIn("loss", ret)
        self.assertNotIn("y_true", ret)

    def test_backward_gradients(self):
        """Test that loss.backward() produces gradients on all trainable params."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters received gradients")

    def test_loss_is_finite(self):
        """Test that loss is finite (not NaN or Inf)."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertTrue(torch.isfinite(ret["loss"]).item(), "Loss is not finite")

    def test_custom_hyperparameters(self):
        """Test model with different num_experts and hidden_size."""
        model = MixLSTM(
            dataset=self.dataset,
            num_experts=4,
            hidden_size=32,
            prev_used_timestamps=3,
        )
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[2], 1)


class TestMixLSTMClassification(unittest.TestCase):
    """Test MixLSTM in classification mode (standard PyHealth label task)."""

    def setUp(self):
        """Set up small synthetic classification dataset and model."""
        self.tmp_dir = tempfile.mkdtemp()

        T = 8
        input_dim = 3
        n = 16
        n_classes = 3

        rng = np.random.RandomState(0)
        self.samples = [
            {
                "patient_id": f"p-{i}",
                "visit_id": "v-0",
                "series": rng.randn(T, input_dim).tolist(),
                "label": int(i % n_classes),
            }
            for i in range(n)
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"series": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_mixlstm_cls",
        )

        self.model = MixLSTM(
            dataset=self.dataset,
            num_experts=2,
            hidden_size=16,
        )
        self.batch_size = 4

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_instantiation(self):
        """Test that classification model initializes correctly."""
        self.assertIsInstance(self.model, MixLSTM)
        self.assertFalse(self.model._per_timestep)
        self.assertEqual(self.model.mode, "multiclass")

    def test_forward_output_keys(self):
        """Test that forward returns expected keys for classification."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("logit", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

    def test_forward_output_shapes(self):
        """Test output tensor shapes for classification (last timestep)."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        bs = ret["logit"].shape[0]
        n_classes = 3
        # logit: (batch, n_classes)
        self.assertEqual(ret["logit"].shape, (bs, n_classes))
        # y_prob: (batch, n_classes) — softmax output
        self.assertEqual(ret["y_prob"].shape, (bs, n_classes))
        # y_true: (batch,)
        self.assertEqual(ret["y_true"].shape[0], bs)
        # loss is scalar
        self.assertEqual(ret["loss"].dim(), 0)

    def test_forward_no_labels(self):
        """Test forward without labels returns logit/y_prob but no loss."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))
        del batch["label"]

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("logit", ret)
        self.assertIn("y_prob", ret)
        self.assertNotIn("loss", ret)
        self.assertNotIn("y_true", ret)

    def test_backward_gradients(self):
        """Test that loss.backward() produces gradients."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters received gradients")

    def test_y_prob_sums_to_one(self):
        """Test that y_prob (softmax) sums to ~1 for each sample."""
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        prob_sums = ret["y_prob"].sum(dim=1)
        self.assertTrue(
            torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5),
            "y_prob rows do not sum to 1",
        )


if __name__ == "__main__":
    unittest.main()
