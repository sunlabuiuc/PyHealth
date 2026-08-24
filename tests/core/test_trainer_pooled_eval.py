"""Tests for pooled (batch-invariant) evaluation metrics in Trainer.

Regression tests for https://github.com/sunlabuiuc/PyHealth/issues/859:
``trainer.evaluate`` must accumulate predictions and labels across batches
and compute every metric exactly once on the pooled arrays, and must pool
the loss as an example-weighted mean. Otherwise the reported scores depend
on the evaluation batch size and on the order in which batches arrive, so
repeated benchmark runs disagree even with a fixed seed.
"""

import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pyhealth.trainer import Trainer


class SyntheticBinaryDataset(Dataset):
    """Small deterministic binary-classification dataset."""

    def __init__(self, n_samples: int = 100, n_features: int = 4, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n_samples, n_features)).astype("float32")
        self.weights = rng.normal(size=(n_features,)).astype("float32")
        noise = rng.normal(scale=2.0, size=n_samples).astype("float32")
        self.y = (self.x @ self.weights + noise > 0).astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}


class DeterministicBinaryModel(nn.Module):
    """Fixed-weight logistic model exposing the PyHealth output contract."""

    def __init__(self, weights: np.ndarray):
        super().__init__()
        self.mode = "binary"
        self.linear = nn.Linear(len(weights), 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weights).reshape(1, -1))
            self.linear.bias.zero_()

    def forward(self, x, y, **kwargs):
        logits = self.linear(x).squeeze(-1)
        y_true = y.float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_true)
        return {"loss": loss, "y_true": y_true, "y_prob": torch.sigmoid(logits)}


class LossOnlyModel(nn.Module):
    """Model without prediction outputs (exercises the mode=None path)."""

    def __init__(self, n_features: int = 4):
        super().__init__()
        self.mode = None
        self.linear = nn.Linear(n_features, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            self.linear.bias.zero_()

    def forward(self, x, y, **kwargs):
        logits = self.linear(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float())
        return {"loss": loss}


METRICS = ["roc_auc", "pr_auc", "f1", "accuracy"]


class TestPooledEvaluation(unittest.TestCase):
    """trainer.evaluate must be deterministic and batch-invariant."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = SyntheticBinaryDataset(n_samples=100)
        cls.model = DeterministicBinaryModel(cls.dataset.weights)
        cls.trainer = Trainer(
            model=cls.model,
            metrics=METRICS,
            device="cpu",
            enable_logging=False,
        )

    def _evaluate(self, batch_size, shuffle_seed=None):
        if shuffle_seed is None:
            loader = DataLoader(self.dataset, batch_size=batch_size)
        else:
            generator = torch.Generator().manual_seed(shuffle_seed)
            loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
        return self.trainer.evaluate(loader)

    def test_determinism_same_seed(self):
        """Two evaluate() calls with the same seed give identical metrics."""
        scores_1 = self._evaluate(batch_size=16, shuffle_seed=0)
        scores_2 = self._evaluate(batch_size=16, shuffle_seed=0)
        self.assertEqual(set(scores_1), set(scores_2))
        for key in scores_1:
            self.assertEqual(scores_1[key], scores_2[key], msg=key)

    def test_batch_order_invariance(self):
        """Metrics do not depend on the order in which batches arrive."""
        scores_1 = self._evaluate(batch_size=16, shuffle_seed=0)
        scores_2 = self._evaluate(batch_size=16, shuffle_seed=1)
        self.assertEqual(set(scores_1), set(scores_2))
        for key in scores_1:
            np.testing.assert_allclose(
                scores_1[key], scores_2[key], rtol=1e-6, err_msg=key
            )

    def test_batch_size_invariance(self):
        """Same data at batch_size 16 vs 64 gives identical pooled metrics.

        100 samples do not divide evenly into either batch size, so any
        per-batch averaging would over-weight the partial final batch and
        make the two runs disagree.
        """
        scores_16 = self._evaluate(batch_size=16)
        scores_64 = self._evaluate(batch_size=64)
        self.assertEqual(set(scores_16), set(scores_64))
        for key in scores_16:
            np.testing.assert_allclose(
                scores_16[key], scores_64[key], rtol=1e-6, err_msg=key
            )

    def test_pooled_metrics_match_sklearn(self):
        """Pooled AUROC/AUPRC match sklearn computed directly on all data."""
        with torch.no_grad():
            logits = self.model.linear(torch.from_numpy(self.dataset.x))
            y_prob = torch.sigmoid(logits.squeeze(-1))
        y_true = self.dataset.y
        expected_roc_auc = roc_auc_score(y_true, y_prob.numpy())
        expected_pr_auc = average_precision_score(y_true, y_prob.numpy())
        expected_loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), torch.from_numpy(y_true)
        ).item()

        scores = self._evaluate(batch_size=16)
        np.testing.assert_allclose(scores["roc_auc"], expected_roc_auc, rtol=1e-6)
        np.testing.assert_allclose(scores["pr_auc"], expected_pr_auc, rtol=1e-6)
        np.testing.assert_allclose(scores["loss"], expected_loss, rtol=1e-5)

    def test_loss_only_model_batch_size_invariance(self):
        """The mode=None loss path is also example-weighted, not batch-averaged."""
        trainer = Trainer(
            model=LossOnlyModel(),
            device="cpu",
            enable_logging=False,
        )
        loss_16 = trainer.evaluate(
            DataLoader(self.dataset, batch_size=16)
        )["loss"]
        loss_64 = trainer.evaluate(
            DataLoader(self.dataset, batch_size=64)
        )["loss"]
        np.testing.assert_allclose(loss_16, loss_64, rtol=1e-6)

    def test_inference_returns_all_samples_in_order(self):
        """inference() pools every sample exactly once, in dataloader order."""
        loader = DataLoader(self.dataset, batch_size=16)
        y_true_all, y_prob_all, _ = self.trainer.inference(loader)
        self.assertEqual(y_true_all.shape[0], len(self.dataset))
        self.assertEqual(y_prob_all.shape[0], len(self.dataset))
        np.testing.assert_array_equal(y_true_all, self.dataset.y)


if __name__ == "__main__":
    unittest.main()
