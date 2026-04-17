"""Tests for ConformalMetaAnalysisModel.

Uses synthetic batches to exercise the KRR + conformal pipeline.
No real dataset is required.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch


def _fake_sample_dataset(d: int = 3):
    """Create a minimal SampleDataset-like object for model init."""
    return SimpleNamespace(
        input_schema={"features": "tensor"},
        output_schema={"true_effect": "regression"},
    )


class TestConformalMetaAnalysisModel(unittest.TestCase):
    """Tests for ConformalMetaAnalysisModel."""

    def test_init_validates_alpha(self):
        """alpha must be in (0, 1)."""
        from pyhealth.models.conformal_meta_analysis_krr import (
            ConformalMetaAnalysisModel,
        )
        ds = _fake_sample_dataset()
        with self.assertRaises(ValueError):
            ConformalMetaAnalysisModel(dataset=ds, alpha=-0.1)
        with self.assertRaises(ValueError):
            ConformalMetaAnalysisModel(dataset=ds, alpha=1.0)

    def test_init_validates_eta(self):
        """eta must be non-negative."""
        from pyhealth.models.conformal_meta_analysis_krr import (
            ConformalMetaAnalysisModel,
        )
        ds = _fake_sample_dataset()
        with self.assertRaises(ValueError):
            ConformalMetaAnalysisModel(dataset=ds, eta=-0.5)

    def test_init_rejects_bad_kernel(self):
        """Only 'gaussian' and 'laplace' kernels are supported."""
        from pyhealth.models.conformal_meta_analysis_krr import (
            ConformalMetaAnalysisModel,
        )
        ds = _fake_sample_dataset()
        with self.assertRaises(ValueError):
            ConformalMetaAnalysisModel(dataset=ds, kernel_type="cosine")

    def test_forward_returns_expected_keys(self):
        """Forward pass returns y_pred, interval_lower, interval_upper."""
        from pyhealth.models.conformal_meta_analysis_krr import (
            ConformalMetaAnalysisModel,
        )
        ds = _fake_sample_dataset()
        model = ConformalMetaAnalysisModel(dataset=ds, alpha=0.2)

        n, d = 20, 3
        rng = np.random.RandomState(0)
        features = torch.tensor(rng.randn(n, d), dtype=torch.float32)
        true_effect = torch.tensor(rng.randn(n), dtype=torch.float32)
        observed = true_effect + 0.1 * torch.randn(n)
        variance = torch.full((n,), 0.01)
        prior_mean = true_effect + 0.05 * torch.randn(n)

        out = model(
            features=features,
            observed_effect=observed,
            variance=variance,
            prior_mean=prior_mean,
            true_effect=true_effect,
        )
        for key in ("y_pred", "interval_lower", "interval_upper",
                    "loss", "y_true"):
            self.assertIn(key, out)
        self.assertEqual(out["y_pred"].shape[0], n)

    def test_interval_bounds_ordered(self):
        """Interval lower <= upper for each trial."""
        from pyhealth.models.conformal_meta_analysis_krr import (
            ConformalMetaAnalysisModel,
        )
        ds = _fake_sample_dataset()
        model = ConformalMetaAnalysisModel(dataset=ds, alpha=0.2)

        n, d = 15, 2
        rng = np.random.RandomState(1)
        features = torch.tensor(rng.randn(n, d), dtype=torch.float32)
        y = torch.tensor(rng.randn(n), dtype=torch.float32)
        v = torch.full((n,), 0.05)
        m = y + 0.1 * torch.randn(n)
        u = y + 0.1 * torch.randn(n)

        out = model(features=features, observed_effect=y,
                    variance=v, prior_mean=m, true_effect=u)
        lo = out["interval_lower"].numpy().ravel()
        hi = out["interval_upper"].numpy().ravel()
        finite = np.isfinite(lo) & np.isfinite(hi)
        self.assertTrue((hi[finite] >= lo[finite]).all())

    def test_eta_changes_output(self):
        """eta=0 and eta>0 produce different intervals on noisy data."""
        from pyhealth.models.conformal_meta_analysis_krr import (
            ConformalMetaAnalysisModel,
        )
        ds = _fake_sample_dataset()

        n, d = 15, 2
        rng = np.random.RandomState(2)
        features = torch.tensor(rng.randn(n, d), dtype=torch.float32)
        y = torch.tensor(rng.randn(n), dtype=torch.float32)
        v = torch.full((n,), 2.0)  # non-trivial variance
        m = y + 0.2 * torch.randn(n)

        m0 = ConformalMetaAnalysisModel(dataset=ds, alpha=0.2, eta=0.0)
        m1 = ConformalMetaAnalysisModel(dataset=ds, alpha=0.2, eta=0.5)
        out0 = m0(features=features, observed_effect=y,
                  variance=v, prior_mean=m)
        out1 = m1(features=features, observed_effect=y,
                  variance=v, prior_mean=m)

        w0 = (out0["interval_upper"] - out0["interval_lower"]).numpy()
        w1 = (out1["interval_upper"] - out1["interval_lower"]).numpy()
        # Widths should differ for at least one trial
        self.assertFalse(np.allclose(w0, w1))


if __name__ == "__main__":
    unittest.main()
