"""Tests for SCRIB, focused on the fill_max inference behavior and the
overall-risk loss's ambiguity term.
"""

import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import SCRIB
from pyhealth.calib.predictionset.scrib.quicksearch import loss_overall_py
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP


class TestSCRIB(unittest.TestCase):
    """Test cases for the SCRIB prediction set constructor."""

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        # 5 classes, enough samples per class for stable class-specific
        # thresholds during calibration.
        self.samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "procedures": np.random.randn(6).tolist(),
                "label": i % 5,
            }
            for i in range(60)
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"procedures": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test",
        )
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["procedures"],
            label_key="label",
            mode="multiclass",
        )
        self.model.eval()

    def test_fill_max_default_resolves_true(self):
        """fill_max=True is the constructor default and should be reflected
        in the resolved self.fill_max used by forward()."""
        m = SCRIB(self.model, risk=0.2)
        self.assertTrue(m.fill_max)

    def test_fill_max_false(self):
        m = SCRIB(self.model, risk=0.2, fill_max=False)
        self.assertFalse(m.fill_max)

    def test_loss_kwargs_without_fill_max_key_resolves_false(self):
        """Passing loss_kwargs explicitly without a 'fill_max' key must not
        silently inherit the fill_max=True constructor default -- it should
        match the underlying search routines' own default of False."""
        m = SCRIB(self.model, risk=0.2, loss_kwargs={"lk": 500.0})
        self.assertFalse(m.fill_max)

    def test_loss_kwargs_with_explicit_fill_max_key(self):
        m = SCRIB(self.model, risk=0.2, loss_kwargs={"lk": 500.0, "fill_max": True})
        self.assertTrue(m.fill_max)

    def test_forward_never_empty_when_fill_max_true(self):
        """The core regression test: forward() must apply the same
        fill_max behavior used during calibration, so no empty prediction
        sets should ever be returned when fill_max=True -- even when
        thresholds are (artificially) set so high that no class clears
        them naturally."""
        m = SCRIB(self.model, risk=0.1, fill_max=True)
        m.calibrate(cal_dataset=self.dataset)
        # Force every threshold above any predicted probability, so every
        # sample's natural prediction set is empty prior to the fill_max
        # fallback.
        m.t = torch.nn.Parameter(torch.ones_like(m.t) * 0.999)

        loader = get_dataloader(self.dataset, batch_size=len(self.samples), shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = m(**batch)

        set_sizes = out["y_predset"].sum(dim=1)
        self.assertTrue(torch.all(set_sizes == 1), "fill_max=True must fill every empty set with exactly the argmax class")

    def test_forward_allows_empty_when_fill_max_false(self):
        """Sanity check that fill_max actually gates the behavior: with the
        same forced thresholds, fill_max=False should leave sets empty."""
        m = SCRIB(self.model, risk=0.1, fill_max=False)
        m.calibrate(cal_dataset=self.dataset)
        m.t = torch.nn.Parameter(torch.ones_like(m.t) * 0.999)

        loader = get_dataloader(self.dataset, batch_size=len(self.samples), shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = m(**batch)

        set_sizes = out["y_predset"].sum(dim=1)
        self.assertTrue(torch.all(set_sizes == 0), "fill_max=False should leave sets empty when no class clears threshold")

    def test_forward_fills_with_argmax_class(self):
        """The filled-in class for an empty set must be the model's own
        argmax prediction, not an arbitrary class."""
        m = SCRIB(self.model, risk=0.1, fill_max=True)
        m.calibrate(cal_dataset=self.dataset)
        m.t = torch.nn.Parameter(torch.ones_like(m.t) * 0.999)

        loader = get_dataloader(self.dataset, batch_size=len(self.samples), shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            base_out = self.model(**batch)
            out = m(**batch)

        argmax_idx = base_out["y_prob"].argmax(dim=1)
        predicted_idx = out["y_predset"].float().argmax(dim=1)
        torch.testing.assert_close(predicted_idx, argmax_idx)

    def test_overall_risk_calibration_runs_and_controls_risk(self):
        """Overall (float) risk mode should calibrate without error and the
        resulting sure-prediction error rate should be near the target."""
        m = SCRIB(self.model, risk=0.2)
        m.calibrate(cal_dataset=self.dataset)
        self.assertIsNotNone(m.t)
        self.assertEqual(m.t.shape[0], 5)

    def test_class_specific_risk_calibration_runs(self):
        """Class-specific (array) risk mode should calibrate without error
        and produce one threshold per class."""
        risk = np.array([0.2, 0.3, 0.15, 0.25, 0.2])
        m = SCRIB(self.model, risk=risk)
        m.calibrate(cal_dataset=self.dataset)
        self.assertEqual(m.t.shape[0], 5)

    def test_forward_before_calibration_uses_none_threshold(self):
        m = SCRIB(self.model, risk=0.2)
        self.assertIsNone(m.t)

    def test_ambiguity_term_is_not_squared(self):
        """The overall-risk loss's chance-ambiguity term must be linear
        (1 - total_sure/N), matching the paper's Eq. 2 / Algorithm 2, not
        squared. Regression test for the fixed bug."""
        n, total_sure = 20, 12
        preds = np.zeros((n, 3), dtype=np.int32)
        # First `total_sure` rows: exactly one class included (|H|=1).
        preds[:total_sure, 0] = 1
        # Remaining rows: two classes included (|H|=2, ambiguous).
        preds[total_sure:, 0] = 1
        preds[total_sure:, 1] = 1
        labels = np.zeros((n, 3))
        labels[:, 0] = 1  # every true label is class 0
        max_classes = np.zeros(n, dtype=np.int32)

        loss = loss_overall_py(preds, labels, max_classes, risk=0.5, lk=1e4, fill_max=False)
        # total_err=0 (every "sure" row correctly includes class 0), so the
        # risk-penalty term is 0 and the loss should equal the unsquared
        # ambiguity term exactly: 1 - total_sure/N.
        expected = 1.0 - total_sure / n
        self.assertAlmostEqual(loss, expected, places=10)


if __name__ == "__main__":
    unittest.main()
