import unittest
import warnings

import torch

from pyhealth.calib.calibration import LogisticRecalibration
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.trainer import Trainer


def _dataset(mode, n=12):
    labels = {
        # three positives for every negative, so a fitted intercept is positive
        "binary": lambda i: int(i % 4 != 0),
        "multilabel": lambda i: [["x"], ["y"], ["x", "y"]][i % 3],
        "multiclass": lambda i: i % 3,
    }[mode]
    samples = [
        {
            "patient_id": f"patient-{i}",
            "visit_id": f"visit-{i}",
            "conditions": ["cond-1", "cond-2", "cond-3", "cond-4"][: 1 + i % 4],
            "label": labels(i),
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "sequence"},
        output_schema={"label": mode},
        dataset_name="test_logistic_recalibration",
    )


def _miscalibrated(true_a, true_b, n=20000, seed=0):
    """Logits whose true log-odds are ``true_a + true_b * logit``."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 1, generator=gen) * 2.0
    prob = torch.sigmoid(true_a + true_b * logits)
    label = torch.bernoulli(prob, generator=gen)
    return logits, label


class TestLogisticRecalibration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.binary_dataset = _dataset("binary")
        self.binary_model = RNN(dataset=self.binary_dataset)

    # --- coefficient recovery on synthetic logits ---

    def test_intercept_slope_recovers_known_values(self):
        logits, label = _miscalibrated(true_a=-1.0, true_b=0.5)
        cal = LogisticRecalibration(self.binary_model, method="intercept_slope")
        cal._fit(logits, label)
        self.assertAlmostEqual(cal.intercept.item(), -1.0, delta=0.1)
        self.assertAlmostEqual(cal.slope.item(), 0.5, delta=0.05)

    def test_intercept_only_keeps_slope_at_one(self):
        logits, label = _miscalibrated(true_a=0.8, true_b=1.0)
        cal = LogisticRecalibration(self.binary_model, method="intercept")
        cal._fit(logits, label)
        self.assertAlmostEqual(cal.intercept.item(), 0.8, delta=0.1)
        self.assertEqual(cal.slope.item(), 1.0)
        self.assertFalse(cal.slope.requires_grad)

    def test_already_calibrated_logits_are_left_alone(self):
        logits, label = _miscalibrated(true_a=0.0, true_b=1.0)
        cal = LogisticRecalibration(self.binary_model, method="intercept_slope")
        cal._fit(logits, label)
        self.assertAlmostEqual(cal.intercept.item(), 0.0, delta=0.1)
        self.assertAlmostEqual(cal.slope.item(), 1.0, delta=0.05)

    # --- public API ---

    def test_forward_applies_fitted_intercept_and_slope(self):
        cal = LogisticRecalibration(self.binary_model, method="intercept")
        cal.calibrate(cal_dataset=self.binary_dataset)
        # 9 of 12 labels are positive and the model is untrained, so the
        # fitted intercept must move the predictions up.
        self.assertGreater(cal.intercept.item(), 0.0)

        batch = next(iter(get_dataloader(self.binary_dataset, batch_size=12)))
        with torch.no_grad():
            base = self.binary_model(**batch)
            out = cal(**batch)
        expected_logit = cal.intercept + cal.slope * base["logit"]
        self.assertTrue(torch.allclose(out["logit"], expected_logit))
        self.assertFalse(torch.allclose(out["logit"], base["logit"]))
        self.assertTrue(torch.allclose(out["y_prob"], torch.sigmoid(expected_logit)))
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            expected_logit, base["y_true"]
        )
        self.assertTrue(torch.allclose(out["loss"], expected_loss))

    def test_evaluates_through_trainer(self):
        cal = LogisticRecalibration(self.binary_model, method="intercept")
        cal.calibrate(cal_dataset=self.binary_dataset)
        loader = get_dataloader(self.binary_dataset, batch_size=12, shuffle=False)
        trainer = Trainer(model=cal, metrics=["roc_auc"], enable_logging=False)
        self.assertIn("roc_auc", trainer.evaluate(loader))

    def test_multilabel_fits_one_pair_per_label_and_round_trips(self):
        dataset = _dataset("multilabel")
        model = RNN(dataset=dataset)
        cal = LogisticRecalibration(model)
        cal.calibrate(cal_dataset=dataset)
        self.assertEqual(cal.intercept.shape, (2,))
        self.assertEqual(cal.slope.shape, (2,))
        self.assertNotEqual(cal.intercept[0].item(), cal.intercept[1].item())

        fresh = LogisticRecalibration(model)
        fresh.load_state_dict(cal.state_dict())
        batch = next(iter(get_dataloader(dataset, batch_size=12)))
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(cal(**batch)["logit"], fresh(**batch)["logit"])
            )

    # --- edge cases ---

    def test_rejects_when_every_label_is_single_valued(self):
        logits = torch.randn(50, 1)
        cal = LogisticRecalibration(self.binary_model)
        with self.assertRaises(ValueError):
            cal._fit(logits, torch.zeros(50, 1))

    def test_multilabel_skips_single_valued_label(self):
        model = RNN(dataset=_dataset("multilabel"))
        gen = torch.Generator().manual_seed(1)
        logits = torch.randn(4000, 2, generator=gen)
        label = torch.bernoulli(torch.sigmoid(1.0 + logits), generator=gen)
        label[:, 1] = 0.0
        cal = LogisticRecalibration(model, method="intercept")
        with self.assertWarns(RuntimeWarning):
            cal._fit(logits, label)
        self.assertAlmostEqual(cal.intercept[0].item(), 1.0, delta=0.15)
        self.assertEqual(cal.intercept[1].item(), 0.0)
        self.assertEqual(cal.slope[1].item(), 1.0)

    def test_warns_on_separation(self):
        logits = torch.linspace(-3, 3, 50).unsqueeze(1)
        label = (logits > 0).float()
        cal = LogisticRecalibration(self.binary_model, method="intercept_slope")
        with self.assertWarns(RuntimeWarning):
            cal._fit(logits, label)

    def test_no_warning_for_large_shift_or_large_slope(self):
        # a large but real intercept shift
        logits, label = _miscalibrated(true_a=4.0, true_b=1.0)
        cal = LogisticRecalibration(self.binary_model, method="intercept")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cal._fit(logits, label)
        self.assertAlmostEqual(cal.intercept.item(), 4.0, delta=0.3)
        # compressed logits, so the true slope is large
        logits, label = _miscalibrated(true_a=0.0, true_b=1.0)
        logits = logits / 20.0
        cal = LogisticRecalibration(self.binary_model, method="intercept_slope")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cal._fit(logits, label)
        self.assertAlmostEqual(cal.slope.item(), 20.0, delta=2.0)

    def test_raises_on_non_finite_fit(self):
        logits, label = _miscalibrated(true_a=0.0, true_b=1.0, n=200)
        logits[0, 0] = float("nan")
        cal = LogisticRecalibration(self.binary_model, method="intercept_slope")
        with self.assertRaises(RuntimeError):
            cal._fit(logits, label)

    def test_rejects_multiclass_and_unknown_method(self):
        with self.assertRaises(ValueError):
            LogisticRecalibration(RNN(dataset=_dataset("multiclass")))
        with self.assertRaises(ValueError):
            LogisticRecalibration(self.binary_model, method="slope")


if __name__ == "__main__":
    unittest.main()
