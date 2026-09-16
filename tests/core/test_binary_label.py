import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import LABEL
from pyhealth.calib.utils import binary_to_2col
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MLP


class TestBinaryToCol(unittest.TestCase):
    """The (N,) / (N,1) -> (N,2) probability expansion."""

    def test_shapes_and_values(self):
        out = binary_to_2col([0.2, 0.9, 0.5])
        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_allclose(
            out, [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]]
        )
        # (N, 1) input is accepted and gives the same result.
        np.testing.assert_allclose(
            binary_to_2col(np.array([[0.2], [0.9], [0.5]])), out
        )


class TestBinaryLabel(unittest.TestCase):
    """LABEL on a binary base model (the binary-mode conformal path)."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

        # 12 binary samples: indices 0-5 train, 6-11 calibration/test.
        self.samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": [f"cond-{i}", f"cond-{i + 1}"],
                "procedures": [1.0 * i, 2.0, 3.5, 4.0],
                "label": i % 2,
            }
            for i in range(12)
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test-binary",
        )
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="binary",
        )
        self.model.eval()
        self.cal_dataset = self.dataset.subset([6, 7, 8, 9, 10, 11])

    def test_binary_mode_is_accepted(self):
        """A binary base model no longer raises, and mode stays honest."""
        cal_model = LABEL(self.model, alpha=0.3)
        self.assertEqual(cal_model.mode, "binary")
        self.assertFalse(hasattr(cal_model, "_binary"))

    def test_calibrate_sets_threshold(self):
        cal_model = LABEL(self.model, alpha=0.3)
        cal_model.calibrate(cal_dataset=self.cal_dataset)
        self.assertIsNotNone(cal_model.t)
        self.assertIsInstance(cal_model.t, torch.Tensor)

    def test_forward_returns_two_column_predset(self):
        cal_model = LABEL(self.model, alpha=0.3)
        cal_model.calibrate(cal_dataset=self.cal_dataset)

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        with torch.no_grad():
            out = cal_model(**next(iter(loader)))

        # The set ranges over both classes -> (N, 2) bool.
        self.assertEqual(out["y_predset"].dtype, torch.bool)
        self.assertEqual(out["y_predset"].dim(), 2)
        self.assertEqual(out["y_predset"].shape[1], 2)
        # y_prob stays the model's native positive-class probability.
        self.assertEqual(out["y_prob"].shape[1], 1)
        # Labels retain the base model's native binary shape and dtype.
        self.assertEqual(out["y_true"].shape, out["y_prob"].shape)
        self.assertEqual(out["y_true"].dtype, torch.float32)

    def test_binary_metrics_accepts_predset(self):
        cal_model = LABEL(self.model, alpha=0.3)
        cal_model.calibrate(cal_dataset=self.cal_dataset)

        loader = get_dataloader(self.dataset, batch_size=12, shuffle=False)
        with torch.no_grad():
            out = cal_model(**next(iter(loader)))

        res = binary_metrics_fn(
            out["y_true"].numpy(),
            out["y_prob"].numpy().reshape(-1),
            metrics=["set_size", "rejection_rate", "miscoverage_ps"],
            y_predset=out["y_predset"].numpy(),
        )
        self.assertIn("set_size", res)
        self.assertIn("rejection_rate", res)
        # Two classes: a set can hold at most both.
        self.assertLessEqual(res["set_size"], 2.0)
        self.assertEqual(len(res["miscoverage_ps"]), 2)


if __name__ == "__main__":
    unittest.main()
