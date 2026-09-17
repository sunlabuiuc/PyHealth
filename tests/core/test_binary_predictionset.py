"""Binary-mode conformal prediction across the SetPredictor subclasses.

Each method re-presents a binary base model as a 2-class problem, so its
``y_predset`` must come out ``(N, 2)`` boolean and be scorable with the binary
prediction-set metrics. FavMac is intentionally excluded: it is multilabel-only.
"""

import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import (
    LABEL,
    SCRIB,
    BaseConformal,
    ClusterLabel,
    CovariateLabel,
    NeighborhoodLabel,
)
from pyhealth.calib.utils import binary_to_2col, extract_embeddings
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MLP


class TestBinaryToCol(unittest.TestCase):
    """The (N,) / (N,1) -> (N,2) probability expansion."""

    def test_shapes_and_values(self):
        out = binary_to_2col([0.2, 0.9, 0.5])
        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_allclose(out, [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
        # (N, 1) input is accepted and gives the same result.
        np.testing.assert_allclose(
            binary_to_2col(np.array([[0.2], [0.9], [0.5]])), out
        )


class TestBinaryPredictionSet(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        self.samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": [f"cond-{i % 5}", f"cond-{(i + 1) % 5}"],
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

        self.train_ds = self.dataset.subset([0, 1, 2, 3, 4, 5])
        self.cal_ds = self.dataset.subset([6, 7, 8, 9, 10, 11])

    def _embeddings(self, ds):
        return extract_embeddings(self.model, ds, batch_size=32, device="cpu")

    def _assert_binary_set(self, out):
        """Every method must yield an (N, 2) bool set, native (N, 1) y_prob,
        native binary labels, and metrics that compute through binary_metrics_fn."""
        self.assertEqual(out["y_predset"].dtype, torch.bool)
        self.assertEqual(out["y_predset"].dim(), 2)
        self.assertEqual(out["y_predset"].shape[1], 2)
        self.assertEqual(out["y_prob"].shape[1], 1)
        self.assertEqual(out["y_true"].shape, out["y_prob"].shape)
        self.assertEqual(out["y_true"].dtype, torch.float32)

        res = binary_metrics_fn(
            out["y_true"].numpy(),
            out["y_prob"].numpy().reshape(-1),
            metrics=["set_size", "rejection_rate", "miscoverage_ps"],
            y_predset=out["y_predset"].numpy(),
        )
        self.assertIn("rejection_rate", res)
        self.assertLessEqual(res["set_size"], 2.0)
        self.assertEqual(len(res["miscoverage_ps"]), 2)

    def _forward(self, cal_model):
        loader = get_dataloader(self.dataset, batch_size=12, shuffle=False)
        with torch.no_grad():
            return cal_model(**next(iter(loader)))

    def test_label(self):
        m = LABEL(self.model, alpha=0.3)
        self.assertEqual(m.mode, "binary")
        m.calibrate(cal_dataset=self.cal_ds)
        self.assertIsInstance(m.t, torch.Tensor)
        self._assert_binary_set(self._forward(m))

    def test_base_conformal(self):
        m = BaseConformal(self.model, alpha=0.3, score_type="threshold")
        m.calibrate(cal_dataset=self.cal_ds)
        self._assert_binary_set(self._forward(m))

    def test_scrib(self):
        m = SCRIB(self.model, risk=0.3)
        m.calibrate(cal_dataset=self.cal_ds)
        self._assert_binary_set(self._forward(m))

    def test_covariate_label(self):
        m = CovariateLabel(self.model, alpha=0.3)
        # Custom-weights path (uniform weights) avoids needing a shifted set.
        m.calibrate(
            cal_dataset=self.cal_ds,
            cal_weights=np.ones(len(self.cal_ds)),
        )
        self._assert_binary_set(self._forward(m))

    def test_cluster_label(self):
        m = ClusterLabel(self.model, alpha=0.3, n_clusters=2, random_state=0)
        m.calibrate(
            cal_dataset=self.cal_ds,
            train_embeddings=self._embeddings(self.train_ds),
            cal_embeddings=self._embeddings(self.cal_ds),
        )
        self._assert_binary_set(self._forward(m))

    def test_neighborhood_label(self):
        m = NeighborhoodLabel(self.model, alpha=0.3, k_neighbors=3, lambda_L=50.0)
        m.calibrate(
            cal_dataset=self.cal_ds,
            cal_embeddings=self._embeddings(self.cal_ds),
        )
        self._assert_binary_set(self._forward(m))


if __name__ == "__main__":
    unittest.main()
