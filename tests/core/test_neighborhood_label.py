"""Tests for NeighborhoodLabel (NCP) prediction set constructor."""

import unittest
import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.calib.predictionset.cluster import NeighborhoodLabel
from pyhealth.calib.utils import extract_embeddings


class TestNeighborhoodLabel(unittest.TestCase):
    """Test cases for the NeighborhoodLabel (NCP) prediction set constructor."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {"patient_id": "p0", "visit_id": "v0", "conditions": ["c1"], "procedures": [1.0], "label": 0},
            {"patient_id": "p1", "visit_id": "v1", "conditions": ["c2"], "procedures": [2.0], "label": 1},
            {"patient_id": "p2", "visit_id": "v2", "conditions": ["c3"], "procedures": [3.0], "label": 2},
            {"patient_id": "p3", "visit_id": "v3", "conditions": ["c4"], "procedures": [1.5], "label": 0},
            {"patient_id": "p4", "visit_id": "v4", "conditions": ["c5"], "procedures": [2.5], "label": 1},
            {"patient_id": "p5", "visit_id": "v5", "conditions": ["c6"], "procedures": [3.5], "label": 2},
        ]
        self.input_schema = {"conditions": "sequence", "procedures": "tensor"}
        self.output_schema = {"label": "multiclass"}
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )
        self.model.eval()

    def _get_embeddings(self, dataset):
        return extract_embeddings(self.model, dataset, batch_size=32, device="cpu")

    def test_initialization(self):
        ncp = NeighborhoodLabel(
            model=self.model,
            alpha=0.1,
            k_neighbors=5,
            lambda_L=100.0,
        )
        self.assertIsInstance(ncp, NeighborhoodLabel)
        self.assertEqual(ncp.mode, "multiclass")
        self.assertEqual(ncp.alpha, 0.1)
        self.assertEqual(ncp.k_neighbors, 5)
        self.assertEqual(ncp.lambda_L, 100.0)
        self.assertIsNone(ncp.cal_embeddings_)
        self.assertIsNone(ncp.cal_conformity_scores_)

    def test_initialization_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            NeighborhoodLabel(model=self.model, alpha=0.0, k_neighbors=5)
        with self.assertRaises(ValueError):
            NeighborhoodLabel(model=self.model, alpha=1.0, k_neighbors=5)
        with self.assertRaises(ValueError):
            NeighborhoodLabel(model=self.model, alpha=-0.1, k_neighbors=5)

    def test_initialization_invalid_k_neighbors_raises(self):
        with self.assertRaises(ValueError):
            NeighborhoodLabel(model=self.model, alpha=0.1, k_neighbors=0)
        with self.assertRaises(ValueError):
            NeighborhoodLabel(model=self.model, alpha=0.1, k_neighbors=-1)
        with self.assertRaises(ValueError):
            NeighborhoodLabel(model=self.model, alpha=0.1, k_neighbors=2.5)

    def test_initialization_non_multiclass_raises(self):
        binary_samples = [
            {"patient_id": "a", "visit_id": "a", "conditions": ["c"], "procedures": [1.0], "label": 0},
            {"patient_id": "b", "visit_id": "b", "conditions": ["d"], "procedures": [2.0], "label": 1},
        ]
        binary_ds = create_sample_dataset(
            samples=binary_samples,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test",
        )
        binary_model = MLP(
            dataset=binary_ds, feature_keys=["conditions"], label_key="label", mode="binary"
        )
        with self.assertRaises(NotImplementedError):
            NeighborhoodLabel(model=binary_model, alpha=0.1, k_neighbors=2)

    def test_calibrate_and_forward_returns_predset(self):
        ncp = NeighborhoodLabel(model=self.model, alpha=0.2, k_neighbors=3, lambda_L=50.0)
        cal_indices = [3, 4, 5]
        cal_dataset = self.dataset.subset(cal_indices)
        cal_embeddings = self._get_embeddings(cal_dataset)
        ncp.calibrate(cal_dataset=cal_dataset, cal_embeddings=cal_embeddings)

        self.assertIsNotNone(ncp.cal_embeddings_)
        self.assertIsNotNone(ncp.cal_conformity_scores_)
        self.assertEqual(ncp.cal_conformity_scores_.shape[0], 3)

        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(test_loader))
        with torch.no_grad():
            out = ncp(**batch)

        self.assertIn("y_predset", out)
        self.assertIn("y_prob", out)
        self.assertEqual(out["y_predset"].dtype, torch.bool)
        self.assertEqual(out["y_predset"].shape, out["y_prob"].shape)

    def test_forward_before_calibration_raises(self):
        ncp = NeighborhoodLabel(model=self.model, alpha=0.1, k_neighbors=5)
        loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                ncp(**batch)

    def test_prediction_sets_nonempty_batch(self):
        ncp = NeighborhoodLabel(model=self.model, alpha=0.3, k_neighbors=2, lambda_L=100.0)
        cal_dataset = self.dataset.subset([2, 3, 4, 5])
        cal_emb = self._get_embeddings(cal_dataset)
        ncp.calibrate(cal_dataset=cal_dataset, cal_embeddings=cal_emb)

        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                out = ncp(**batch)
                set_sizes = out["y_predset"].sum(dim=1)
                self.assertTrue(torch.all(set_sizes > 0), "Prediction sets should be non-empty")

    def test_calibrate_without_embeddings_extracts(self):
        ncp = NeighborhoodLabel(model=self.model, alpha=0.1, k_neighbors=2)
        cal_dataset = self.dataset.subset([3, 4, 5])
        ncp.calibrate(cal_dataset=cal_dataset, batch_size=2)
        self.assertIsNotNone(ncp.cal_embeddings_)
        self.assertIsNotNone(ncp.cal_conformity_scores_)

    def test_calibration_empirical_coverage_at_least_1_minus_alpha(self):
        """After calibrate(), empirical coverage on calibration set >= 1-alpha."""
        from pyhealth.calib.predictionset.base_conformal import _query_weighted_quantile

        ncp = NeighborhoodLabel(model=self.model, alpha=0.2, k_neighbors=3, lambda_L=50.0)
        cal_indices = [0, 1, 2, 3, 4, 5]
        cal_dataset = self.dataset.subset(cal_indices)
        cal_emb = self._get_embeddings(cal_dataset)
        ncp.calibrate(cal_dataset=cal_dataset, cal_embeddings=cal_emb)

        self.assertIsNotNone(ncp.alpha_tilde_)
        self.assertGreaterEqual(ncp.alpha_tilde_, 0.0)
        self.assertLessEqual(ncp.alpha_tilde_, 1.0)

        # Recompute per-sample thresholds using alpha_tilde (Q^NCP definition: alpha_tilde-quantile of conformity)
        N = ncp.cal_conformity_scores_.shape[0]
        k = min(ncp.k_neighbors, N)
        distances_cal, indices_cal = ncp._nn.kneighbors(
            ncp.cal_embeddings_, n_neighbors=k
        )
        cal_weights = np.exp(-distances_cal / ncp.lambda_L)
        cal_weights = cal_weights / cal_weights.sum(axis=1, keepdims=True)

        covered = 0
        for i in range(N):
            t_i = _query_weighted_quantile(
                ncp.cal_conformity_scores_[indices_cal[i]],
                ncp.alpha_tilde_,
                cal_weights[i],
            )
            # Covered = true label in set = conformity_i >= threshold_i (paper: V_i <= t in non-conf space)
            if ncp.cal_conformity_scores_[i] >= t_i:
                covered += 1
        empirical_coverage = covered / N
        self.assertGreaterEqual(
            empirical_coverage,
            1.0 - ncp.alpha - 1e-6,
            msg=f"Calibration empirical coverage {empirical_coverage:.4f} should be >= 1-alpha={1 - ncp.alpha}",
        )


if __name__ == "__main__":
    unittest.main()
