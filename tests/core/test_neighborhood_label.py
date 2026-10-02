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

    def test_score_type_aps_runs_end_to_end(self):
        """score_type='aps' should calibrate and produce non-empty,
        correctly-typed prediction sets, just like the default 'threshold'
        (NeighborhoodLabel always guarantees non-empty sets via its own
        argmax fallback, independent of score_type)."""
        ncp = NeighborhoodLabel(
            model=self.model, alpha=0.3, k_neighbors=2, lambda_L=100.0,
            score_type="aps", random_state=42,
        )
        cal_dataset = self.dataset.subset([2, 3, 4, 5])
        cal_emb = self._get_embeddings(cal_dataset)
        ncp.calibrate(cal_dataset=cal_dataset, cal_embeddings=cal_emb)

        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                out = ncp(**batch)
                self.assertEqual(out["y_predset"].dtype, torch.bool)
                self.assertEqual(out["y_predset"].shape, out["y_prob"].shape)
                set_sizes = out["y_predset"].sum(dim=1)
                self.assertTrue(torch.all(set_sizes > 0))

    def test_calibrate_without_embeddings_extracts(self):
        ncp = NeighborhoodLabel(model=self.model, alpha=0.1, k_neighbors=2)
        cal_dataset = self.dataset.subset([3, 4, 5])
        ncp.calibrate(cal_dataset=cal_dataset, batch_size=2)
        self.assertIsNotNone(ncp.cal_embeddings_)
        self.assertIsNotNone(ncp.cal_conformity_scores_)

    def test_calibration_empirical_coverage_at_least_1_minus_alpha(self):
        """After calibrate(), empirical coverage on calibration set >= 1-alpha,
        recomputed the same leave-one-out way calibrate() itself does (a
        calibration point's own score must never appear in its own
        neighbor set -- see test_calibrate_excludes_self_from_neighbors for
        why: querying kneighbors() with an explicit X equal to the fitted
        set makes each point its own nearest neighbor at distance 0, which
        would leak a point's own score into its own threshold and trivially
        inflate this exact coverage check if not excluded).

        Uses its own seeded, larger (N=30), *trained* calibration set rather
        than the shared 6-sample untrained fixture. Two independent issues
        made the original version flaky/wrong, not just one:

        1. With only 6 points, achievable coverage values are exactly
           {0, 1/6, ..., 1}, so the target 1-alpha=0.8 sits squarely between
           4/6=0.667 and 5/6=0.833 -- a single point tipping either way
           (driven only by the model's unseeded random init, since setUp()
           never seeds torch) flips the assertion.
        2. More fundamentally: the shared fixture's model is never trained,
           so its predicted probabilities -- and therefore the conformity
           scores NCP calibrates on -- are just noise from a random init,
           uncorrelated with embedding-space locality. NCP's per-point
           threshold comes from each point's k-nearest *neighbors only*
           (itself excluded), not a global quantile over all N points the
           way plain split conformal's is, so it has no automatic "any
           alpha_tilde >= 0 must reach 1-alpha coverage" guarantee the way
           a global quantile would -- that guarantee is a property of the
           *underlying scores actually correlating with locality*, which an
           untrained model doesn't provide. Confirmed directly: with the
           untrained model, alpha_tilde_ converges to its floor of 0.0 (the
           search's most permissive setting) and still only covers 23/30,
           because roughly 1/(k+1) of points have a lower score than all k
           of their neighbors purely by chance when scores are pure noise.
           Training the model so scores genuinely reflect how "easy" each
           point is relative to its neighborhood removes this floor issue.

        N=30 gives a 3.3%-wide step size (fine enough not to sit on a knife
        edge), a fixed seed makes the outcome reproducible, and training
        makes the target coverage actually achievable by the method's own
        theory rather than accidentally testing an unmet precondition.
        """
        from pyhealth.calib.predictionset.base_conformal import _query_weighted_quantile

        torch.manual_seed(0)
        np.random.seed(0)

        n_per_class = 10
        samples = []
        for label in range(3):
            for j in range(n_per_class):
                idx = label * n_per_class + j
                samples.append(
                    {
                        "patient_id": f"p{idx}",
                        "visit_id": f"v{idx}",
                        "conditions": [f"c{idx}"],
                        "procedures": [float(label) + 0.01 * j],
                        "label": label,
                    }
                )
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_coverage",
        )
        model = MLP(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )

        # Brief training: this trivially class-separable fixture (procedures
        # cleanly bucketed by label) only needs a handful of epochs for
        # predicted probabilities -- and thus conformity scores -- to
        # reflect real structure instead of random-init noise.
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        train_loader = get_dataloader(dataset, batch_size=len(samples), shuffle=True)
        for _ in range(50):
            for batch in train_loader:
                optimizer.zero_grad()
                ret = model(**batch)
                ret["loss"].backward()
                optimizer.step()
        model.eval()

        ncp = NeighborhoodLabel(model=model, alpha=0.2, k_neighbors=3, lambda_L=50.0)
        cal_dataset = dataset.subset(list(range(len(samples))))
        cal_emb = extract_embeddings(model, cal_dataset, batch_size=32, device="cpu")
        ncp.calibrate(cal_dataset=cal_dataset, cal_embeddings=cal_emb)

        self.assertIsNotNone(ncp.alpha_tilde_)
        self.assertGreaterEqual(ncp.alpha_tilde_, 0.0)
        self.assertLessEqual(ncp.alpha_tilde_, 1.0)

        # Recompute per-sample thresholds using alpha_tilde (Q^NCP definition: alpha_tilde-quantile of conformity)
        N = ncp.cal_conformity_scores_.shape[0]
        k = min(ncp.k_neighbors, N)
        k_query = min(k + 1, N)
        distances_all, indices_all = ncp._nn.kneighbors(
            ncp.cal_embeddings_, n_neighbors=k_query
        )
        n_loo = k_query - 1
        distances_cal = np.zeros((N, n_loo))
        indices_cal = np.zeros((N, n_loo), dtype=int)
        for i in range(N):
            mask = indices_all[i] != i
            distances_cal[i] = distances_all[i][mask][:n_loo]
            indices_cal[i] = indices_all[i][mask][:n_loo]
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

    def test_calibrate_excludes_self_from_neighbors(self):
        """Regression test: calibrate()'s alpha_tilde search must not let a
        calibration point be its own nearest neighbor.

        Querying sklearn's NearestNeighbors.kneighbors() with an explicit X
        argument equal to the fitted set returns each point as its own
        nearest neighbor at distance 0 (unlike the implicit no-argument
        form, which sklearn special-cases to exclude self-matches). Without
        excluding this self-match, a calibration point's own score leaks
        into its own threshold computation during calibration -- something
        a genuine test point (never part of the calibration set) can't
        benefit from -- which biases the alpha_tilde search toward an
        overly permissive threshold and causes real under-coverage at test
        time (empirically ~0.82-0.83 actual vs. 0.90 target in isolated
        simulation, before this fix).
        """
        ncp = NeighborhoodLabel(model=self.model, alpha=0.2, k_neighbors=3, lambda_L=50.0)
        cal_dataset = self.dataset.subset([0, 1, 2, 3, 4, 5])
        cal_emb = self._get_embeddings(cal_dataset)
        ncp.calibrate(cal_dataset=cal_dataset, cal_embeddings=cal_emb)

        N = ncp.cal_conformity_scores_.shape[0]
        k = min(ncp.k_neighbors, N)
        k_query = min(k + 1, N)
        _, indices_all = ncp._nn.kneighbors(ncp.cal_embeddings_, n_neighbors=k_query)

        for i in range(N):
            with self.subTest(point=i):
                mask = indices_all[i] != i
                kept = indices_all[i][mask][: k_query - 1]
                self.assertNotIn(
                    i,
                    kept,
                    f"calibration point {i} leaked into its own neighbor set",
                )


if __name__ == "__main__":
    unittest.main()
