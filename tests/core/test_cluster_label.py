import unittest
import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.calib.predictionset.cluster import ClusterLabel
from pyhealth.calib.utils import extract_embeddings


class TestClusterLabel(unittest.TestCase):
    """Test cases for the ClusterLabel prediction set constructor."""

    def setUp(self):
        """Set up test data and model."""
        # Stabilize model initialization and downstream calibration behavior.
        np.random.seed(42)
        torch.manual_seed(42)

        # Create samples with 3 classes for multiclass classification.
        # 12 samples (2 per class for train, 2 per class for cal) ensure
        # calibration samples are spread across clusters during K-means.
        self.samples = [
            # --- train set (indices 0–5) ---
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4.0],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4.0],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ["cond-10", "cond-20", "cond-30"],
                "procedures": [2.0, 3.0, 4.5, 5.0],
                "label": 2,
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-3",
                "conditions": ["cond-40", "cond-50"],
                "procedures": [1.5, 2.5, 3.0, 4.5],
                "label": 0,
            },
            {
                "patient_id": "patient-4",
                "visit_id": "visit-4",
                "conditions": ["cond-60", "cond-70", "cond-80"],
                "procedures": [3.0, 4.0, 5.0, 6.0],
                "label": 1,
            },
            {
                "patient_id": "patient-5",
                "visit_id": "visit-5",
                "conditions": ["cond-90", "cond-100"],
                "procedures": [2.5, 3.5, 4.0, 5.5],
                "label": 2,
            },
            # --- calibration set (indices 6–11) ---
            {
                "patient_id": "patient-6",
                "visit_id": "visit-6",
                "conditions": ["cond-11", "cond-22", "cond-33"],
                "procedures": [6.0, 1.0, 2.5, 3.5],
                "label": 0,
            },
            {
                "patient_id": "patient-7",
                "visit_id": "visit-7",
                "conditions": ["cond-44", "cond-55", "cond-66"],
                "procedures": [4.5, 5.5, 1.5, 2.0],
                "label": 1,
            },
            {
                "patient_id": "patient-8",
                "visit_id": "visit-8",
                "conditions": ["cond-77", "cond-88"],
                "procedures": [3.5, 6.5, 2.0, 1.0],
                "label": 2,
            },
            {
                "patient_id": "patient-9",
                "visit_id": "visit-9",
                "conditions": ["cond-15", "cond-25", "cond-35", "cond-45"],
                "procedures": [7.0, 1.5, 3.0, 4.0],
                "label": 0,
            },
            {
                "patient_id": "patient-10",
                "visit_id": "visit-10",
                "conditions": ["cond-55", "cond-65"],
                "procedures": [2.0, 5.0, 6.5, 1.5],
                "label": 1,
            },
            {
                "patient_id": "patient-11",
                "visit_id": "visit-11",
                "conditions": ["cond-75", "cond-85", "cond-95"],
                "procedures": [5.5, 2.5, 1.0, 3.0],
                "label": 2,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        self.output_schema = {"label": "multiclass"}

        # Create dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create and set up base model
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )
        self.model.eval()

    def _get_embeddings(self, dataset):
        """Helper to extract embeddings from dataset."""
        return extract_embeddings(self.model, dataset, batch_size=32, device="cpu")

    def test_initialization(self):
        """Test that ClusterLabel initializes correctly."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.1,
            n_clusters=3,
            random_state=42,
        )

        self.assertIsInstance(cluster_model, ClusterLabel)
        self.assertEqual(cluster_model.mode, "multiclass")
        self.assertEqual(cluster_model.alpha, 0.1)
        self.assertEqual(cluster_model.n_clusters, 3)
        self.assertEqual(cluster_model.random_state, 42)
        self.assertIsNone(cluster_model.kmeans_model)
        self.assertIsNone(cluster_model.cluster_thresholds)

    def test_initialization_with_array_alpha(self):
        """Test initialization with class-specific alpha values."""
        alpha_per_class = [0.1, 0.15, 0.2]
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=alpha_per_class,
            n_clusters=3,
        )

        self.assertIsInstance(cluster_model.alpha, np.ndarray)
        np.testing.assert_array_equal(cluster_model.alpha, alpha_per_class)

    def test_initialization_non_multiclass_raises_error(self):
        """Test that non-multiclass models raise an error."""
        # Create a binary classification dataset
        binary_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-1"],
                "procedures": [1.0],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-2"],
                "procedures": [2.0],
                "label": 1,
            },
        ]
        binary_dataset = create_sample_dataset(
            samples=binary_samples,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test",
        )
        binary_model = MLP(
            dataset=binary_dataset,
            feature_keys=["conditions"],
            label_key="label",
            mode="binary",
        )

        with self.assertRaises(NotImplementedError):
            ClusterLabel(
                model=binary_model,
                alpha=0.1,
                n_clusters=2,
            )

    def test_initialization_invalid_n_clusters_raises_error(self):
        """Test that invalid n_clusters (non-positive or non-int) raises ValueError."""
        with self.assertRaises(ValueError):
            ClusterLabel(
                model=self.model,
                alpha=0.1,
                n_clusters=0,
            )
        with self.assertRaises(ValueError):
            ClusterLabel(
                model=self.model,
                alpha=0.1,
                n_clusters=-1,
            )
        with self.assertRaises(ValueError):
            ClusterLabel(
                model=self.model,
                alpha=0.1,
                n_clusters=2.5,
            )

    def test_calibrate_marginal(self):
        """Test calibration with marginal coverage (single alpha)."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.3,
            n_clusters=2,
            random_state=42,
        )

        # Split into train and cal sets (6 train, 6 cal)
        train_indices = [0, 1, 2, 3, 4, 5]
        cal_indices = [6, 7, 8, 9, 10, 11]
        train_dataset = self.dataset.subset(train_indices)
        cal_dataset = self.dataset.subset(cal_indices)

        # Extract embeddings
        train_embeddings = self._get_embeddings(train_dataset)
        cal_embeddings = self._get_embeddings(cal_dataset)

        cluster_model.calibrate(
            cal_dataset=cal_dataset,
            train_embeddings=train_embeddings,
            cal_embeddings=cal_embeddings,
        )

        # Check that K-means model is fitted
        self.assertIsNotNone(cluster_model.kmeans_model)
        self.assertEqual(cluster_model.kmeans_model.n_clusters, 2)

        # Check that cluster thresholds are set
        self.assertIsNotNone(cluster_model.cluster_thresholds)
        self.assertIsInstance(cluster_model.cluster_thresholds, dict)
        self.assertEqual(len(cluster_model.cluster_thresholds), 2)

        # Check that each cluster has a threshold
        for cluster_id in range(2):
            self.assertIn(cluster_id, cluster_model.cluster_thresholds)
            threshold = cluster_model.cluster_thresholds[cluster_id]
            self.assertIsInstance(threshold, (float, np.floating))

    def test_calibrate_class_conditional(self):
        """Test calibration with class-conditional coverage."""
        alpha_per_class = [0.2, 0.25, 0.3]
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=alpha_per_class,
            n_clusters=2,
            random_state=42,
        )

        # Split into train and cal sets (6 train, 6 cal)
        train_indices = [0, 1, 2, 3, 4, 5]
        cal_indices = [6, 7, 8, 9, 10, 11]
        train_dataset = self.dataset.subset(train_indices)
        cal_dataset = self.dataset.subset(cal_indices)

        # Extract embeddings
        train_embeddings = self._get_embeddings(train_dataset)
        cal_embeddings = self._get_embeddings(cal_dataset)

        cluster_model.calibrate(
            cal_dataset=cal_dataset,
            train_embeddings=train_embeddings,
            cal_embeddings=cal_embeddings,
        )

        # Check that cluster thresholds are set (one per class per cluster)
        self.assertIsNotNone(cluster_model.cluster_thresholds)
        for cluster_id in cluster_model.cluster_thresholds:
            threshold = cluster_model.cluster_thresholds[cluster_id]
            self.assertIsInstance(threshold, np.ndarray)
            self.assertEqual(len(threshold), 3)  # 3 classes

    def test_forward_returns_predset(self):
        """Test that forward pass returns prediction sets."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.2,
            n_clusters=2,
            random_state=42,
        )

        # Calibrate
        train_indices = [0, 1, 2, 3, 4, 5]
        cal_indices = [6, 7, 8, 9, 10, 11]
        train_dataset = self.dataset.subset(train_indices)
        cal_dataset = self.dataset.subset(cal_indices)

        train_embeddings = self._get_embeddings(train_dataset)
        cal_embeddings = self._get_embeddings(cal_dataset)

        cluster_model.calibrate(
            cal_dataset=cal_dataset,
            train_embeddings=train_embeddings,
            cal_embeddings=cal_embeddings,
        )

        # Test forward pass
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))

        with torch.no_grad():
            output = cluster_model(**data_batch)

        # Check output contains prediction set
        self.assertIn("y_predset", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)

        # Check prediction set is boolean
        self.assertEqual(output["y_predset"].dtype, torch.bool)

        # Check prediction set shape matches probability shape
        self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)

    def test_prediction_sets_nonempty(self):
        """Test that prediction sets are non-empty for most examples."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.3,
            n_clusters=2,
            random_state=42,
        )

        # Calibrate
        train_indices = [0, 1, 2, 3, 4, 5]
        cal_indices = [6, 7, 8, 9, 10, 11]
        train_dataset = self.dataset.subset(train_indices)
        cal_dataset = self.dataset.subset(cal_indices)

        train_embeddings = self._get_embeddings(train_dataset)
        cal_embeddings = self._get_embeddings(cal_dataset)

        cluster_model.calibrate(
            cal_dataset=cal_dataset,
            train_embeddings=train_embeddings,
            cal_embeddings=cal_embeddings,
        )

        # Test on all samples
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)

        with torch.no_grad():
            for data_batch in test_loader:
                output = cluster_model(**data_batch)
                # Each example should have at least one class in prediction set
                set_sizes = output["y_predset"].sum(dim=1)
                self.assertTrue(
                    torch.all(set_sizes > 0), "Some prediction sets are empty"
                )

    def test_score_type_aps_runs_end_to_end(self):
        """score_type='aps' should calibrate and produce non-empty,
        correctly-typed prediction sets, just like the default 'threshold'."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.3,
            n_clusters=2,
            random_state=42,
            score_type="aps",
        )

        train_indices = [0, 1, 2, 3, 4, 5]
        cal_indices = [6, 7, 8, 9, 10, 11]
        train_dataset = self.dataset.subset(train_indices)
        cal_dataset = self.dataset.subset(cal_indices)

        train_embeddings = self._get_embeddings(train_dataset)
        cal_embeddings = self._get_embeddings(cal_dataset)

        cluster_model.calibrate(
            cal_dataset=cal_dataset,
            train_embeddings=train_embeddings,
            cal_embeddings=cal_embeddings,
        )

        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for data_batch in test_loader:
                output = cluster_model(**data_batch)
                self.assertEqual(output["y_predset"].dtype, torch.bool)
                self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)
                set_sizes = output["y_predset"].sum(dim=1)
                self.assertTrue(torch.all(set_sizes > 0))

    def test_calibrate_requires_train_embeddings(self):
        """Test that calibrate requires train_embeddings."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.2,
            n_clusters=3,
        )

        cal_indices = [6, 7, 8, 9, 10, 11]
        cal_dataset = self.dataset.subset(cal_indices)
        cal_embeddings = self._get_embeddings(cal_dataset)

        with self.assertRaises(ValueError):
            cluster_model.calibrate(
                cal_dataset=cal_dataset,
                train_embeddings=None,
                cal_embeddings=cal_embeddings,
            )

    def test_forward_before_calibration_raises_error(self):
        """Test that forward pass raises error before calibration."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.2,
            n_clusters=3,
        )

        test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(test_loader))

        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                cluster_model(**data_batch)

    def test_different_cluster_counts(self):
        """Test that different cluster counts work.

        Cluster counts are capped at 3 (half of the 6 cal samples) so that
        K-means can reliably assign at least one calibration sample per cluster
        with a small test dataset.
        """
        for n_clusters in [2, 3]:
            cluster_model = ClusterLabel(
                model=self.model,
                alpha=0.2,
                n_clusters=n_clusters,
                random_state=42,
            )

            train_indices = [0, 1, 2, 3, 4, 5]
            cal_indices = [6, 7, 8, 9, 10, 11]
            train_dataset = self.dataset.subset(train_indices)
            cal_dataset = self.dataset.subset(cal_indices)

            train_embeddings = self._get_embeddings(train_dataset)
            cal_embeddings = self._get_embeddings(cal_dataset)

            cluster_model.calibrate(
                cal_dataset=cal_dataset,
                train_embeddings=train_embeddings,
                cal_embeddings=cal_embeddings,
            )

            self.assertEqual(cluster_model.kmeans_model.n_clusters, n_clusters)
            self.assertEqual(len(cluster_model.cluster_thresholds), n_clusters)

    def test_model_device_handling(self):
        """Test that the calibrator handles device correctly."""
        device = self.model.device

        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.2,
            n_clusters=2,
            random_state=42,
        )

        train_indices = [0, 1, 2, 3, 4, 5]
        cal_indices = [6, 7, 8, 9, 10, 11]
        train_dataset = self.dataset.subset(train_indices)
        cal_dataset = self.dataset.subset(cal_indices)

        train_embeddings = self._get_embeddings(train_dataset)
        cal_embeddings = self._get_embeddings(cal_dataset)

        cluster_model.calibrate(
            cal_dataset=cal_dataset,
            train_embeddings=train_embeddings,
            cal_embeddings=cal_embeddings,
        )

        # Check that device is set correctly
        self.assertEqual(cluster_model.device.type, device.type)

        # Test forward pass and check output device
        test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(test_loader))

        with torch.no_grad():
            output = cluster_model(**data_batch)
            self.assertEqual(output["y_predset"].device.type, device.type)


class TestClusterLabelCoverage(unittest.TestCase):
    """Monte Carlo verification of ClusterLabel's core statistical claim:
    per-cluster (Mondrian) coverage, at scale a full trained-model pipeline
    can't practically reach. Exercises the same calibration logic
    ClusterLabel.calibrate()/forward() use (KMeans + _query_quantile),
    directly, the same way test_scores.py's TestScoresCoverage does for
    the shared score module.
    """

    def _run_trial(self, rng, n_train, n_cal, n_test, n_kmeans_clusters, alpha):
        from sklearn.cluster import KMeans
        from pyhealth.calib.predictionset.base_conformal import _query_quantile

        n_true_clusters = 3
        embed_dim = 5
        centers = rng.normal(scale=8.0, size=(n_true_clusters, embed_dim))
        beta_params = [(2, 8), (5, 5), (8, 2)]

        def sample(n):
            true_c = rng.integers(0, n_true_clusters, size=n)
            emb = centers[true_c] + rng.normal(scale=1.0, size=(n, embed_dim))
            scores = np.array([rng.beta(*beta_params[c]) for c in true_c])
            return emb, scores

        train_emb, _ = sample(n_train)
        cal_emb, cal_scores = sample(n_cal)
        test_emb, test_scores = sample(n_test)

        # Mirrors ClusterLabel.calibrate(): fit K-means on training
        # embeddings only, assign calibration/test points out-of-sample.
        km = KMeans(n_clusters=n_kmeans_clusters, random_state=0, n_init=10)
        km.fit(train_emb)
        cal_cluster = km.predict(cal_emb)
        test_cluster = km.predict(test_emb)

        thresholds = {}
        for c in range(n_kmeans_clusters):
            mask = cal_cluster == c
            thresholds[c] = (
                _query_quantile(cal_scores[mask], alpha) if mask.sum() > 0 else np.inf
            )
        t_test = np.array([thresholds[c] for c in test_cluster])
        return (test_scores <= t_test).mean()

    def test_per_cluster_coverage_matches_target(self):
        """The core claim: ClusterLabel's calibration logic (K-means fit on
        training embeddings only, calibration points assigned via
        .predict()) should achieve approximately the target 1-alpha
        coverage, matching the standard split-conformal quantile guarantee
        applied within each Mondrian category (cluster)."""
        rng = np.random.default_rng(42)
        alpha = 0.1
        coverages = [
            self._run_trial(rng, n_train=600, n_cal=300, n_test=2000,
                             n_kmeans_clusters=3, alpha=alpha)
            for _ in range(30)
        ]
        mean_coverage = np.mean(coverages)
        self.assertGreaterEqual(
            mean_coverage, 1 - alpha - 0.03,
            f"Mean coverage {mean_coverage:.4f} too far below target {1 - alpha}",
        )


class TestClusterLabelKMeansFitIsOutOfSample(unittest.TestCase):
    """Regression test: calibrate() must not let calibration data influence
    the K-means cluster boundaries used to assign calibration points' own
    thresholds. K-means must be fit on train_embeddings only, and
    calibration points assigned via out-of-sample .predict() -- the
    Mondrian conformal guarantee (Vovk, Lindsay, Nouretdinov, and Gammerman
    2003) requires the category function to be independent of the
    calibration data it's later evaluated against.
    """

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "conditions": [f"c{i}"],
                "procedures": [float(i % 3)],
                "label": i % 3,
            }
            for i in range(12)
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_cluster_out_of_sample",
        )
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )

    def test_kmeans_fit_receives_only_train_embeddings(self):
        from unittest.mock import patch
        from sklearn.cluster import KMeans

        train_ds = self.dataset.subset(list(range(6)))
        cal_ds = self.dataset.subset(list(range(6, 12)))
        train_embeddings = extract_embeddings(self.model, train_ds, batch_size=32)
        cal_embeddings = extract_embeddings(self.model, cal_ds, batch_size=32)

        cluster_predictor = ClusterLabel(model=self.model, alpha=0.2, n_clusters=2)

        with patch.object(KMeans, "fit", autospec=True) as mock_fit, \
             patch.object(KMeans, "predict", autospec=True, return_value=np.zeros(6, dtype=int)) as mock_predict:
            mock_fit.side_effect = lambda self, X, *a, **kw: setattr(
                self, "cluster_centers_", np.zeros((2, X.shape[1]))
            )
            cluster_predictor.calibrate(
                cal_dataset=cal_ds,
                train_embeddings=train_embeddings,
                cal_embeddings=cal_embeddings,
            )

        fit_X = mock_fit.call_args.args[1]
        self.assertEqual(fit_X.shape[0], len(train_embeddings))
        np.testing.assert_array_equal(fit_X, train_embeddings)

        predict_X = mock_predict.call_args.args[1]
        self.assertEqual(predict_X.shape[0], len(cal_embeddings))
        np.testing.assert_array_equal(predict_X, cal_embeddings)


if __name__ == "__main__":
    unittest.main()
