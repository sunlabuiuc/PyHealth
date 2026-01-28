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
        # Create samples with 3 classes for multiclass classification
        self.samples = [
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

    def test_calibrate_marginal(self):
        """Test calibration with marginal coverage (single alpha)."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.3,
            n_clusters=3,
            random_state=42,
        )

        # Split into train and cal sets
        train_indices = [0, 1, 2]
        cal_indices = [3, 4, 5]
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
        self.assertEqual(cluster_model.kmeans_model.n_clusters, 3)

        # Check that cluster thresholds are set
        self.assertIsNotNone(cluster_model.cluster_thresholds)
        self.assertIsInstance(cluster_model.cluster_thresholds, dict)
        self.assertEqual(len(cluster_model.cluster_thresholds), 3)

        # Check that each cluster has a threshold
        for cluster_id in range(3):
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

        # Split into train and cal sets
        train_indices = [0, 1, 2]
        cal_indices = [3, 4, 5]
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
            n_clusters=3,
            random_state=42,
        )

        # Calibrate
        train_indices = [0, 1, 2]
        cal_indices = [3, 4, 5]
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
        train_indices = [0, 1, 2]
        cal_indices = [3, 4, 5]
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

    def test_calibrate_requires_train_embeddings(self):
        """Test that calibrate requires train_embeddings."""
        cluster_model = ClusterLabel(
            model=self.model,
            alpha=0.2,
            n_clusters=3,
        )

        cal_indices = [3, 4, 5]
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
        """Test that different cluster counts work."""
        for n_clusters in [2, 3, 4]:
            cluster_model = ClusterLabel(
                model=self.model,
                alpha=0.2,
                n_clusters=n_clusters,
                random_state=42,
            )

            train_indices = [0, 1, 2]
            cal_indices = [3, 4, 5]
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
            n_clusters=3,
            random_state=42,
        )

        train_indices = [0, 1, 2]
        cal_indices = [3, 4, 5]
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


if __name__ == "__main__":
    unittest.main()
