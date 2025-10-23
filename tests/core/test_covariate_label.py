import unittest
import numpy as np
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.calib.predictionset.covariate import CovariateLabel, fit_kde
from pyhealth.calib.utils import extract_embeddings


class TestCovariateLabel(unittest.TestCase):
    """Test cases for the CovariateLabel prediction set constructor."""

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
        self.dataset = SampleDataset(
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

        # Create dummy KDE estimators that return uniform densities
        def dummy_kde_cal(data):
            """Dummy calibration KDE that returns uniform density."""
            if isinstance(data, (list, np.ndarray)):
                return np.ones(len(data))
            return np.array([1.0])

        def dummy_kde_test(data):
            """Dummy test KDE that returns slightly different density."""
            if isinstance(data, (list, np.ndarray)):
                return np.ones(len(data)) * 1.1
            return np.array([1.1])

        self.kde_cal = dummy_kde_cal
        self.kde_test = dummy_kde_test

    def _get_embeddings(self, dataset):
        """Helper to extract embeddings from dataset."""
        return extract_embeddings(self.model, dataset, batch_size=32, device="cpu")

    def test_initialization(self):
        """Test that CovariateLabel initializes correctly."""
        cal_model = CovariateLabel(
            model=self.model,
            alpha=0.1,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        self.assertIsInstance(cal_model, CovariateLabel)
        self.assertEqual(cal_model.mode, "multiclass")
        self.assertEqual(cal_model.alpha, 0.1)
        self.assertIsNone(cal_model.t)
        self.assertIsNone(cal_model._sum_cal_weights)

    def test_initialization_with_array_alpha(self):
        """Test initialization with class-specific alpha values."""
        alpha_per_class = [0.1, 0.15, 0.2]
        cal_model = CovariateLabel(
            model=self.model,
            alpha=alpha_per_class,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        self.assertIsInstance(cal_model.alpha, np.ndarray)
        np.testing.assert_array_equal(cal_model.alpha, alpha_per_class)

    def test_initialization_non_multiclass_raises_error(self):
        """Test that non-multiclass models raise an error."""
        # Create a binary classification dataset with both labels
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
        binary_dataset = SampleDataset(
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
            CovariateLabel(
                model=binary_model,
                alpha=0.1,
                kde_test=self.kde_test,
                kde_cal=self.kde_cal,
            )

    def test_calibrate_marginal(self):
        """Test calibration with marginal coverage (single alpha)."""
        cal_model = CovariateLabel(
            model=self.model,
            alpha=0.3,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        # Calibrate on first 4 samples
        cal_indices = [0, 1, 2, 3]
        cal_dataset = torch.utils.data.Subset(self.dataset, cal_indices)

        # Extract embeddings
        cal_embeddings = self._get_embeddings(cal_dataset)
        test_embeddings = self._get_embeddings(self.dataset)

        cal_model.calibrate(
            cal_dataset=cal_dataset,
            cal_embeddings=cal_embeddings,
            test_embeddings=test_embeddings,
        )

        # Check that threshold is set
        self.assertIsNotNone(cal_model.t)
        self.assertIsInstance(cal_model.t, torch.Tensor)
        self.assertEqual(cal_model.t.dim(), 0)  # scalar threshold

        # Check that sum of calibration weights is set
        self.assertIsNotNone(cal_model._sum_cal_weights)
        self.assertGreater(cal_model._sum_cal_weights, 0)

    def test_calibrate_class_conditional(self):
        """Test calibration with class-conditional coverage."""
        alpha_per_class = [0.2, 0.25, 0.3]
        cal_model = CovariateLabel(
            model=self.model,
            alpha=alpha_per_class,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        # Extract embeddings
        cal_embeddings = self._get_embeddings(self.dataset)
        test_embeddings = self._get_embeddings(self.dataset)

        # Calibrate on all samples
        cal_model.calibrate(
            cal_dataset=self.dataset,
            cal_embeddings=cal_embeddings,
            test_embeddings=test_embeddings,
        )

        # Check that thresholds are set (one per class)
        self.assertIsNotNone(cal_model.t)
        self.assertIsInstance(cal_model.t, torch.Tensor)
        self.assertEqual(cal_model.t.shape[0], 3)  # 3 classes

    def test_forward_returns_predset(self):
        """Test that forward pass returns prediction sets."""
        cal_model = CovariateLabel(
            model=self.model,
            alpha=0.2,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        # Calibrate
        cal_indices = [0, 1, 2, 3]
        cal_dataset = torch.utils.data.Subset(self.dataset, cal_indices)

        # Extract embeddings
        cal_embeddings = self._get_embeddings(cal_dataset)
        test_embeddings = self._get_embeddings(self.dataset)

        cal_model.calibrate(
            cal_dataset=cal_dataset,
            cal_embeddings=cal_embeddings,
            test_embeddings=test_embeddings,
        )

        # Test forward pass
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))

        with torch.no_grad():
            output = cal_model(**data_batch)

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
        cal_model = CovariateLabel(
            model=self.model,
            alpha=0.3,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        # Calibrate on first 4 samples
        cal_indices = [0, 1, 2, 3]
        cal_dataset = torch.utils.data.Subset(self.dataset, cal_indices)

        # Extract embeddings
        cal_embeddings = self._get_embeddings(cal_dataset)
        test_embeddings = self._get_embeddings(self.dataset)

        cal_model.calibrate(
            cal_dataset=cal_dataset,
            cal_embeddings=cal_embeddings,
            test_embeddings=test_embeddings,
        )

        # Test on remaining samples
        test_indices = [4, 5]
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        test_loader = get_dataloader(test_dataset, batch_size=2, shuffle=False)

        with torch.no_grad():
            for data_batch in test_loader:
                output = cal_model(**data_batch)
                # Each example should have at least one class in prediction set
                set_sizes = output["y_predset"].sum(dim=1)
                self.assertTrue(
                    torch.all(set_sizes > 0), "Some prediction sets are empty"
                )

    def test_weighted_quantile_function(self):
        """Test the weighted quantile helper function."""
        from pyhealth.calib.predictionset.covariate.covariate_label import (
            _query_weighted_quantile,
        )

        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        alpha = 0.5

        quantile = _query_weighted_quantile(scores, alpha, weights)

        self.assertIsInstance(quantile, (float, np.floating))
        self.assertGreaterEqual(quantile, scores.min())
        self.assertLessEqual(quantile, scores.max())

    def test_likelihood_ratio_function(self):
        """Test the likelihood ratio computation."""
        from pyhealth.calib.predictionset.covariate.covariate_label import (
            _compute_likelihood_ratio,
        )

        # Create dummy data
        data = np.array([["a", "b"], ["c", "d"]])

        # Define simple KDEs
        def kde1(x):
            return np.ones(len(x)) * 2.0

        def kde2(x):
            return np.ones(len(x)) * 1.0

        ratios = _compute_likelihood_ratio(kde1, kde2, data)

        self.assertEqual(len(ratios), 2)
        self.assertTrue(np.all(ratios > 0))
        # Should be approximately 2.0 / 1.0 = 2.0
        np.testing.assert_allclose(ratios, 2.0, rtol=1e-5)

    def test_edge_case_empty_class(self):
        """Test calibration with a class that has no samples."""
        # This tests the edge case handling in class-conditional calibration
        alpha_per_class = [0.1, 0.2, 0.3]
        cal_model = CovariateLabel(
            model=self.model,
            alpha=alpha_per_class,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        # Use subset that doesn't have all classes
        cal_indices = [0, 1]  # Only classes 0 and 1
        cal_dataset = torch.utils.data.Subset(self.dataset, cal_indices)

        # Extract embeddings
        cal_embeddings = self._get_embeddings(cal_dataset)
        test_embeddings = self._get_embeddings(self.dataset)

        cal_model.calibrate(
            cal_dataset=cal_dataset,
            cal_embeddings=cal_embeddings,
            test_embeddings=test_embeddings,
        )

    def test_model_device_handling(self):
        """Test that the calibrator handles device correctly."""
        # This test assumes CPU; modify if GPU is available
        device = self.model.device

        cal_model = CovariateLabel(
            model=self.model,
            alpha=0.2,
            kde_test=self.kde_test,
            kde_cal=self.kde_cal,
        )

        # Extract embeddings
        cal_embeddings = self._get_embeddings(self.dataset)
        test_embeddings = self._get_embeddings(self.dataset)

        cal_model.calibrate(
            cal_dataset=self.dataset,
            cal_embeddings=cal_embeddings,
            test_embeddings=test_embeddings,
        )

        # Check that threshold is on the same device as the model
        self.assertEqual(cal_model.t.device.type, device.type)
        self.assertEqual(cal_model.device.type, device.type)


if __name__ == "__main__":
    unittest.main()
