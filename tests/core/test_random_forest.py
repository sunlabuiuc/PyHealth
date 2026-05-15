"""
Unit tests for the RandomForest model.

Tests cover:
- Model initialization
- Classification and regression behavior
- Forward pass (fitted vs. unfitted)
- Model parameters
- Expected no backward/gradients
"""
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RandomForest


class TestRandomForest(unittest.TestCase):
    """Test cases for the Random Forest model."""

    def setUp(self):
        """Set up minimal synthetic test data."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "label": 1,
            },
        ]

        self.batch_size = len(self.samples)
        self.num_samples = len(self.samples)
        self.num_classes = len(set(sample["label"] for sample in self.samples))

        self.input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        self.output_schema = {"label": "binary"}
        self.regression_output_schema = {"label": "regression"}

        self.dataset = create_sample_dataset(
            samples = self.samples,
            input_schema = self.input_schema,
            output_schema = self.output_schema,
            dataset_name = "test",
        )
        self.dataset_regression = create_sample_dataset(
            samples = self.samples,
            input_schema = self.input_schema,
            output_schema = self.regression_output_schema,
            dataset_name = "test",
        )

        # models are instantiated within individual test cases

    def test_model_initialization_classification(self):
        """Test model initialization using default parameters"""

        model = RandomForest(
            dataset = self.dataset
        )

        expected = {
            "pad_batches": True,
            "f1_average": "macro",
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None,
            "monotonic_cst": None
        }

        self.assertIsInstance(model, RandomForest)
        with self.assertRaises(AttributeError):
            # Random forests do not have embeddings
            self.assertEqual(model.embedding_dim, None)
        self.assertEqual(len(model.feature_keys), 2)
        self.assertIn("conditions", model.feature_keys)
        self.assertIn("procedures", model.feature_keys)
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.get_params(), expected)

    def test_model_initialization_regression(self):
        """Test regression model initialization with default params"""

        model = RandomForest(
            dataset = self.dataset_regression
        )

        expected = {
            "pad_batches": True,
            "f1_average": "macro",
            "n_estimators": 100,
            "criterion": "friedman_mse",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False,
            "ccp_alpha": 0.0,
            "max_samples": None,
            "monotonic_cst": None
        }

        self.assertIsInstance(model, RandomForest)
        with self.assertRaises(AttributeError):
            # Random forests do not have embeddings
            self.assertEqual(model.embedding_dim, None)
        self.assertEqual(len(model.feature_keys), 2)
        self.assertIn("conditions", model.feature_keys)
        self.assertIn("procedures", model.feature_keys)
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.get_params(), expected)

    def test_model_initialization_non_defaults(self):
        """Test regression model initialization with custom parameters"""

        model = RandomForest(
            dataset = self.dataset_regression,
            pad_batches = False,
            f1_average = "weighted",
            n_estimators = 50,
            criterion = "poisson",
            max_depth = 10,
            min_samples_split = 5,
            min_samples_leaf = 2,
            min_weight_fraction_leaf = 0.1,
            max_features = "log2",
            max_leaf_nodes = 20,
            min_impurity_decrease = 0.01,
            bootstrap = False,
            oob_score = True,
            n_jobs = 1,
            random_state = 123,
            verbose = 1,
            warm_start = True,
            class_weight = "balanced",
            ccp_alpha = 0.05,
            max_samples = 2
        )

        expected = {
            "pad_batches": False,
            "f1_average": "weighted",
            "n_estimators": 50,
            "criterion": "poisson",
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "min_weight_fraction_leaf": 0.1,
            "max_features": "log2",
            "max_leaf_nodes": 20,
            "min_impurity_decrease": 0.01,
            "bootstrap": False,
            "oob_score": True,
            "n_jobs": 1,
            "random_state": 123,
            "verbose": 1,
            "warm_start": True,
            "ccp_alpha": 0.05,
            "max_samples": 2,
            "monotonic_cst": None
        }

        self.assertIsInstance(model, RandomForest)
        with self.assertRaises(AttributeError):
            # Random forests do not have embeddings
            self.assertEqual(model.embedding_dim, None)
        self.assertEqual(len(model.feature_keys), 2)
        self.assertIn("conditions", model.feature_keys)
        self.assertIn("procedures", model.feature_keys)
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.get_params(), expected)

    def test_model_initialization_invalid_f1_average(self):
        with self.assertRaises(ValueError):
            RandomForest(dataset = self.dataset, f1_average = "any")

    def test_model_initialization_invalid_criterion(self):
        with self.assertRaises(ValueError):
            RandomForest(dataset = self.dataset, criterion = "invalid")

    def test_forward_unfitted_classification(self):
        """
        Tests forward call on the Random Forest model tasked for classification without
        having called fit, which is expected to elicit a runtime error
        """
        model = RandomForest(
            dataset = self.dataset)

        loader = get_dataloader(self.dataset, batch_size = self.batch_size,
                                shuffle = False)
        batch = next(iter(loader))

        # Calling forward without fitting should cause a RuntimeError to be raised
        with self.assertRaises(RuntimeError):
            model(**batch)

    def test_forward_unfitted_regression(self):
        """
        Tests forward call on the Random Forest model tasked for regression without
        having called fit, which is expected to elicit a runtime error
        """
        model = RandomForest(
            dataset = self.dataset_regression
        )

        loader = get_dataloader(self.dataset_regression, batch_size = self.batch_size,
                                shuffle = False)
        batch = next(iter(loader))

        # Calling forward without fitting should cause a RuntimeError to be raised
        with self.assertRaises(RuntimeError):
            model(**batch)

    def test_forward_fitted_classification(self):
        """
        Tests forward call on the Random Forest model tasked for classification with
        fit called
        """
        model = RandomForest(
            dataset = self.dataset)

        loader = get_dataloader(
            self.dataset, batch_size = self.batch_size, shuffle = False)

        model.fit(loader)

        batch = next(iter(loader))
        out = model(**batch)

        # Verify all expected keys exist in forward output: loss, y_prob, y_true, logit
        self.assertIn("loss",   out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit",  out)

        # Verify output shapes
        self.assertEqual(out["logit"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(out["y_prob"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(out["y_true"].shape[0], self.batch_size)

    def test_forward_fitted_regression(self):
        """
        Tests forward call on the Random Forest model tasked for regression with
        fit called
        """
        model = RandomForest(
            dataset = self.dataset_regression
        )

        loader = get_dataloader(
            self.dataset_regression, batch_size = self.batch_size, shuffle = False)

        model.fit(loader)

        batch = next(iter(loader))
        out = model(**batch)

        # Verify all expected keys exist in forward output: loss, y_prob, y_true, logit
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)

        # Verify output shapes
        self.assertEqual(out["logit"].shape, (self.batch_size, 1))
        self.assertEqual(out["y_prob"].shape, (self.batch_size, 1))
        self.assertEqual(out["y_true"].shape[0], self.batch_size)

    def test_backward(self):
        """Test that the Random Forest model does not support a backward pass as
        expected."""

        model = RandomForest(dataset = self.dataset)

        # Create data loader
        train_loader = get_dataloader(
            self.dataset, batch_size = self.batch_size, shuffle = True)
        data_batch = next(iter(train_loader))

        # Fit and forward pass
        model.fit(train_loader)
        out = model(**data_batch)

        # Backward is not applicable for random forest models. Verify a runtime error
        # is raised when a backward pass is attempted
        with self.assertRaises(RuntimeError):
            out["loss"].backward()

        # Check that no gradients are required
        has_gradient = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertFalse(has_gradient)

    def test_fit_and_predict_pipeline(self):
        """Test fitting and inference produce probability outputs."""
        model = RandomForest(dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size = self.batch_size, shuffle=False)

        model.fit(loader)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(**batch)

        self.assertIn("y_prob", out)

    def test_multiple_batches(self):
        """Test model processes all batches and returns outputs per sample."""
        model = RandomForest(dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size = 1, shuffle=False)

        model.fit(loader)

        outputs = []
        for batch in loader:
            outputs.append(model(**batch)["y_prob"])

        self.assertEqual(len(outputs), self.num_samples)


if __name__ == "__main__":
    unittest.main()
