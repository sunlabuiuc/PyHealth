"""Tests for CaliForest model.

These tests use minimal synthetic data to ensure fast execution.
All tests should complete in milliseconds.
"""

import numpy as np
import pytest
from sklearn.utils.validation import NotFittedError

from pyhealth.models.califorest import CaliForest


# Shared minimal fixtures - created once, reused
@pytest.fixture(scope="module")
def minimal_data():
    """Minimal synthetic dataset - only 20 samples, 3 features."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted_model(minimal_data):
    """Pre-fitted model for prediction tests."""
    X, y = minimal_data
    model = CaliForest(n_estimators=5, random_state=42)  # Only 5 trees!
    model.fit(X, y)
    return model


class TestCaliForestInstantiation:
    """Tests for model instantiation - no fitting required."""

    def test_default_instantiation(self):
        """Test model instantiation with default parameters."""
        model = CaliForest()
        assert model.n_estimators == 100
        assert model.calibration_method == "isotonic"
        assert model.min_oob_trees == 1

    def test_custom_parameters(self):
        """Test model instantiation with custom parameters."""
        model = CaliForest(
            n_estimators=50,
            max_depth=10,
            calibration_method="platt",
            min_oob_trees=3,
            random_state=42,
        )
        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.calibration_method == "platt"
        assert model.min_oob_trees == 3


class TestCaliForestFit:
    """Tests for model fitting - use minimal data."""

    def test_fit_isotonic(self, minimal_data):
        """Test fitting with isotonic calibration."""
        X, y = minimal_data
        model = CaliForest(n_estimators=5, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "rf_")
        assert hasattr(model, "calibrator_")
        assert hasattr(model, "classes_")

    def test_fit_platt(self, minimal_data):
        """Test fitting with Platt scaling."""
        X, y = minimal_data
        model = CaliForest(
            n_estimators=5, calibration_method="platt", random_state=42
        )
        model.fit(X, y)

        assert hasattr(model, "rf_")
        assert hasattr(model, "calibrator_")

    def test_fit_returns_self(self, minimal_data):
        """Test that fit returns self for method chaining."""
        X, y = minimal_data
        model = CaliForest(n_estimators=5, random_state=42)
        result = model.fit(X, y)
        assert result is model

    def test_invalid_calibration_method(self, minimal_data):
        """Test that invalid calibration method raises error."""
        X, y = minimal_data
        model = CaliForest(n_estimators=5, calibration_method="invalid")
        with pytest.raises(ValueError, match="Unsupported calibration"):
            model.fit(X, y)


class TestCaliForestPredict:
    """Tests for model prediction - use pre-fitted model."""

    def test_predict_proba_shape(self, fitted_model):
        """Test predict_proba output shape."""
        X_test = np.random.randn(5, 3)
        probas = fitted_model.predict_proba(X_test)
        assert probas.shape == (5, 2)

    def test_predict_proba_valid_range(self, fitted_model):
        """Test that probabilities are in valid range."""
        X_test = np.random.randn(5, 3)
        probas = fitted_model.predict_proba(X_test)

        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_predict_shape(self, fitted_model):
        """Test predict output shape."""
        X_test = np.random.randn(5, 3)
        predictions = fitted_model.predict(X_test)

        assert predictions.shape == (5,)
        assert set(predictions).issubset(set(fitted_model.classes_))

    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises error."""
        model = CaliForest()
        X_test = np.random.randn(3, 3)

        with pytest.raises(NotFittedError):
            model.predict_proba(X_test)


class TestCaliForestOOB:
    """Tests for OOB functionality."""

    def test_oob_tree_counts_computed(self, minimal_data):
        """Test that OOB tree counts are properly computed."""
        X, y = minimal_data
        model = CaliForest(n_estimators=5, random_state=42)
        model.fit(X, y)

        assert len(model.oob_tree_counts_) == len(y)
        assert model.oob_tree_counts_.min() >= 0

    def test_get_oob_calibration_data(self, minimal_data):
        """Test OOB data retrieval."""
        X, y = minimal_data
        model = CaliForest(n_estimators=5, random_state=42)
        model.fit(X, y)

        counts, ratio = model.get_oob_calibration_data()
        assert len(counts) == len(y)
        assert 0.0 <= ratio <= 1.0


class TestCaliForestReproducibility:
    """Tests for reproducibility."""

    def test_same_random_state_reproducible(self, minimal_data):
        """Test that same random state produces reproducible results."""
        X, y = minimal_data
        X_test = np.random.randn(3, 3)

        model1 = CaliForest(n_estimators=5, random_state=42)
        model2 = CaliForest(n_estimators=5, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        probas1 = model1.predict_proba(X_test)
        probas2 = model2.predict_proba(X_test)

        assert np.allclose(probas1, probas2)
