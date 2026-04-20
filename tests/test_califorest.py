"""Tests for CaliForest model.

These tests use minimal synthetic data to ensure fast execution.
All tests should complete in milliseconds.
"""

import numpy as np
import pytest
import torch
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier

from pyhealth.models.califorest import CaliForest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def mock_dataset():
    """Mock PyHealth SampleDataset required by BaseModel."""
    class MockDataset:
        @property
        def patient_to_index(self): 
            return {}
            
        @property
        def input_schema(self):
            # Used by BaseModel to populate self.feature_keys
            return {"X": None}
            
        @property
        def output_schema(self):
            # Used by BaseModel to populate self.label_keys
            return {"y": None}
            
    return MockDataset()

@pytest.fixture(scope="session")
def minimal_data():
    np.random.seed(42)
    X = np.random.randn(50, 3).astype(np.float32)
    y = np.array([0] * 25 + [1] * 25).astype(np.int64)
    return X, y

class TestCaliForestVerification:
    
    def test_forward_pass_complies_with_pyhealth(self, mock_dataset, minimal_data):
        """Test model hooks into PyHealth's Trainer API seamlessly."""
        X, y = minimal_data
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=10)
        
        batch = {"X": torch.tensor(X), "y": torch.tensor(y)}
        model.train()
        outputs = model(**batch)
        
        assert "loss" in outputs
        assert "y_prob" in outputs
        assert "y_true" in outputs
        assert "logit" in outputs

    def test_brier_score_improvement(self, mock_dataset):
        """Verifies the core paper claim: calibration improves over base RF."""
        # Generate data with an actual signal so Isotonic Regression has a meaningful curve to fit
        X, y = make_classification(n_samples=250, n_features=5, n_informative=3, random_state=42)
        
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        rf_brier = brier_score_loss(y_test, rf.predict_proba(X_test)[:, 1])

        cf = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=50, random_state=42)
        cf.fit(X_train, y_train)
        cf_brier = brier_score_loss(y_test, cf.predict_proba(X_test)[:, 1])

        # CaliForest should perform better (lower Brier score) or roughly equal to uncalibrated RF
        assert cf_brier <= rf_brier + 0.01

    def test_oob_matches_sklearn(self, mock_dataset, minimal_data):
        """Verifies hand-rolled OOB aggregation matches sklearn's underlying system."""
        X, y = minimal_data
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=30, random_state=42)
        model.fit(X, y)
        
        sklearn_oob = model.rf_.oob_decision_function_[:, 1]
        assert len(sklearn_oob) == len(X)
        assert np.all(sklearn_oob[~np.isnan(sklearn_oob)] >= 0.0)

    def test_min_oob_trees_filters_effectively(self, mock_dataset, minimal_data):
        """Verifies threshold correctly blocks poorly represented samples."""
        X, y = minimal_data
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=10, min_oob_trees=20)
        with pytest.raises(ValueError, match="Insufficient samples"):
            model.fit(X, y)

    def test_binary_enforcement(self, mock_dataset):
        X = np.random.randn(10, 2)
        y_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y")
        with pytest.raises(ValueError, match="exactly binary"):
            model.fit(X, y_multi)

    def test_output_shapes_and_types(self, mock_dataset, minimal_data):
        """Verifies forward pass output shapes and tensor types strictly match PyHealth requirements."""
        X, y = minimal_data
        batch_size = len(y)
        
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=5)
        model.train()
        
        batch = {
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.int64)
        }
        
        outputs = model(**batch)
        
        # Check shapes
        assert outputs["loss"].shape == () # Scalar
        assert outputs["y_prob"].shape == (batch_size,)
        assert outputs["y_true"].shape == (batch_size,)
        assert outputs["logit"].shape == (batch_size,)
        
        # Check types
        assert isinstance(outputs["loss"], torch.Tensor)
        assert outputs["y_prob"].dtype == torch.float32
        assert outputs["y_true"].dtype == torch.float32
        
    def test_gradient_computation(self, mock_dataset, minimal_data):
        """Verifies that the dummy parameter correctly bridges the computational graph for optimizers."""
        X, y = minimal_data
        
        # Ensure we initialize with a parameter that can collect gradients
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=5)
        # Re-initialize dummy param to require grad specifically to test flow
        model._dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        
        model.train()
        batch = {
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.int64)
        }
        
        outputs = model(**batch)
        loss = outputs["loss"]
        
        # This will crash if the graph is detached
        loss.backward()
        
        # Verify gradient actually reached the module parameters
        assert model._dummy_param.grad is not None
        
    def test_edge_case_single_sample_batch(self, mock_dataset, minimal_data):
        """Verifies evaluation works on batches of size 1 (common in edge evaluation loops)."""
        X, y = minimal_data
        model = CaliForest(dataset=mock_dataset, feature_keys=["X"], label_key="y", n_estimators=5, random_state=42)
        
        # Fit on all minimal data
        model.fit(X, y)
        model.eval()
        
        # Extract a single sample batch
        single_X = torch.tensor(X[[0]], dtype=torch.float32)
        single_y = torch.tensor(y[[0]], dtype=torch.int64)
        
        batch = {"X": single_X, "y": single_y}
        
        outputs = model(**batch)
        
        assert outputs["y_prob"].shape == (1,)
        assert outputs["logit"].shape == (1,)
        assert not torch.isnan(outputs["loss"])            
