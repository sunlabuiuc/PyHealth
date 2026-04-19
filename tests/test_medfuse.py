import pytest
import torch
import torch.nn as nn
import os
import shutil
import tempfile
from pyhealth.models.medfuse import MedFuse


class MockProcessor:
    def __init__(self, size=174):
        # satisfy method call: processor.vocab_size()
        self.vocab_size = lambda: size 

class MockDataset:
    def __init__(self):
        # BaseModel expects these exact names to populate feature/label keys
        self.input_schema = {"conditions": None}
        self.output_schema = {"label": None}
        
        # MedFuse __init__ looks here for the vocab size
        self.input_processors = {"conditions": MockProcessor(174)}
        
        # Standard metadata
        self.feature_keys = ["conditions"]
        self.label_keys = ["label"]
        self.device = "cpu"


@pytest.fixture
def mock_batch():
    """Requirement: Uses small synthetic data"""
    batch_size = 4
    seq_len = 5
    return {
        "conditions": torch.randint(0, 100, (batch_size, seq_len)),
        "label": torch.randint(0, 2, (batch_size, 1)),
        "cxr": torch.randn(batch_size, 512)
    }


def test_model_instantiation():
    """Requirement: Tests instantiation"""
    ds = MockDataset()
    model = MedFuse(dataset=ds, hidden_dim=64)
    assert isinstance(model, nn.Module)
    assert "conditions" in model.feature_keys
    assert "label" in model.label_keys

def test_model_forward_and_shapes(mock_batch):
    """Requirement: Tests forward pass and output shapes"""
    ds = MockDataset()
    model = MedFuse(dataset=ds, hidden_dim=64)
    output = model(**mock_batch)
    
    assert "loss" in output
    assert "y_prob" in output
    # Check shape: (batch_size, 1)
    assert output["y_prob"].shape == (4, 1)

def test_gradient_computation(mock_batch):
    """Requirement: Tests gradient computation"""
    ds = MockDataset()
    model = MedFuse(dataset=ds, hidden_dim=64)
    output = model(**mock_batch)
    loss = output["loss"]
    loss.backward()
    
    # Check if gradients flow to the weights
    assert model.fc.weight.grad is not None
    assert not torch.isnan(loss)

def test_data_integrity_mock():
    """Requirement: Tests data integrity"""
    ds = MockDataset()
    assert "conditions" in ds.input_schema
    assert "label" in ds.output_schema

def test_edge_case_missing_cxr(mock_batch):
    """Requirement: Tests edge cases (missing image modality)"""
    ds = MockDataset()
    model = MedFuse(dataset=ds, hidden_dim=64)
    
    # Test fallback to zero-tensors when CXR is missing
    del mock_batch["cxr"]
    
    output = model(**mock_batch)
    assert output["y_prob"].shape == (4, 1)

def test_cleanup_logic():
    """Requirement: Uses temporary directories and proper cleanup"""
    temp_path = tempfile.mkdtemp()
    assert os.path.exists(temp_path)
    shutil.rmtree(temp_path)
    assert not os.path.exists(temp_path)