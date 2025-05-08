# unittests/test_clinical_t5.py
import pytest
import torch
from pyhealth.datasets import SampleDataset
from pyhealth.models import ClinicalT5

@pytest.fixture
def mock_mednli_dataset():
    """Mock dataset for MedNLI classification testing."""
    dataset = SampleDataset()
    dataset.input_schema = {
        "sentence1": "text",
        "sentence2": "text"
    }
    dataset.output_schema = {"label": "multiclass"}
    
    class MockProcessor:
        def get_label_space(self):
            return ["entailment", "neutral", "contradiction"]
    
    dataset.output_processors = {"label": MockProcessor()}
    return dataset

@pytest.fixture
def mock_mimiccxr_dataset():
    """Mock dataset for MIMIC-CXR generation testing."""
    dataset = SampleDataset()
    dataset.input_schema = {"findings": "text"}
    dataset.output_schema = {"report": "text"}
    return dataset

def test_classification_forward(mock_mednli_dataset):
    """Test classification task forward pass."""
    model = ClinicalT5(
        dataset=mock_mednli_dataset,
        model_name="t5-small",
        mode="classification"
    )
    
    batch = {
        "sentence1": ["Patient has pneumonia"],
        "sentence2": ["Lung infection present"],
        "label": torch.tensor([0])
    }
    
    outputs = model(**batch)
    assert outputs["y_prob"].shape == (1, 3)  # batch_size x num_classes

def test_generation_predict(mock_mimiccxr_dataset):
    """Test generation task prediction."""
    model = ClinicalT5(
        dataset=mock_mimiccxr_dataset,
        model_name="t5-small",
        mode="generation"
    )
    
    report = model.predict("Lungs are clear", max_length=50)
    assert isinstance(report, str)
    assert len(report) > 0
