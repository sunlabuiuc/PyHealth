import pytest
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.ehr_mamba import EhrMamba
from pyhealth.models.gru_baseline import GRUBaseline
 
SAMPLES = [
    {"patient_id": "p-0", "visit_id": "v-0", "conditions": ["cond-1", "cond-2", "cond-3"], "label": 1},
    {"patient_id": "p-1", "visit_id": "v-1", "conditions": ["cond-2", "cond-4"], "label": 0},
    {"patient_id": "p-2", "visit_id": "v-2", "conditions": ["cond-1", "cond-5"], "label": 1},
]
 
 
@pytest.fixture
def dataset():
    return create_sample_dataset(
        samples=SAMPLES,
        input_schema={"conditions": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="test_ehrmamba",
    )
 
 
@pytest.fixture
def batch(dataset):
    loader = get_dataloader(dataset, batch_size=3, shuffle=False)
    return next(iter(loader))
 
 
class TestEhrMamba:
    def test_instantiation(self, dataset):
        model = EhrMamba(dataset=dataset)
        assert model is not None
 
    def test_forward_output_keys(self, dataset, batch):
        model = EhrMamba(dataset=dataset)
        output = model(**batch)
        assert "loss" in output
        assert "y_prob" in output
        assert "y_true" in output
 
    def test_output_shape(self, dataset, batch):
        model = EhrMamba(dataset=dataset)
        output = model(**batch)
        assert output["y_prob"].shape[0] == 3
 
    def test_loss_is_scalar(self, dataset, batch):
        model = EhrMamba(dataset=dataset)
        output = model(**batch)
        assert output["loss"].ndim == 0
 
    def test_probabilities_in_range(self, dataset, batch):
        model = EhrMamba(dataset=dataset)
        output = model(**batch)
        probs = output["y_prob"]
        assert (probs >= 0).all() and (probs <= 1).all()
 
    def test_gradients_flow(self, dataset, batch):
        model = EhrMamba(dataset=dataset)
        output = model(**batch)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and not name.endswith("_dummy_param"):
                assert param.grad is not None, f"No gradient for {name}"
 
    def test_different_hidden_dims(self, dataset, batch):
        for hidden_dim in [64, 128, 256]:
            model = EhrMamba(dataset=dataset, hidden_dim=hidden_dim)
            output = model(**batch)
            assert output["y_prob"].shape[0] == 3
 
    def test_different_dropout(self, dataset, batch):
        for dropout in [0.1, 0.3, 0.5]:
            model = EhrMamba(dataset=dataset, dropout=dropout)
            output = model(**batch)
            assert "loss" in output
 
 
class TestGRUBaseline:
    def test_instantiation(self, dataset):
        model = GRUBaseline(dataset=dataset)
        assert model is not None
 
    def test_forward_output_keys(self, dataset, batch):
        model = GRUBaseline(dataset=dataset)
        output = model(**batch)
        assert "loss" in output
        assert "y_prob" in output
 
    def test_output_shape(self, dataset, batch):
        model = GRUBaseline(dataset=dataset)
        output = model(**batch)
        assert output["y_prob"].shape[0] == 3
 
    def test_gradients_flow(self, dataset, batch):
        model = GRUBaseline(dataset=dataset)
        output = model(**batch)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and not name.endswith("_dummy_param"):
                assert param.grad is not None, f"No gradient for {name}"