import torch

from pyhealth.models.temporal_fusion_mlp import TemporalFusionMLP


class DummyDataset:
    """Minimal dataset stub for BaseModel initialization."""
    
    def __init__(self):
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
            "drugs": "sequence",
            "admission_year": "tensor",
        }
        self.output_schema = {
            "mortality": "binary"
        }


def test_temporal_fusion_mlp_instantiation() -> None:
    model = TemporalFusionMLP(
        dataset=DummyDataset(),
        input_dim=10,
        hidden_dim=8,
        dropout=0.1,
    )
    assert model.input_dim == 10
    assert model.hidden_dim == 8


def test_temporal_fusion_mlp_forward_shape() -> None:
    torch.manual_seed(42)
    model = TemporalFusionMLP(
        dataset=DummyDataset(),
        input_dim=10,
        hidden_dim=8,
        dropout=0.1,
    )
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2,)


def test_temporal_fusion_mlp_gradient_computation() -> None:
    torch.manual_seed(42)
    model = TemporalFusionMLP(
        dataset=DummyDataset(),
        input_dim=10,
        hidden_dim=8,
        dropout=0.1,
    )
    x = torch.randn(2, 10)
    y = torch.tensor([0.0, 1.0])
    logits = model(x)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
