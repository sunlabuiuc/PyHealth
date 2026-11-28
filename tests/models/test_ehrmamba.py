"""End-to-end model tests using only synthetic pseudo-patients."""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel, EHRMamba


@pytest.fixture(scope="module")
def dataset():
    samples = [
        {"patient_id": "p1", "codes": ["A", "B"], "label": 1},
        {"patient_id": "p2", "codes": ["B"], "label": 0},
        {"patient_id": "p3", "codes": ["A", "C", "D"], "label": 1},
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="ehrmamba_synthetic",
        # The PyHealth helper manages its own cached dataset.
    )


def test_initialization(dataset):
    model = EHRMamba(dataset, embedding_dim=8, num_layers=1, state_dim=3)
    assert isinstance(model, BaseModel)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_forward_and_backward(dataset):
    model = EHRMamba(dataset, embedding_dim=8, num_layers=1, state_dim=3, dropout=0)
    batch = next(iter(get_dataloader(dataset, batch_size=3)))
    output = model(**batch)
    assert output["y_prob"].shape == (3, 1)
    assert output["logit"].shape == (3, 1)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_predict_without_labels(dataset):
    model = EHRMamba(dataset, embedding_dim=8, num_layers=1, state_dim=3)
    batch = next(iter(get_dataloader(dataset, batch_size=2)))
    batch.pop("label")
    output = model(**batch, embed=True)
    assert "loss" not in output
    assert output["embed"].shape == (2, 8)


def test_empty_sequence_rejected(dataset):
    model = EHRMamba(dataset, embedding_dim=8, num_layers=1, state_dim=3)
    with pytest.raises(ValueError):
        model(codes=torch.zeros(1, 3, dtype=torch.long))
