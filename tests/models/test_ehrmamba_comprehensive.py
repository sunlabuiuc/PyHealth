"""Fast EHRMamba contract tests using three synthetic patients only."""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EHRMamba


@pytest.fixture(scope="module")
def sample_dataset():
    samples = [
        {"patient_id": "p0", "conditions": ["A", "B"], "drugs": ["X"], "label": 1},
        {"patient_id": "p1", "conditions": ["B"], "drugs": ["Y", "X"], "label": 0},
        {"patient_id": "p2", "conditions": ["A", "C", "B"], "drugs": ["Y"], "label": 1},
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "sequence", "drugs": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="ehrmamba_comprehensive_synthetic",
    )


def make_model(dataset, **overrides):
    kwargs = {"embedding_dim": 8, "num_layers": 1, "state_dim": 3, "dropout": 0.0}
    kwargs.update(overrides)
    return EHRMamba(dataset=dataset, **kwargs)


def batch_for(dataset):
    return next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))


@pytest.mark.parametrize(
    "embedding_dim,num_layers,state_dim", [(4, 1, 2), (8, 1, 3), (8, 2, 4)]
)
def test_initialization_and_shapes(
    sample_dataset, embedding_dim, num_layers, state_dim
):
    model = make_model(
        sample_dataset,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        state_dim=state_dim,
    )
    model.eval()
    with torch.no_grad():
        output = model(**batch_for(sample_dataset), embed=True)
    assert output["logit"].shape == (3, 1)
    assert output["y_prob"].shape == (3, 1)
    assert output["y_true"].shape == (3, 1)
    assert output["embed"].shape == (3, 2 * embedding_dim)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    assert torch.all((output["y_prob"] >= 0) & (output["y_prob"] <= 1))
    assert len(model.blocks["conditions"]) == num_layers


def test_gradients_reach_backbone_embeddings_and_classifier(sample_dataset):
    model = make_model(sample_dataset)
    model.train()
    loss = model(**batch_for(sample_dataset))["loss"]
    loss.backward()
    for name, module in [
        ("condition embedding", model.embeddings["conditions"]),
        ("drug embedding", model.embeddings["drugs"]),
        ("condition SSM", model.blocks["conditions"]),
        ("drug SSM", model.blocks["drugs"]),
        ("classifier", model.classifier),
    ]:
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None for g in gradients), name
        assert all(torch.isfinite(g).all() for g in gradients if g is not None), name


def test_optimizer_updates_weights(sample_dataset):
    model = make_model(sample_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.classifier.weight.detach().clone()
    optimizer.zero_grad()
    model(**batch_for(sample_dataset))["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight.detach())


def test_inference_without_labels(sample_dataset):
    model = make_model(sample_dataset)
    batch = batch_for(sample_dataset)
    batch.pop("label")
    output = model(**batch, embed=True)
    assert "loss" not in output
    assert "y_true" not in output
    assert output["embed"].shape == (3, 16)


def test_empty_patient_rejected(sample_dataset):
    model = make_model(sample_dataset)
    batch = batch_for(sample_dataset)
    batch["conditions"] = torch.zeros_like(torch.as_tensor(batch["conditions"]))
    with pytest.raises(ValueError, match="empty patient sequence"):
        model(**batch)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"num_layers": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_invalid_hyperparameters(sample_dataset, kwargs):
    with pytest.raises(ValueError):
        make_model(sample_dataset, **kwargs)


def test_reproducible_inference(sample_dataset):
    torch.manual_seed(123)
    first = make_model(sample_dataset)
    torch.manual_seed(123)
    second = make_model(sample_dataset)
    first.eval()
    second.eval()
    batch = batch_for(sample_dataset)
    with torch.no_grad():
        a = first(**batch)["logit"]
        b = second(**batch)["logit"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
