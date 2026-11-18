import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.base_model import BaseModel
from pyhealth.models.neural_fg import NeuralFG, PositiveLinear


# ---------------------------------------------------------------------
# Shared synthetic test data
# ---------------------------------------------------------------------


@pytest.fixture
def dataset():
    """Create a tiny synthetic competing-risk dataset."""

    samples = [
        {
            "patient_id": "p0",
            "features": [
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
            ],
            "duration": [1.0],
            "event": [1],
        },
        {
            "patient_id": "p1",
            "features": [
                0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9,
                1.0, 1.1, 1.2, 1.3,
            ],
            "duration": [2.0],
            "event": [0],
        },
        {
            "patient_id": "p2",
            "features": [
                0.4, 0.5, 0.6, 0.7,
                0.8, 0.9, 1.0, 1.1,
                1.2, 1.3, 1.4, 1.5,
            ],
            "duration": [3.0],
            "event": [2],
        },
        {
            "patient_id": "p3",
            "features": [
                0.7, 0.8, 0.9, 1.0,
                1.1, 1.2, 1.3, 1.4,
                1.5, 1.6, 1.7, 1.8,
            ],
            "duration": [4.0],
            "event": [1],
        },
    ]

    return create_sample_dataset(
        samples=samples,
        input_schema={
            "features": "tensor",
        },
        output_schema={
            "duration": "tensor",
            "event": "tensor",
        },
        dataset_name="neuralfg_test",
    )


@pytest.fixture
def model(dataset):
    """Create a deterministic small NeuralFG model."""

    torch.manual_seed(42)

    return NeuralFG(
        dataset=dataset,
        num_risks=2,
        hidden_dim=8,
        embedding_dim=8,
        embedding_layers=2,
        monotonic_layers=2,
        dropout=0.0,
        time_grid=torch.tensor(
            [0.5, 1.0, 2.0]
        ),
    )


@pytest.fixture
def batch(dataset):
    """Create one small PyHealth batch."""

    loader = get_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    return next(iter(loader))


# ---------------------------------------------------------------------
# 1. Model instantiation
# ---------------------------------------------------------------------


def test_model_instantiation(model):
    """NeuralFG should instantiate as a PyHealth BaseModel."""

    assert isinstance(model, NeuralFG)
    assert isinstance(model, BaseModel)
    assert model.input_dim == 12
    assert model.num_risks == 2
    assert model.network is not None


# ---------------------------------------------------------------------
# 2. Forward pass
# ---------------------------------------------------------------------


def test_forward_pass(model, batch):
    """Forward pass should run and return expected fields."""
    model.eval()
    result = model(**batch)

    expected_keys = {
        "loss",
        "logit",
        "y_prob",
        "y_true",
        "observed_cif",
    }

    assert expected_keys.issubset(
        result.keys()
    )

    assert torch.isfinite(
        result["loss"]
    )

# ---------------------------------------------------------------------
# 3. Output shapes
# ---------------------------------------------------------------------


def test_output_shapes(model, batch):
    """NeuralFG outputs should have the expected dimensions."""

    model.eval()
    result = model(**batch)

    batch_size = 4
    num_times = 3
    num_risks = 2

    # Scalar loss
    assert result["loss"].ndim == 0

    # Raw output contains:
    # balancing logits + monotonic outputs
    assert result["logit"].shape == (
        batch_size,
        2 * num_risks,
    )

    # CIF curves
    assert result["y_prob"].shape == (
        batch_size,
        num_times,
        num_risks,
    )

    # duration + event
    assert result["y_true"].shape == (
        batch_size,
        2,
    )

    # CIF at observed patient time
    assert result["observed_cif"].shape == (
        batch_size,
        num_risks,
    )


# ---------------------------------------------------------------------
# 4. Gradient computation
# ---------------------------------------------------------------------


def test_gradient_computation(model, batch):
    """Loss backward pass should produce finite gradients."""

    model.train()
    model.zero_grad(
        set_to_none=True
    )

    result = model(**batch)
    result["loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            and parameter.grad is not None
        )
    ]

    assert len(gradients) > 0

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    assert any(
        gradient.abs().sum() > 0
        for gradient in gradients
    )


# ---------------------------------------------------------------------
# 5. PositiveLinear constraint
# ---------------------------------------------------------------------

def test_positive_linear_weights():
    """Effective PositiveLinear weights should be non-negative."""

    torch.manual_seed(42)
    layer = PositiveLinear(
        in_features=4,
        out_features=3,
    )

    weights = layer.positive_weight
    assert weights.shape == (3, 4)
    assert torch.all(
        weights >= 0
    )


# ---------------------------------------------------------------------
# 6. Balancing probabilities
# ---------------------------------------------------------------------
def test_balancing_probabilities(model):
    """Risk-balancing probabilities should sum to one."""
    model.eval()
    torch.manual_seed(42)
    features = torch.randn(
        4,
        12,
    )

    embedding = model.network.embed(
        features
    )

    probabilities = (
        model.network.balancing_probabilities(
            embedding
        )
    )

    assert probabilities.shape == (
        4,
        2,
    )

    assert torch.all(
        probabilities >= 0
    )

    assert torch.all(
        probabilities <= 1
    )

    row_sums = probabilities.sum(
        dim=-1
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# 7. CIF validity
# ---------------------------------------------------------------------
def test_cif_bounds(model):
    """CIF values should be valid competing-risk probabilities."""
    model.eval()
    torch.manual_seed(42)
    features = torch.randn(
        4,
        12,
    )

    times = torch.tensor(
        [0.5, 1.0, 2.0, 4.0]
    )

    cif = model.network.cif(
        features=features,
        time=times,
    )

    assert cif.shape == (
        4,
        2,
    )

    # Individual CIF values must be valid probabilities.
    assert torch.all(
        cif >= 0
    )

    assert torch.all(
        cif <= 1
    )

    # Total competing-risk incidence cannot exceed 1.
    total_cif = cif.sum(
        dim=-1
    )

    assert torch.all(
        total_cif <= 1.0 + 1e-6
    )


# ---------------------------------------------------------------------
# 8. CIF monotonicity over time
# ---------------------------------------------------------------------


def test_cif_monotonicity(model):
    """CIF for a fixed patient should not decrease with time."""
    model.eval()
    torch.manual_seed(42)
    features = torch.randn(
        1,
        12,
    )

    times = torch.tensor(
        [
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
        ]
    )

    cif = model.predict_cif(
        features=features,
        times=times,
    )

    assert cif.shape == (
        1,
        5,
        2,
    )

    differences = (
        cif[:, 1:, :]
        - cif[:, :-1, :]
    )

    assert torch.all(
        differences >= -1e-6
    )


# ---------------------------------------------------------------------
# 9. Invalid event labels
# ---------------------------------------------------------------------


def test_invalid_event_label(model):
    """Events greater than num_risks should raise an error."""
    features = torch.randn(
        2,
        12,
    )

    duration = torch.tensor(
        [1.0, 2.0]
    )

    # num_risks = 2, so event 3 is invalid.
    event = torch.tensor(
        [1, 3]
    )

    with pytest.raises(
        ValueError,
        match="event labels cannot exceed",
    ):
        model.network.competing_risk_nll(
            features=features,
            time=duration,
            event=event,
        )