import torch
import torch.nn as nn

from pyhealth.models.adaptive_transfer import AdaptiveTransferModel


class _DummyInputProcessor:
    """Input processor stub for model tests."""

    def schema(self):
        return ("value",)


class _DummyOutputProcessor:
    """Output processor stub for model tests."""

    def __init__(self, size: int):
        self._size = size

    def size(self):
        return self._size


class _DummyDataset:
    """Dataset stub satisfying BaseModel requirements."""

    def __init__(self, num_classes: int = 3, input_dim: int = 4):
        self.input_schema = {"signal": "tensor"}
        self.output_schema = {"label": "multiclass"}
        self.input_processors = {"signal": _DummyInputProcessor()}
        self.output_processors = {"label": _DummyOutputProcessor(num_classes)}
        self.input_info = {"signal": {"dim": input_dim}}


class _MeanPoolBackbone(nn.Module):
    """Custom backbone for dependency-injection tests."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        pooled = x.mean(dim=1)
        return self.proj(pooled)


def _make_batch(
    batch_size: int = 4,
    seq_len: int = 6,
    input_dim: int = 4,
    num_classes: int = 3,
):
    """Small synthetic batch for testing."""
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return {"signal": x, "label": y}


def test_adaptive_transfer_instantiation_default():
    """Test model instantiation with the default backbone."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)
    model = AdaptiveTransferModel(dataset=dataset, feature_key="signal")

    assert isinstance(model, AdaptiveTransferModel)
    assert model.feature_key == "signal"
    assert model.label_key == "label"
    assert isinstance(model.encoder, nn.LSTM)


def test_adaptive_transfer_forward_shapes():
    """Test forward pass output keys and tensor shapes."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)
    model = AdaptiveTransferModel(dataset=dataset, feature_key="signal")

    batch = _make_batch(batch_size=5, seq_len=7, input_dim=4, num_classes=3)
    output = model(**batch)

    assert "logit" in output
    assert "y_prob" in output
    assert "loss" in output
    assert "y_true" in output

    assert output["logit"].shape == (5, 3)
    assert output["y_prob"].shape == (5, 3)
    assert output["y_true"].shape == (5,)
    assert output["loss"].dim() == 0


def test_adaptive_transfer_backward_computes_gradients():
    """Test backward pass and gradient propagation."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)
    model = AdaptiveTransferModel(dataset=dataset, feature_key="signal")

    batch = _make_batch(batch_size=4, seq_len=5, input_dim=4, num_classes=3)
    output = model(**batch)
    loss = output["loss"]
    loss.backward()

    grads = [
        param.grad
        for param in model.parameters()
        if param.requires_grad
    ]
    assert any(grad is not None for grad in grads)


def test_adaptive_transfer_custom_backbone_forward():
    """Test dependency injection with a custom backbone."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)
    backbone = _MeanPoolBackbone(input_dim=4, output_dim=8)

    model = AdaptiveTransferModel(
        dataset=dataset,
        feature_key="signal",
        backbone=backbone,
        backbone_output_dim=8,
    )

    batch = _make_batch(batch_size=3, seq_len=6, input_dim=4, num_classes=3)
    output = model(**batch)

    assert output["logit"].shape == (3, 3)
    assert output["y_prob"].shape == (3, 3)


def test_adaptive_transfer_compute_ipd_with_string_distance():
    """Test IPD computation with a built-in string distance function."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)
    model = AdaptiveTransferModel(
        dataset=dataset,
        feature_key="signal",
        distance_fn="cosine",
        use_kde_smoothing=False,
    )

    source_batch = {"signal": torch.randn(4, 5, 4)}
    target_batch = {"signal": torch.randn(4, 5, 4)}

    ipd = model.compute_ipd(source_batch, target_batch)

    assert isinstance(ipd, float)
    assert ipd >= 0.0


def test_adaptive_transfer_compute_ipd_with_callable_distance():
    """Test IPD computation with a custom callable distance function."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)

    def l1_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, p=1, dim=1)

    model = AdaptiveTransferModel(
        dataset=dataset,
        feature_key="signal",
        distance_fn=l1_distance,
        use_kde_smoothing=False,
    )

    source_batch = {"signal": torch.randn(4, 5, 4)}
    target_batch = {"signal": torch.randn(4, 5, 4)}

    distances = model.compute_pairwise_distances(source_batch, target_batch)

    assert distances.shape == (4,)
    assert torch.all(distances >= 0)


def test_adaptive_transfer_rank_source_domains():
    """Test source-domain ranking by similarity."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)
    model = AdaptiveTransferModel(
        dataset=dataset,
        feature_key="signal",
        use_kde_smoothing=False,
    )

    target_batch = {"signal": torch.zeros(4, 5, 4)}

    source_batches = [
        {"signal": torch.zeros(4, 5, 4)},
        {"signal": torch.ones(4, 5, 4)},
        {"signal": torch.full((4, 5, 4), 2.0)},
    ]

    ranked = model.rank_source_domains(source_batches, target_batch)

    assert len(ranked) == 3
    assert sorted(ranked) == [0, 1, 2]
    assert ranked[0] == 0


def test_adaptive_transfer_get_adaptive_lr():
    """Test similarity-weighted learning-rate scaling."""
    dataset = _DummyDataset(num_classes=3, input_dim=4)

    model_weighted = AdaptiveTransferModel(
        dataset=dataset,
        feature_key="signal",
        use_similarity_weighting=True,
    )
    model_unweighted = AdaptiveTransferModel(
        dataset=dataset,
        feature_key="signal",
        use_similarity_weighting=False,
    )

    base_lr = 1e-3
    similarity = 2.0

    assert model_weighted.get_adaptive_lr(base_lr, similarity) == base_lr * similarity
    assert model_unweighted.get_adaptive_lr(base_lr, similarity) == base_lr
