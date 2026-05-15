"""
Unit tests for the Temporal Pointwise Convolution (TPC) PyHealth model.

This module contains fast synthetic tests for validating the dataset-backed
``BaseModel`` implementation of TPC. The tests are designed to run quickly
without requiring real eICU or MIMIC-IV data.

Overview:
    The test suite checks:

    1. Model instantiation under a dataset-backed BaseModel contract.
    2. Forward-pass correctness and required output keys.
    3. Output shapes for scalar regression.
    4. Forward-pass behavior across major ablation variants.
    5. Gradient propagation through the model during backpropagation.
    6. Failure behavior when required batch fields are missing.

Implementation Notes:
    - Uses only small synthetic tensors for speed and reproducibility.
    - Uses a minimal mocked dataset/processor contract to exercise the model's
      true BaseModel-facing path.
    - Does not depend on real datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

import pyhealth.models.tpc as tpc_module
from pyhealth.models.tpc import TPC


@dataclass
class DummyProcessor:
    """Minimal processor stub exposing a schema with a ``value`` field."""

    names: List[str]

    def schema(self) -> List[str]:
        """Return the processor schema."""
        return self.names


class DummyDataset:
    """Minimal dataset stub matching the fields TPC expects from BaseModel.

    This object is intentionally lightweight. It provides only the fields
    needed by the rewritten TPC model and the patched BaseModel methods in
    this test module.
    """

    def __init__(self) -> None:
        """Initialize the dummy dataset."""
        self.input_schema = {
            "time_series": "timeseries",
            "static": "tensor",
        }
        self.output_schema = {
            "target_los_hours": "regression",
        }
        self.input_processors: Dict[str, DummyProcessor] = {
            "time_series": DummyProcessor(["value"]),
            "static": DummyProcessor(["value"]),
        }
        self.output_processors = {}


@pytest.fixture()
def patch_basemodel(monkeypatch):
    """Patch BaseModel helpers so TPC can be tested with synthetic inputs only.

    The project rubric requires fast synthetic model tests, and the current
    test environment does not rely on the full trainer/runtime stack.
    This fixture patches the minimum BaseModel behaviors the TPC model uses:
        - dataset-backed initialization
        - output size resolution
        - regression loss resolution
        - prediction preparation
    """

    def fake_init(self, dataset):
        nn.Module.__init__(self)
        self.dataset = dataset
        self.feature_keys = list(dataset.input_schema.keys())
        self.label_keys = list(dataset.output_schema.keys())
        self.__dict__["device"] = torch.device("cpu")

        class DummyProcessor:
            def size(self):
                return 1

        self.dataset.output_processors = {
            self.label_keys[0]: DummyProcessor()
        }

    monkeypatch.setattr(tpc_module.BaseModel, "__init__", fake_init)
    monkeypatch.setattr(tpc_module.BaseModel, "get_output_size", lambda self: 1)
    monkeypatch.setattr(
        tpc_module.BaseModel,
        "get_loss_function",
        lambda self: nn.MSELoss(),
    )
    monkeypatch.setattr(
        tpc_module.BaseModel,
        "prepare_y_prob",
        lambda self, logit: logit,
    )
    monkeypatch.setattr(
        tpc_module.BaseModel,
        "device",
        property(lambda self: torch.device("cpu")),
    )


def make_batch(
    batch_size: int = 2,
    seq_len: int = 6,
    num_features: int = 5,
    static_dim: int = 3,
) -> Dict[str, torch.Tensor | tuple[torch.Tensor]]:
    """Create a small synthetic batch for BaseModel-style testing.

    The ``time_series`` tensor uses the task/model contract shape [B, T, 3F]
    with interleaved [value, mask, decay] channels per feature.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_features: Number of base time-series features.
        static_dim: Static feature dimension.

    Returns:
        A batch dictionary compatible with ``TPC.forward(**kwargs)``.
    """
    values = torch.randn(batch_size, seq_len, num_features)
    masks = torch.randint(0, 2, (batch_size, seq_len, num_features)).float()
    decay = torch.rand(batch_size, seq_len, num_features)

    pieces = []
    for feature_idx in range(num_features):
        pieces.append(values[:, :, feature_idx].unsqueeze(-1))
        pieces.append(masks[:, :, feature_idx].unsqueeze(-1))
        pieces.append(decay[:, :, feature_idx].unsqueeze(-1))
    time_series = torch.cat(pieces, dim=-1)  # [B, T, 3F]

    static = torch.randn(batch_size, static_dim)
    target = torch.rand(batch_size, 1) * 10.0

    return {
        "time_series": (time_series,),
        "static": (static,),
        "target_los_hours": target,
    }


def test_tpc_init_requires_expected_feature_keys(patch_basemodel) -> None:
    """Test that TPC instantiates with the expected dataset-backed contract."""
    dataset = DummyDataset()

    model = TPC(
        dataset=dataset,
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
    )

    assert model.label_key == "target_los_hours"
    assert set(model.feature_keys) == {"time_series", "static"}


def test_tpc_forward_returns_required_basemodel_keys(patch_basemodel) -> None:
    """Test that forward returns the required BaseModel output dictionary."""
    dataset = DummyDataset()
    batch = make_batch()

    model = TPC(
        dataset=dataset,
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
    )

    outputs = model(**batch)

    assert set(outputs.keys()) == {"loss", "y_prob", "y_true", "logit"}
    assert torch.is_tensor(outputs["loss"])
    assert outputs["y_prob"].shape == (2, 1)
    assert outputs["y_true"].shape == (2, 1)
    assert outputs["logit"].shape == (2, 1)
    assert torch.isfinite(outputs["logit"]).all()


def test_tpc_forward_accepts_tensor_inputs_without_processor_tuple(
    patch_basemodel,
) -> None:
    """Test that forward accepts raw tensors as well as processor tuples."""
    dataset = DummyDataset()
    batch = make_batch()

    raw_batch = {
        "time_series": batch["time_series"][0],
        "static": batch["static"][0],
        "target_los_hours": batch["target_los_hours"],
    }

    model = TPC(
        dataset=dataset,
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
    )

    outputs = model(**raw_batch)
    assert outputs["logit"].shape == (2, 1)
    assert torch.isfinite(outputs["logit"]).all()


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"use_temporal": True, "use_pointwise": True},
        {"use_temporal": True, "use_pointwise": False},
        {"use_temporal": False, "use_pointwise": True},
        {"use_temporal": True, "use_pointwise": True, "shared_temporal": True},
        {"use_temporal": True, "use_pointwise": True, "use_skip_connections": False},
    ],
)
def test_tpc_ablation_variants_forward(
    patch_basemodel,
    model_kwargs: dict,
) -> None:
    """Test forward pass across major ablation variants."""
    dataset = DummyDataset()
    batch = make_batch()

    model = TPC(
        dataset=dataset,
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
        **model_kwargs,
    )

    outputs = model(**batch)
    assert outputs["logit"].shape == (2, 1)
    assert torch.isfinite(outputs["logit"]).all()
    assert torch.isfinite(outputs["loss"]).all()


def test_tpc_requires_at_least_one_branch(patch_basemodel) -> None:
    """Test that TPC rejects configuration with both branches disabled."""
    dataset = DummyDataset()

    with pytest.raises(ValueError):
        TPC(
            dataset=dataset,
            input_dim=5,
            static_dim=3,
            temporal_channels=4,
            pointwise_channels=4,
            num_layers=2,
            kernel_size=3,
            fc_dim=8,
            use_temporal=False,
            use_pointwise=False,
        )


def test_tpc_missing_label_raises(patch_basemodel) -> None:
    """Test that missing label data raises a clear error."""
    dataset = DummyDataset()
    batch = make_batch()
    batch.pop("target_los_hours")

    model = TPC(
        dataset=dataset,
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
    )

    with pytest.raises(ValueError, match="Missing required label field"):
        model(**batch)


def test_tpc_backward_pass(patch_basemodel) -> None:
    """Test that gradients propagate through the dataset-backed model path."""
    dataset = DummyDataset()
    batch = make_batch()

    model = TPC(
        dataset=dataset,
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
    )

    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()

    has_grad = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert has_grad