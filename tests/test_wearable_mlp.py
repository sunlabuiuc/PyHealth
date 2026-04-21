from __future__ import annotations

import pytest
import torch

from pyhealth.models import WearableMLP


class DummyOutputProcessor:
    """Minimal output processor stub used by BaseModel."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class DummyBinaryDataset:
    """Minimal dataset stub for testing WearableMLP."""

    def __init__(self) -> None:
        self.input_schema = {"wearable": {"shape": (13,)}}
        self.output_schema = {"label": "binary"}

        self.feature_keys = ["wearable"]
        self.label_keys = ["label"]

        self.output_processors = {"label": DummyOutputProcessor(2)}

    def get_all_tokens(self, key):
        return None

    def get_label_tokenizer(self):
        return None

    def get_feature_tokenizer(self, key):
        return None


class DummyMulticlassDataset:
    """Minimal multiclass dataset stub for testing WearableMLP."""

    def __init__(self) -> None:
        self.input_schema = {"wearable": {"shape": (13,)}}
        self.output_schema = {"label": "multiclass"}

        self.feature_keys = ["wearable"]
        self.label_keys = ["label"]

        self.output_processors = {"label": DummyOutputProcessor(3)}

    def get_all_tokens(self, key):
        return None

    def get_label_tokenizer(self):
        return None

    def get_feature_tokenizer(self, key):
        return None


@pytest.fixture
def binary_batch():
    batch_size = 4
    x = torch.randn(batch_size, 13)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    return x, y


@pytest.fixture
def multiclass_batch():
    batch_size = 5
    x = torch.randn(batch_size, 13)
    y = torch.randint(0, 3, (batch_size,))
    return x, y


def test_wearable_mlp_binary_forward(binary_batch):
    dataset = DummyBinaryDataset()
    model = WearableMLP(
        dataset=dataset,
        feature_key="wearable",
        hidden_dim=32,
        dropout=0.1,
    )

    x, y = binary_batch
    out = model(wearable=x, label=y)

    assert "logit" in out
    assert "y_prob" in out
    assert "loss" in out
    assert "y_true" in out

    assert out["logit"].shape == (4, 1)
    assert out["y_prob"].shape == (4, 1)
    assert out["y_true"].shape == (4, 1)
    assert out["loss"].ndim == 0


def test_wearable_mlp_binary_backward(binary_batch):
    dataset = DummyBinaryDataset()
    model = WearableMLP(dataset=dataset, feature_key="wearable")

    x, y = binary_batch
    out = model(wearable=x, label=y)
    out["loss"].backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_wearable_mlp_accepts_sequence_input():
    dataset = DummyBinaryDataset()
    dataset.input_schema = {"wearable": {"shape": (3, 4)}}

    model = WearableMLP(dataset=dataset, feature_key="wearable", hidden_dim=16)

    x = torch.randn(2, 3, 4)
    y = torch.randint(0, 2, (2, 1)).float()

    out = model(wearable=x, label=y)

    assert out["logit"].shape == (2, 1)
    assert out["y_prob"].shape == (2, 1)


def test_wearable_mlp_multiclass_forward(multiclass_batch):
    dataset = DummyMulticlassDataset()
    model = WearableMLP(
        dataset=dataset,
        feature_key="wearable",
        hidden_dim=16,
        dropout=0.0,
    )

    x, y = multiclass_batch
    out = model(wearable=x, label=y)

    assert out["logit"].shape == (5, 3)
    assert out["y_prob"].shape == (5, 3)
    assert out["loss"].ndim == 0


def test_wearable_mlp_requires_valid_feature_key():
    dataset = DummyBinaryDataset()
    with pytest.raises(ValueError):
        WearableMLP(dataset=dataset, feature_key="not_a_real_key")


def test_wearable_mlp_requires_single_feature_key_when_omitted():
    dataset = DummyBinaryDataset()
    dataset.input_schema = {
        "wearable": {"shape": (13,)},
        "extra": {"shape": (5,)},
    }
    dataset.feature_keys = ["wearable", "extra"]

    with pytest.raises(ValueError):
        WearableMLP(dataset=dataset, feature_key=None)