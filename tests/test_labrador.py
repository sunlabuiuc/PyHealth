"""Fast synthetic tests for LabradorModel.

These tests satisfy the project requirement for model tests with minimal
synthetic tensors and no demo datasets.
"""

from types import SimpleNamespace

import pytest
import torch

from pyhealth.models.labrador import LabradorEmbedding, LabradorModel


def _build_mock_dataset(mode: str = "binary"):
    """Creates a minimal dataset-like object expected by BaseModel."""
    output_processor = SimpleNamespace(size=lambda: 1)
    return SimpleNamespace(
        output_schema={"label": mode},
        input_schema={"lab_codes": "sequence", "lab_values": "sequence"},
        output_processors={"label": output_processor},
    )


@pytest.fixture
def mock_dataset_binary():
    return _build_mock_dataset(mode="binary")


class TestLabradorEmbedding:
    def test_output_shape_and_dtype(self):
        emb = LabradorEmbedding(vocab_size=32, hidden_dim=16)
        codes = torch.randint(0, 32, (2, 4))
        values = torch.rand(2, 4)

        out = emb(codes, values)

        assert out.shape == (2, 4, 16)
        assert out.dtype == torch.float32


class TestLabradorModel:
    def test_instantiation(self, mock_dataset_binary):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=32,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        assert model.mode == "binary"

    def test_forward_contains_required_keys_when_label_given(self, mock_dataset_binary):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=32,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        codes = torch.randint(0, 32, (2, 5))
        values = torch.rand(2, 5)
        label = torch.tensor([0, 1])

        out = model(lab_codes=codes, lab_values=values, label=label)

        assert set(["loss", "y_prob", "y_true", "logit"]).issubset(out.keys())
        assert out["logit"].shape == (2, 1)
        assert out["y_prob"].shape == (2, 1)
        assert out["y_true"].shape == (2, 1)

    def test_forward_without_label_returns_inference_outputs(self, mock_dataset_binary):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=32,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        codes = torch.randint(0, 32, (2, 3))
        values = torch.rand(2, 3)

        out = model(lab_codes=codes, lab_values=values)

        assert "loss" not in out
        assert out["logit"].shape == (2, 1)

    def test_padding_mask_forward_and_gradients(self, mock_dataset_binary):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=32,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        codes = torch.randint(0, 32, (2, 5))
        values = torch.rand(2, 5)
        label = torch.tensor([1, 0])
        padding_mask = torch.tensor(
            [[False, False, False, True, True], [False, False, True, True, True]]
        )

        out = model(
            lab_codes=codes,
            lab_values=values,
            padding_mask=padding_mask,
            label=label,
        )
        out["loss"].backward()

        assert out["logit"].shape == (2, 1)
        assert any(param.grad is not None for param in model.parameters())
