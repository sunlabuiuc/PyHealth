"""Tests for LabradorModel.

All tests use small synthetic tensors — no real datasets.
Tests complete in milliseconds.
"""

import pytest
import torch

from pyhealth.models.labrador import LabradorEmbedding, LabradorModel


@pytest.fixture
def mock_dataset(mocker):
    """Minimal mock of a PyHealth dataset for model init."""
    dataset = mocker.MagicMock()
    dataset.output_schema = {"label": "binary"}
    dataset.get_output_size = mocker.MagicMock(return_value=2)
    return dataset


class TestLabradorEmbedding:
    def test_output_shape(self):
        emb = LabradorEmbedding(vocab_size=100, hidden_dim=32)
        codes = torch.randint(0, 100, (2, 10))
        values = torch.rand(2, 10)
        out = emb(codes, values)
        assert out.shape == (2, 10, 32)

    def test_output_dtype(self):
        emb = LabradorEmbedding(vocab_size=100, hidden_dim=32)
        codes = torch.randint(0, 100, (2, 10))
        values = torch.rand(2, 10)
        out = emb(codes, values)
        assert out.dtype == torch.float32


class TestLabradorModel:
    def test_instantiation(self, mock_dataset):
        model = LabradorModel(
            dataset=mock_dataset,
            vocab_size=100,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
        )
        assert model is not None

    def test_forward_output_keys(self, mock_dataset):
        model = LabradorModel(
            dataset=mock_dataset, vocab_size=100, hidden_dim=32,
            num_heads=2, num_layers=1
        )
        codes = torch.randint(0, 100, (2, 5))
        values = torch.rand(2, 5)
        labels = torch.randint(0, 2, (2,))
        out = model(lab_codes=codes, lab_values=values, labels=labels)
        assert "loss" in out
        assert "y_prob" in out
        assert "y_true" in out
        assert "logit" in out

    def test_forward_output_shapes(self, mock_dataset):
        model = LabradorModel(
            dataset=mock_dataset, vocab_size=100, hidden_dim=32,
            num_heads=2, num_layers=1
        )
        codes = torch.randint(0, 100, (3, 8))
        values = torch.rand(3, 8)
        labels = torch.randint(0, 2, (3,))
        out = model(lab_codes=codes, lab_values=values, labels=labels)
        assert out["logit"].shape == (3, 2)
        assert out["y_prob"].shape == (3, 2)

    def test_gradients_flow(self, mock_dataset):
        model = LabradorModel(
            dataset=mock_dataset, vocab_size=100, hidden_dim=32,
            num_heads=2, num_layers=1
        )
        codes = torch.randint(0, 100, (2, 5))
        values = torch.rand(2, 5)
        labels = torch.randint(0, 2, (2,))
        out = model(lab_codes=codes, lab_values=values, labels=labels)
        out["loss"].backward()
        # Check at least one parameter has a gradient
        has_grad = any(
            p.grad is not None for p in model.parameters()
        )
        assert has_grad

    def test_padding_mask(self, mock_dataset):
        model = LabradorModel(
            dataset=mock_dataset, vocab_size=100, hidden_dim=32,
            num_heads=2, num_layers=1
        )
        codes = torch.randint(0, 100, (2, 5))
        values = torch.rand(2, 5)
        # Last 2 positions are padding
        mask = torch.tensor([[False, False, False, True, True],
                              [False, False, True, True, True]])
        out = model(lab_codes=codes, lab_values=values, padding_mask=mask)
        assert out["logit"].shape == (2, 2)