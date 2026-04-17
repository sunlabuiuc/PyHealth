"""Fast synthetic tests for Labrador model components."""

from types import SimpleNamespace

import pytest
import torch

from pyhealth.models.labrador import LabradorEmbedding, LabradorMLMHead, LabradorModel


def _build_mock_dataset(mode: str = "binary"):
    output_processor = SimpleNamespace(size=lambda: 1)
    return SimpleNamespace(
        output_schema={"label": mode},
        input_schema={"lab_codes": "sequence", "lab_values": "sequence"},
        output_processors={"label": output_processor},
    )


@pytest.fixture
def mock_dataset_binary():
    return _build_mock_dataset(mode="binary")


@pytest.fixture
def tiny_batch():
    return {
        "lab_codes": torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
        "lab_values": torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32),
        "label": torch.tensor([0, 1], dtype=torch.long),
        "padding_mask": torch.tensor([[False, False, True], [False, True, True]]),
    }


class TestLabradorEmbedding:
    def test_output_shape_and_dtype(self):
        emb = LabradorEmbedding(vocab_size=8, hidden_dim=4)
        out = emb(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        )
        assert out.shape == (2, 2, 4)
        assert out.dtype == torch.float32


class TestLabradorModelValidation:
    def test_invalid_hidden_dim_num_heads_raises(self, mock_dataset_binary):
        with pytest.raises(ValueError, match="divisible"):
            LabradorModel(
                dataset=mock_dataset_binary,
                vocab_size=8,
                hidden_dim=5,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
            )

    def test_invalid_vocab_size_raises(self, mock_dataset_binary):
        with pytest.raises(ValueError, match="vocab_size"):
            LabradorModel(
                dataset=mock_dataset_binary,
                vocab_size=0,
                hidden_dim=4,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
            )


class TestLabradorMLMUtilities:
    def test_mlm_head_output_shapes(self):
        head = LabradorMLMHead(hidden_dim=4, vocab_size=8)
        hidden = torch.randn(2, 3, 4)
        out = head(hidden)
        assert out["categorical_logits"].shape == (2, 3, 8)
        assert out["categorical_output"].shape == (2, 3, 8)
        assert out["continuous_output"].shape == (2, 3, 1)

    def test_categorical_mlm_loss_computes(self, mock_dataset_binary):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=8,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        logits = torch.randn(2, 3, 8)
        targets = torch.tensor([[1, 2, -1], [3, -1, -1]])
        loss = model.categorical_mlm_loss(logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_continuous_mlm_loss_computes(self, mock_dataset_binary):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=8,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        preds = torch.tensor([[[0.1], [0.2], [0.0]], [[0.3], [0.0], [0.0]]])
        targets = torch.tensor([[0.0, 0.2, -1.0], [0.5, -1.0, -1.0]])
        loss = model.continuous_mlm_loss(preds, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestLabradorForward:
    def test_include_classifier_head_false_with_label_raises(
        self, mock_dataset_binary, tiny_batch
    ):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=8,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            include_classifier_head=False,
        )
        with pytest.raises(ValueError, match="include_classifier_head=False"):
            model(
                lab_codes=tiny_batch["lab_codes"],
                lab_values=tiny_batch["lab_values"],
                label=tiny_batch["label"],
            )

    def test_forward_without_mlm_head_has_no_categorical_continuous_outputs(
        self, mock_dataset_binary, tiny_batch
    ):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=8,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            include_mlm_head=False,
        )
        out = model(
            lab_codes=tiny_batch["lab_codes"],
            lab_values=tiny_batch["lab_values"],
            label=tiny_batch["label"],
        )
        assert "categorical_output" not in out
        assert "continuous_output" not in out

    def test_all_padding_true_no_nan(self, mock_dataset_binary, tiny_batch):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=8,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        all_padding = torch.ones_like(tiny_batch["padding_mask"], dtype=torch.bool)
        out = model(
            lab_codes=tiny_batch["lab_codes"],
            lab_values=tiny_batch["lab_values"],
            padding_mask=all_padding,
            label=tiny_batch["label"],
        )
        assert torch.isfinite(out["logit"]).all()
        assert torch.isfinite(out["loss"])

    def test_single_backward_path(self, mock_dataset_binary, tiny_batch):
        model = LabradorModel(
            dataset=mock_dataset_binary,
            vocab_size=8,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            include_mlm_head=True,
        )
        masked_codes = torch.tensor([[1, -1, -1], [3, -1, -1]])
        masked_values = torch.tensor([[0.1, -1.0, -1.0], [0.3, -1.0, -1.0]])
        out = model(
            lab_codes=tiny_batch["lab_codes"],
            lab_values=tiny_batch["lab_values"],
            padding_mask=tiny_batch["padding_mask"],
            label=tiny_batch["label"],
            masked_lab_codes=masked_codes,
            masked_lab_values=masked_values,
        )
        assert "categorical_mlm_loss" in out
        assert "continuous_mlm_loss" in out
        total_loss = out["loss"] + out["categorical_mlm_loss"] + out["continuous_mlm_loss"]
        total_loss.backward()
        assert any(param.grad is not None for param in model.parameters())
