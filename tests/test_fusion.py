import torch

from pyhealth.models.fusion import TransformerFusion


def test_transformer_fusion_all_modalities():
    fusion = TransformerFusion(hidden_dim=64, num_heads=4, num_layers=2)

    inputs = {
        "ecg": torch.randn(2, 100, 64),
        "ppg": torch.randn(2, 100, 64),
        "resp": torch.randn(2, 100, 64),
    }

    out = fusion(inputs)
    assert out.shape == (2, 100, 64)


def test_transformer_fusion_missing_modality():
    fusion = TransformerFusion(hidden_dim=64, num_heads=4, num_layers=2)

    inputs = {
        "ecg": torch.randn(2, 100, 64),
        "resp": torch.randn(2, 100, 64),
    }

    out = fusion(inputs)
    assert out.shape == (2, 100, 64)


def test_transformer_fusion_attention_mask():
    fusion = TransformerFusion(hidden_dim=64, num_heads=4, num_layers=2)

    inputs = {
        "ecg": torch.randn(2, 100, 64),
        "ppg": torch.randn(2, 100, 64),
        "resp": torch.randn(2, 100, 64),
    }

    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    out = fusion(inputs, attention_mask=mask)
    assert out.shape == (2, 100, 64)
