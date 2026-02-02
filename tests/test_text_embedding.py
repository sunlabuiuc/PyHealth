import torch
import pytest
import warnings
from pyhealth.models.text_embedding import TextEmbedding


def test_text_embedding_initialization():
    """Test module initialization with default parameters."""
    model = TextEmbedding(embedding_dim=64, chunk_size=128)
    assert model.embedding_dim == 64
    assert model.chunk_size == 128
    assert model.pooling == "none"
    assert model.return_mask is True
    assert model.max_chunks == 64
    assert model.transformer is not None
    assert model.tokenizer is not None
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.fc.out_features == 64


def test_text_embedding_forward_short_text():
    """Test forward pass with text shorter than chunk_size."""
    model = TextEmbedding(embedding_dim=32, chunk_size=128, pooling="none")
    texts = ["This is a test sentence.", "Another medical report."]

    embeddings, mask = model(texts)

    # Check shape: [B, T, E']
    assert embeddings.dim() == 3
    assert embeddings.shape[0] == 2
    assert embeddings.shape[2] == 32
    # Check mask shape matches
    assert mask.shape[0] == 2
    assert mask.shape[1] == embeddings.shape[1]
    # Check mask dtype is bool
    assert mask.dtype == torch.bool


def test_text_embedding_backward_compat_no_mask():
    """Test backward compatibility with return_mask=False."""
    model = TextEmbedding(embedding_dim=32, return_mask=False)
    texts = ["Test sentence."]

    result = model(texts)

    # Should return just tensor, not tuple
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 3


def test_text_embedding_chunking_long_text():
    """Test that long texts are chunked into 128-token segments."""
    model = TextEmbedding(embedding_dim=32, chunk_size=128, pooling="none", max_chunks=None)
    # Create a long text that will exceed 128 tokens
    long_text = " ".join(["word"] * 500)  # ~500 tokens

    embeddings, mask = model([long_text])

    # Should have more than 128 tokens due to chunking
    assert embeddings.shape[1] > 128
    # Valid tokens should match mask
    valid_count = mask[0].sum().item()
    assert valid_count > 128


def test_text_embedding_max_chunks_warning():
    """Test that exceeding max_chunks triggers a warning."""
    model = TextEmbedding(embedding_dim=32, chunk_size=128, max_chunks=2)
    # Create text that needs more than 2 chunks
    long_text = " ".join(["word"] * 500)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        embeddings, mask = model([long_text])
        # Should have triggered a warning
        assert len(w) == 1
        assert "truncated" in str(w[0].message).lower()


def test_text_embedding_pooling_cls():
    """Test CLS pooling returns one embedding per chunk."""
    model = TextEmbedding(embedding_dim=32, chunk_size=128, pooling="cls", max_chunks=None)
    long_text = " ".join(["word"] * 500)

    embeddings, mask = model([long_text])

    # With CLS pooling, output is [B, num_chunks, E']
    assert embeddings.dim() == 3
    assert embeddings.shape[2] == 32
    assert embeddings.shape[1] >= 4  # At least 4 chunks for 500 words


def test_text_embedding_pooling_mean():
    """Test mean pooling returns one embedding per chunk."""
    model = TextEmbedding(embedding_dim=32, chunk_size=128, pooling="mean")
    texts = ["Short text.", "Another short one."]

    embeddings, mask = model(texts)

    # With mean pooling, each short text should produce 1 chunk
    assert embeddings.shape[0] == 2
    assert embeddings.shape[2] == 32


def test_text_embedding_freeze():
    """Test that freeze option works correctly."""
    model = TextEmbedding(embedding_dim=32, freeze=True)
    for param in model.transformer.parameters():
        assert param.requires_grad is False
    # fc layer should still be trainable
    assert model.fc.weight.requires_grad is True


def test_text_embedding_eval_mode_deterministic():
    """Test that eval mode produces deterministic outputs (dropout disabled)."""
    model = TextEmbedding(embedding_dim=32)
    model.eval()
    texts = ["Test determinism."]

    with torch.no_grad():
        emb1, _ = model(texts)
        emb2, _ = model(texts)

    # Should be identical in eval mode
    assert torch.allclose(emb1, emb2)


def test_text_embedding_device():
    """Test that model works on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    model = TextEmbedding(embedding_dim=32).to(device)
    texts = ["Test on gpu"]

    embeddings, mask = model(texts)
    assert embeddings.device.type == "cuda"
    assert mask.device.type == "cuda"


def test_text_embedding_returns_mask():
    """Test that returned mask correctly indicates valid positions."""
    model = TextEmbedding(embedding_dim=32, chunk_size=128, pooling="none")
    # Two texts of different lengths should have different valid token counts
    texts = ["Short.", "This is a longer sentence with more tokens."]

    embeddings, mask = model(texts)

    # Both should be padded to same length
    assert embeddings.shape[1] == mask.shape[1]
    # Second text should have more valid tokens
    valid_1 = mask[0].sum().item()
    valid_2 = mask[1].sum().item()
    assert valid_2 > valid_1


def test_text_embedding_mask_convention():
    """Test mask convention: True=valid, False=padding."""
    model = TextEmbedding(embedding_dim=32, pooling="none")
    texts = ["Short.", "Longer text here."]

    embeddings, mask = model(texts)

    # First sample should have fewer valid tokens
    # Mask should have True for valid, False for padding
    valid_1 = mask[0].sum().item()
    valid_2 = mask[1].sum().item()

    # After padding, both rows have same length but different valid counts
    assert valid_1 < valid_2
    # All positions that are True in mask should have non-zero embeddings
    # (projections from actual tokens, not padding)


def test_text_embedding_empty_text():
    """Test handling of empty text."""
    model = TextEmbedding(embedding_dim=32)

    embeddings, mask = model([""])

    # Should produce at least [CLS][SEP]
    assert embeddings.shape[1] >= 2
    assert mask.sum().item() >= 2
