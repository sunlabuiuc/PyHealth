"""Fast synthetic CEHR embedding tests."""

import pytest
import torch

from pyhealth.models.ehrmamba import CEHREmbeddings


def test_shape_and_padding():
    model = CEHREmbeddings(12, 8, dropout=0)
    ids = torch.tensor([[0, 2, 3], [4, 0, 0]])
    result = model(ids)
    assert result.shape == (2, 3, 8)
    torch.testing.assert_close(result[0, 0], torch.zeros(8))
    torch.testing.assert_close(result[1, 1:], torch.zeros(2, 8))


def test_temporal_metadata_and_gradients():
    model = CEHREmbeddings(12, 8, dropout=0)
    ids = torch.tensor([[2, 3]])
    output = model(
        ids,
        ages=torch.tensor([[20, 21]]),
        times=torch.tensor([[1, 2]]),
        segments=torch.tensor([[0, 1]]),
    )
    output.sum().backward()
    assert model.code.weight.grad is not None
    assert torch.isfinite(model.code.weight.grad).all()


def test_invalid_shape():
    with pytest.raises(ValueError):
        CEHREmbeddings(12, 8)(torch.tensor([2, 3]))
