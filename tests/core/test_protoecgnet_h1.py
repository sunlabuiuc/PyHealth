"""Fast H1 objective checks on synthetic tensors; no ECGs or model training."""

import pytest
import torch
import torch.nn.functional as F
from pyhealth.models.protoecgnet_h1 import paired_similarity_objective


def test_known_answer_and_analytic_two_sided_gradients():
    """Check values and both view gradients against hand-derived numbers."""
    clean = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    noisy = torch.tensor([[2.0, 4.0]], dtype=torch.float64, requires_grad=True)
    clean_loss = torch.tensor(2.0, requires_grad=True)
    noisy_loss = torch.tensor(4.0, requires_grad=True)
    result = paired_similarity_objective(
        clean_loss, noisy_loss, clean, noisy, lambda_cons=2.0
    )
    assert result["base_loss"].item() == 3.0
    assert result["consistency_loss"].item() == 2.5
    assert result["weighted_consistency_loss"].item() == 5.0
    assert result["loss"].item() == 8.0
    result["loss"].backward()
    torch.testing.assert_close(clean.grad, torch.tensor([[-2.0, -4.0]]).double())
    torch.testing.assert_close(noisy.grad, torch.tensor([[2.0, 4.0]]).double())
    assert clean_loss.grad.item() == noisy_loss.grad.item() == 0.5


def test_complete_loss_regularizer_count_and_lambda_zero_recovery():
    """Do not double a shared prototype regularizer or lose its gradient."""
    weight = torch.tensor([[0.2, -0.5], [0.4, 0.3]], requires_grad=True)
    prototype = torch.tensor([0.2, -0.1], requires_grad=True)
    views = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    scores = views @ weight + prototype
    labels = torch.tensor([[1.0, 0.0]])
    regularizer = 3.0 * prototype.square().sum()
    clean_bce = F.binary_cross_entropy_with_logits(scores[:1], labels)
    noisy_bce = F.binary_cross_entropy_with_logits(scores[1:], labels)
    result = paired_similarity_objective(
        clean_bce + regularizer,
        noisy_bce + regularizer,
        scores[:1],
        scores[1:],
        lambda_cons=0.0,
    )
    independent = 0.5 * (clean_bce + noisy_bce) + regularizer
    torch.testing.assert_close(result["loss"], independent, rtol=0, atol=0)
    actual = torch.autograd.grad(result["loss"], (weight, prototype), retain_graph=True)
    expected = torch.autograd.grad(independent, (weight, prototype))
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_replication_preserves_mean_and_shared_parameter_gradient():
    """Duplicating pairs/prototypes must not multiply consistency strength."""
    parameter = torch.tensor([1.0, 3.0], requires_grad=True)
    clean = parameter[None]
    noisy = 2.0 * parameter[None]
    loss = torch.tensor(0.0)
    result = paired_similarity_objective(loss, loss, clean, noisy, 7.0)
    repeated = paired_similarity_objective(
        loss, loss, clean.repeat(3, 4), noisy.repeat(3, 4), 7.0
    )
    torch.testing.assert_close(result["loss"], repeated["loss"])
    first = torch.autograd.grad(result["loss"], parameter, retain_graph=True)[0]
    second = torch.autograd.grad(repeated["loss"], parameter)[0]
    torch.testing.assert_close(first, second)


def test_zero_noise_has_zero_consistency_and_zero_consistency_gradient():
    clean = torch.tensor([[1.0, 2.0]], requires_grad=True)
    noisy = clean.detach().clone().requires_grad_(True)
    result = paired_similarity_objective(
        clean.sum(), noisy.sum(), clean, noisy, lambda_cons=117608.0
    )
    assert result["consistency_loss"].item() == 0.0
    for value in torch.autograd.grad(
        result["weighted_consistency_loss"], (clean, noisy)
    ):
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf")])
def test_invalid_coefficient_is_rejected(coefficient):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        paired_similarity_objective(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.ones(1, 2),
            torch.ones(1, 2),
            coefficient,
        )


@pytest.mark.parametrize("coefficient", [True, "1", torch.tensor(1.0)])
def test_non_numeric_coefficient_is_rejected(coefficient):
    with pytest.raises(TypeError, match="real number"):
        paired_similarity_objective(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.ones(1, 2),
            torch.ones(1, 2),
            coefficient,
        )


@pytest.mark.parametrize("shape", [(2,), (0, 2), (2, 0), (1, 2, 3)])
def test_invalid_similarity_rank_or_empty_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="nonempty"):
        paired_similarity_objective(
            torch.tensor(0.0), torch.tensor(0.0), torch.ones(shape), torch.ones(shape)
        )


def test_mismatched_shapes_are_rejected_instead_of_broadcasting():
    with pytest.raises(ValueError, match="identical shapes"):
        paired_similarity_objective(
            torch.tensor(0.0), torch.tensor(0.0), torch.ones(2, 3), torch.ones(1, 3)
        )


def test_unreduced_original_loss_is_rejected():
    with pytest.raises(ValueError, match="scalar"):
        paired_similarity_objective(
            torch.ones(2), torch.tensor(0.0), torch.ones(2, 3), torch.ones(2, 3)
        )


def test_integer_scores_are_rejected():
    with pytest.raises(TypeError, match="floating"):
        paired_similarity_objective(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2),
        )


def test_score_dtype_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same dtype"):
        paired_similarity_objective(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.ones(1, 2).double(),
            torch.ones(1, 2),
        )


def test_device_mismatch_is_rejected_before_computation():
    with pytest.raises(ValueError, match="one device"):
        paired_similarity_objective(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.ones(1, 2),
            torch.ones(1, 2, device="meta"),
        )
