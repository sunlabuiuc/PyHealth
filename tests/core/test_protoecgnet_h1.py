"""Fast H1 objective checks on synthetic tensors; no ECGs or model training.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., MLHC 2025.
ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning.
https://proceedings.mlr.press/v298/sethi25a.html
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.models.protoecgnet_h1 import (
    MorphologyPrototypeBranch,
    paired_similarity_objective,
)


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


class TinyMorphologyBackbone(nn.Module):
    """Small substitute for branch unit tests, not a ResNet parity claim."""

    def __init__(self, in_leads: int, pretrained: bool = False) -> None:
        super().__init__()
        assert pretrained is False
        self.out_channels = 4
        self.conv = nn.Conv1d(in_leads, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def morphology(monkeypatch):
    from pyhealth.models import protoecgnet

    monkeypatch.setattr(protoecgnet, "ResNet2dBackbone", TinyMorphologyBackbone)
    with torch.random.fork_rng():
        torch.manual_seed(681)
        return MorphologyPrototypeBranch(2, 2, proto_dim=4, prototypes_per_label=2)


def test_morphology_configuration_and_bank_ownership(morphology):
    assert morphology.topk == 5
    assert morphology.window == 3
    assert morphology.prototype_vectors.shape == (4, 4, 3)
    assert morphology.classifier.bias is None
    torch.testing.assert_close(
        morphology.prototype_class_identity,
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )


def test_reference_scale_and_epsilon_have_known_answers(morphology):
    # D=4, L=3 => int(sqrt(12))/3 = 1. Both vectors are twelve ones.
    with torch.no_grad():
        morphology.prototype_vectors.fill_(1.0)
    windows = torch.ones(1, 2, 12)
    actual = morphology.similarity_windows(windows)
    torch.testing.assert_close(actual, torch.full((1, 2, 4), 12 / (12 + 1e-4)))
    # At tiny norms, F.normalize would hide this epsilon placement difference.
    windows = torch.full((1, 2, 12), 1e-5)
    expected = 12e-5 / ((12e-10 + 1e-4) * (12 + 1e-4)) ** 0.5
    torch.testing.assert_close(
        morphology.similarity_windows(windows), torch.full((1, 2, 4), expected)
    )


@pytest.mark.parametrize(
    "window_magnitude,prototype_magnitude",
    [(1.0, 1.0), (1e-5, 1.0), (1.0, 1e-5), (1e-5, 1e-5)],
)
def test_research_width_scale_and_epsilon_have_known_answers(
    monkeypatch, window_magnitude, prototype_magnitude
):
    """Check scale 13 and both normalization epsilons at width 512 cheaply."""
    from pyhealth.models import protoecgnet

    monkeypatch.setattr(protoecgnet, "ResNet2dBackbone", TinyMorphologyBackbone)
    with torch.random.fork_rng():
        branch = MorphologyPrototypeBranch(2, 2, proto_dim=512, prototypes_per_label=1)
    # Only latent windows are evaluated; no ResNet forward or ECG data is needed.
    ones = torch.ones(512 * 3)
    alternating = ones.clone()
    alternating[1::2] = -1.0
    with torch.no_grad():
        branch.prototype_vectors.copy_(
            torch.stack((ones, alternating)).reshape(2, 512, 3) * prototype_magnitude
        )
        windows = torch.stack((ones, -ones, alternating)).unsqueeze(0)
        actual = branch.similarity_windows(windows * window_magnitude)

    # Independent known answer: aligned/opposed cosines +/-1 and orthogonal 0.
    # The research factor is int(sqrt(1536)) / 3 = 39 / 3 = 13.
    numerator = 13.0 * 1536 * window_magnitude * prototype_magnitude
    denominator = (
        (1536 * window_magnitude**2 + 1e-4) * (1536 * prototype_magnitude**2 + 1e-4)
    ) ** 0.5
    expected = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    expected *= numerator / denominator
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_zero_vectors_are_finite_and_keep_gradients(morphology):
    with torch.no_grad():
        morphology.prototype_vectors.zero_()
    windows = torch.zeros(1, 5, 12, requires_grad=True)
    scores = morphology.similarity_windows(windows)
    gradients = torch.autograd.grad(
        scores.sum(), (windows, morphology.prototype_vectors)
    )
    assert torch.count_nonzero(scores) == 0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_reference_activation_buffer_dtype_is_preserved(morphology):
    morphology.double()
    windows = torch.ones(1, 5, 12, dtype=torch.float64, requires_grad=True)
    scores = morphology.similarity_windows(windows)
    assert scores.dtype == torch.get_default_dtype()
    gradients = torch.autograd.grad(
        scores.sum(), (windows, morphology.prototype_vectors)
    )
    assert all(gradient.dtype == torch.float64 for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_classwise_reduction_includes_all_zero_targets(morphology):
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    terms = morphology.prototype_terms(scores, targets)
    assert terms["clst"].item() == -0.5
    assert terms["sep"].item() == 4.0
    assert terms["spars"].item() == 10.0
    assert terms["cnrst"].item() == 0.0


def test_zero_mask_entries_participate_in_maximum(morphology):
    terms = morphology.prototype_terms(
        torch.tensor([[-1.0, -2.0, -3.0, -4.0]]), torch.tensor([[1.0, 0.0]])
    )
    assert terms["clst"].item() == 0.0
    assert terms["sep"].item() == -0.5
    assert terms["spars"].item() == 0.0


def test_explicit_zero_cooccurrence_does_not_disable_contrastive(morphology):
    scores = torch.ones(1, 4)
    targets = torch.zeros(1, 2)
    # Identical vectors have off-diagonal cosine 1: the negative mask is active.
    with torch.no_grad():
        morphology.prototype_vectors.fill_(1.0)
    actual = morphology.prototype_terms(scores, targets, torch.zeros(2, 2))
    torch.testing.assert_close(actual["cnrst"], torch.tensor((1 - 1e-6) / 2))


def test_complete_original_loss_has_all_regularizers_once(morphology):
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    logits = morphology.classifier(scores)
    weights = torch.tensor([2.0, 3.0])
    result = morphology.original_objective(
        logits,
        targets,
        scores,
        weights,
        torch.eye(2),
        lam_clst=2.0,
        lam_sep=3.0,
        lam_spars=4.0,
        lam_div=5.0,
        lam_cnrst=6.0,
    )
    expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)
    expected = expected + sum(
        coefficient * result[name]
        for name, coefficient in zip(
            ("clst", "sep", "spars", "div", "cnrst"), range(2, 7)
        )
    )
    torch.testing.assert_close(result["loss"], expected)
    gradients = torch.autograd.grad(
        result["loss"],
        (scores, morphology.prototype_vectors, morphology.classifier.weight),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_forward_returns_top5_mean_and_reconstructable_logits(morphology):
    x = torch.arange(36).reshape(2, 2, 9).float() / 36
    logits, pooled, best = morphology(x)
    similarities = morphology.similarity_windows(
        morphology.windows(morphology.feature_map(x))
    )
    assert logits.shape == (2, 2)
    assert pooled.shape == best.shape == (2, 4)
    torch.testing.assert_close(
        pooled, similarities.sort(dim=1, descending=True).values[:, :5].mean(1)
    )
    torch.testing.assert_close(best, similarities.argmax(1))
    contributions = pooled[:, None, :] * morphology.classifier.weight[None]
    torch.testing.assert_close(logits, contributions.sum(-1))
    logits.sum().backward()
    assert morphology.backbone.conv.weight.grad is not None
    assert morphology.prototype_vectors.grad is not None


def test_fewer_than_five_windows_is_rejected(morphology):
    with pytest.raises(ValueError, match="at least five windows"):
        morphology(torch.ones(1, 2, 6))


@pytest.mark.parametrize(
    "kwargs", [{"num_labels": 0}, {"proto_dim": True}, {"prototypes_per_label": -1}]
)
def test_invalid_morphology_configuration_is_rejected(kwargs):
    options = {"in_channels": 2, "num_labels": 2, **kwargs}
    with pytest.raises(ValueError, match="positive integer"):
        MorphologyPrototypeBranch(**options)


@pytest.mark.parametrize("shape", [(2, 12), (1, 2, 11), (0, 2, 12), (1, 0, 12)])
def test_invalid_window_shapes_are_rejected(morphology, shape):
    with pytest.raises(ValueError):
        morphology.similarity_windows(torch.ones(shape))


@pytest.mark.parametrize(
    "matrix", [torch.eye(3), torch.full((2, 2), float("nan")), torch.full((2, 2), 1.1)]
)
def test_invalid_cooccurrence_is_rejected(morphology, matrix):
    with pytest.raises(ValueError, match="cooccurrence"):
        morphology.prototype_terms(torch.ones(1, 4), torch.ones(1, 2), matrix)
