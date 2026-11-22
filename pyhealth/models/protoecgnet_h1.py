"""Paired-view objective building blocks for ProtoECGNet H1.

Contributor: Phil Tsai (UIUC NetID: ct68).
Paper: Sethi et al., ProtoECGNet, MLHC 2025.
https://proceedings.mlr.press/v298/sethi25a.html

H1 adds pooled-similarity consistency to the average of two complete original
objectives. This module does not yet provide a model or a training runner.
"""

import math
from numbers import Real

import torch
from torch import Tensor


def paired_similarity_objective(
    clean_original_loss: Tensor,
    noisy_original_loss: Tensor,
    clean_similarity: Tensor,
    noisy_similarity: Tensor,
    lambda_cons: float = 0.0,
) -> dict[str, Tensor]:
    """Combine complete original losses and a two-sided consistency penalty.

    The caller must obtain both views from one combined clean/noisy forward
    pass, using the same targets and the intended BatchNorm/dropout policy.
    Each original loss must already include all original regularizers with
    their original coefficients and reduction. Averaging both complete losses
    counts prototype-only regularizers once. No input is detached.

    A zero coefficient recovers paired-view training without consistency; it
    does not recover single-view training. The experimental value 117608 is
    specific to the locked reference computation, not a portable default.

    Args:
        clean_original_loss: Scalar complete original objective for clean ECGs.
        noisy_original_loss: Scalar complete original objective for noisy ECGs.
        clean_similarity: Pooled similarities of shape ``(batch, prototypes)``.
        noisy_similarity: Corresponding pooled similarities, same shape, dtype
            and device as ``clean_similarity``.
        lambda_cons: Finite, nonnegative consistency coefficient.

    Returns:
        Differentiable scalar tensors under ``loss``, ``base_loss``,
        ``consistency_loss`` and ``weighted_consistency_loss``. Consistency
        uses the mean over all batch/prototype elements, not their sum.

    Raises:
        TypeError: A tensor has a non-floating dtype or the coefficient is not
            a real number.
        ValueError: A coefficient, shape or device is incompatible.

    Examples:
        >>> import torch
        >>> from pyhealth.models.protoecgnet_h1 import paired_similarity_objective
        >>> clean = torch.tensor([[1., 2.]], requires_grad=True)
        >>> noisy = torch.tensor([[2., 4.]], requires_grad=True)
        >>> result = paired_similarity_objective(
        ...     torch.tensor(2.), torch.tensor(4.), clean, noisy, lambda_cons=2.
        ... )
        >>> result["loss"].item()
        8.0
        >>> result["loss"].backward()
        >>> noisy.grad is not None and clean.grad is not None
        True
    """
    if not isinstance(lambda_cons, Real) or isinstance(lambda_cons, bool):
        raise TypeError("lambda_cons must be a finite nonnegative real number.")
    if not math.isfinite(lambda_cons) or lambda_cons < 0:
        raise ValueError("lambda_cons must be finite and nonnegative.")
    tensors = (
        clean_original_loss,
        noisy_original_loss,
        clean_similarity,
        noisy_similarity,
    )
    if any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError("Losses and similarities must be torch tensors.")
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("Losses and similarities must have floating dtypes.")
    if clean_original_loss.ndim != 0 or noisy_original_loss.ndim != 0:
        raise ValueError("Each complete original loss must be a scalar.")
    if clean_similarity.ndim != 2 or min(clean_similarity.shape) == 0:
        raise ValueError("Similarity shape must be nonempty (batch, prototypes).")
    if clean_similarity.shape != noisy_similarity.shape:
        raise ValueError("Clean/noisy similarities must have identical shapes.")
    if clean_similarity.dtype != noisy_similarity.dtype:
        raise ValueError("Clean/noisy similarities must have the same dtype.")
    if any(value.device != clean_similarity.device for value in tensors):
        raise ValueError("All losses and similarities must share one device.")

    base = 0.5 * (clean_original_loss + noisy_original_loss)
    consistency = torch.mean((clean_similarity - noisy_similarity).square())
    weighted = float(lambda_cons) * consistency
    return {
        "loss": base + weighted,
        "base_loss": base,
        "consistency_loss": consistency,
        "weighted_consistency_loss": weighted,
    }
