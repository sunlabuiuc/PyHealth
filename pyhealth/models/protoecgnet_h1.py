"""Reference-compatible morphology building blocks for ProtoECGNet H1.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., ProtoECGNet, MLHC 2025.
https://proceedings.mlr.press/v298/sethi25a.html

H1 adds pooled-similarity consistency to the average of two complete original
objectives. The morphology branch below preserves the arithmetic of author
commit fefbe6166a87a961c9c65d220d6344e8d53f2c49, rather than changing the
existing ProtoECGNet defaults. A complete PyHealth model is added separately.
"""

# Reference-compatible calculations adapted from bbj-lab/protoecgnet.
# MIT License
# Copyright (c) 2025 Sahil Sethi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor

from .protoecgnet import PrototypeBranch


class MorphologyPrototypeBranch(PrototypeBranch):
    """Local morphology branch with the pinned research similarity and losses.

    Reuses the existing 2D ResNet18, add-on layers, prototype assignment and
    bias-free classifier. Prototypes have three latent steps and use temporal
    top-5 mean pooling. No pretrained weights are downloaded. The default bank
    has 18 prototypes per label; 52 labels give 936 prototypes.

    The reference divides both normalized windows and normalized prototypes by
    ``sqrt(3)`` and scales windows by ``int(sqrt(proto_dim * 3))``. Thus the
    effective scale is 13 at width 512. Epsilon is inside the square root.
    Clustering/separation average over samples AND labels, including all-zero
    targets. These choices differ from the existing general-purpose branch.

    Args:
        in_channels: Number of ECG leads.
        num_labels: Number of ordered morphology labels.
        proto_dim: Prototype feature width; 512 for the research model.
        prototypes_per_label: Number assigned to each label; 18 in research.

    Note:
        Branch inputs use ``(batch, leads, time)``. The containing model must
        handle data conversion, paired forwards, stages and provenance. This
        branch alone is not a PyHealth Trainer model or checkpoint converter.
    """

    def __init__(
        self,
        in_channels: int,
        num_labels: int,
        proto_dim: int = 512,
        prototypes_per_label: int = 18,
    ) -> None:
        for name, value in (
            ("in_channels", in_channels),
            ("num_labels", num_labels),
            ("proto_dim", proto_dim),
            ("prototypes_per_label", prototypes_per_label),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        super().__init__(
            in_channels=in_channels,
            num_labels=num_labels,
            proto_dim=proto_dim,
            prototypes_per_label=prototypes_per_label,
            joint_prototypes_per_pair=0,
            proto_time_len=3,
            backbone="resnet2d",
            pretrained=False,
            similarity="cosine",
            topk=5,
            similarity_scale=math.isqrt(proto_dim * 3) / 3,
            epsilon=1e-4,
        )

    def similarity_windows(self, windows: Tensor) -> Tensor:
        """Return reference similarities with shape ``(batch, windows, bank)``.

        Mirrors the author's default-dtype activation buffer, including its
        cast when inputs use float64. Normalization retains both input and
        prototype gradients. Compare reference and port on the same runtime.
        """
        if windows.ndim != 3 or windows.shape[-1] != self.proto_dim * self.window:
            raise ValueError("windows must have shape (batch, windows, dim * 3).")
        if windows.shape[0] == 0 or windows.shape[1] == 0:
            raise ValueError("At least one sample and window are required.")
        factor = math.sqrt(self.window)
        prototypes = self.prototype_vectors.flatten(1)
        proto_norm = torch.sqrt(prototypes.square().sum(1, keepdim=True) + self.epsilon)
        normalized_p = prototypes / proto_norm / factor
        window_norm = torch.sqrt(windows.square().sum(-1, keepdim=True) + self.epsilon)
        normalized_x = math.isqrt(self.proto_dim * self.window) * windows / window_norm
        similarities = (normalized_x / factor) @ normalized_p.t()
        return similarities.to(dtype=torch.get_default_dtype())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, top-5 mean similarities and best latent-window indices.

        Reject fewer than five windows rather than silently lowering top-k.
        """
        windows = self.windows(self.feature_map(x))
        if windows.shape[1] < self.topk:
            raise ValueError("Morphology top-5 pooling requires at least five windows.")
        similarities = self.similarity_windows(windows)
        pooled = similarities.topk(self.topk, dim=1).values.mean(dim=1)
        return self.classifier(pooled), pooled, similarities.argmax(dim=1)

    def prototype_terms(
        self,
        similarity: Tensor,
        y: Tensor,
        cooccurrence: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return unweighted reference ``clst/sep/spars/div/cnrst`` losses.

        Co-occurrence is optional; None disables contrastive loss. An explicit
        all-zero matrix still enables it, as in the reference. It must use the
        same label order and device, use float32 as in the reference, and come
        from training data. Class-specific masks multiply similarities before
        maximization; zero entries participate in the maximum.
        """
        if similarity.ndim != 2 or similarity.shape[1] != self.num_prototypes:
            raise ValueError("similarity must have shape (batch, num_prototypes).")
        if similarity.shape[0] == 0 or y.shape != (len(similarity), self.num_labels):
            raise ValueError("targets must have shape (nonempty batch, num_labels).")
        if cooccurrence is not None:
            if cooccurrence.shape != (self.num_labels, self.num_labels):
                raise ValueError(
                    "cooccurrence must have shape (num_labels, num_labels)."
                )
            if not bool(torch.isfinite(cooccurrence).all()) or bool(
                ((cooccurrence < 0) | (cooccurrence > 1)).any()
            ):
                raise ValueError("cooccurrence must contain finite values in [0, 1].")
        correct = self.prototype_class_identity[None] * y.float()[:, None, :]
        expanded = similarity[:, :, None]
        prototypes = F.normalize(self.prototype_vectors.flatten(1), dim=1)
        gram = prototypes @ prototypes.t()
        count = self.num_prototypes
        eye = torch.eye(count, device=gram.device)
        terms = {
            "clst": -(expanded * correct).max(dim=1).values.mean(),
            "sep": (expanded * (1 - correct)).max(dim=1).values.mean(),
            "spars": similarity.clamp_min(0).sum(dim=1).mean(),
            "div": (gram - eye).square().sum() / count**2,
            "cnrst": gram.new_zeros(()),
        }
        if cooccurrence is not None:
            assignment = self.prototype_class_identity.float()
            positive = assignment @ cooccurrence @ assignment.t()
            negative = 1 - positive
            positive = positive / (positive.sum() + 1e-6)
            negative = negative / (negative.sum() + 1e-6)
            cosine = gram.clamp(-1 + 1e-6, 1 - 1e-6)
            terms["cnrst"] = (
                -((cosine * positive).sum() - (cosine * negative).sum()) / count**0.5
            )
        return terms

    def original_objective(
        self,
        logits: Tensor,
        targets: Tensor,
        similarity: Tensor,
        class_weights: Tensor | None = None,
        cooccurrence: Tensor | None = None,
        *,
        lam_clst: float = 0.004,
        lam_sep: float = 0.0004,
        lam_spars: float = 0.0,
        lam_div: float = 250.0,
        lam_cnrst: float = 300.0,
    ) -> dict[str, Tensor]:
        """Return one complete original objective and unweighted components.

        Labels are cast to float32 as in the author loss. ``classification``
        is BCE with positive-class weights; ``clst/sep/spars/div/cnrst`` are
        unweighted terms. ``loss`` is their weighted sum, without H1 or refit
        L1. Pass the two complete ``loss`` tensors to
        :func:`paired_similarity_objective` during paired joint training.

        Both views must receive identical targets and training-only statistics.
        This method does not infer a stage or generate noisy inputs.
        """
        if logits.shape != (len(similarity), self.num_labels):
            raise ValueError("logits must have shape (batch, num_labels).")
        targets = targets.float()
        terms = self.prototype_terms(similarity, targets, cooccurrence)
        classification = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=class_weights
        )
        loss = (
            classification
            + lam_clst * terms["clst"]
            + lam_sep * terms["sep"]
            + lam_spars * terms["spars"]
            + lam_div * terms["div"]
            + lam_cnrst * terms["cnrst"]
        )
        return {"loss": loss, "classification": classification, **terms}


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
