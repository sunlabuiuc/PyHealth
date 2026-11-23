"""Reference-compatible morphology building blocks for ProtoECGNet H1.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., ProtoECGNet, MLHC 2025.
https://proceedings.mlr.press/v298/sethi25a.html

H1 adds pooled-similarity consistency to the average of two complete original
objectives. The morphology branch preserves the arithmetic of author
commit fefbe6166a87a961c9c65d220d6344e8d53f2c49, rather than changing the
existing ProtoECGNet defaults. ProtoECGNetH1 adds the paired model interface.
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
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..datasets import SampleDataset
from .protoecgnet import ProtoECGNet, PrototypeBranch


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

    def feature_map(self, x: Tensor) -> Tensor:
        """Accept both PyHealth ECG tensors and reference one-channel images."""
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        return super().feature_map(x)

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


class ProtoECGNetH1(ProtoECGNet):
    """Morphology-only ProtoECGNet with a paired similarity objective.

    Uses one combined clean/noisy forward in joint training, averages BOTH
    complete original objectives, and adds mean squared pooled-similarity
    consistency. Neither view is detached. A zero coefficient is the paired
    augmentation control. The research value 117608 must be supplied explicitly;
    it is not recalibrated here and is not a general-purpose default.

    In joint training, the classifier is frozen, BatchNorm trains and dropout
    is disabled. The classifier stage uses clean inputs, weighted BCE and
    off-class L1; feature parameters, prototypes and BN buffers stay fixed.
    Start a new Trainer.train call after switching stages to reset the optimizer.

    Args:
        dataset: One tensor signal input and one multilabel output. The label
            processor defines the exact ordered label vocabulary. Metadata
            such as ECG IDs can remain outside the input/output schemas.
        lambda_cons: Nonnegative coefficient of mean squared consistency.
        in_channels: Number of ECG leads; 12 in the research configuration.
        proto_dim: Prototype width; 512 in the research configuration.
        prototypes_per_label: Bank size per label; 18 in the research setup.
        class_weights: Training-only BCE positive weights in label order.
            If omitted, inherits the dataset processor's weights or ones.
        label_cooccurrence: Training-only float32 co-occurrence in label order.
            None disables contrastive loss; an explicit zero matrix enables it.
        lam_clst: Clustering coefficient.
        lam_sep: Separation coefficient.
        lam_spars: Positive-similarity sparsity coefficient, zero in research.
        lam_div: Diversity coefficient.
        lam_cnrst: Co-occurrence contrastive coefficient.
        lam_l1: Off-class classifier L1 coefficient during refit.

    Note:
        Clean evaluation requires model.eval(). Its loss follows the current
        stage and excludes consistency; classification_loss exposes BCE alone.
        Projection uses the inherited training-loader algorithm and returns
        loader-relative sources. Durable ECG-linked provenance and conversion
        of historical research checkpoints require a separate adapter.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        *,
        lambda_cons: float = 0.0,
        in_channels: int = 12,
        proto_dim: int = 512,
        prototypes_per_label: int = 18,
        class_weights: Tensor | None = None,
        label_cooccurrence: Tensor | None = None,
        lam_clst: float = 0.004,
        lam_sep: float = 0.0004,
        lam_spars: float = 0.0,
        lam_div: float = 250.0,
        lam_cnrst: float = 300.0,
        lam_l1: float = 1e-4,
    ) -> None:
        coefficients = {
            name: self._coefficient(name, value)
            for name, value in {
                "lambda_cons": lambda_cons,
                "lam_clst": lam_clst,
                "lam_sep": lam_sep,
                "lam_spars": lam_spars,
                "lam_div": lam_div,
                "lam_cnrst": lam_cnrst,
                "lam_l1": lam_l1,
            }.items()
        }
        if len(dataset.input_schema) != 1 or len(dataset.output_schema) != 1:
            raise ValueError("ProtoECGNetH1 expects one signal input and one label.")
        label_key = next(iter(dataset.output_schema))
        vocab = getattr(dataset.output_processors[label_key], "label_vocab", None)
        if not vocab or sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError(
                "A nonempty multilabel vocabulary with contiguous indices is required."
            )
        ordered_labels = sorted(vocab, key=vocab.get)
        super().__init__(
            dataset=dataset,
            label_groups={"morphology": ordered_labels},
            in_channels=in_channels,
            proto_dim=proto_dim,
            prototypes_per_label=prototypes_per_label,
            proto_time_len=3,
            topk=5,
            lam_clst=lam_clst,
            lam_sep=lam_sep,
            lam_div=lam_div,
            lam_cnrst=lam_cnrst,
            lam_l1=lam_l1,
            class_weights=class_weights,
            label_cooccurrence=label_cooccurrence,
        )
        if "noisy_signal" in (self.signal_key, self.label_key):
            raise ValueError("noisy_signal is reserved for the paired view.")
        self.in_channels = in_channels
        self._contrastive_enabled = label_cooccurrence is not None
        for name, value in coefficients.items():
            setattr(self, name, value)
        self.set_label_statistics(class_weights, label_cooccurrence)
        self.set_stage("joint")

    @staticmethod
    def _coefficient(name: str, value: Real) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a nonnegative real number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        return float(value)

    def _make_branch(
        self, in_channels: int, num_labels: int, **options: Any
    ) -> MorphologyPrototypeBranch:
        return MorphologyPrototypeBranch(
            in_channels,
            num_labels,
            proto_dim=options["proto_dim"],
            prototypes_per_label=options["prototypes_per_label"],
        )

    def set_label_statistics(
        self,
        class_weights: Tensor | None = None,
        label_cooccurrence: Tensor | None = None,
    ) -> None:
        """Validate and set training-only, ordered label statistics.

        Args:
            class_weights: Positive finite weights with one entry per label.
            label_cooccurrence: Finite symmetric matrix in [0, 1]. None leaves
                the current matrix and its enabled/disabled setting unchanged.
        """
        weights = self.class_weights if class_weights is None else class_weights
        if weights.shape != (self.num_labels,) or not bool(
            torch.isfinite(weights).all() and (weights > 0).all()
        ):
            raise ValueError(
                "class_weights must be positive finite weights in label order."
            )
        if label_cooccurrence is not None:
            matrix = label_cooccurrence
            if matrix.shape != (self.num_labels, self.num_labels) or not bool(
                torch.isfinite(matrix).all()
            ):
                raise ValueError(
                    "label_cooccurrence must be a finite label-by-label matrix."
                )
            if bool(((matrix < 0) | (matrix > 1)).any()) or not torch.allclose(
                matrix.float(), matrix.t().float(), atol=1e-6, rtol=1e-5
            ):
                raise ValueError("label_cooccurrence must be symmetric and in [0, 1].")
        super().set_label_statistics(class_weights, label_cooccurrence)
        if label_cooccurrence is not None:
            self._contrastive_enabled = True

    def set_stage(self, stage: str) -> None:
        """Select joint or classifier training and immediately apply mode policy."""
        if stage not in ("joint", "classifier"):
            raise ValueError("ProtoECGNetH1 stage must be 'joint' or 'classifier'.")
        super().set_stage(stage)
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.grad = None
        self.train(self.training)

    def train(self, mode: bool = True) -> "ProtoECGNetH1":
        """Preserve BN/dropout policy when Trainer calls train each epoch."""
        super().train(mode)
        if mode:
            branch = self.branches["morphology"]
            if self.stage == "classifier":
                # Also freezes buffer-only BN modules with affine=False.
                branch.backbone.eval()
                branch.add_on.eval()
            dropout_types = (
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.AlphaDropout,
                torch.nn.FeatureAlphaDropout,
            )
            for module in self.modules():
                if isinstance(module, dropout_types):
                    module.eval()
        return self

    def _signal(self, value: Tensor) -> Tensor:
        x = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != self.in_channels:
            raise ValueError(
                "Signal shape must be (batch, leads, time) or (batch, 1, leads, time)."
            )
        return x

    def _original(self, logits: Tensor, scores: Tensor, y: Tensor) -> dict[str, Tensor]:
        branch = self.branches["morphology"]
        if self.stage == "classifier":
            classification = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.class_weights
            )
            l1 = self._l1_penalty()
            return {
                "loss": classification + self.lam_l1 * l1,
                "classification": classification,
                "l1": l1,
            }
        return branch.original_objective(
            logits,
            y,
            scores,
            self.class_weights,
            self.label_cooccurrence if self._contrastive_enabled else None,
            lam_clst=self.lam_clst,
            lam_sep=self.lam_sep,
            lam_spars=self.lam_spars,
            lam_div=self.lam_div,
            lam_cnrst=self.lam_cnrst,
        )

    def forward(
        self, noisy_signal: Tensor | None = None, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Return predictions, similarities and the stage-specific objective.

        Args:
            noisy_signal: Optional precomputed noisy view in the same order as
                the clean batch. Required for joint training, also at lambda=0.
                Noise generation stays outside the model.
            **kwargs: Clean tensor under the dataset signal key and optional
                multi-hot targets under its label key. ECG metadata is ignored.

        Returns:
            Standard logit/y_prob/similarity and optional loss/y_true, plus
            best_window and unweighted loss components. Paired output includes
            noisy_logit/noisy_y_prob/noisy_similarity/noisy_best_window,
            base_loss/consistency_loss/weighted_consistency_loss and both full
            original losses. Standard predictions refer to the clean view.
        """
        if self.training and self.stage == "joint" and noisy_signal is None:
            raise ValueError(
                "Joint training requires noisy_signal, including at lambda_cons=0."
            )
        if noisy_signal is not None and self.stage != "joint":
            raise ValueError("noisy_signal is supported only in the joint stage.")
        clean = self._signal(kwargs[self.signal_key])
        batch = len(clean)
        if noisy_signal is None:
            combined = clean
        else:
            noisy = self._signal(noisy_signal)
            if noisy.shape != clean.shape:
                raise ValueError("Clean/noisy signals must have identical shapes.")
            combined = torch.cat([clean, noisy], dim=0)
        # A single branch invocation shares the intended BN statistics.
        logits, scores, best = self.branches["morphology"](combined)
        result = {
            "logit": logits[:batch],
            "y_prob": logits[:batch].sigmoid(),
            "similarity": scores[:batch],
            "best_window": best[:batch],
        }
        if noisy_signal is not None:
            result.update(
                noisy_logit=logits[batch:],
                noisy_y_prob=logits[batch:].sigmoid(),
                noisy_similarity=scores[batch:],
                noisy_best_window=best[batch:],
            )
        if kwargs.get(self.label_key) is None:
            return result
        y = torch.as_tensor(
            kwargs[self.label_key], device=self.device, dtype=torch.float32
        )
        if y.shape != (batch, self.num_labels) or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError("Targets must be binary multi-hot (batch, num_labels).")
        clean_loss = self._original(logits[:batch], scores[:batch], y)
        result["y_true"] = y
        result["clean_original_loss"] = clean_loss["loss"]
        if noisy_signal is None:
            result["loss"] = clean_loss["loss"]
            result.update(
                {
                    f"{name}_loss": value
                    for name, value in clean_loss.items()
                    if name != "loss"
                }
            )
        else:
            noisy_loss = self._original(logits[batch:], scores[batch:], y)
            result["noisy_original_loss"] = noisy_loss["loss"]
            result.update(
                paired_similarity_objective(
                    clean_loss["loss"],
                    noisy_loss["loss"],
                    scores[:batch],
                    scores[batch:],
                    self.lambda_cons,
                )
            )
            result.update(
                {
                    f"{name}_loss": 0.5 * (value + noisy_loss[name])
                    for name, value in clean_loss.items()
                    if name != "loss"
                }
            )
        return result

    def _architecture(self) -> dict[str, Any]:
        branch = self.branches["morphology"]
        return {
            "in_channels": self.in_channels,
            "proto_dim": branch.proto_dim,
            "num_prototypes": branch.num_prototypes,
            "window": 3,
            "topk": 5,
            "reference_commit": "fefbe6166a87a961c9c65d220d6344e8d53f2c49",
        }

    def get_extra_state(self) -> dict[str, Any]:
        """Save ordered labels, architecture, objective settings and stage."""
        names = (
            "lambda_cons",
            "lam_clst",
            "lam_sep",
            "lam_spars",
            "lam_div",
            "lam_cnrst",
            "lam_l1",
        )
        return {
            "h1_format_version": 1,
            "stage": self.stage,
            "label_names": list(self.label_names),
            "architecture": self._architecture(),
            "objective": {name: getattr(self, name) for name in names},
            "contrastive_enabled": self._contrastive_enabled,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Validate labels/architecture and restore the objective and stage."""
        if not isinstance(state, dict) or state.get("h1_format_version") != 1:
            raise ValueError("Missing or unsupported H1 checkpoint metadata.")
        if state.get("label_names") != self.label_names:
            raise ValueError("H1 checkpoint label order does not match the dataset.")
        if state.get("architecture") != self._architecture():
            raise ValueError("H1 checkpoint architecture does not match the model.")
        names = (
            "lambda_cons",
            "lam_clst",
            "lam_sep",
            "lam_spars",
            "lam_div",
            "lam_cnrst",
            "lam_l1",
        )
        objective = state.get("objective", {})
        if set(objective) != set(names) or not isinstance(
            state.get("contrastive_enabled"), bool
        ):
            raise ValueError("Incomplete H1 checkpoint objective metadata.")
        coefficients = {
            name: self._coefficient(name, objective[name]) for name in names
        }
        for name, value in coefficients.items():
            setattr(self, name, value)
        self._contrastive_enabled = state["contrastive_enabled"]
        self.set_stage(state["stage"])
