"""Output processors for segmentation masks.

The mask itself is read by an image processor such as
:class:`~pyhealth.processors.NiftiProcessor`; these label kinds tell models and
the :class:`~pyhealth.trainer.Trainer` how to activate, gather and score
variable-shaped masks. Keeping them in their own module leaves
``label_processor.py`` focused on the general classification and regression
label kinds.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .label_processor import LabelProcessor

__all__ = [
    "SegmentationBinaryLabelProcessor",
    "SegmentationMultiClassLabelProcessor",
]


def _segmentation_batches(value: Any) -> list[np.ndarray]:
    """Split one batch of masks or probabilities into per-sample arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return [value[index] for index in range(len(value))]


class _SegmentationLabelProcessor(LabelProcessor):
    """Shared behaviour of the segmentation label kinds.

    Segmentation masks keep their spatial shape, so batches cannot be stacked
    into one array the way classification labels can. :meth:`to_numpy` returns
    one array per sample and :meth:`gather` keeps that list, which is what
    :func:`~pyhealth.metrics.segmentation.segmentation_metrics_fn` expects.

    Args:
        num_classes: Number of classes including background. It is used by
            :meth:`size` and by the metrics so every case scores the same
            class set.
        ignore_index: Label value used for padding, excluded from the metrics.
            ``PatchDataset`` pads masks with 255 by default.

    These label kinds are supplied by the model through
    :attr:`~pyhealth.models.UNet.label_processors`; they are intentionally not
    registered for string schemas or ``get_metrics_fn``. Use the model, or
    call :func:`~pyhealth.metrics.segmentation.segmentation_metrics_fn`
    directly. The ``Trainer`` uses the instance attribute bound in
    ``__init__``, which adds ``ignore_index`` and ``num_classes``.
    """

    stores_tensor = False

    def __init__(
        self,
        num_classes: int | None = None,
        ignore_index: int | None = 255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Trainer calls ``label.metrics_fn(...)`` on the instance. Bind the
        # per-instance options here so padded masks and a fixed class count
        # reach the metrics, while the classmethod below satisfies the
        # ``LabelProcessor`` interface.
        from pyhealth.metrics.segmentation import segmentation_metrics_fn

        self.metrics_fn = functools.partial(
            segmentation_metrics_fn,
            mode=type(self).mode,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def size(self) -> int:
        if self.num_classes is None:
            raise ValueError(
                "A segmentation label processor needs num_classes to report "
                "its size."
            )
        return int(self.num_classes)

    def process(self, value: Any) -> torch.Tensor:
        if isinstance(value, (str, Path, dict)):
            raise TypeError(
                "Segmentation label processors expect a materialised mask. "
                "Use an image processor such as NiftiProcessor for the "
                "output field, or materialise the value first."
            )
        return torch.as_tensor(value)

    def to_numpy(
        self, y_true: Any, y_prob: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """One mask and one probability map per sample."""
        return _segmentation_batches(y_true), _segmentation_batches(y_prob)

    def gather(
        self, y_true: list[np.ndarray], y_prob: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Keep one array per sample instead of concatenating."""
        return y_true, y_prob

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        from pyhealth.metrics.segmentation import segmentation_metrics_fn

        return segmentation_metrics_fn(
            y_true, y_prob, metrics=metrics, mode=cls.mode
        )


class SegmentationBinaryLabelProcessor(_SegmentationLabelProcessor):
    """Binary segmentation labels, one mask channel.

    The mask is already read by an image processor; this processor defines the
    label kind so models and the ``Trainer`` know how to activate, score and
    gather variable-shaped masks. Supplied by
    :attr:`~pyhealth.models.UNet.label_processors`; not available as a string
    schema or through ``get_metrics_fn``.

    Examples:
        >>> import torch
        >>> p = SegmentationBinaryLabelProcessor()
        >>> p.mode
        'segmentation_binary'
        >>> p.activation(torch.tensor([[0.0]])).item()
        0.5
    """

    mode = "segmentation_binary"

    def size(self) -> int:
        """Binary segmentation models have one output channel."""
        return 1

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def loss(
        self, logits: torch.Tensor, y_true: Any, **kwargs: Any
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, y_true, **kwargs)


class SegmentationMultiClassLabelProcessor(_SegmentationLabelProcessor):
    """Multi-class segmentation labels: one integer class per voxel.

    Supplied by :attr:`~pyhealth.models.UNet.label_processors`; not available
    as a string schema or through ``get_metrics_fn``.

    Examples:
        >>> import torch
        >>> p = SegmentationMultiClassLabelProcessor(num_classes=3)
        >>> p.mode
        'segmentation_multiclass'
        >>> p.size()
        3
        >>> probs = p.activation(torch.zeros(1, 3, 2))
        >>> probs.shape
        torch.Size([1, 3, 2])
        >>> float(probs[0, :, 0].sum())
        1.0
    """

    mode = "segmentation_multiclass"

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=1)

    def loss(
        self, logits: torch.Tensor, y_true: Any, **kwargs: Any
    ) -> torch.Tensor:
        if y_true.ndim == logits.ndim and y_true.shape[1] == 1:
            y_true = y_true[:, 0]
        return F.cross_entropy(logits, y_true.long(), **kwargs)
