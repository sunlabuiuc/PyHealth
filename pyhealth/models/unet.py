"""U-Net models, losses and inference helpers for 2D/3D segmentation.

The U-Net architecture is implemented locally so its defaults follow the
original U-Net paper (64 base channels, depth 5, no normalisation,
up-convolutions). The differentiable Dice term and sliding-window inference
use MONAI, which is an optional dependency in the ``imaging`` extra.
``DiceLoss`` and ``SegmentationLoss`` remain here as thin, model-specific
wrappers so the repository keeps its convention of not adding a separate
``losses/`` package.

Examples:
    >>> import torch
    >>> from pyhealth.datasets import create_sample_dataset
    >>> from pyhealth.models import UNet2D
    >>> samples = [
    ...     {
    ...         "patient_id": "p0",
    ...         "image": torch.zeros(1, 16, 16),
    ...         "mask": torch.zeros(1, 16, 16, dtype=torch.long),
    ...     }
    ... ]
    >>> dataset = create_sample_dataset(
    ...     samples, {"image": "raw"}, {"mask": "raw"}
    ... )
    >>> model = UNet2D(dataset, in_channels=1, num_classes=2, base_channels=4, depth=2)
    >>> model.mode
    'segmentation_multiclass'
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.metrics.segmentation import (
    remap_class_mask,
    segmentation_metrics_fn,
)
from pyhealth.processors import (
    LabelProcessor,
    SegmentationBinaryLabelProcessor,
    SegmentationMultiClassLabelProcessor,
)

from .base_model import BaseModel

logger = logging.getLogger(__name__)


def _prepare_binary_target(
    target: torch.Tensor,
    logits: torch.Tensor,
    ignore_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return a float target shaped like the binary logits and a valid mask."""
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    if target.shape != logits.shape:
        raise ValueError(
            f"Binary target shape {tuple(target.shape)} does not match logits "
            f"shape {tuple(logits.shape)}."
        )
    target = target.float()
    valid = None
    if ignore_index is not None:
        valid = target != float(ignore_index)
        target = torch.where(valid, target, torch.zeros_like(target))
    return target, valid


def _require_monai_dice():
    """Import MONAI's DiceLoss with an actionable optional-dependency error."""
    try:
        from monai.losses import DiceLoss as MonaiDiceLoss
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Segmentation losses require MONAI. Install it with "
            "`pip install 'monai>=1.6.1,<2'` or `pip install pyhealth[imaging]`."
        ) from exc
    return MonaiDiceLoss


def _require_monai_inferer():
    """Import MONAI's sliding-window inferer with an actionable error."""
    try:
        from monai.inferers import sliding_window_inference as monai_swi
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Sliding-window inference requires MONAI. Install it with "
            "`pip install 'monai>=1.6.1,<2'` or `pip install pyhealth[imaging]`."
        ) from exc
    return monai_swi


class DiceLoss(nn.Module):
    """Soft Dice loss backed by :class:`monai.losses.DiceLoss`.

    Args:
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        include_background: Whether to include class 0 in the multiclass
            average. Defaults to False.
        smooth: Smoothing constant added to the Dice numerator and
            denominator.
        ignore_index: Optional label value to ignore in the loss.

    Examples:
        >>> import torch
        >>> from pyhealth.models.unet import DiceLoss
        >>> logits = torch.zeros(2, 3, 4, 4)
        >>> target = torch.zeros(2, 1, 4, 4, dtype=torch.long)
        >>> loss = DiceLoss("segmentation_multiclass")(logits, target)
        >>> float(loss) >= 0
        True
    """

    def __init__(
        self,
        mode: str = "segmentation_multiclass",
        include_background: bool = False,
        smooth: float = 1e-5,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("segmentation_binary", "segmentation_multiclass"):
            raise ValueError(f"Unsupported Dice mode: {mode!r}.")
        monai_dice = _require_monai_dice()
        self.mode = mode
        self.include_background = bool(include_background)
        common = {
            "include_background": (
                True if mode == "segmentation_binary" else include_background
            ),
            "smooth_nr": float(smooth),
            "smooth_dr": float(smooth),
            "ignore_index": ignore_index,
        }
        if mode == "segmentation_binary":
            common.update(
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
            )
        else:
            common.update(
                to_onehot_y=True,
                sigmoid=False,
                softmax=True,
            )
        self._loss = monai_dice(reduction="mean", **common)
        self._none_loss = monai_dice(reduction="none", **common)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the soft Dice loss.

        Args:
            logits: ``(batch, 1, *spatial)`` for binary or
                ``(batch, num_classes, *spatial)`` for multiclass.
            target: Integer mask with shape ``(batch, *spatial)`` or
                ``(batch, 1, *spatial)``.
            class_mask: Optional ``(batch, num_classes)`` boolean mask. False
                classes are excluded from the multiclass average.

        Returns:
            A scalar loss.
        """
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        if class_mask is None:
            return self._loss(logits, target)
        if self.mode == "segmentation_binary":
            raise ValueError("class_mask is only supported for multiclass Dice.")
        mask = torch.as_tensor(class_mask, device=logits.device, dtype=torch.bool)
        num_classes = logits.shape[1]
        if mask.ndim != 2 or mask.shape != (logits.shape[0], num_classes):
            raise ValueError(
                "class_mask must have shape "
                f"{(logits.shape[0], num_classes)}, got {tuple(mask.shape)}."
            )
        per_class = self._none_loss(logits, target)
        if not self.include_background and num_classes > 1:
            mask = mask[:, 1:]
        per_class = per_class[mask]
        if per_class.numel() == 0:
            return logits.new_zeros(())
        return per_class.mean()


class SegmentationLoss(nn.Module):
    """Cross-entropy/BCE plus MONAI Dice for segmentation.

    The Dice term is :class:`monai.losses.DiceLoss`; CE/BCE use PyTorch's
    built-in losses. The optional ``class_mask`` argument excludes classes that
    are not annotated in a particular sample.

    Args:
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        ce_weight: Weight of the cross-entropy/BCE term.
        dice_weight: Weight of the Dice term.
        include_background: Whether Dice includes class 0.
        ignore_index: Optional label value to ignore.
        smooth: Smoothing constant for Dice.

    Examples:
        >>> import torch
        >>> from pyhealth.models.unet import SegmentationLoss
        >>> logits = torch.zeros(1, 2, 4, 4)
        >>> target = torch.zeros(1, 4, 4, dtype=torch.long)
        >>> float(SegmentationLoss()(logits, target)) > 0
        True
    """

    def __init__(
        self,
        mode: str = "segmentation_multiclass",
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = False,
        ignore_index: int | None = None,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if mode not in ("segmentation_binary", "segmentation_multiclass"):
            raise ValueError(f"Unsupported segmentation mode: {mode!r}.")
        self.mode = mode
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = ignore_index
        # Internally all ignored/padding voxels are mapped to -100 so the same
        # Dice instance serves both standard and partial-label batches.
        self.dice = DiceLoss(
            mode=mode,
            include_background=include_background,
            smooth=smooth,
            ignore_index=-100,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined segmentation loss.

        Args:
            logits: Model output logits.
            target: Integer mask.
            class_mask: Optional ``(batch, num_classes)`` boolean mask.  It
                marks which classes are annotated for each sample.  Classes
                marked False are excluded from both the cross-entropy and Dice
                terms, which is useful for partially annotated datasets such
                as CHAOS task 4 (CT annotates the liver only).

        Returns:
            A scalar loss.
        """
        if class_mask is not None:
            if self.mode != "segmentation_multiclass":
                raise ValueError(
                    "class_mask is only supported for multiclass segmentation."
                )
            return self._forward_with_class_mask(logits, target, class_mask)
        if self.mode == "segmentation_binary":
            target_long = target.long()
            if self.ignore_index is not None:
                target_long = torch.where(
                    target_long == int(self.ignore_index),
                    torch.full_like(target_long, -100),
                    target_long,
                )
            target_float, valid = _prepare_binary_target(
                target_long, logits, -100
            )
            elementwise = F.binary_cross_entropy_with_logits(
                logits, target_float, reduction="none"
            )
            if valid is not None:
                elementwise = elementwise * valid
                normalizer = valid.sum().clamp(min=1)
            else:
                normalizer = torch.tensor(
                    elementwise.numel(), device=elementwise.device
                )
            ce_loss = elementwise.sum() / normalizer
        else:
            target_long = target.long()
            if (
                target_long.ndim == logits.ndim
                and target_long.shape[1] == 1
            ):
                target_long = target_long[:, 0]
            if target_long.ndim != logits.ndim - 1:
                raise ValueError(
                    "Multiclass target shape does not match logits: "
                    f"{tuple(target.shape)} vs {tuple(logits.shape)}."
                )
            if self.ignore_index is not None:
                target_long = torch.where(
                    target_long == int(self.ignore_index),
                    torch.full_like(target_long, -100),
                    target_long,
                )
            if bool((target_long != -100).any()):
                ce_loss = F.cross_entropy(
                    logits,
                    target_long,
                    reduction="mean",
                    ignore_index=-100,
                )
            else:
                ce_loss = logits.new_zeros(())
        target_for_dice = (
            target_long.unsqueeze(1)
            if target_long.ndim == logits.ndim - 1
            else target_long
        )
        return self.ce_weight * ce_loss + self.dice_weight * self.dice(
            logits, target_for_dice
        )

    def _forward_with_class_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use ``ignore_index`` and masked logits for partial labels.

        PyTorch's ``cross_entropy`` can ignore target voxels, but it cannot
        ignore an output class.  Invalid classes are therefore filled with a
        very small logit so their softmax probability is zero, while target
        voxels whose label is not in ``class_mask`` are mapped to
        ``ignore_index``.  This keeps the loss vectorized and lets every task
        decide which classes it annotates.
        """
        mask = torch.as_tensor(
            class_mask, device=logits.device, dtype=torch.bool
        )
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]
        if mask.ndim != 2 or mask.shape != (batch_size, num_classes):
            raise ValueError(
                "class_mask must have shape "
                f"{(batch_size, num_classes)}, got {tuple(mask.shape)}."
            )
        target_long = target.long()
        if target_long.ndim == logits.ndim and target_long.shape[1] == 1:
            target_long = target_long[:, 0]
        if target_long.ndim != logits.ndim - 1:
            raise ValueError(
                "Multiclass target shape does not match logits: "
                f"{tuple(target.shape)} vs {tuple(logits.shape)}."
            )

        ignore_index = -100
        in_range = (target_long >= 0) & (target_long < num_classes)
        safe_target = target_long.clamp(0, num_classes - 1)
        target_valid = in_range & mask.gather(
            1, safe_target.reshape(batch_size, -1)
        ).reshape_as(target_long)
        if self.ignore_index is not None:
            target_valid = target_valid & (
                target_long != int(self.ignore_index)
            )
        target_for_ce = torch.where(
            target_valid,
            target_long,
            torch.full_like(target_long, ignore_index),
        )

        logits_mask = mask.reshape(
            batch_size, num_classes, *([1] * (logits.ndim - 2))
        )
        masked_logits = logits.masked_fill(
            ~logits_mask, torch.finfo(logits.dtype).min
        )
        if bool(target_valid.any()):
            ce_loss = F.cross_entropy(
                masked_logits,
                target_for_ce,
                reduction="mean",
                ignore_index=ignore_index,
            )
        else:
            ce_loss = logits.new_zeros(())
        target_for_dice = (
            target_for_ce.unsqueeze(1)
            if target_for_ce.ndim == masked_logits.ndim - 1
            else target_for_ce
        )
        dice_loss = self.dice(
            masked_logits, target_for_dice, class_mask=mask
        )
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def _conv_cls(spatial_dim: int):
    if spatial_dim == 2:
        return nn.Conv2d
    if spatial_dim == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported spatial dimension: {spatial_dim}.")


def _norm_cls(spatial_dim: int, norm: str | None):
    if norm in (None, "none"):
        return None
    if norm == "batch":
        return nn.BatchNorm2d if spatial_dim == 2 else nn.BatchNorm3d
    if norm == "instance":
        return nn.InstanceNorm2d if spatial_dim == 2 else nn.InstanceNorm3d
    raise ValueError(f"Unsupported norm: {norm!r}.")


def _interpolate(
    tensor: torch.Tensor,
    size: Sequence[int],
    spatial_dim: int,
    mode: str | None = None,
) -> torch.Tensor:
    """Resize the last spatial dimensions of a batched tensor."""
    if tuple(tensor.shape[2:]) == tuple(size):
        return tensor
    if mode is None:
        mode = "bilinear" if spatial_dim == 2 else "trilinear"
    return F.interpolate(
        tensor,
        size=tuple(size),
        mode=mode,
        align_corners=False,
    )


class _DoubleConv(nn.Module):
    """Two convolution-normalisation-activation blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        norm: str | None = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        conv = _conv_cls(spatial_dim)
        norm_cls = _norm_cls(spatial_dim, norm)
        layers: list[nn.Module] = [conv(in_channels, out_channels, 3, padding=1)]
        if norm_cls is not None:
            layers.append(norm_cls(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv(out_channels, out_channels, 3, padding=1))
        if norm_cls is not None:
            layers.append(norm_cls(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Downsampling block: max-pool followed by a double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        norm: str | None = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pool = nn.MaxPool2d(2) if spatial_dim == 2 else nn.MaxPool3d(2)
        self.block = nn.Sequential(
            pool,
            _DoubleConv(
                in_channels, out_channels, spatial_dim, norm=norm, dropout=dropout
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    """Upsampling block with a skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        spatial_dim: int,
        bilinear: bool,
        norm: str | None = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        if bilinear:
            mode = "bilinear" if spatial_dim == 2 else "trilinear"
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
            up_channels = in_channels
        else:
            conv_transpose = (
                nn.ConvTranspose2d if spatial_dim == 2 else nn.ConvTranspose3d
            )
            self.up = conv_transpose(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            up_channels = in_channels // 2
        self.conv = _DoubleConv(
            up_channels + skip_channels,
            out_channels,
            spatial_dim,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _interpolate(x, skip.shape[2:], self.spatial_dim)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(BaseModel):
    """Configurable U-Net for 2D or 3D segmentation.

    Args:
        dataset: A ``SampleDataset`` with one image input and one mask output.
        in_channels: Number of input channels. Inferred from the dataset when
            omitted.
        num_classes: Number of output classes. ``1`` selects binary
            segmentation; values above 1 select multiclass segmentation.
            Inferred from the mask processor or the first sample when omitted.
        spatial_dim: ``2`` or ``3``. Inferred from the input processor or the
            first sample when omitted.
        base_channels: Number of channels after the first convolution.
            Defaults to 64, matching the original U-Net paper.
        depth: Number of resolution levels. Defaults to 5, matching the
            original U-Net paper. Input spatial sizes should be divisible by
            ``2 ** (depth - 1)``; smaller or odd inputs are padded internally.
        bilinear: Use bilinear/trilinear upsampling instead of transposed
            convolutions. Defaults to False, matching the original
            up-convolution.
        dropout: Dropout probability in each double convolution.
        norm: ``"batch"``, ``"instance"`` or ``None``. Defaults to None,
            matching the original U-Net paper.
        dice_weight: Weight of the Dice term in the training loss.
        ce_weight: Weight of the cross-entropy/BCE term.
        include_background: Whether Dice includes class 0.
        ignore_index: Optional mask value ignored by the loss.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import UNet3D
        >>> samples = [{
        ...     "patient_id": "p0",
        ...     "image": torch.zeros(1, 8, 8, 8),
        ...     "mask": torch.zeros(1, 8, 8, 8, dtype=torch.long),
        ... }]
        >>> dataset = create_sample_dataset(
        ...     samples, {"image": "raw"}, {"mask": "raw"}
        ... )
        >>> model = UNet3D(dataset, in_channels=1, num_classes=2,
        ...                base_channels=4, depth=2)
        >>> model.mode
        'segmentation_multiclass'
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int | None = None,
        num_classes: int | None = None,
        spatial_dim: int | None = None,
        base_channels: int = 64,
        depth: int = 5,
        bilinear: bool = False,
        dropout: float = 0.0,
        norm: str | None = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
        ignore_index: int | None = 255,
    ) -> None:
        super().__init__(dataset)
        assert len(self.feature_keys) == 1, (
            "UNet supports exactly one image input feature."
        )
        assert len(self.label_keys) == 1, (
            "UNet supports exactly one mask output feature."
        )
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.ignore_index = ignore_index
        self.segmentation_ignore_index = ignore_index
        self.spatial_dim = spatial_dim or self._infer_spatial_dim()
        self.in_channels = in_channels or self._infer_in_channels()
        self.num_classes = num_classes or self._infer_num_classes()
        if self.num_classes < 1:
            raise ValueError("num_classes must be at least 1.")
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.bilinear = bool(bilinear)
        if self.depth < 1:
            raise ValueError("depth must be at least 1.")
        if self.base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        if self.depth > 1 and not self.bilinear and self.base_channels < 2:
            raise ValueError(
                "base_channels must be at least 2 when using transposed "
                "convolutions with depth > 1."
            )

        self._segmentation_label_processor = (
            SegmentationMultiClassLabelProcessor(
                self.num_classes, ignore_index=ignore_index
            )
            if self.num_classes > 1
            else SegmentationBinaryLabelProcessor(
                num_classes=2, ignore_index=ignore_index
            )
        )
        self.loss_fn = SegmentationLoss(
            mode=self.mode,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            include_background=include_background,
            ignore_index=ignore_index,
        )
        self.network = self._build_network(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            depth=self.depth,
            bilinear=self.bilinear,
            dropout=dropout,
            norm=norm,
        )

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The mask label kind used by :class:`~pyhealth.trainer.Trainer`.

        The mask is read by an image processor, which is not a label
        processor. UNet supplies the segmentation label kind itself, the same
        way :class:`~pyhealth.models.APN` supplies its forecast label, so the
        Trainer can activate, gather and score variable-shaped masks without
        changing the shared trainer code.
        """
        return {self.label_key: self._segmentation_label_processor}

    def _infer_spatial_dim(self) -> int:
        """Infer 2D vs 3D from the input processor or the first sample."""
        processor = self.dataset.input_processors.get(self.feature_key)
        if processor is not None and hasattr(processor, "dim"):
            try:
                rank = processor.dim()[0]
            except (NotImplementedError, TypeError, IndexError):
                rank = None
            if rank == 3:
                return 2
            if rank == 4:
                return 3
        sample = self.dataset[0][self.feature_key]
        if isinstance(sample, tuple):
            sample = sample[0]
        sample = _materialize(sample, processor)
        if sample.dim() == 3:
            return 2
        if sample.dim() == 4:
            return 3
        raise ValueError(
            "Unable to infer spatial_dim; pass spatial_dim=2 or 3 explicitly."
        )

    def _infer_in_channels(self) -> int:
        """Infer the number of input channels."""
        processor = self.dataset.input_processors.get(self.feature_key)
        channels = getattr(processor, "channels", None)
        if channels is not None:
            return int(channels)
        sample = self.dataset[0][self.feature_key]
        if isinstance(sample, tuple):
            sample = sample[0]
        sample = _materialize(sample, processor)
        return int(sample.shape[0])

    def _infer_num_classes(self) -> int:
        """Infer the number of mask classes and warn when guessing."""
        processor = self.dataset.output_processors.get(self.label_key)
        num_classes = getattr(processor, "num_classes", None)
        if num_classes is not None:
            return int(num_classes)
        sample = self.dataset[0][self.label_key]
        if isinstance(sample, tuple):
            sample = sample[0]
        sample = _materialize(sample, processor)
        values = sample
        if self.ignore_index is not None:
            values = sample[sample != int(self.ignore_index)]
        inferred = int(values.max().item()) + 1 if values.numel() else 1
        logger.warning(
            "num_classes was inferred from the first mask as %d; pass "
            "num_classes explicitly if the dataset contains rarer labels.",
            inferred,
        )
        return inferred

    @staticmethod
    def _build_network(
        spatial_dim: int,
        in_channels: int,
        num_classes: int,
        base_channels: int,
        depth: int,
        bilinear: bool,
        dropout: float,
        norm: str | None,
    ) -> nn.Module:
        """Build the encoder-decoder network."""
        channels = [base_channels * (2**i) for i in range(depth)]
        down_blocks = nn.ModuleList(
            [
                _Down(
                    channels[i],
                    channels[i + 1],
                    spatial_dim,
                    norm=norm,
                    dropout=dropout,
                )
                for i in range(depth - 1)
            ]
        )
        up_blocks = nn.ModuleList(
            [
                _Up(
                    channels[i + 1],
                    channels[i],
                    channels[i],
                    spatial_dim,
                    bilinear=bilinear,
                    norm=norm,
                    dropout=dropout,
                )
                for i in reversed(range(depth - 1))
            ]
        )
        return _UNetNetwork(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
            bilinear=bilinear,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass.

        The label is optional: without it, ``loss`` and ``y_true`` are omitted,
        which is useful for sliding-window inference.
        """
        image = kwargs[self.feature_key]
        if isinstance(image, tuple):
            image = image[0]
        image = image.to(self.device).float()
        if image.dim() != self.spatial_dim + 2:
            raise ValueError(
                f"Expected a {self.spatial_dim + 2}-D image tensor, got "
                f"{tuple(image.shape)}."
            )
        original_shape = tuple(int(v) for v in image.shape[2:])
        factor = 2 ** (self.depth - 1)
        padded_shape = tuple(
            max(factor, ((size + factor - 1) // factor) * factor)
            for size in original_shape
        )
        if padded_shape != original_shape:
            pads: list[int] = []
            for size, padded in zip(
                reversed(original_shape), reversed(padded_shape)
            ):
                pads.extend([0, padded - size])
            image = F.pad(image, pads, mode="replicate")
        logits = self.network(image)
        if tuple(logits.shape[2:]) != original_shape:
            slices = (slice(None), slice(None)) + tuple(
                slice(0, size) for size in original_shape
            )
            logits = logits[slices]
        if self.num_classes > 1:
            y_prob = F.softmax(logits, dim=1)
        else:
            y_prob = torch.sigmoid(logits)
        results: dict[str, torch.Tensor] = {"logit": logits, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            loss_kwargs = {}
            if "class_mask" in kwargs:
                loss_kwargs["class_mask"] = kwargs["class_mask"]
            results["loss"] = self.loss_fn(logits, y_true, **loss_kwargs)
            results["y_true"] = y_true
        return results


class _UNetNetwork(nn.Module):
    """Inner network used by :class:`UNet`."""

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        num_classes: int,
        channels: list[int],
        down_blocks: nn.ModuleList,
        up_blocks: nn.ModuleList,
        bilinear: bool,
        norm: str | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.inc = _DoubleConv(
            in_channels, channels[0], spatial_dim, norm=norm, dropout=dropout
        )
        self.downs = down_blocks
        self.ups = up_blocks
        conv = _conv_cls(spatial_dim)
        self.outc = conv(channels[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.inc(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)


class UNet2D(UNet):
    """2D U-Net. See :class:`UNet` for arguments.

    Examples:
        >>> from pyhealth.models import UNet2D
        >>> UNet2D.__name__
        'UNet2D'
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["spatial_dim"] = 2
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    """3D U-Net. See :class:`UNet` for arguments.

    Examples:
        >>> from pyhealth.models import UNet3D
        >>> UNet3D.__name__
        'UNet3D'
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["spatial_dim"] = 3
        super().__init__(*args, **kwargs)


def sliding_window_inference(
    model: UNet,
    image: torch.Tensor,
    patch_size: int | Sequence[int],
    overlap: float = 0.5,
    feature_key: str | None = None,
    batch_size: int = 8,
) -> torch.Tensor:
    """Run sliding-window inference and return per-class probabilities.

    Args:
        model: A :class:`UNet`-style model.
        image: ``(C, *spatial)`` image tensor.
        patch_size: Patch size as one int or a 2/3-tuple.
        overlap: Fractional overlap between neighboring patches. Defaults to
            0.5.
        feature_key: Input field name. Defaults to ``model.feature_key``.
        batch_size: Number of patches processed per forward pass.

    Returns:
        A ``(num_classes, *spatial)`` probability tensor cropped to the
        original spatial shape.

    Examples:
        >>> import torch
        >>> from pyhealth.models import UNet3D, sliding_window_inference
        >>> model = UNet3D(...)  # doctest: +SKIP
        >>> probability = sliding_window_inference(  # doctest: +SKIP
        ...     model, torch.zeros(1, 64, 64, 64), patch_size=32
        ... )
    """
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    feature_key = feature_key or model.feature_key
    spatial_dim = image.dim() - 1
    if spatial_dim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image, got {tuple(image.shape)}.")
    if isinstance(patch_size, int):
        patch = (int(patch_size),) * spatial_dim
    else:
        patch = tuple(int(v) for v in patch_size)
        if len(patch) != spatial_dim:
            raise ValueError(
                f"patch_size {patch} does not match image spatial rank "
                f"{spatial_dim}."
            )
    if any(v <= 0 for v in patch):
        raise ValueError("patch_size values must be positive.")

    monai_swi = _require_monai_inferer()
    image = image.to(model.device).float()
    was_training = model.training
    model.eval()

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        return model(**{feature_key: patch})["y_prob"]

    try:
        with torch.no_grad():
            probability = monai_swi(
                image.unsqueeze(0),
                roi_size=patch,
                sw_batch_size=max(1, int(batch_size)),
                predictor=predictor,
                overlap=float(overlap),
                mode="constant",
            )
    finally:
        model.train(was_training)
    return probability[0]


def _materialize(value: Any, processor: Any) -> Any:
    """Load a lazy image or mask with its processor's ``materialize()``."""
    if processor is None or not hasattr(processor, "materialize"):
        return value
    if isinstance(value, (str, Path, dict)) or (
        isinstance(value, (list, tuple))
        and all(isinstance(item, (str, Path)) for item in value)
    ):
        return processor.materialize(value)
    return value


def evaluate_segmentation(
    model: UNet,
    dataset: Any,
    patch_size: int | Sequence[int] | None = None,
    overlap: float = 0.5,
    metrics: Sequence[str] | None = None,
    input_key: str | None = None,
    label_key: str | None = None,
    batch_size: int = 8,
    ignore_index: int | None = None,
    include_background: bool = False,
) -> dict[str, float]:
    """Evaluate one segmentation model at the case level.

    Args:
        model: A :class:`UNet`-style model.
        dataset: A dataset whose items contain an image and a mask.
        patch_size: Patch size for sliding-window inference. ``None`` runs the
            whole volume in one forward pass.
        overlap: Sliding-window overlap.
        metrics: Metric names. Defaults to ``["dice", "iou"]``.
        input_key: Image field name. Defaults to ``model.feature_key``.
        label_key: Mask field name. Defaults to ``model.label_key``.
        batch_size: Sliding-window batch size.
        ignore_index: Optional label value excluded from the metrics.
        include_background: Whether class 0 is included in multiclass Dice/IoU.

    Returns:
        Case-level metrics averaged over the dataset.

    Each sample may include a boolean ``class_mask`` to exclude unannotated
    classes; the metrics are remapped per case. ``Trainer.evaluate`` does not
    read ``class_mask``.

    Examples:
        >>> from pyhealth.models import UNet3D, evaluate_segmentation
        >>> model = UNet3D(...)  # doctest: +SKIP
        >>> scores = evaluate_segmentation(  # doctest: +SKIP
        ...     model, val_dataset, patch_size=(64, 64, 64)
        ... )
    """
    input_key = input_key or model.feature_key
    label_key = label_key or model.label_key
    if ignore_index is None:
        ignore_index = getattr(model, "segmentation_ignore_index", None)
    processors = {
        **getattr(dataset, "input_processors", {}),
        **getattr(dataset, "output_processors", {}),
    }
    true_cases = []
    prob_cases = []
    case_classes: list[int] = []
    class_mask_used = False
    was_training = model.training
    model.eval()
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            image = _materialize(
                sample[input_key], processors.get(input_key)
            )
            target = _materialize(
                sample[label_key], processors.get(label_key)
            )
            if isinstance(image, tuple):
                image = image[0]
            image = image.to(model.device).float()
            if patch_size is None:
                with torch.no_grad():
                    output = model(**{input_key: image.unsqueeze(0)})
                probability = output["y_prob"][0]
            else:
                probability = sliding_window_inference(
                    model,
                    image,
                    patch_size=patch_size,
                    overlap=overlap,
                    feature_key=input_key,
                    batch_size=batch_size,
                )
            class_mask = (
                sample.get("class_mask") if hasattr(sample, "get") else None
            )
            if class_mask is not None:
                mask_ignore_index = (
                    ignore_index if ignore_index is not None else 255
                )
                target, probability = remap_class_mask(
                    target,
                    probability,
                    class_mask,
                    ignore_index=mask_ignore_index,
                )
                true_cases.append(target)
                prob_cases.append(probability)
                case_classes.append(
                    int(torch.as_tensor(class_mask).sum().item())
                )
                class_mask_used = True
            else:
                true_cases.append(target)
                prob_cases.append(probability.cpu())
                case_classes.append(int(model.num_classes))
        if class_mask_used and ignore_index is None:
            ignore_index = 255
        return segmentation_metrics_fn(
            true_cases,
            prob_cases,
            metrics=metrics,
            mode=model.mode,
            ignore_index=ignore_index,
            include_background=include_background,
            num_classes=(
                case_classes if class_mask_used else model.num_classes
            ),
        )
    finally:
        model.train(was_training)
