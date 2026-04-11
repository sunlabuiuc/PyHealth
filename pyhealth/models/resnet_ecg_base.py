"""Shared building blocks for 1-D ResNet-based ECG models.

This module provides:

* :class:`BasicBlock1d` – the two-conv residual block for ResNet-18/34.
* :class:`Bottleneck1d` – the three-conv bottleneck block for ResNet-50+.
* :class:`ResNet1d` – a generic 1-D ResNet backbone whose block type and
  layer counts are fully configurable via constructor arguments.
* :class:`ECGBackboneModel` – an abstract :class:`~pyhealth.models.BaseModel`
  that owns the shared prediction head, :meth:`forward`, and
  :meth:`forward_sliding_window` inherited by every ECG ResNet variant.

References:
    He K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

    Nonaka N. & Seita J. (2021). In-depth Benchmarking of Deep Neural Network
    Architectures for ECG Diagnosis. *PMLR* 149:1–19.
    https://proceedings.mlr.press/v149/nonaka21a.html
"""

from typing import Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


# ---------------------------------------------------------------------------
# BasicBlock1d
# ---------------------------------------------------------------------------

class BasicBlock1d(nn.Module):
    """1-D two-conv residual basic block (ResNet-18/34).

    Directly mirrors ``torchvision.models.resnet.BasicBlock`` with all 2-D
    operations replaced by 1-D equivalents.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first convolution. Default ``1``.

    Examples:
        >>> block = BasicBlock1d(64, 128, stride=2)
        >>> block(torch.randn(4, 64, 500)).shape
        torch.Size([4, 128, 250])
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def _conv_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Conv1 → BN1 → ReLU → Conv2 → BN2.

        Extracted so SE/Lambda subclasses can insert attention after BN2
        and before the residual addition.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self._conv_branch(x) + identity)


# ---------------------------------------------------------------------------
# Bottleneck1d
# ---------------------------------------------------------------------------

class Bottleneck1d(nn.Module):
    """1-D three-conv bottleneck residual block (ResNet-50+).

    Mirrors ``torchvision.models.resnet.Bottleneck`` with 1-D operations.
    The channel expansion factor is 4 (``planes * 4`` output channels).

    Args:
        in_channels (int): Number of input channels.
        planes (int): Base channel width; output is ``planes * 4``.
        stride (int): Stride of the 3×1 convolution. Default ``1``.

    Examples:
        >>> block = Bottleneck1d(64, 64)   # 64 → 256 channels
        >>> block(torch.randn(4, 64, 500)).shape
        torch.Size([4, 256, 500])
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv1d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def _conv_branch(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.bn3(self.conv3(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self._conv_branch(x) + identity)


# ---------------------------------------------------------------------------
# ResNet1d
# ---------------------------------------------------------------------------

BlockType = Union[Type[BasicBlock1d], Type[Bottleneck1d]]


class ResNet1d(nn.Module):
    """Generic configurable 1-D ResNet backbone.

    Args:
        in_channels (int): Input channels (ECG leads).
        layers (List[int]): Blocks per stage, e.g. ``[2, 2, 2, 2]``.
        block (BlockType): Block constructor (:class:`BasicBlock1d` or
            :class:`Bottleneck1d`, or an augmented subclass).
        base_channels (int): Width of the first residual stage. Default ``64``.
        output_dim (int, optional): Projection output dimension.  ``None``
            returns the raw GAP output. Default ``256``.
        block_kwargs (dict, optional): Extra keyword arguments forwarded to
            every block constructor call.

    Examples:
        >>> bb = ResNet1d(12, [2, 2, 2, 2], BasicBlock1d, output_dim=256)
        >>> bb(torch.randn(4, 12, 1250)).shape
        torch.Size([4, 256])
    """

    def __init__(
        self,
        in_channels: int,
        layers: List[int],
        block: BlockType,
        base_channels: int = 64,
        output_dim: Optional[int] = 256,
        block_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        block_kwargs = block_kwargs or {}

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        channel_widths = [base_channels * (2 ** i) for i in range(4)]
        strides = [1, 2, 2, 2]

        self.stages = nn.ModuleList()
        in_ch = base_channels
        for n_blocks, stride, out_ch in zip(layers, strides, channel_widths):
            # First block may change spatial resolution and/or channel width;
            # remaining blocks keep stride=1.
            stage: List[nn.Module] = [
                block(in_ch, out_ch, stride=stride, **block_kwargs)
            ]
            in_ch = out_ch * block.expansion  # type: ignore[attr-defined]
            for _ in range(1, n_blocks):
                stage.append(block(in_ch, out_ch, stride=1, **block_kwargs))
            self.stages.append(nn.Sequential(*stage))

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.proj: Optional[nn.Linear] = None
        final_ch = channel_widths[-1] * block.expansion  # type: ignore[attr-defined]
        if output_dim is not None:
            self.proj = nn.Linear(final_ch, output_dim)
            self.out_channels = output_dim
        else:
            self.out_channels = final_ch

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x).squeeze(-1)
        if self.proj is not None:
            x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# ECGBackboneModel – shared PyHealth BaseModel
# ---------------------------------------------------------------------------

class ECGBackboneModel(BaseModel):
    """Abstract base class for ECG ResNet-variant PyHealth models.

    Subclass template::

        class MyECGModel(ECGBackboneModel):
            def __init__(self, dataset, ...):
                super().__init__(dataset)
                self.backbone = MyBackbone(...)
                self._build_head(backbone_output_dim, dropout)

    Provides :meth:`_build_head`, :meth:`forward`, and
    :meth:`forward_sliding_window`.  Subclasses only need to set
    ``self.backbone`` and call ``self._build_head()``.
    """

    def __init__(self, dataset: SampleDataset) -> None:
        super().__init__(dataset=dataset)
        assert len(self.feature_keys) == 1, (
            f"{type(self).__name__} expects exactly one feature key."
        )
        assert len(self.label_keys) == 1, (
            f"{type(self).__name__} expects exactly one label key."
        )
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.backbone: nn.Module  # assigned by subclass before _build_head()

    def _build_head(self, backbone_output_dim: int, dropout: float = 0.25) -> None:
        """Build the prediction head from Nonaka & Seita (2021).

        ``Linear(d, 128) → ReLU → BN(128) → Dropout(0.25) → Linear(128, n_classes)``

        Must be called after ``self.backbone`` has been assigned.

        Args:
            backbone_output_dim (int): Output dimension of the backbone.
            dropout (float): Dropout probability. Default ``0.25``.
        """
        output_size = self.get_output_size()
        self.head = nn.Sequential(
            nn.Linear(backbone_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for a single fixed-length window.

        Args:
            **kwargs: Must contain the feature key (tensor ``(batch, n_leads,
                window_length)``) and the label key.

        Returns:
            Dict with keys ``loss``, ``y_prob``, ``y_true``, ``logit``, and
            optionally ``embed`` when ``kwargs["embed"]`` is ``True``.
        """
        x: torch.Tensor = kwargs[self.feature_key].to(self.device)
        emb = self.backbone(x)
        logits = self.head(emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results

    def forward_sliding_window(
        self,
        signal: torch.Tensor,
        window_size: int,
        step_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Sliding-window evaluation (Nonaka & Seita, 2021, Section 4.2).

        Splits *signal* into overlapping windows, runs the model on each, and
        returns the per-class **maximum** probability across windows.

        Args:
            signal (torch.Tensor): ``(batch, n_leads, total_length)``.
            window_size (int): Samples per window (e.g. ``1250`` for 2.5 s at
                500 Hz).
            step_size (int, optional): Stride between windows. Defaults to
                ``window_size // 2`` (50 % overlap).

        Returns:
            torch.Tensor: ``(batch, n_classes)``.
        """
        if step_size is None:
            step_size = window_size // 2

        total_length = signal.shape[-1]
        starts = list(range(0, total_length - window_size + 1, step_size))
        if not starts:
            signal = F.pad(signal, (0, window_size - total_length))
            starts = [0]

        all_probs: List[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for start in starts:
                window = signal[..., start: start + window_size].to(self.device)
                logits = self.head(self.backbone(window))
                all_probs.append(self.prepare_y_prob(logits))

        return torch.stack(all_probs, dim=1).max(dim=1).values
