"""1-D SE-ResNet-50 ECG model.

Implements the ``se_resnet1d50`` backbone used in:

    Nonaka N. & Seita J. (2021). In-depth Benchmarking of Deep Neural Network
    Architectures for ECG Diagnosis. *PMLR* 149:1–19.
    https://proceedings.mlr.press/v149/nonaka21a.html

The SE block is described in:

    Hu J., Shen L. & Sun G. (2018). Squeeze-and-Excitation Networks. *CVPR*.

The paper benchmarks SE-ResNet-**50** (not SE-ResNet-18).  The backbone uses
three-conv bottleneck blocks (expansion = 4) with an SE module applied after
the third convolution and before the residual addition (Figure 3 of Hu et al.).

The reference implementation (``senet1d.py``) uses ``Conv1d(1×1)`` rather
than ``nn.Linear`` for the SE excitation bottleneck, and the downsample
convolution in each stage uses ``kernel_size=1, padding=0`` (not 3/1 as in
the SENet-154 variant).

See :mod:`pyhealth.models.resnet_ecg_base` for shared building blocks.
"""

from typing import Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.resnet_ecg_base import ECGBackboneModel, ResNet1d


# ---------------------------------------------------------------------------
# SE module
# ---------------------------------------------------------------------------

class SEModule1d(nn.Module):
    """1-D Squeeze-and-Excitation module (Hu et al., 2018).

    Uses ``Conv1d(1×1)`` projections (matching the reference ``senet1d.py``).

    Squeeze: ``AdaptiveAvgPool1d(1)`` → ``(batch, C, 1)``
    Excitation: ``Conv1d(C, C//r, 1) → ReLU → Conv1d(C//r, C, 1) → Sigmoid``
    Scale: element-wise multiply input by the channel weights.

    Args:
        channels (int): Number of input channels.
        reduction (int): Bottleneck reduction ratio. Default ``16``.

    Examples:
        >>> se = SEModule1d(256)
        >>> se(torch.randn(4, 256, 312)).shape
        torch.Size([4, 256, 312])
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction,
                             kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels,
                             kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s


# ---------------------------------------------------------------------------
# SE bottleneck block
# ---------------------------------------------------------------------------

class SEResNetBottleneck1d(nn.Module):
    """1-D SE-ResNet bottleneck block.

    Three-conv bottleneck (1×1 → 3×1 → 1×1) with an SE module applied
    after the third convolution and before the residual addition.

    Matches ``SEResNetBottleneck1d`` in the reference ``senet1d.py``.

    Args:
        in_channels (int): Number of input channels.
        planes (int): Base channel width; output is ``planes * 4``.
        stride (int): Stride of the 3×1 convolution. Default ``1``.
        reduction (int): SE reduction ratio. Default ``16``.

    Examples:
        >>> block = SEResNetBottleneck1d(64, 64)   # 64 → 256 channels
        >>> block(torch.randn(4, 64, 500)).shape
        torch.Size([4, 256, 500])
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # Reference uses stride in conv1 (Caffe convention), not in conv2.
        self.conv1 = nn.Conv1d(in_channels, planes,
                               kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule1d(out_channels, reduction=reduction)

        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            # Reference uses kernel_size=1, padding=0 for SE-ResNet variants.
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se_module(out) + identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# SEResNet50ECG  (PyHealth BaseModel)
# ---------------------------------------------------------------------------

class SEResNet50ECG(ECGBackboneModel):
    """SE-ResNet-50 backbone for ECG classification (Nonaka & Seita, 2021).

    Augments a ResNet-50 backbone by inserting a
    :class:`~pyhealth.models.SEModule1d` channel-attention gate into every
    bottleneck block, following Hu et al. (2018).

    This is the variant the paper benchmarks as "SE-ResNet" (``se_resnet1d50``
    in the reference code).  It uses three-conv bottleneck blocks
    (``expansion = 4``) with layer counts ``[3, 4, 6, 3]``.

    All training/evaluation conventions (backbone output dimension, prediction
    head, sliding-window protocol) are identical to
    :class:`~pyhealth.models.ResNet18ECG`.

    Args:
        dataset (SampleDataset): Dataset used to infer feature/label keys,
            output size, and loss function.
        in_channels (int): Number of ECG leads. Default ``12``.
        base_channels (int): Width of the first residual stage. Default ``64``.
        backbone_output_dim (int): Backbone projection output dimension.
            Default ``256``.
        dropout (float): Dropout probability in the prediction head.
            Default ``0.25``.
        reduction (int): SE bottleneck reduction ratio. Default ``16``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "signal": np.random.randn(12, 1250).astype(np.float32),
        ...      "label": [1, 0, 1, 0, 0]},
        ...     {"patient_id": "p1", "visit_id": "v1",
        ...      "signal": np.random.randn(12, 1250).astype(np.float32),
        ...      "label": [0, 1, 0, 1, 0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>> model = SEResNet50ECG(dataset=dataset)
        >>> out = model(**next(iter(get_dataloader(dataset, batch_size=2))))
        >>> sorted(out.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 12,
        base_channels: int = 64,
        backbone_output_dim: int = 256,
        dropout: float = 0.25,
        reduction: int = 16,
    ) -> None:
        super().__init__(dataset=dataset)

        self.backbone = ResNet1d(
            in_channels=in_channels,
            layers=[3, 4, 6, 3],           # ResNet-50 layer counts
            block=SEResNetBottleneck1d,
            base_channels=base_channels,
            output_dim=backbone_output_dim,
            block_kwargs={"reduction": reduction},
        )
        self._build_head(backbone_output_dim, dropout)
