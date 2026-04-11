"""1-D Lambda-ResNet-18 ECG model.

Implements the ``lambda_resnet1d18`` backbone used in:

    Nonaka N. & Seita J. (2021). In-depth Benchmarking of Deep Neural Network
    Architectures for ECG Diagnosis. *PMLR* 149:1–19.
    https://proceedings.mlr.press/v149/nonaka21a.html

The lambda layer is described in:

    Bello I. (2021). LambdaNetworks: Modeling Long-Range Interactions Without
    Attention. *ICLR 2021* (Spotlight).
    https://openreview.net/forum?id=xTJEN-ggl1b

**Architecture notes** (from the reference ``lambdanet1d.py``):

* The model uses **bottleneck blocks** (expansion = 4) for all four stages,
  not basic blocks — giving effective channel widths of 256 / 512 / 1024 / 2048.
* Every bottleneck replaces its middle 3×1 convolution with a
  :class:`LambdaConv1d` layer followed by optional average-pooling (for
  downsampling stages) and BN + ReLU.
* ``dim_u = 4`` (intra-depth dimension) and ``nhead = 4``.
* The local positional context uses a learnable ``nn.Parameter`` shaped as a
  Conv2d weight ``(dim_k, dim_u, 1, dim_m)`` applied via ``F.conv2d``,
  exactly as in the reference.
* A ``Dropout(0.3)`` is placed *inside* the backbone's final FC (before the
  linear projection to ``backbone_out_dim``).
* Input is clamped to ``[-20, 20]`` at the start of each forward pass and
  after each of the first three stages — matching the reference's numerical
  stability guards.

See :mod:`pyhealth.models.resnet_ecg_base` for shared building blocks.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.resnet_ecg_base import ECGBackboneModel

_CLAMP = 20.0


# ---------------------------------------------------------------------------
# Lambda convolution layer
# ---------------------------------------------------------------------------

class LambdaConv1d(nn.Module):
    """1-D lambda layer (Bello, 2021).

    Captures content-based and position-based long-range interactions across
    the full temporal sequence without materialising an attention map.

    Matches ``LambdaConv1d`` in the reference ``lambdanet1d.py``:

    * Queries projected and BN-normalised.
    * Keys projected (no BN), softmax-normalised over the time dimension.
    * Values projected and BN-normalised.
    * **Content lambda**: ``λ_c = softmax(K)ᵀ V`` → ``(B, dim_k, dim_v)``.
    * **Position lambda**: learnable ``nn.Parameter`` shaped as a Conv2d
      weight ``(dim_k, dim_u, 1, dim_m)`` applied to the reshaped values via
      ``F.conv2d``, producing per-position context ``(B, dim_k, dim_v, N)``.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (must be divisible by
            ``nhead``).
        nhead (int): Number of query heads.
        dim_k (int): Key/query depth. Default ``16``.
        dim_u (int): Intra-depth dimension. Default ``4``.
        dim_m (int): Receptive field for the local positional embedding
            (must be odd; ``0`` disables the positional term). Default ``7``.

    Examples:
        >>> layer = LambdaConv1d(256, 256, nhead=4)
        >>> layer(torch.randn(2, 256, 312)).shape
        torch.Size([2, 256, 312])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int,
        dim_k: int = 16,
        dim_u: int = 4,
        dim_m: int = 7,
    ) -> None:
        super().__init__()

        assert out_channels % nhead == 0, (
            f"out_channels ({out_channels}) must be divisible by nhead ({nhead})"
        )

        self.nhead = nhead
        self.dim_k = dim_k
        self.dim_u = dim_u
        self.dim_m = dim_m
        self.dim_v = out_channels // nhead

        self.local_context = dim_m > 0
        self.padding = (dim_m - 1) // 2

        # Projections
        self.to_queries = nn.Sequential(
            nn.Conv1d(in_channels, dim_k * nhead, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim_k * nhead),
        )
        self.to_keys = nn.Sequential(
            nn.Conv1d(in_channels, dim_k * dim_u, kernel_size=1, bias=False),
        )
        self.to_values = nn.Sequential(
            nn.Conv1d(in_channels, self.dim_v * dim_u, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim_v * dim_u),
        )

        # Positional embedding: stored as a Conv2d-shaped parameter so that
        # F.conv2d can apply it efficiently, matching the reference exactly.
        if self.local_context:
            self.embedding = nn.Parameter(
                torch.randn(dim_k, dim_u, 1, dim_m), requires_grad=True
            )
        else:
            self.embedding = nn.Parameter(
                torch.randn(dim_k, dim_u), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N = x.shape
        k, u, v = self.dim_k, self.dim_u, self.dim_v
        h = self.nhead

        queries = self.to_queries(x).view(B, h, k, N)          # (B, h, k, N)

        keys = self.to_keys(x).view(B, k, u, N)                # (B, k, u, N)
        keys = F.softmax(keys, dim=-1)                          # softmax over N

        values = self.to_values(x).view(B, v, u, N)            # (B, v, u, N)

        # Content lambda: λ_c = keys^T values → (B, k, v)
        lambda_c = torch.einsum("bkum,bvum->bkv", keys, values)
        y_c = torch.einsum("bhkn,bkv->bhvn", queries, lambda_c)  # (B, h, v, N)

        # Position lambda
        if self.local_context:
            # values reshaped to (B, u, v, N) then treated as a 2D feature
            # map of size (v, N) with u channels for the Conv2d application.
            values_2d = values.view(B, u, v, N)  # (B, u, v, N)
            # F.conv2d: weight (k, u, 1, dim_m) applied to (B, u, v, N)
            # → (B, k, v, N)
            lambda_p = F.conv2d(values_2d, self.embedding,
                                padding=(0, self.padding))      # (B, k, v, N)
            y_p = torch.einsum("bhkn,bkvn->bhvn", queries, lambda_p)
        else:
            lambda_p = torch.einsum("ku,bvun->bkvn", self.embedding, values)
            y_p = torch.einsum("bhkn,bkvn->bhvn", queries, lambda_p)

        return (y_c + y_p).contiguous().view(B, h * v, N)      # (B, out_ch, N)


# ---------------------------------------------------------------------------
# Lambda bottleneck block
# ---------------------------------------------------------------------------

class LambdaBottleneck1d(nn.Module):
    """1-D Lambda bottleneck block.

    Three-conv bottleneck where the middle 3×1 convolution is replaced by a
    :class:`LambdaConv1d` layer, following ``LambdaBottleneck1d`` in the
    reference ``lambdanet1d.py``.

    For downsampling stages (``stride > 1``) an ``AvgPool1d`` is appended
    after the lambda layer (before BN + ReLU), matching the reference.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Base channel width; output is ``planes * 4``.
        stride (int): Downsampling stride. Default ``1``.

    Examples:
        >>> block = LambdaBottleneck1d(64, 64)   # 64 → 256 channels
        >>> block(torch.randn(4, 64, 312)).shape
        torch.Size([4, 256, 312])
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        # 1×1 bottleneck-down
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        # Lambda layer (replaces 3×1 conv) + optional avg-pool + BN + ReLU
        lambda_layers: List[nn.Module] = [
            LambdaConv1d(planes, planes, nhead=4)
        ]
        if stride != 1 or in_planes != self.expansion * planes:
            lambda_layers.append(
                nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
            )
        lambda_layers.append(nn.BatchNorm1d(planes))
        lambda_layers.append(nn.ReLU())
        self.conv2 = nn.Sequential(*lambda_layers)

        # 1×1 bottleneck-up
        self.conv3 = nn.Conv1d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        # Shortcut
        self.shortcut: nn.Module
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# ---------------------------------------------------------------------------
# LambdaResNet1d backbone
# ---------------------------------------------------------------------------

class LambdaResNet1d(nn.Module):
    """1-D Lambda-ResNet backbone.

    All four stages use :class:`LambdaBottleneck1d` blocks, matching
    ``LambdaResNet1d`` in the reference ``lambdanet1d.py``.

    The backbone's final FC wraps the linear projection with
    ``Dropout(0.3)``, and ``torch.clamp([-20, 20])`` is applied after the
    stem and after each of the first three stages for numerical stability
    (both matching the reference).

    Args:
        num_blocks (List[int]): Blocks per stage.
        num_lead (int): Number of input channels (ECG leads). Default ``12``.
        backbone_out_dim (int): Projection output dimension. Default ``256``.

    Examples:
        >>> bb = LambdaResNet1d([2, 2, 2, 2])
        >>> bb(torch.randn(2, 12, 1250)).shape
        torch.Size([2, 256])
    """

    def __init__(
        self,
        num_blocks: List[int],
        num_lead: int = 12,
        backbone_out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(num_lead, 64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(LambdaBottleneck1d, 64,  num_blocks[0])
        self.layer2 = self._make_layer(LambdaBottleneck1d, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBottleneck1d, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBottleneck1d, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Dropout inside the backbone FC, matching the reference.
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * LambdaBottleneck1d.expansion, backbone_out_dim),
        )

    def _make_layer(
        self,
        block: type,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=-_CLAMP, max=_CLAMP)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = torch.clamp(x, min=-_CLAMP, max=_CLAMP)
        x = self.layer2(x)
        x = torch.clamp(x, min=-_CLAMP, max=_CLAMP)
        x = self.layer3(x)
        x = torch.clamp(x, min=-_CLAMP, max=_CLAMP)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# LambdaResNet18ECG  (PyHealth BaseModel)
# ---------------------------------------------------------------------------

class LambdaResNet18ECG(ECGBackboneModel):
    """Lambda-ResNet-18 backbone for ECG classification (Nonaka & Seita, 2021).

    Replaces the 3×1 convolution in every bottleneck block with a
    :class:`LambdaConv1d` layer, enabling global temporal context modelling
    without explicit attention maps (Bello, 2021).

    This is the ``lambda_resnet1d18`` variant from the reference code: four
    stages of :class:`LambdaBottleneck1d` with layer counts ``[2, 2, 2, 2]``
    and effective channel widths of 256 / 512 / 1024 / 2048 (bottleneck
    expansion = 4).

    All training/evaluation conventions (prediction head, sliding-window
    protocol) are identical to :class:`~pyhealth.models.ResNet18ECG`.

    Args:
        dataset (SampleDataset): Dataset used to infer feature/label keys,
            output size, and loss function.
        in_channels (int): Number of ECG leads. Default ``12``.
        backbone_output_dim (int): Backbone projection output dimension.
            Default ``256``.
        dropout (float): Dropout probability in the prediction head.
            Default ``0.25``.

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
        >>> model = LambdaResNet18ECG(dataset=dataset)
        >>> out = model(**next(iter(get_dataloader(dataset, batch_size=2))))
        >>> sorted(out.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 12,
        backbone_output_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(dataset=dataset)

        self.backbone = LambdaResNet1d(
            num_blocks=[2, 2, 2, 2],
            num_lead=in_channels,
            backbone_out_dim=backbone_output_dim,
        )
        self._build_head(backbone_output_dim, dropout)
