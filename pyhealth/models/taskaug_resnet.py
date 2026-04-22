# Authors:     Paul Garcia (alanpg2), Rogelio Medina (orm9), Cesar Nava (can14)
# Paper:       Data Augmentation for Electrocardiograms (Raghu et al., CHIL 2022)
# Link:        https://proceedings.mlr.press/v174/raghu22a.html
# Description: TaskAug differentiable augmentation policy with 1-D ResNet-18
#              backbone for binary ECG classification on PTB-XL.

"""TaskAug: Task-Adaptive Data Augmentation with a 1-D ResNet-18 backbone.

Implements the TaskAug framework from:

    Raghu et al. (2022). Data Augmentation for Electrocardiograms.
    Conference on Health, Inference, and Learning (CHIL), PMLR 174.
    https://proceedings.mlr.press/v174/raghu22a.html

Architecture
------------
* :class:`TaskAugPolicy` — K-stage differentiable augmentation policy.
  At each stage one of seven operations is selected via the Gumbel-Softmax
  trick and applied with a class-specific magnitude drawn from learnable
  parameters (``mag_neg`` for label=0, ``mag_pos`` for label=1).

* :class:`_ResNet1D` — 1-D adaptation of ResNet-18 with kernel size 7.

* :class:`TaskAugResNet` — :class:`~pyhealth.models.BaseModel` subclass that
  wires the policy and backbone together.  During training the policy
  augments the input before it reaches the backbone; at inference the raw
  signal is forwarded directly.

For bi-level optimisation (inner loop: backbone on augmented training data;
outer loop: policy on clean validation loss) use the ``BiLevelTrainer``
provided in ``examples/ptbxl_ecg_classification_taskaug_resnet.py``.
"""
from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


# ---------------------------------------------------------------------------
# Seven augmentation operations
# Each accepts x: (B, C, T) and mag: (B,) and returns (B, C, T).
# ---------------------------------------------------------------------------

def _gaussian_noise(x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Add i.i.d. Gaussian noise scaled per sample by *mag*.

    Args:
        x: Input signal of shape ``(B, C, T)``.
        mag: Per-sample noise standard deviation of shape ``(B,)``.

    Returns:
        Noisy signal of shape ``(B, C, T)``.
    """
    return x + mag.view(-1, 1, 1) * torch.randn_like(x)


def _magnitude_scale(x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Scale signal amplitude by ``(1 + mag)`` per sample.

    Args:
        x: Input signal of shape ``(B, C, T)``.
        mag: Per-sample scaling offset of shape ``(B,)``.

    Returns:
        Scaled signal of shape ``(B, C, T)``.
    """
    return x * (1.0 + mag.view(-1, 1, 1))


def _time_mask(x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Zero-out a random contiguous time segment of length ``floor(|mag|*T)``.

    Args:
        x: Input signal of shape ``(B, C, T)``.
        mag: Per-sample mask fraction in ``[0, 1]`` of shape ``(B,)``.

    Returns:
        Masked signal of shape ``(B, C, T)``.
    """
    B, C, T = x.shape
    out = x.clone()
    mask_len = (mag.abs() * T).long().clamp(0, T)
    for b in range(B):
        ml = int(mask_len[b].item())
        if ml > 0:
            start = int(torch.randint(0, max(T - ml + 1, 1), (1,)).item())
            out[b, :, start : start + ml] = 0.0
    return out


def _baseline_wander(x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Add a low-frequency sinusoidal baseline drift to each sample.

    Args:
        x: Input signal of shape ``(B, C, T)``.
        mag: Per-sample drift amplitude of shape ``(B,)``.

    Returns:
        Signal with baseline drift added, shape ``(B, C, T)``.
    """
    B, C, T = x.shape
    device = x.device
    t = torch.linspace(0, 2 * math.pi, T, device=device)
    freq = torch.rand(B, device=device) * 0.3 + 0.05   # normalised 0.05–0.35
    phase = torch.rand(B, device=device) * 2 * math.pi
    wander = torch.sin(freq.unsqueeze(-1) * t + phase.unsqueeze(-1))  # (B, T)
    return x + mag.view(-1, 1, 1) * wander.unsqueeze(1)


def _temporal_warp(x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Apply differentiable non-linear temporal warping via ``grid_sample``.

    Args:
        x: Input signal of shape ``(B, C, T)``.
        mag: Per-sample warp strength of shape ``(B,)``.

    Returns:
        Warped signal of shape ``(B, C, T)``.
    """
    B, C, T = x.shape
    device = x.device
    base = torch.linspace(-1.0, 1.0, T, device=device).unsqueeze(0).expand(B, -1)
    freq = torch.rand(B, device=device) * 2.0 + 1.0    # 1–3 cycles
    phase = torch.rand(B, device=device) * 2 * math.pi
    t_norm = torch.linspace(0, 2 * math.pi, T, device=device)
    disp = 0.1 * mag.unsqueeze(-1) * torch.sin(
        freq.unsqueeze(-1) * t_norm + phase.unsqueeze(-1)
    )
    grid_x = (base + disp).clamp(-1.0, 1.0)            # (B, T)
    grid_y = torch.zeros_like(grid_x)
    # grid_sample expects (B, H_out, W_out, 2) — treat H=1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # (B, 1, T, 2)
    warped = F.grid_sample(
        x.unsqueeze(2), grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return warped.squeeze(2)                            # (B, C, T)


def _temporal_displacement(x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Circularly shift each sample along the time axis by ``floor(|mag|*T)``.

    Args:
        x: Input signal of shape ``(B, C, T)``.
        mag: Per-sample shift fraction in ``[0, 1]`` of shape ``(B,)``.

    Returns:
        Shifted signal of shape ``(B, C, T)``.
    """
    B, C, T = x.shape
    shifts = (mag.abs() * T).long()
    return torch.stack(
        [torch.roll(x[b], int(shifts[b].item()), dims=-1) for b in range(B)]
    )


def _no_op(x: torch.Tensor, _mag: torch.Tensor) -> torch.Tensor:
    """Return the signal unchanged (identity operation).

    Args:
        x: Input signal of shape ``(B, C, T)``.
        _mag: Unused magnitude placeholder of shape ``(B,)``.

    Returns:
        The unmodified input signal.
    """
    return x


_OPS: List = [
    _gaussian_noise,
    _magnitude_scale,
    _time_mask,
    _baseline_wander,
    _temporal_warp,
    _temporal_displacement,
    _no_op,
]
_NUM_OPS: int = len(_OPS)  # 7


# ---------------------------------------------------------------------------
# TaskAugPolicy
# ---------------------------------------------------------------------------

class TaskAugPolicy(nn.Module):
    """Differentiable task-adaptive augmentation policy (Raghu et al., 2022).

    Applies *num_stages* sequential augmentation stages.  In each stage the
    operation weights are obtained via Gumbel-Softmax, then a soft weighted
    sum of all augmented versions is computed (differentiable at training
    time).  Each operation has two learnable scalar magnitudes — one for the
    negative class (``mag_neg``) and one for the positive class
    (``mag_pos``) — enabling asymmetric augmentation intensities.

    Args:
        num_stages: Number of sequential augmentation stages ``K``. Default: 2.
        temperature: Gumbel-Softmax temperature ``τ``. Default: 1.0.

    Attributes:
        logits: ``(K, N_ops)`` selection logits.
        mag_neg: ``(K, N_ops)`` per-stage magnitudes for label=0 samples.
        mag_pos: ``(K, N_ops)`` per-stage magnitudes for label=1 samples.
    """

    def __init__(self, num_stages: int = 2, temperature: float = 1.0) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.temperature = temperature

        self.logits = nn.Parameter(torch.zeros(num_stages, _NUM_OPS))
        self.mag_neg = nn.Parameter(0.1 * torch.ones(num_stages, _NUM_OPS))
        self.mag_pos = nn.Parameter(0.1 * torch.ones(num_stages, _NUM_OPS))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply K augmentation stages to *x*.

        Args:
            x: ECG tensor of shape ``(B, C, T)``.
            labels: Binary class labels of shape ``(B,)`` (0 or 1).

        Returns:
            Augmented tensor of the same shape as *x*.
        """
        labels_f = labels.float()  # (B,)

        for k in range(self.num_stages):
            # Gumbel-Softmax operation weights: (N_ops,)
            weights = F.gumbel_softmax(
                self.logits[k], tau=self.temperature, hard=False
            )

            # Per-sample class-specific magnitude: (B, N_ops)
            mag_k = (
                self.mag_neg[k].unsqueeze(0)
                + labels_f.unsqueeze(-1)
                * (self.mag_pos[k] - self.mag_neg[k]).unsqueeze(0)
            ).abs()

            # Soft weighted combination of all augmented versions
            augmented = torch.zeros_like(x)
            for i, op in enumerate(_OPS):
                augmented = augmented + weights[i] * op(x, mag_k[:, i])
            x = augmented

        return x


# ---------------------------------------------------------------------------
# 1-D ResNet-18 backbone
# ---------------------------------------------------------------------------

class _BasicBlock1D(nn.Module):
    """Residual block for 1-D signals (kernel size 7, BN + ReLU)."""

    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute residual block output.

        Args:
            x: Input feature map of shape ``(B, in_ch, T)``.

        Returns:
            Output feature map of shape ``(B, out_ch, T//stride)``.
        """
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _ResNet1D(nn.Module):
    """1-D ResNet-18 adapted for multi-lead ECG signals.

    Args:
        in_channels: Number of input channels (ECG leads). Default: 12.
        num_classes: Output dimension (1 for binary classification).

    Input shape:
        ``(B, in_channels, T)`` — e.g. ``(B, 12, 1000)``.

    Output shape:
        ``(B, num_classes)``.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, n_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build a stage of *n_blocks* residual blocks.

        Args:
            in_ch: Input channel count for the first block.
            out_ch: Output channel count for all blocks in this stage.
            n_blocks: Number of :class:`_BasicBlock1D` blocks.
            stride: Stride for the first block (subsequent blocks use 1).

        Returns:
            Sequential module containing all blocks for this stage.
        """
        blocks: List[nn.Module] = [_BasicBlock1D(in_ch, out_ch, stride=stride)]
        blocks += [_BasicBlock1D(out_ch, out_ch) for _ in range(1, n_blocks)]
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Initialise Conv1d with Kaiming normal and BatchNorm with identity.

        Returns:
            None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full ResNet-18 forward pass.

        Args:
            x: Input tensor of shape ``(B, in_channels, T)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).squeeze(-1))


# ---------------------------------------------------------------------------
# TaskAugResNet — PyHealth BaseModel subclass
# ---------------------------------------------------------------------------

class TaskAugResNet(BaseModel):
    """1-D ResNet-18 with a learned TaskAug augmentation policy.

    Reproduces the TaskAug framework of Raghu et al. (2022) within the
    PyHealth ``BaseModel`` interface.  The :class:`TaskAugPolicy` is applied
    to the input only during training; at inference the raw signal is passed
    directly to the backbone.

    For bi-level optimisation — inner loop updating backbone weights on
    augmented training data, outer loop updating policy weights on clean
    validation loss — use the ``BiLevelTrainer`` helper in the accompanying
    examples script.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` produced by
            :class:`~pyhealth.tasks.ecg_classification.ECGBinaryClassification`.
        num_leads: Number of ECG leads (input channels). Default: 12.
        policy_stages: Number of sequential augmentation stages *K*. Default: 2.
        temperature: Gumbel-Softmax temperature. Default: 1.0.

    Examples:
        >>> model = TaskAugResNet(sample_dataset)
        >>> out = model(ecg=ecg_tensor, label=label_tensor)
        >>> out.keys()
        dict_keys(['logit', 'y_prob', 'loss', 'y_true'])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_leads: int = 12,
        policy_stages: int = 2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(dataset)
        self.mode = "binary"  # binary classification throughout

        self.policy = TaskAugPolicy(
            num_stages=policy_stages,
            temperature=temperature,
        )
        self.backbone = _ResNet1D(
            in_channels=num_leads,
            num_classes=self.get_output_size(),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        ecg: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Augment (training only) and classify ECG signals.

        Args:
            ecg: Float tensor of shape ``(B, 12, T)``.
            label: Optional binary label tensor of shape ``(B,)``.  Enables
                class-specific augmentation magnitudes and loss computation.

        Returns:
            Dict containing:

            * ``logit``  — raw logits ``(B, 1)``
            * ``y_prob`` — sigmoid probabilities ``(B, 1)``
            * ``loss``   — scalar BCE loss *(only when label is provided)*
            * ``y_true`` — label tensor *(only when label is provided)*
        """
        if self.training and label is not None:
            ecg = self.policy(ecg, label)

        logits = self.backbone(ecg)         # (B, 1)
        y_prob = torch.sigmoid(logits)

        output: Dict[str, torch.Tensor] = {"logit": logits, "y_prob": y_prob}
        if label is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), label.float()
            )
            output["y_true"] = label

        return output

    # ------------------------------------------------------------------
    # Parameter group helpers (used by BiLevelTrainer)
    # ------------------------------------------------------------------

    def policy_parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over augmentation policy parameters.

        Returns:
            Iterator of :class:`torch.nn.Parameter` objects belonging to
            the :class:`TaskAugPolicy` (logits and magnitudes).
        """
        return self.policy.parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over ResNet backbone parameters.

        Returns:
            Iterator of :class:`torch.nn.Parameter` objects belonging to
            the :class:`_ResNet1D` backbone.
        """
        return self.backbone.parameters()
