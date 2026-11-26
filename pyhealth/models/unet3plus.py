"""UNet 3+ for one-dimensional signal segmentation (ECG delineation).

Paper (architecture):
    H. Huang et al., "UNet 3+: A Full-Scale Connected UNet for Medical Image
    Segmentation," ICASSP 2020 - 2020 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020,
    pp. 1055-1059, doi: 10.1109/ICASSP40776.2020.9053405

Reproduced paper (1D ECG delineation, best model on LUDB):
    Park, J., Park, T., Kwon, J. & Jo, Y.. (2025). Benchmarking ECG
    Delineation using Deep Neural Network-based Semantic Segmentation
    Models. Proceedings of the sixth Conference on Health, Inference, and
    Learning, in Proceedings of Machine Learning Research 287:63-88
    Available from https://proceedings.mlr.press/v287/park25a.html.

Authors:
    Mateo Ignacio (Mateo.Ignacio@gatech.edu), Minwoo Jo (mjo38@gatech.edu)

Description:
    Implements ``UNet3Plus``, a BaseModel that maps a ``(channels, length)``
    signal to per-timestep logits for each target channel (P, QRS and T
    waves for :class:`~pyhealth.tasks.ECGDelineationLUDB`). It is trained
    with binary cross-entropy plus Dice loss and scored with
    :func:`segmentation_iou`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    """Return ``Conv1d -> BatchNorm1d -> ReLU`` with "same" padding.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Odd convolution kernel size.

    Returns:
        The block as an ``nn.Sequential``.
    """
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _resize(x: torch.Tensor, length: int) -> torch.Tensor:
    """Bring ``x`` to ``length`` time steps: max-pool down or upsample.

    Args:
        x: Tensor of shape ``(batch, channels, time)``.
        length: Target number of time steps.

    Returns:
        ``x`` resized along time.
    """
    if x.shape[-1] > length:
        return F.adaptive_max_pool1d(x, length)
    if x.shape[-1] < length:
        return F.interpolate(x, size=length, mode="linear", align_corners=False)
    return x


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss averaged over samples and channels.

    Args:
        logits: Raw scores of shape ``(batch, channels, time)``.
        target: Binary target of the same shape.

    Returns:
        A scalar ``1 - Dice``; channels absent from both target and
        prediction count as a perfect match.

    Examples:
        >>> target = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        >>> right = torch.tensor([[[9.0, 9.0, -9.0, -9.0]]])
        >>> round(dice_loss(right, target).item(), 3)
        0.0
        >>> round(dice_loss(-right, target).item(), 3)
        0.8
    """
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(-1)
    denom = prob.sum(-1) + target.sum(-1)
    return (1 - (2 * inter + 1.0) / (denom + 1.0)).mean()


def segmentation_iou(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    names: tuple[str, ...] = ("P", "QRS", "T"),
    annotated_only: bool = False,
) -> dict[str, float]:
    """Per-channel IoU and their mean (mIoU) for time-step segmentation.

    Intersections and unions are summed over all samples and time steps
    before dividing, so a channel absent from a sample does not count as a
    perfect or a failed sample.

    Args:
        y_true: Binary targets of shape ``(n, channels, time)``.
        y_prob: Probabilities of the same shape.
        threshold: A time step is predicted positive when ``y_prob >=
            threshold``.
        names: Channel names used as keys.
        annotated_only: If True, predictions outside each sample's annotated
            span (first to last positive time step of ``y_true`` over all
            channels) are ignored. LUDB, for example, leaves the first and
            last beats of each record unannotated although they contain
            real waves, so predicting them would otherwise count as false
            positives. A sample without any annotation then contributes
            nothing.

    Returns:
        ``{"iou_<name>": ..., "miou": ...}``.

    Raises:
        ValueError: If the shapes differ or ``names`` does not match the
            channel count.

    Examples:
        >>> t = np.array([[[1, 1, 0, 0]]])
        >>> p = np.array([[[0.9, 0.2, 0.7, 0.1]]])
        >>> segmentation_iou(t, p, names=("QRS",))
        {'iou_QRS': 0.3333333333333333, 'miou': 0.3333333333333333}
    """
    y_true = np.asarray(
        y_true.detach().cpu() if isinstance(y_true, torch.Tensor) else y_true
    )
    y_prob = np.asarray(
        y_prob.detach().cpu() if isinstance(y_prob, torch.Tensor) else y_prob
    )
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if y_true.shape[1] != len(names):
        raise ValueError(f"Got {y_true.shape[1]} channels but {len(names)} names")
    true = y_true > 0.5
    pred = y_prob >= threshold
    if annotated_only:
        labelled = true.any(axis=1)  # (n, time)
        steps = np.arange(true.shape[2])
        first = np.where(labelled.any(axis=1), labelled.argmax(axis=1), true.shape[2])
        last = true.shape[2] - 1 - labelled[:, ::-1].argmax(axis=1)
        inside = (steps >= first[:, None]) & (steps <= last[:, None])
        pred = pred & inside[:, None, :]
    inter = (true & pred).sum(axis=(0, 2))
    union = (true | pred).sum(axis=(0, 2))
    ious = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    scores = {f"iou_{name}": float(v) for name, v in zip(names, ious)}
    scores["miou"] = float(ious.mean())
    return scores


class UNet3Plus(BaseModel):
    """UNet 3+ with full-scale skip connections for 1D segmentation.

    The encoder halves the time resolution ``depth - 1`` times. Every
    decoder level combines all encoder levels at or above it (max-pooled to
    its resolution) with all deeper decoder levels (upsampled), which is the
    full-scale skip connection of UNet 3+. With ``deep_supervision`` every
    decoder level also predicts the masks and the logits are summed. This is
    one of the two options in the implementation released with Park et al.
    (2025), where UNet 3+ gave the best LUDB mIoU (0.854); the other option
    computes a separate loss for each level.

    The model expects a ``"tensor"`` input of shape ``(channels, length)``
    and a ``"tensor"`` target of shape ``(n_masks, length)`` with values in
    {0, 1}. Its loss is ``bce_weight * BCE + dice_weight * Dice``. The
    paper's search also included an MS-SSIM loss term and the
    classification-guided module (CGM) of UNet 3+; both are left out here.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` with one tensor
            input and one tensor target, e.g. from
            :class:`~pyhealth.tasks.ECGDelineationLUDB`.
        base_channels: Channels at the first depth; depth ``i`` has
            ``base_channels * 2**i``. The paper searched 32, 64 and 128.
        depth: Number of resolutions (encoder stages). The paper found 5 best.
        kernel_size: Odd convolution kernel size.
        encoder_convs: Conv-BN-ReLU blocks per encoder stage.
        deep_supervision: Sum predictions from every decoder level.
        bce_weight: Weight of the binary cross-entropy term.
        dice_weight: Weight of the Dice term.

    Raises:
        ValueError: If an argument is out of range or the dataset does not
            have exactly one input and one target.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": str(i), "signal": np.random.randn(1, 256).tolist(),
        ...      "mask": np.zeros((3, 256)).tolist()}
        ...     for i in range(2)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"signal": "tensor"}, {"mask": "tensor"}
        ... )  # doctest: +SKIP
        >>> model = UNet3Plus(dataset, base_channels=8, depth=3)  # doctest: +SKIP
    """

    def __init__(
        self,
        dataset: SampleDataset,
        base_channels: int = 32,
        depth: int = 5,
        kernel_size: int = 7,
        encoder_convs: int = 2,
        deep_supervision: bool = True,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError("UNet3Plus needs exactly one input and one target key.")
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        if base_channels < 1 or encoder_convs < 1:
            raise ValueError("base_channels and encoder_convs must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd number, got {kernel_size}"
            )
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("bce_weight and dice_weight must be >= 0, not both 0")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.in_channels, self.n_masks = self._infer_channels()
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        widths = [base_channels * 2**i for i in range(depth)]
        self.encoders = nn.ModuleList()
        in_ch = self.in_channels
        for width in widths:
            blocks = []
            for _ in range(encoder_convs):
                blocks.append(_conv_bn_relu(in_ch, width, kernel_size))
                in_ch = width
            self.encoders.append(nn.Sequential(*blocks))

        # Decoder level i (0 = full resolution) for i = depth - 2 .. 0. Every
        # level has ``depth`` inputs, each reduced to ``base_channels``.
        cat_channels = base_channels * depth
        self.decoders = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for i in range(depth - 1):
            branches = nn.ModuleList()
            for j in range(depth):
                source = widths[j] if (j <= i or j == depth - 1) else cat_channels
                branches.append(_conv_bn_relu(source, base_channels, kernel_size))
            self.decoders.append(branches)
            self.fusions.append(_conv_bn_relu(cat_channels, cat_channels, kernel_size))

        # Heads: one per decoder level plus the bottleneck, or only the
        # full-resolution head without deep supervision.
        head_inputs = [cat_channels] * (depth - 1) + [widths[-1]]
        if not deep_supervision:
            head_inputs = head_inputs[:1]
        self.heads = nn.ModuleList(
            nn.Conv1d(ch, self.n_masks, kernel_size, padding=kernel_size // 2)
            for ch in head_inputs
        )

    def _infer_channels(self) -> tuple[int, int]:
        """Read input and target channel counts from the first sample.

        Returns:
            ``(in_channels, n_masks)``.

        Raises:
            ValueError: If the tensors are not two-dimensional.
        """
        sample = self.dataset[0]
        x, y = sample[self.feature_key], sample[self.label_key]
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError(
                f"Expected 2D '{self.feature_key}' and '{self.label_key}' tensors "
                f"(channels, length), got {tuple(x.shape)} and {tuple(y.shape)}"
            )
        return int(x.shape[0]), int(y.shape[0])

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep logits.

        Args:
            x: Signal of shape ``(batch, in_channels, length)``.

        Returns:
            Logits of shape ``(batch, n_masks, length)``.
        """
        length = x.shape[-1]
        enc: list[torch.Tensor] = []
        out = x
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                out = F.max_pool1d(out, 2)
            out = encoder(out)
            enc.append(out)

        # dec[j] holds the feature map at depth j; the bottleneck is enc[-1].
        dec: dict[int, torch.Tensor] = {self.depth - 1: enc[-1]}
        for i in reversed(range(self.depth - 1)):
            size = enc[i].shape[-1]
            parts = []
            for j, branch in enumerate(self.decoders[i]):
                source = enc[j] if j <= i else dec[j]
                parts.append(branch(_resize(source, size)))
            dec[i] = self.fusions[i](torch.cat(parts, dim=1))

        if not self.deep_supervision:
            return self.heads[0](dec[0])
        logits = 0
        for level, head in enumerate(self.heads):
            logits = logits + _resize(head(dec[level]), length)
        return logits

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run the model on a batch from the PyHealth data loader.

        Args:
            **kwargs: Batch dict with the input key and, when present, the
                target key.

        Returns:
            ``{"logit", "y_prob"}`` and, when the target is in the batch,
            ``"y_true"`` and ``"loss"``.
        """
        x = kwargs[self.feature_key].to(self.device, dtype=torch.float32)
        logits = self.segment(x)
        results = {"logit": logits, "y_prob": torch.sigmoid(logits)}
        if self.label_key in kwargs:
            y = kwargs[self.label_key].to(self.device, dtype=torch.float32)
            loss = self.bce_weight * F.binary_cross_entropy_with_logits(logits, y)
            if self.dice_weight > 0:
                loss = loss + self.dice_weight * dice_loss(logits, y)
            results["y_true"] = y
            results["loss"] = loss
        return results
