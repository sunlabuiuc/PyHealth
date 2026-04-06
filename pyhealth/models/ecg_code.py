from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class ConvBNAct1d(nn.Module):
    """Conv1d + BatchNorm1d + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = activation if activation is not None else nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual1d(nn.Module):
    """MobileNetV2-style inverted residual block for 1D signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBNAct1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                )
            )

        layers.append(
            ConvBNAct1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            )
        )

        layers.append(
            ConvBNAct1d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                activation=nn.Identity(),
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        return out


class ECGCODE(BaseModel):
    """
    ECG-CODE-like delineation model with a MobileNet-style 1D backbone.

    The model predicts, for each interval and each ECG object (P/QRS/T):
      1) confidence (presence in interval)
      2) normalized start position in interval
      3) normalized end position in interval

    Output shape (after sigmoid):
      [batch, n_intervals, 3_waves, 3_values(conf,start,end)]

    Custom interval loss (inspired by arXiv:2406.02711):
      CL  = 0 if |pc - tc| < conf_tolerance else (pc - tc)^2
      SS  = (ps - ts)^2 + (pe - te)^2
      SEL = 0 if SS < se_tolerance else SS * tc
      total = mean(CL + SEL)
    """

    WAVE_LABELS = (1, 2, 3)  # P, QRS, T in mask encoding

    def __init__(
        self,
        dataset: SampleDataset,
        signal_key: str = "signal",
        mask_key: str = "mask",
        width_mult: float = 1.0,
        interval_size: int = 16,
        conf_tolerance: float = 0.25,
        se_tolerance: float = 0.15,
    ) -> None:
        super().__init__(dataset=dataset)

        if interval_size <= 0:
            raise ValueError("interval_size must be positive.")

        self.signal_key = signal_key
        self.mask_key = mask_key
        self.interval_size = int(interval_size)
        self.conf_tolerance = float(conf_tolerance)
        self.se_tolerance = float(se_tolerance)

        stem_channels = self._make_divisible(32 * width_mult)
        self.stem = ConvBNAct1d(1, stem_channels, kernel_size=7, stride=2)

        # MobileNetV2-like configuration: (expand_ratio, channels, repeats, stride)
        cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 3, 2),
            (6, 96, 2, 1),
            (6, 160, 2, 2),
        ]

        blocks = []
        in_ch = stem_channels
        for t, c, n, s in cfg:
            out_ch = self._make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(
                    InvertedResidual1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        expand_ratio=t,
                    )
                )
                in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

        proj_ch = self._make_divisible(128 * width_mult)
        self.proj = ConvBNAct1d(in_ch, proj_ch, kernel_size=1, stride=1)

        # 3 waves x 3 outputs(conf,start,end)
        self.head = nn.Conv1d(proj_ch, 9, kernel_size=1, bias=True)

    @staticmethod
    def _make_divisible(v: float, divisor: int = 8) -> int:
        return int(math.ceil(v / divisor) * divisor)

    @staticmethod
    def _extract_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, tuple):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        raise ValueError("Expected a tensor or tuple containing a tensor.")

    def _normalize_signal_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize to [B, C, T], where C is typically 1 for ECG.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,T]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        elif x.dim() == 3:
            # Heuristic for [B,T,C]
            if x.shape[-1] <= 4 and x.shape[1] > x.shape[-1]:
                x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported signal shape: {tuple(x.shape)}")
        return x.float()

    def _build_interval_targets(
        self,
        masks: torch.Tensor,
        n_intervals: int,
    ) -> torch.Tensor:
        """
        Build interval targets: [B, n_intervals, 3_waves, 3_values].
        """
        if masks.dim() == 3 and masks.shape[1] == 1:
            masks = masks[:, 0, :]
        if masks.dim() != 2:
            raise ValueError(
                f"Expected mask shape [B,T] or [B,1,T], got {tuple(masks.shape)}"
            )

        bsz, seq_len = masks.shape
        device = masks.device
        targets = torch.zeros(
            (bsz, n_intervals, 3, 3), device=device, dtype=torch.float32
        )

        for b in range(bsz):
            mask_b = masks[b].long()
            for i in range(n_intervals):
                start = i * self.interval_size
                end = min((i + 1) * self.interval_size, seq_len)
                if start >= end:
                    continue

                seg = mask_b[start:end]
                seg_len = int(seg.numel())
                denom = max(seg_len - 1, 1)

                for w_idx, wave_label in enumerate(self.WAVE_LABELS):
                    idx = torch.where(seg == wave_label)[0]
                    if idx.numel() == 0:
                        continue

                    targets[b, i, w_idx, 0] = 1.0  # confidence target
                    targets[b, i, w_idx, 1] = float(idx.min().item()) / float(denom)
                    targets[b, i, w_idx, 2] = float(idx.max().item()) / float(denom)

        return targets

    def _interval_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pred/target shape: [B, N, 3, 3(conf,start,end)]
        """
        pc = pred[..., 0]
        ps = pred[..., 1]
        pe = pred[..., 2]

        tc = target[..., 0]
        ts = target[..., 1]
        te = target[..., 2]

        diff_c = torch.abs(pc - tc)
        cl = torch.where(
            diff_c < self.conf_tolerance,
            torch.zeros_like(diff_c),
            (pc - tc) ** 2,
        )

        ss = (ps - ts) ** 2 + (pe - te) ** 2
        sel = torch.where(
            ss < self.se_tolerance,
            torch.zeros_like(ss),
            ss * tc,
        )

        loss = (cl + sel).mean()
        return loss, cl.mean(), sel.mean()

    def _predict_intervals(
        self, signal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          raw_logits: [B, N, 3, 3]
          pred:       [B, N, 3, 3] (sigmoid)
        """
        x = self._normalize_signal_shape(signal)
        bsz, _, seq_len = x.shape
        n_intervals = max(1, math.ceil(seq_len / self.interval_size))

        feat = self.stem(x)
        feat = self.backbone(feat)
        feat = self.proj(feat)
        feat = F.adaptive_avg_pool1d(feat, n_intervals)
        logits = self.head(feat)  # [B, 9, N]

        raw_logits = logits.permute(0, 2, 1).reshape(bsz, n_intervals, 3, 3)
        pred = torch.sigmoid(raw_logits)
        return raw_logits, pred

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        signal = self._extract_tensor(kwargs[self.signal_key]).to(self.device)
        raw_logits, pred = self._predict_intervals(signal)

        if self.mask_key in kwargs:
            mask = self._extract_tensor(kwargs[self.mask_key]).to(self.device)
            targets = self._build_interval_targets(mask, n_intervals=pred.shape[1])
            loss, cl_mean, sel_mean = self._interval_loss(pred, targets)
        else:
            targets = torch.zeros_like(pred, device=self.device)
            loss = torch.tensor(0.0, device=self.device)
            cl_mean = torch.tensor(0.0, device=self.device)
            sel_mean = torch.tensor(0.0, device=self.device)

        return {
            "loss": loss,
            "y_prob": pred,  # interval-level predictions
            "y_true": targets,  # interval-level targets
            "logit": raw_logits,
            "cl_loss": cl_mean,
            "sel_loss": sel_mean,
        }

    def forward_from_embedding(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        For compatibility with interpretability utilities.
        This model consumes dense signal tensors directly.
        """
        return self.forward(**kwargs)
