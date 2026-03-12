from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class ConvBlock(nn.Module):
    """A two-layer convolutional block used by RetinaUNet."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RetinaUNet(BaseModel):
    """Retina U-Net style model with an auxiliary segmentation branch.

    This implementation is intentionally lightweight for reproducibility
    experiments in PyHealth:

    - An encoder backbone extracts image features.
    - A classification head predicts image-level labels.
    - A U-Net-like decoder predicts an auxiliary segmentation map.

    The final training loss is:

    ``loss = cls_loss + seg_loss_weight * seg_loss``

    where ``seg_loss`` is computed from either a provided ``seg_target`` or a
    pseudo-mask created from image intensity.

    Args:
        dataset: SampleDataset used to infer feature/label keys and output size.
        in_channels: Expected number of input channels. Default is 1.
        base_channels: Width of the first encoder stage. Default is 32.
        seg_loss_weight: Weight for auxiliary segmentation loss. Default is 0.1.
        dropout: Dropout used in the classification head. Default is 0.1.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 1,
        base_channels: int = 32,
        seg_loss_weight: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("RetinaUNet supports exactly one image-like feature key.")
        if len(self.label_keys) != 1:
            raise ValueError("RetinaUNet supports exactly one label key.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if seg_loss_weight < 0:
            raise ValueError("seg_loss_weight must be non-negative.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.seg_loss_weight = seg_loss_weight
        self.dropout = dropout

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

        output_size = self.get_output_size()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c4, output_size),
        )

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to NCHW format."""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # Treat as NHW by default.
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(x.shape)}.")
        if x.dim() == 4 and x.shape[1] not in {1, 3} and x.shape[-1] in {1, 3}:
            # NHWC -> NCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _align_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Match input channel count to model configuration."""
        if x.shape[1] == self.in_channels:
            return x
        if self.in_channels == 1:
            return x.mean(dim=1, keepdim=True)
        if x.shape[1] == 1 and self.in_channels == 3:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] > self.in_channels:
            return x[:, : self.in_channels]
        repeats = (self.in_channels + x.shape[1] - 1) // x.shape[1]
        x = x.repeat(1, repeats, 1, 1)
        return x[:, : self.in_channels]

    @staticmethod
    def _build_pseudo_mask(x: torch.Tensor) -> torch.Tensor:
        """Create a pseudo segmentation target from image intensity."""
        intensity = x.mean(dim=1, keepdim=True)
        threshold = intensity.mean(dim=(2, 3), keepdim=True)
        return (intensity > threshold).float()

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def _encode_decode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder-decoder and return (class_logits, seg_logits, embedding)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))

        class_logits = self.classifier(bottleneck)
        embed = F.adaptive_avg_pool2d(bottleneck, output_size=(1, 1)).flatten(1)

        d3 = self.up3(bottleneck)
        d3 = self._resize_like(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._resize_like(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._resize_like(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg_logits = self.seg_head(d1)
        seg_logits = self._resize_like(seg_logits, x)
        return class_logits, seg_logits, embed

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Required inputs:
            - ``feature_key`` inferred from dataset schema
            - ``label_key`` during training

        Optional inputs:
            - ``seg_target``: tensor of shape (B, 1, H, W) or broadcastable form
            - ``embed``: if True, only returns pooled encoder embedding
        """
        x = kwargs[self.feature_key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(self.device, dtype=torch.float32)
        x = self._to_nchw(x)
        x = self._align_channels(x)

        class_logits, seg_logits, embed = self._encode_decode(x)

        if kwargs.get("embed", False):
            return {"embed": embed}

        results: Dict[str, torch.Tensor] = {
            "logit": class_logits,
            "y_prob": self.prepare_y_prob(class_logits),
            "seg_logit": seg_logits,
        }

        if self.label_key not in kwargs:
            return results

        y_true = kwargs[self.label_key].to(self.device)
        cls_loss = self.get_loss_function()(class_logits, y_true)

        seg_target = kwargs.get("seg_target")
        if seg_target is None:
            seg_target = self._build_pseudo_mask(x)
        else:
            if not isinstance(seg_target, torch.Tensor):
                seg_target = torch.as_tensor(seg_target)
            seg_target = seg_target.to(self.device, dtype=torch.float32)
            if seg_target.dim() == 3:
                seg_target = seg_target.unsqueeze(1)
            if seg_target.dim() == 2:
                seg_target = seg_target.unsqueeze(0).unsqueeze(0)
            if seg_target.shape[1] != 1:
                seg_target = seg_target.mean(dim=1, keepdim=True)
            if seg_target.shape[-2:] != seg_logits.shape[-2:]:
                seg_target = F.interpolate(
                    seg_target,
                    size=seg_logits.shape[-2:],
                    mode="nearest",
                )

        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
        loss = cls_loss + self.seg_loss_weight * seg_loss

        results.update(
            {
                "loss": loss,
                "cls_loss": cls_loss,
                "seg_loss": seg_loss,
                "y_true": y_true,
            }
        )
        return results
