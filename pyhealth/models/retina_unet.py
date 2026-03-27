"""Retina U-Net skeleton model for segmentation.

Provides a minimal UNet-like implementation that subclasses `BaseModel` so
it can be used with PyHealth datasets and trainers.

This file intentionally keeps dependencies to PyTorch and PyHealth core.
"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleDataset
from pyhealth.processors import ImageProcessor


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY or diffX:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RetinaUNet(BaseModel):
    """Minimal Retina U-Net style segmentation model.

    Args:
        dataset: PyHealth `SampleDataset` used to infer image fields.
        in_channels: number of image channels (overridden per-field if needed).
        num_classes: number of output channels/classes.
        features: sequence of feature widths for UNet stages.
        field_name: optional specific input field name to use (image key).
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 1,
        num_classes: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        field_name: Optional[str] = None,
    ) -> None:
        super().__init__(dataset)
        self.num_classes = num_classes
        self.features = list(features)
        self.field_name = field_name

        # build encoder
        self.inc = DoubleConv(in_channels, self.features[0])
        self.downs = nn.ModuleList()
        for i in range(len(self.features) - 1):
            self.downs.append(Down(self.features[i], self.features[i + 1]))

        # bottleneck
        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2)

        # build decoder (reverse)
        self.ups = nn.ModuleList()
        rev = [self.features[-1] * 2] + self.features[::-1]
        for i in range(len(self.features)):
            in_ch = rev[i]
            out_ch = self.features[::-1][i]
            self.ups.append(Up(in_ch, out_ch))

        self.outc = OutConv(self.features[0], self.num_classes)

    def _select_field(self, inputs: Dict[str, torch.Tensor]) -> str:
        if self.field_name is not None:
            return self.field_name
        # prefer dataset image processors
        for name, proc in getattr(self.dataset, "input_processors", {}).items():
            if isinstance(proc, ImageProcessor):
                return name
        # fallback to the first provided input
        return next(iter(inputs.keys()))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward expects a dict of image tensors keyed by field name.

        Returns a dict with keys `logit` and `y_prob` where `logit` is the
        raw logits with shape (B, C, H, W) and `y_prob` is the per-pixel
        probability map (sigmoid for binary, softmax for multi-class).
        """
        field = self._select_field(inputs)
        x = inputs[field].to(self.device)

        x1 = self.inc(x)
        encs = [x1]
        out = x1
        for down in self.downs:
            out = down(out)
            encs.append(out)

        out = self.bottleneck(out)

        # reverse encoders (skip connection order)
        for i, up in enumerate(self.ups):
            enc = encs[-(i + 1)]
            out = up(out, enc)

        logits = self.outc(out)

        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        return {"logit": logits, "y_prob": probs}


if __name__ == "__main__":
    # quick smoke-test with random input
    from pyhealth.datasets import create_sample_dataset
    import numpy as np

    samples = [
        {"patient_id": "p1", "visit_id": "v1", "img": np.zeros((64, 64), dtype=np.uint8), "label": 0}
    ]
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"img": ("image", {"image_size": 64, "mode": "L"})},
        output_schema={"label": "binary"},
        dataset_name="retina_smoke",
    )
    model = RetinaUNet(dataset, in_channels=1, num_classes=1)
    imgs = torch.randn(2, 1, 64, 64)
    out = model({"img": imgs})
    print(out["logit"].shape, out["y_prob"].shape)
