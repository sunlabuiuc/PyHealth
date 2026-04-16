"""
Retina U-Net Implementation for Medical Image Object Detection

Reference: Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision 
for Medical Object Detection (https://arxiv.org/abs/1811.08661)

This implementation uses only standard PyTorch without custom CUDA operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class ConvBlock(nn.Module):
    """Reusable convolution block with normalization and activation."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm_type: str = None,
        activation: str = 'relu',
        dim: int = 2
    ):
        super().__init__()
        self.dim = dim
        
        # Select appropriate convolution
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        
        # Build block
        layers = []
        layers.append(Conv(in_channels, out_channels, kernel_size, stride, padding))
        
        # Normalization
        if norm_type == 'batch':
            if dim == 2:
                layers.append(nn.BatchNorm2d(out_channels))
            else:
                layers.append(nn.BatchNorm3d(out_channels))
        elif norm_type == 'instance':
            if dim == 2:
                layers.append(nn.InstanceNorm2d(out_channels))
            else:
                layers.append(nn.InstanceNorm3d(out_channels))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block for feature learning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        norm_type: str = None,
        activation: str = 'relu',
        dim: int = 2
    ):
        super(ResidualBlock, self).__init__()

        self.dim = dim
        hidden_channels = out_channels // expansion

        self.conv1 = ConvBlock(
            in_channels, 
            hidden_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=0, 
            norm_type=norm_type, 
            activation=activation, 
            dim=dim
        )
        self.conv2 = ConvBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            norm_type=norm_type, 
            activation=activation, 
            dim=dim
            )
        self.conv3 = ConvBlock(
            hidden_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            norm_type=norm_type, 
            activation=None, 
            dim=dim
        )
        
        # Shortcut
        self.shortcut = None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                norm_type=norm_type,
                activation=None,  
                dim=dim
            )
        
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class FPN(nn.Module):
    """Feature Pyramid Network backbone for multi-scale feature extraction."""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        out_channels: int = 192,
        num_blocks: List[int] = None,
        norm_type: str = None,
        activation: str = 'relu',
        dim: int = 2,
    ):
        super().__init__()
        self.activation = activation
        self.norm_type = norm_type
        self.dim = dim
        self.block_expansion = 4

        if num_blocks is None:
            num_blocks = [3, 4, 6, 3]  # ResNet50-like

        stride = 2 if dim == 2 else (2, 2, 1) # For 3D, preserve depth resolution
        
        # Initial convolution
        self.c0 = nn.Sequential(
            ConvBlock(
                in_channels, 
                base_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                norm_type=norm_type, 
                activation=activation, 
                dim=dim
            ),
            ConvBlock(
                base_channels, 
                base_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                norm_type=norm_type, 
                activation=activation, 
                dim=dim
            )
        )

        self.c1 = ConvBlock(
            base_channels, 
            base_channels, 
            kernel_size=7, 
            stride=stride, 
            padding=3, 
            norm_type=norm_type, 
            activation=activation, 
            dim=dim
        )
        
        # Residual blocks
        c2_out_channels = base_channels * self.block_expansion
        self.c2 = self._make_layer(
            base_channels, 
            c2_out_channels, 
            num_blocks[0], 
            stride=1, 
            pool=True
        )

        c3_out_channels = c2_out_channels * 2
        self.c3 = self._make_layer(
            c2_out_channels, 
            c3_out_channels, 
            num_blocks[1], 
            stride=2
        )
        c4_out_channels = c3_out_channels * 2
        self.c4 = self._make_layer(
            c3_out_channels, 
            c4_out_channels,
            num_blocks[2], 
            stride=2
        )
        
        c5_out_channels = c4_out_channels * 2
        self.c5 = self._make_layer(
            c4_out_channels, 
            c5_out_channels, 
            num_blocks[3], 
            stride=2
        )

        if self.dim == 2:
            self.p1_upsample = nn.Upsample(
                scale_factor=stride, mode="bilinear", align_corners=False
            )
            self.p2_upsample = nn.Upsample(
                scale_factor=stride, mode="bilinear", align_corners=False
            )
        else:
            self.p1_upsample = nn.Upsample(
                scale_factor=stride, mode="trilinear", align_corners=False
            )
            self.p2_upsample = nn.Upsample(
                scale_factor=stride, mode="trilinear", align_corners=False
            )

        
        # FPN lateral connections and smoothing
        self.p5 = ConvBlock(
            base_channels * 32, 
            out_channels, 
            kernel_size=1, 
            activation=None, 
            dim=dim
        )
        self.p4 = ConvBlock(
            base_channels * 16, 
            out_channels, 
            kernel_size=1, 
            activation=None,
            dim=dim
        )
        self.p3 = ConvBlock(
            base_channels * 8, 
            out_channels, 
            kernel_size=1, 
            activation=None,
            dim=dim
        )
        self.p2 = ConvBlock(
            base_channels * 4, 
            out_channels, 
            kernel_size=1, 
            activation=None,
            dim=dim
        )
        self.p1 = ConvBlock(
            base_channels, 
            out_channels, 
            kernel_size=1, 
            activation=None,
            dim=dim
        )
        self.p0 = ConvBlock(
            base_channels, 
            out_channels, 
            kernel_size=1, 
            activation=None,
            dim=dim
        )
        
        self.smooth_p5 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=None, 
            dim=dim
        )
        self.smooth_p4 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=None, 
            dim=dim
        )
        self.smooth_p3 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=None, 
            dim=dim
        )
        self.smooth_p2 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=None, 
            dim=dim
        )
        self.smooth_p1 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=None, 
            dim=dim
        )
        self.smooth_p0 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=None, 
            dim=dim
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, pool=False):
        """Build residual layer."""
        layers = []

        # For the first block
        if pool:
            MaxPool = nn.MaxPool2d if self.dim == 2 else nn.MaxPool3d
            stride_pool = 2 if self.dim == 2 else (2, 2, 1)
            
            layers.append(MaxPool(kernel_size=3, stride=stride_pool, padding=1))
   
        layers.append(
            ResidualBlock(
                in_channels, 
                out_channels, 
                stride=stride, 
                expansion=self.block_expansion,
                norm_type=self.norm_type, 
                activation=self.activation, 
                dim=self.dim
            )
        )
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(
                    out_channels, 
                    out_channels, 
                    stride=1,
                    expansion=self.block_expansion,
                    norm_type=self.norm_type, 
                    activation=self.activation, 
                    dim=self.dim
                )
            )
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale feature maps.
        Returns: [p0, p1, p2, p3, p4, p5] from highest to lowest resolution
        """
        # Stem
        c0_out = self.c0(x)
        c1_out = self.c1(c0_out)
        
        # Backbone
        c2_out = self.c2(c1_out)
        c3_out = self.c3(c2_out)
        c4_out = self.c4(c3_out)
        c5_out = self.c5(c4_out)
        
        # FPN top-down path
        p5_pre_out = self.p5(c5_out)
        p4_pre_out = self.p4(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.p3(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.p2(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)
        p1_pre_out = self.p1(c1_out) + self.p2_upsample(p2_pre_out)
        p0_pre_out = self.p0(c0_out) + self.p1_upsample(p1_pre_out)
        
        # Smooth
        p5 = self.smooth_p5(p5_pre_out)
        p4 = self.smooth_p4(p4_pre_out)
        p3 = self.smooth_p3(p3_pre_out)
        p2 = self.smooth_p2(p2_pre_out)
        p1 = self.smooth_p1(p1_pre_out)
        p0 = self.smooth_p0(p0_pre_out)
        
        return [p0, p1, p2, p3, p4, p5]


class AnchorGenerator(nn.Module):
    """Generates anchor boxes for object detection using vectorized operations."""
    
    def __init__(
        self,
        rpn_anchor_scales: Dict[str, Dict[str, List[float]]] = None,
        rpn_anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        rpn_anchor_stride: int = 1,
        pyramid_levels: List[int] = [2, 3, 4, 5], # Corresponding to P2, P3, P4, P5
        dim: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.pyramid_levels = pyramid_levels

        # Default scales but need to adjust based on expected objects size to detect
        if rpn_anchor_scales is None:
            rpn_anchor_scales = {
                'xy': {
                    "P2": 8,
                    "P3": 16,
                    "P4": 32,
                    "P5": 64
                },
                'z': {
                    "P2": 2,
                    "P3": 4,
                    "P4": 8,
                    "P5": 16
                }
            }

        self.rpn_anchor_scales = rpn_anchor_scales

        # Anchor Sub-scaling: 
        # For each base scale, to create a dense scaling, 
        # add two additional sub scales by multiplying with 2^(1/3) and 2^(2/3)
        self.rpn_anchor_scales['xy'] = {
            key: [value, value * (2 ** (1 / 3)), value * (2 ** (2 / 3))] 
            for key, value in rpn_anchor_scales['xy'].items()
        }
        self.rpn_anchor_scales['z'] = {
            key: [value, value * (2 ** (1 / 3)), value * (2 ** (2 / 3))] 
            for key, value in rpn_anchor_scales['z'].items()
        }

        self.rpn_anchor_ratios = torch.tensor(rpn_anchor_ratios, dtype=torch.float32)
        self.rpn_anchor_stride = rpn_anchor_stride
        # mapping from pyramid level to orignal image, based on architercture design
        self.feature_strides = {
                'xy': {
                    "P2": 4,
                    "P3": 8,
                    "P4": 16,
                    "P5": 32
                },
                'z': {
                    "P2": 1,
                    "P3": 2,
                    "P4": 4,
                    "P5": 8
                }
            }
    def generate_anchors_2d(
            self,
            level: int,
            h: int,
            w: int,
            device: torch.device = torch.device('cpu')
        ) -> torch.Tensor:
        """Vectorized 2D anchor generation, matching original generate_anchors."""
        # Get feature stride and anchor scales for the current level
        scales_xy = torch.tensor(
            self.rpn_anchor_scales['xy'][f'P{level}'], 
            dtype=torch.float32, 
            device=device
        )
        ratios = self.rpn_anchor_ratios.to(device)

        fs_xy = self.feature_strides['xy'][f'P{level}']
        stride = self.rpn_anchor_stride

        # Generate Anchor Shapes (Matching NP Meshgrid order)
        scales_mesh, ratios_mesh = torch.meshgrid(scales_xy, ratios, indexing='xy')
        heights = (scales_mesh / torch.sqrt(ratios_mesh)).flatten()
        widths = (scales_mesh * torch.sqrt(ratios_mesh)).flatten()

        # Generate Grid Shifts
        shifts_y = torch.arange(0, h, stride, dtype=torch.float32, device=device) * fs_xy
        shifts_x = torch.arange(0, w, stride, dtype=torch.float32, device=device) * fs_xy

        # Match NumPy np.meshgrid(x, y) output shape (len(y), len(x))
        sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

        # Combine and Flatten
        grid_centers = torch.stack([sy, sx], dim=-1).reshape(-1, 1, 2)
        anchor_sizes = torch.stack([heights, widths], dim=-1).reshape(1, -1, 2)
        # Use broadcasting to get [N_grid, N_anchors, 3]
        box_min = grid_centers - 0.5 * anchor_sizes
        box_max = grid_centers + 0.5 * anchor_sizes

        # Extract and Reorder Columns to: [y1, x1, y2, x2]
        y1, x1 = box_min.unbind(-1)
        y2, x2 = box_max.unbind(-1)

        # Reshape to final list of boxes
        boxes = torch.stack([y1, x1, y2, x2], dim=-1).reshape(-1, 4)

        return boxes
    
    def generate_anchors_3d(
            self, 
            level: int,
            h: int, 
            w: int, 
            d: int, 
            device: torch.device = torch.device('cpu')
        ) -> torch.Tensor:
        """Vectorized 3D anchor generation, matching original generate_anchors_3D."""
        # Get feature stride and anchor scales for the current level
        scales_xy = torch.tensor(
            self.rpn_anchor_scales['xy'][f'P{level}'], 
            dtype=torch.float32, 
            device=device
        )
        scales_z = torch.tensor(
            self.rpn_anchor_scales['z'][f'P{level}'], 
            dtype=torch.float32, 
            device=device
        )
        ratios = self.rpn_anchor_ratios.to(device)

        fs_xy = self.feature_strides['xy'][f'P{level}']
        fs_z = self.feature_strides['z'][f'P{level}']
        stride = self.rpn_anchor_stride

        # Generate Anchor Shapes (Matching NP Meshgrid order)
        scales_mesh, ratios_mesh = torch.meshgrid(scales_xy, ratios, indexing='xy')
        heights = (scales_mesh / torch.sqrt(ratios_mesh)).flatten()
        widths = (scales_mesh * torch.sqrt(ratios_mesh)).flatten()
        depths = scales_z.repeat(len(heights) // len(scales_z))

        # Generate Grid Shifts
        shifts_y = torch.arange(0, h, stride, dtype=torch.float32, device=device) * fs_xy
        shifts_x = torch.arange(0, w, stride, dtype=torch.float32, device=device) * fs_xy
        shifts_z = torch.arange(0, d, stride, dtype=torch.float32, device=device) * fs_z

        # Match NumPy np.meshgrid(x, y, z) output shape (len(y), len(x), len(z))
        sy, sx, sz = torch.meshgrid(shifts_y, shifts_x, shifts_z, indexing='ij')

        # Combine and Flatten
        grid_centers = torch.stack([sy, sx, sz], dim=-1).reshape(-1, 1, 3)
        anchor_sizes = torch.stack([heights, widths, depths], dim=-1).reshape(1, -1, 3)

        # Use broadcasting to get [N_grid, N_anchors, 3]
        box_min = grid_centers - 0.5 * anchor_sizes
        box_max = grid_centers + 0.5 * anchor_sizes

        # Extract and Reorder Columns to: [y1, x1, y2, x2, z1, z2]
        y1, x1, z1 = box_min.unbind(-1)
        y2, x2, z2 = box_max.unbind(-1)

        # Reshape to final list of boxes
        boxes = torch.stack([y1, x1, y2, x2, z1, z2], dim=-1).reshape(-1, 6)
        
        return boxes
    
    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """Generate all anchors from feature maps."""
        all_anchors = []

        for level in self.pyramid_levels:
            if self.dim == 2:
                _, _, h, w = feature_maps[level].shape
                anchors = self.generate_anchors_2d(
                    level,
                    h,
                    w,
                    device=feature_maps[level].device
                )
            else:
                _, _, h, w, d = feature_maps[level].shape
                anchors = self.generate_anchors_3d(
                    level,
                    h,
                    w,
                    d,
                    device=feature_maps[level].device
                )

            all_anchors.append(anchors)
        
        return torch.cat(all_anchors, dim=0)


def _apply_box_deltas_2d(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply predicted deltas to 2D anchor boxes."""
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    height = y2 - y1
    width = x2 - x1
    ctr_y = y1 + 0.5 * height
    ctr_x = x1 + 0.5 * width

    dy = deltas[:, 0]
    dx = deltas[:, 1]
    dh = deltas[:, 2]
    dw = deltas[:, 3]

    pred_ctr_y = dy * height + ctr_y
    pred_ctr_x = dx * width + ctr_x
    pred_h = torch.exp(dh) * height
    pred_w = torch.exp(dw) * width

    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w

    return torch.stack([pred_y1, pred_x1, pred_y2, pred_x2], dim=1)


def _apply_box_deltas_3d(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply predicted deltas to 3D anchor boxes."""
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    z1 = boxes[:, 4]
    z2 = boxes[:, 5]
    height = y2 - y1
    width = x2 - x1
    depth = z2 - z1
    ctr_y = y1 + 0.5 * height
    ctr_x = x1 + 0.5 * width
    ctr_z = z1 + 0.5 * depth

    dy = deltas[:, 0]
    dx = deltas[:, 1]
    dz = deltas[:, 2]
    dh = deltas[:, 3]
    dw = deltas[:, 4]
    dd = deltas[:, 5]

    pred_ctr_y = dy * height + ctr_y
    pred_ctr_x = dx * width + ctr_x
    pred_ctr_z = dz * depth + ctr_z
    pred_h = torch.exp(dh) * height
    pred_w = torch.exp(dw) * width
    pred_d = torch.exp(dd) * depth

    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_z1 = pred_ctr_z - 0.5 * pred_d
    pred_y2 = pred_ctr_y + 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_z2 = pred_ctr_z + 0.5 * pred_d

    return torch.stack([pred_y1, pred_x1, pred_y2, pred_x2, pred_z1, pred_z2], dim=1)


def _clip_boxes_2d(
        boxes: torch.Tensor, 
        window: Tuple[float, float]
    ) -> torch.Tensor:
    y1 = boxes[:, 0].clamp(min=0, max=window[0])
    x1 = boxes[:, 1].clamp(min=0, max=window[1])
    y2 = boxes[:, 2].clamp(min=0, max=window[0])
    x2 = boxes[:, 3].clamp(min=0, max=window[1])

    return torch.stack([y1, x1, y2, x2], dim=1)


def _clip_boxes_3d(
        boxes: torch.Tensor, 
        window: Tuple[float, float, float]
    ) -> torch.Tensor:
    y1 = boxes[:, 0].clamp(min=0, max=window[0])
    x1 = boxes[:, 1].clamp(min=0, max=window[1])
    y2 = boxes[:, 2].clamp(min=0, max=window[0])
    x2 = boxes[:, 3].clamp(min=0, max=window[1])
    z1 = boxes[:, 4].clamp(min=0, max=window[2])
    z2 = boxes[:, 5].clamp(min=0, max=window[2])

    return torch.stack([y1, x1, y2, x2, z1, z2], dim=1)


def _nms_2d(
        boxes: torch.Tensor, 
        scores: torch.Tensor, 
        iou_threshold: float
    ) -> torch.Tensor:
    """Pure torch NMS for 2D boxes."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    heights = (y2 - y1).clamp(min=0)
    widths = (x2 - x1).clamp(min=0)
    areas = heights * widths
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        others = order[1:]
        yy1 = torch.max(y1[i], y1[others])
        xx1 = torch.max(x1[i], x1[others])
        yy2 = torch.min(y2[i], y2[others])
        xx2 = torch.min(x2[i], x2[others])
        h = (yy2 - yy1).clamp(min=0)
        w = (xx2 - xx1).clamp(min=0)
        inter = h * w
        union = areas[i] + areas[others] - inter
        iou = inter / (union + 1e-6)
        order = others[iou <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

def _nms_3d(
        boxes: torch.Tensor, 
        scores: torch.Tensor, 
        iou_threshold: float
    ) -> torch.Tensor:
    """Pure torch NMS for 3D boxes."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    z1 = boxes[:, 4]
    z2 = boxes[:, 5]
    heights = (y2 - y1).clamp(min=0)
    widths = (x2 - x1).clamp(min=0)
    depths = (z2 - z1).clamp(min=0)
    volumes = heights * widths * depths
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        others = order[1:]
        yy1 = torch.max(y1[i], y1[others])
        xx1 = torch.max(x1[i], x1[others])
        zz1 = torch.max(z1[i], z1[others])
        yy2 = torch.min(y2[i], y2[others])
        xx2 = torch.min(x2[i], x2[others])
        zz2 = torch.min(z2[i], z2[others])
        h = (yy2 - yy1).clamp(min=0)
        w = (xx2 - xx1).clamp(min=0)
        d = (zz2 - zz1).clamp(min=0)
        inter = h * w * d
        union = volumes[i] + volumes[others] - inter
        iou = inter / (union + 1e-6)
        order = others[iou <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)



class ClassificationHead(nn.Module):
    """Classification head for detecting object presence."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        hidden_channels: int = 256,
        activation: str = 'relu',
        dim: int = 2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.dim = dim
        
        self.conv1 = ConvBlock(
            in_channels, 
            hidden_channels, 
            kernel_size=3, 
            padding=1,
            activation=activation, 
            dim=dim
        )
        self.conv2 = ConvBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3, 
            padding=1,
            activation=activation, 
            dim=dim
        )
        self.conv3 = ConvBlock(
            hidden_channels,
            hidden_channels, 
            kernel_size=3,
            padding=1,
            activation=activation, 
            dim=dim
        )
        self.conv4 = ConvBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3,
            padding=1,
            activation=activation, 
            dim=dim
        )
        self.conv_final = ConvBlock(
            hidden_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            activation=None, 
            dim=dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class predictions."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        class_out = self.conv_final(x)
        
        # Reshape to (B, -1, num_classes)
        batch_size = x.size(0)
        if self.dim == 2:
            class_out = class_out.permute(0, 2, 3, 1).contiguous()
        else:
            class_out = class_out.permute(0, 2, 3, 4, 1).contiguous()
        
        class_out = class_out.view(batch_size, -1, self.num_classes)

        return class_out


class BBoxHead(nn.Module):
    """Bounding box regression head."""
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 9,
        hidden_channels: int = 256,
        activation: str = 'relu',
        dim: int = 2
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.dim = dim
        output_channels = num_anchors * (dim * 2)
        
        self.conv1 = ConvBlock(
            in_channels, 
            hidden_channels, 
            kernel_size=3, 
            padding=1, 
            activation=activation, 
            dim=dim
        )
        self.conv2 = ConvBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3,
            padding=1, 
            activation=activation, 
            dim=dim
        )
        self.conv3 = ConvBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3, 
            padding=1, 
            activation=activation, 
            dim=dim
        )
        self.conv4 = ConvBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3, 
            padding=1, 
            activation=activation, 
            dim=dim
        )
        self.conv_final = ConvBlock(
            hidden_channels, 
            output_channels, 
            kernel_size=3,
            padding=1, 
            activation=None, 
            dim=dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning bbox deltas."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        bb_out = self.conv_final(x)
        
        # Reshape to (B, -1, dim*2)
        batch_size = x.size(0)
        if self.dim == 2:
            bb_out = bb_out.permute(0, 2, 3, 1).contiguous()
        else:
            bb_out = bb_out.permute(0, 2, 3, 4, 1).contiguous()
        
        bb_out = bb_out.view(batch_size, -1, self.dim * 2)

        return bb_out


class SegmentationHead(nn.Module):
    """U-Net style decoder for segmentation."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        dim: int = 2
    ):
        super().__init__()
        self.dim = dim
        
        # Simple 1x1 convolution to produce segmentation mask
        self.conv_seg = ConvBlock(
            in_channels, 
            num_classes, 
            kernel_size=1,
            padding=0, 
            norm_type=None,
            activation=None, 
            dim=dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning segmentation mask."""
        seg_out = self.conv_seg(x)

        return seg_out


class RetinaUNetCore(nn.Module):
    """
    Retina U-Net: Multi-task detection and segmentation model for medical images.
    
    Combines:
    - Retina Net architecture for object detection (classification + bbox regression)
    - U-Net style segmentation decoder
    - FPN backbone for multi-scale features
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dim: int = 2,
        fpn_base_channels: int = 48,
        fpn_out_channels: int = 192,
        fpn_num_blocks: List[int] = None,
        rpn_hidden_channels: int = 256,
        norm_type: str = None,
        activation: str = 'relu',
        rpn_anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        rpn_anchor_scales: Dict[str, Dict[str, List[float]]] = None,
        rpn_anchor_stride: int = 1,
        pyramid_levels: List[int] = [2, 3, 4, 5], # Corresponding to P2, P3, P4, P5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes_head = num_classes
        self.num_classes_seg = 2 # Binary segmentation (foreground vs background)
        self.pyramid_levels = pyramid_levels
        self.num_anchors = len(rpn_anchor_ratios) * 3  # Anchor Sub-scaling
        self.dim = dim
        if self.dim == 2:
            self.rpn_bbox_std_dev = torch.tensor(
                [0.1, 0.1, 0.2, 0.2], dtype=torch.float32
            )
        else:
            self.rpn_bbox_std_dev = torch.tensor(
                [0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype=torch.float32
            )
        self.pre_nms_limit = 3000
        self.nms_threshold = 1e-5
        self.max_instances_per_batch_element = 10
        
        # Backbone
        self.fpn = FPN(
            in_channels = in_channels,
            base_channels = fpn_base_channels,
            out_channels = fpn_out_channels,
            num_blocks = fpn_num_blocks,
            norm_type = norm_type,
            activation = activation,
            dim = dim,
        )

        # Heads
        self.classification_head = ClassificationHead(
            fpn_out_channels, 
            num_classes, 
            self.num_anchors, 
            rpn_hidden_channels, 
            activation, 
            dim
        )
        self.bbox_head = BBoxHead(
            fpn_out_channels, 
            self.num_anchors,
            rpn_hidden_channels, 
            activation, 
            dim
        )
        self.segmentation_head = SegmentationHead(
            fpn_out_channels, self.num_classes_seg, dim
        )
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            rpn_anchor_scales = rpn_anchor_scales,
            rpn_anchor_ratios = rpn_anchor_ratios,
            rpn_anchor_stride = rpn_anchor_stride,
            pyramid_levels = pyramid_levels,
            dim = dim
        )

        # LRU cache anchors by (feature_shapes, device) to avoid regenerating on every forward pass
        # OrderedDict maintains insertion order; max 3 cached anchor sets before LRU eviction
        self.anchor_cache = OrderedDict()
        self.max_anchor_cache_size = 3
    
    def _clear_anchor_cache(self):
        """Clear anchor cache and explicitly free GPU memory."""
        for key in list(self.anchor_cache.keys()):
            anchors = self.anchor_cache[key]
            # Explicitly move to CPU and delete to free GPU memory
            if anchors.is_cuda:
                anchors = anchors.cpu()
            del self.anchor_cache[key]
        self.anchor_cache.clear()
        # Force garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _apply(self, fn):
        """Override to clear anchor cache when model moves to a different device."""
        super()._apply(fn)
        self._clear_anchor_cache()
        return self
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) for 2D or (B, C, H, W, D) for 3D
        
        Returns:
            Dictionary with:
                - 'class_logits': (B, num_anchors, num_classes)
                - 'bbox_deltas': (B, num_anchors, dim*2)
                - 'segmentation': (B, 1, H, W) or (B, 1, H, W, D)
                - 'detections': final post-NMS detections
        """

        # Backbone
        features = self.fpn(x)  # List of multi-scale features
        
        # Heads on lowest resolution (highest semantic)
        class_outputs = []
        bbox_outputs = []
        for level in self.pyramid_levels:
            class_out = self.classification_head(features[level])
            bbox_out = self.bbox_head(features[level])
            class_outputs.append(class_out)
            bbox_outputs.append(bbox_out)
        
        # Segmentation on higher resolution features
        seg_features = features[0]  # Highest resolution P0
        segmentation = self.segmentation_head(seg_features)

        # Generate or retrieve cached anchors for the current feature shapes and device
        feature_shapes = tuple(f.shape[2:] for f in features)
        device = features[0].device
        cache_key = (feature_shapes, str(device))
        
        if cache_key not in self.anchor_cache:
            # Generate new anchors
            self.anchor_cache[cache_key] = self.anchor_generator(features)
            # LRU eviction: remove oldest entry if cache exceeds max size
            if len(self.anchor_cache) > self.max_anchor_cache_size:
                # Pop and explicitly free memory of oldest entry
                old_key, old_anchors = self.anchor_cache.popitem(last=False)
                if old_anchors.is_cuda:
                    old_anchors = old_anchors.cpu()
                del old_anchors
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # Move accessed key to end to mark as recently used
            self.anchor_cache.move_to_end(cache_key)
        
        anchors = self.anchor_cache[cache_key]
        class_logits = torch.cat(class_outputs, dim=1)
        bbox_deltas = torch.cat(bbox_outputs, dim=1)

        detections = self.refine_detections(
            anchors=anchors,
            class_logits=class_logits,
            bbox_deltas=bbox_deltas,
            batch_size=x.shape[0],
            image_size=tuple(x.shape[-self.dim:])
        )

        return {
            'class_logits': class_logits,
            'bbox_deltas': bbox_deltas,
            'segmentation': segmentation,
            'detections': detections,
            'anchors': anchors
        }

    def refine_detections(
        self,
        anchors: torch.Tensor,
        class_logits: torch.Tensor,
        bbox_deltas: torch.Tensor,
        batch_size: int,
        image_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """Refine raw network outputs into final detections.

        Uses top-k score filtering, delta decoding, clipping, and NMS.
        """

        # Raw foreground probabilities after background channel
        if class_logits.size(-1) > 1:
            probs = torch.softmax(class_logits, dim=-1)
            fg_probs = probs[:, :, 1:]
        else:
            fg_probs = torch.sigmoid(class_logits)
            fg_probs = fg_probs.unsqueeze(-1)

        flat_probs = fg_probs.view(-1)
        topk = min(self.pre_nms_limit, flat_probs.numel())
        scores, order = flat_probs.sort(descending=True)
        order = order[:topk]
        scores = scores[:topk]

        num_classes = fg_probs.shape[-1]
        proposal_ids = order // num_classes
        class_ids = order % num_classes

        anchor_ids = proposal_ids % anchors.shape[0]
        selected_anchors = anchors[anchor_ids]
        selected_deltas = bbox_deltas.view(batch_size * anchors.shape[0], -1)
        selected_deltas = selected_deltas[proposal_ids]
        selected_scores = scores

        bbox_std_dev = self.rpn_bbox_std_dev.to(selected_deltas.device)
        decoded_deltas = selected_deltas * bbox_std_dev
        if self.dim == 2:
            decoded = _apply_box_deltas_2d(selected_anchors, decoded_deltas)
            decoded = _clip_boxes_2d(decoded, image_size)
        else:
            decoded = _apply_box_deltas_3d(selected_anchors, decoded_deltas)
            decoded = _clip_boxes_3d(decoded, image_size)

        detections = []
        for b in range(batch_size):
            batch_mask = proposal_ids // anchors.shape[0] == b
            if batch_mask.sum() == 0:
                continue
            batch_boxes = decoded[batch_mask]
            batch_scores = selected_scores[batch_mask]
            batch_classes = class_ids[batch_mask] + 1
            keep = []
            for cls in torch.unique(batch_classes):
                cls_mask = batch_classes == cls
                cls_boxes = batch_boxes[cls_mask]
                cls_scores = batch_scores[cls_mask]
                if cls_boxes.numel() == 0:
                    continue
                if self.dim == 2:
                    keep_idx = _nms_2d(cls_boxes, cls_scores, self.nms_threshold)
                else:
                    keep_idx = _nms_3d(cls_boxes, cls_scores, self.nms_threshold)
                keep.append(
                    torch.stack(
                        [
                            torch.where(cls_mask)[0][keep_idx], 
                            torch.full(
                                (keep_idx.numel(),), cls, device=cls_scores.device
                            )
                        ], 
                        dim=1)
                )
            if keep:
                batch_keep = torch.cat(keep, dim=0)
                kept_boxes = batch_boxes[batch_keep[:, 0].long()]
                kept_scores = batch_scores[batch_keep[:, 0].long()]
                kept_classes = batch_keep[:, 1].long()
                batch_ids = torch.full(
                    (kept_boxes.shape[0], 1), 
                    b, 
                    device=kept_boxes.device, 
                    dtype=torch.long
                )
                if kept_boxes.shape[0] > self.max_instances_per_batch_element:
                    selected_idx = kept_scores.argsort(descending=True)
                    selected_idx = selected_idx[:self.max_instances_per_batch_element]
                    kept_boxes = kept_boxes[selected_idx]
                    kept_scores = kept_scores[selected_idx]
                    kept_classes = kept_classes[selected_idx]
                    batch_ids = batch_ids[selected_idx]

                detections.append(
                    torch.cat(
                        [
                            kept_boxes, 
                            batch_ids.float(), 
                            kept_classes.float().unsqueeze(1), 
                            kept_scores.unsqueeze(1)
                        ], 
                        dim=1
                    )
                )
        
        # detections for 2D [y1, x1, y2, x2, batch_id, class_id, score]
        # detections for 3D [y1, x1, y2, x2, z1, z2, batch_id, class_id, score]
        if detections:
            detections = torch.cat(detections, dim=0)
        else:
            detections = torch.empty((0, anchors.shape[1] + 3), device=anchors.device)
        
        return detections


class RetinaUNet(BaseModel):
    """Retina U-Net model wrapper for PyHealth.

    Wraps RetinaUNetLayer with dataset-driven initialization and standard
    PyHealth outputs.

    The dataset argument is kept because BaseModel depends on dataset
    schema metadata. Training-specific detection target preparation and
    loss computation are handled inside this class, so an external
    retina_unet_train helper is no longer required for core model logic.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 1,
        num_classes: int = 2,
        dim: int = 2,
        fpn_base_channels: int = 48,
        fpn_out_channels: int = 192,
        fpn_num_blocks: Optional[List[int]] = None, # default restnet-style [3, 4, 6, 3]
        rpn_hidden_channels: int = 256,
        norm_type: Optional[str] = None,
        activation: str = "relu",
        rpn_anchor_ratios: Optional[List[float]] = [0.5, 1.0, 2.0],
        rpn_anchor_scales: Optional[Dict[str, Dict[str, List[float]]]] = None,
        rpn_anchor_stride: int = 1,
        pyramid_levels: Optional[List[int]] = [2, 3, 4, 5],
    ):
        super().__init__(dataset=dataset)

        self.core = RetinaUNetCore(
            in_channels=in_channels,
            num_classes=num_classes,
            dim=dim,
            fpn_base_channels=fpn_base_channels,
            fpn_out_channels=fpn_out_channels,
            fpn_num_blocks=fpn_num_blocks,
            rpn_hidden_channels=rpn_hidden_channels,
            norm_type=norm_type,
            activation=activation,
            rpn_anchor_ratios=rpn_anchor_ratios,
            rpn_anchor_scales=rpn_anchor_scales,
            rpn_anchor_stride=rpn_anchor_stride,
            pyramid_levels=pyramid_levels,
        ).to(self.device)


    def forward(
            self, 
            images, 
            gt_seg_masks=None, 
            gt_boxes_list=None, 
            gt_classes_list=None, 
            **kwargs
        ) -> dict[str, torch.Tensor]:
        """Forward pass through the model."""

        # Retina U-Net forward pass
        outputs = self.core(images.to(self.device))

        detections = outputs['detections']
        class_logits = outputs['class_logits']
        bbox_deltas = outputs['bbox_deltas']
        seg_logits = outputs['segmentation']
        anchors = outputs['anchors']

        if gt_seg_masks is None:
            batch_class_loss = torch.tensor(0.0, device=self.device)
            batch_bbox_loss = torch.tensor(0.0, device=self.device)
            seg_loss = torch.tensor(0.0, device=self.device)
        else:
            batch_class_loss = 0.0
            batch_bbox_loss = 0.0
            
            # Per-image Matching and Loss
            batch_size = images.shape[0]
            for b in range(batch_size):
                if len(gt_boxes_list[b]) > 0:
                    gt_boxes = torch.stack(gt_boxes_list[b]).to(self.device).contiguous()
                    # 0 index is for PyHealth dataset schema compatibility
                    gt_class_ids = gt_classes_list[b][0].to(self.device).contiguous()
                else:
                    # Create empty tensors with correct shapes and dtypes
                    gt_boxes = torch.zeros((0, 4), device=self.device)
                    gt_class_ids = torch.tensor([], dtype=torch.int64, device=self.device)
                
                anchor_class_match, anchor_target_deltas = self._compute_anchor_matches(
                    anchors, gt_boxes, gt_class_ids
                )
                
                batch_class_loss += self._compute_class_loss(
                    class_logits[b], 
                    anchor_class_match
                    )
                batch_bbox_loss += self._compute_bbox_loss(
                    bbox_deltas[b], anchor_target_deltas, anchor_class_match
                )
            
            # Average batch detection losses
            batch_class_loss /= batch_size
            batch_bbox_loss /= batch_size
            
            # Segmentation Loss (Foreground-only Dice + CE)
            seg_loss = self._compute_segmentation_loss(
                seg_logits, gt_seg_masks.to(self.device)
            )
        
        # Standard weights: Class=1.0, Bbox=1.0, Seg=0.5
        total_loss = batch_class_loss + batch_bbox_loss + (0.5 * seg_loss)

        # Mandatory PyHealth return format
        results = {
            "loss": total_loss,
            "class_loss": batch_class_loss,
            "bbox_loss": batch_bbox_loss,
            "seg_loss": seg_loss,
            "det_bboxes": detections,
            "class_logits": class_logits,
            "bbox_deltas": bbox_deltas,
            "seg_logits": seg_logits,
            "anchors": anchors
        }
        return results

    @staticmethod
    def _compute_iou_matrix_2d(anchors, gt_boxes):
        """
        Corrected Vectorized IoU in PyTorch.
        anchors: (N, 4) tensor [y1, x1, y2, x2]
        gt_boxes: (M, 4) tensor [y1, x1, y2, x2]
        """
        # 1. Areas
        a_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        g_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        # 2. Intersections
        lt = torch.max(anchors[:, None, :2], gt_boxes[None, :, :2]) 
        rb = torch.min(anchors[:, None, 2:], gt_boxes[None, :, 2:]) 
        
        wh = (rb - lt).clamp(min=0)  # [h, w]
        inter_area = wh[:, :, 0] * wh[:, :, 1] # (N, M)

        # 3. Union
        union_area = a_area[:, None] + g_area[None, :] - inter_area
        
        # Return (M, N) matrix to keep GTs as rows and Anchors as columns
        return (inter_area / union_area.clamp(min=1e-6)).T

    @staticmethod
    def _compute_iou_matrix_3d(anchors, gt_boxes):
        """
        Optimized Vectorized 3D IoU.
        Order: [y1, x1, y2, x2, z1, z2]
        """
        if anchors.numel() == 0 or gt_boxes.numel() == 0:
            return torch.zeros(
                (gt_boxes.shape[0], anchors.shape[0]), device=anchors.device
            )

        # 1. Define index groups for [y, x, z]
        mins = [0, 1, 4]
        maxs = [2, 3, 5]

        # 2. Compute Volumes
        # (y2-y1) * (x2-x1) * (z2-z1)
        a_vol = (anchors[:, maxs] - anchors[:, mins]).clamp(min=0).prod(dim=1)
        g_vol = (gt_boxes[:, maxs] - gt_boxes[:, mins]).clamp(min=0).prod(dim=1)

        # 3. Intersections (The Vectorized Step)
        # Resulting 'lt' and 'rb' are (N, M, 3)
        lt = torch.max(anchors[:, None, mins], gt_boxes[None, :, mins]) 
        rb = torch.min(anchors[:, None, maxs], gt_boxes[None, :, maxs]) 
        
        # Compute width, height, and depth at once
        inter_vol = (rb - lt).clamp(min=0).prod(dim=2) # (N, M)

        # 4. Union and Final IoU
        union_vol = a_vol[:, None] + g_vol[None, :] - inter_vol
        
        # Return (M, N) to match your original implementation's shape
        return (inter_vol / union_vol.clamp(min=1e-6)).T

    def _compute_anchor_matches(
            self, 
            anchors, 
            gt_boxes, 
            gt_class_ids, 
            pos_thresh=0.5, 
            neg_thresh=0.4, 
            dim=2
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive Anchor Matching for 2D or 3D.
        dim=2: [y1, x1, y2, x2]
        dim=3: [y1, x1, y2, x2, z1, z2]
        """
        device = anchors.device
        num_anchors = anchors.shape[0]
        
        if gt_boxes.shape[0] == 0:
            # Returns (num_anchors,) for matches and (num_anchors, 4 or 6) for deltas
            return torch.full((num_anchors,), -1, device=device), torch.zeros_like(anchors)

        # 1. Select the correct IoU function based on dimension
        if dim == 2:
            iou_matrix = self._compute_iou_matrix_2d(anchors, gt_boxes)
        else:
            iou_matrix = self._compute_iou_matrix_3d(anchors, gt_boxes)

        # 2. Find best GT for each anchor
        max_iou_per_anchor, best_gt_idx_per_anchor = torch.max(iou_matrix, dim=0)

        # Initialize as Neutral/Ignore (-1)
        anchor_class_match = torch.full((num_anchors,), -1, dtype=torch.int32, device=device)

        # 3. Assign Background (0) and Positives (class_id)
        anchor_class_match[max_iou_per_anchor < neg_thresh] = 0
        
        pos_mask = max_iou_per_anchor >= pos_thresh
        anchor_class_match[pos_mask] = gt_class_ids[best_gt_idx_per_anchor[pos_mask]].to(torch.int32)

        # 4. SAFETY NET: Force best anchor for every GT
        _, best_anchor_idx_per_gt = torch.max(iou_matrix, dim=1)
        
        # Force these to be positive and update the index tracker for deltas
        anchor_class_match[best_anchor_idx_per_gt] = gt_class_ids.to(torch.int32)
        best_gt_idx_per_anchor[best_anchor_idx_per_gt] = torch.arange(len(gt_boxes), device=device)

        # 5. Regression Targets (Deltas)
        matched_gt_boxes = gt_boxes[best_gt_idx_per_anchor]
        anchor_target_deltas = (matched_gt_boxes - anchors) / (anchors + 1e-5)

        return anchor_class_match, anchor_target_deltas

    @staticmethod
    def _compute_class_loss(class_pred_logits, anchor_matches, alpha=0.25, gamma=2.0):
        """
        Handles [-1=ignore, 0=background, 1+=class_id]
        """
        # 1. Mask out anchors labeled as -1 (ignore/neutral)
        valid_indices = torch.nonzero(anchor_matches != -1).reshape(-1)
        if valid_indices.numel() == 0:
            return torch.tensor(0.0, device=class_pred_logits.device, requires_grad=True)

        logits = class_pred_logits[valid_indices]
        targets = anchor_matches[valid_indices].long()

        # 2. Focal Loss Math
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        f_loss = alpha * (1 - pt)**gamma * ce_loss

        # 3. Normalize by POSITIVE anchors only
        num_pos = torch.clamp(torch.sum(anchor_matches > 0).float(), min=1.0)
        return f_loss.sum() / num_pos

    @staticmethod
    def _compute_bbox_loss(
        bbox_deltas, 
        anchor_target_deltas, 
        anchor_class_match, 
        beta=0.11 # beta ~1/9 is standard for many Retina implementations
    ):
        """
        Improved Bbox loss aligned with RetinaNet normalization.
        """
        # 1. Mask for positive anchors only
        pos_mask = anchor_class_match > 0
        num_pos = pos_mask.sum().float()
        
        if num_pos == 0:
            return torch.tensor(0.0, device=bbox_deltas.device, requires_grad=True)
        
        # 2. Extract positive predictions and targets
        pos_deltas = bbox_deltas[pos_mask]
        pos_targets = anchor_target_deltas[pos_mask]
        
        # 3. Smooth L1 Loss 
        # Use reduction='sum' so we can normalize manually
        bbox_loss = F.smooth_l1_loss(
            pos_deltas, pos_targets, beta=beta, reduction='sum'
        )
        
        # 4. Normalize by the SAME num_pos used in Class Loss
        # This keeps the 'weight' of classification and regression balanced.
        return bbox_loss / torch.clamp(num_pos, min=1.0)

    @staticmethod
    def _compute_segmentation_loss(
        seg_logits: torch.Tensor,
        seg_masks: torch.Tensor,
        n_classes=2
    ) -> torch.Tensor:
        # 1. Cross-Entropy: Use weights if possible, but keep reduction='mean'
        target_masks = seg_masks.squeeze(1).long()
        ce_loss = F.cross_entropy(seg_logits, target_masks)

        # 2. Dice Loss: Focus on Foreground
        probs = F.softmax(seg_logits, dim=1)
        
        # One-hot encoding
        target_ohe = F.one_hot(target_masks, num_classes=n_classes)
        target_ohe = target_ohe.permute(0, 3, 1, 2).float() # [B, C, H, W]

        # Compute intersection and cardinality per class
        # Sum over spatial dimensions only (H, W)
        dims = (2, 3) 
        intersection = torch.sum(probs * target_ohe, dim=dims)
        cardinality = torch.sum(probs + target_ohe, dim=dims)
        
        dice_score = (2. * intersection + 1e-5) / (cardinality + 1e-5)
        
        # IMPORTANT: Exclude background (index 0) from the Dice Loss
        # Only average Dice for classes 1, 2, ...
        foreground_dice_loss = 1 - dice_score[:, 1:].mean() 
        
        return 0.5 * ce_loss + 0.5 * foreground_dice_loss

if __name__ == '__main__':
    # Test 2D
    model_2d = RetinaUNetCore(in_channels=1, num_classes=2, dim=2)
    x_2d = torch.randn(2, 1, 64, 64)
    output_2d = model_2d(x_2d)
    print("2D Output shapes:")
    print(f"  class_logits: {output_2d['class_logits'].shape}")
    print(f"  bbox_deltas: {output_2d['bbox_deltas'].shape}")
    print(f"  segmentation: {output_2d['segmentation'].shape}")
    print(f"  detections: {output_2d['detections'].shape}")
    
    # Test 3D
    model_3d = RetinaUNetCore(in_channels=1, num_classes=2, dim=3)
    x_3d = torch.randn(2, 1, 64, 64, 32)
    output_3d = model_3d(x_3d)
    print("\n3D Output shapes:")
    print(f"  class_logits: {output_3d['class_logits'].shape}")
    print(f"  bbox_deltas: {output_3d['bbox_deltas'].shape}")
    print(f"  segmentation: {output_3d['segmentation'].shape}")
    print(f"  detections: {output_3d['detections'].shape}")



