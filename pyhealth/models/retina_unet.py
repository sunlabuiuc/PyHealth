"""
Clean Retina U-Net Implementation for Medical Image Object Detection

Reference: Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision 
for Medical Object Detection (https://arxiv.org/abs/1811.08661)

This implementation uses only standard PyTorch without custom CUDA operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import ImageProcessor, TensorProcessor



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
        scales_xy = torch.tensor(self.rpn_anchor_scales['xy'][f'P{level}'], dtype=torch.float32, device=device)
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
        scales_xy = torch.tensor(self.rpn_anchor_scales['xy'][f'P{level}'], dtype=torch.float32, device=device)
        scales_z = torch.tensor(self.rpn_anchor_scales['z'][f'P{level}'], dtype=torch.float32, device=device)
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


def _clip_boxes_2d(boxes: torch.Tensor, window: Tuple[float, float]) -> torch.Tensor:
    y1 = boxes[:, 0].clamp(min=0, max=window[0])
    x1 = boxes[:, 1].clamp(min=0, max=window[1])
    y2 = boxes[:, 2].clamp(min=0, max=window[0])
    x2 = boxes[:, 3].clamp(min=0, max=window[1])

    return torch.stack([y1, x1, y2, x2], dim=1)


def _clip_boxes_3d(boxes: torch.Tensor, window: Tuple[float, float, float]) -> torch.Tensor:
    y1 = boxes[:, 0].clamp(min=0, max=window[0])
    x1 = boxes[:, 1].clamp(min=0, max=window[1])
    y2 = boxes[:, 2].clamp(min=0, max=window[0])
    x2 = boxes[:, 3].clamp(min=0, max=window[1])
    z1 = boxes[:, 4].clamp(min=0, max=window[2])
    z2 = boxes[:, 5].clamp(min=0, max=window[2])

    return torch.stack([y1, x1, y2, x2, z1, z2], dim=1)


def _nms_2d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
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


def _nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
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


class RetinaUNetLayer(nn.Module):
    """Core Retina U-Net neural network layer for medical image detection and segmentation.
    
    This layer combines a Feature Pyramid Network (FPN) backbone with detection and
    segmentation heads. It is designed to be wrapped by the RetinaUNet BaseModel for
    PyHealth integration.
    
    Args:
        in_channels (int): Number of input channels. Default is 1 (grayscale).
        num_classes (int): Number of detection classes. Default is 2.
        num_seg_classes (int): Number of segmentation classes. Default is 2.
        dim (int): Spatial dimensionality (2 or 3). Default is 2.
        fpn_base_channels (int): Base channels for FPN. Default is 48.
        fpn_out_channels (int): Output channels for FPN features. Default is 192.
        fpn_num_blocks (List[int]): Blocks per FPN stage (ResNet-style). Default is None (auto).
        rpn_hidden_channels (int): Hidden channels in detection/segmentation heads. Default is 256.
        norm_type (str): Normalization type ('batch', 'instance', None). Default is None.
        activation (str): Activation function ('relu', 'leaky_relu'). Default is 'relu'.
        rpn_anchor_ratios (List[float]): Anchor aspect ratios. Default is [0.5, 1.0, 2.0].
        rpn_anchor_scales (Dict): Anchor scales per pyramid level. Default is None (auto).
        rpn_anchor_stride (int): Anchor stride for dense prediction. Default is 1.
        pyramid_levels (List[int]): Pyramid levels to use (e.g., [2,3,4,5]=P2-P5). Default is [2,3,4,5].
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        num_seg_classes: int = 2,
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
        self.num_classes_seg = num_seg_classes
        self.cf = type('cf', (), {'dim': dim})()
        self.pyramid_levels = pyramid_levels
        self.num_anchors = len(rpn_anchor_ratios) * 3  # Anchor Sub-scaling
        self.dim = dim
        if self.dim == 2:
            self.rpn_bbox_std_dev = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32)
        else:
            self.rpn_bbox_std_dev = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype=torch.float32)
        self.pre_nms_limit = 3000
        self.nms_threshold = 1e-5
        self.model_max_instances_per_batch_element = 10
        
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
        
        # Backward-compatible aliases used by existing tests
        self.Fpn = self.fpn
        self.Classifier = self.classification_head
        self.BBRegressor = self.bbox_head
        
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
            'detections': detections
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
        selected_deltas = bbox_deltas.view(batch_size * anchors.shape[0], -1)[proposal_ids]
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
                keep.append(torch.stack([torch.where(cls_mask)[0][keep_idx], torch.full((keep_idx.numel(),), cls, device=cls_scores.device)], dim=1))
            if keep:
                batch_keep = torch.cat(keep, dim=0)
                kept_boxes = batch_boxes[batch_keep[:, 0].long()]
                kept_scores = batch_scores[batch_keep[:, 0].long()]
                kept_classes = batch_keep[:, 1].long()
                batch_ids = torch.full((kept_boxes.shape[0], 1), b, device=kept_boxes.device, dtype=torch.long)
                if kept_boxes.shape[0] > self.model_max_instances_per_batch_element:
                    selected_idx = kept_scores.argsort(descending=True)[:self.model_max_instances_per_batch_element]
                    kept_boxes = kept_boxes[selected_idx]
                    kept_scores = kept_scores[selected_idx]
                    kept_classes = kept_classes[selected_idx]
                    batch_ids = batch_ids[selected_idx]
                detections.append(torch.cat([kept_boxes, batch_ids.float(), kept_classes.float().unsqueeze(1), kept_scores.unsqueeze(1)], dim=1))
        
        # detections for 2D [y1, x1, y2, x2, batch_id, class_id, score]
        # detections for 3D [y1, x1, y2, x2, z1, z2, batch_id, class_id, score]
        if detections:
            detections = torch.cat(detections, dim=0)
        else:
            detections = torch.empty((0, anchors.shape[1] + 3), device=anchors.device)
        
        return detections


class RetinaUNet(BaseModel):
    """Retina U-Net model for medical image object detection and segmentation.
    
    A PyHealth-integrated wrapper around the Retina U-Net architecture that combines
    Retina Net (two-stage object detection with focal loss) with U-Net style segmentation.
    The model uses a Feature Pyramid Network (FPN) backbone to handle multi-scale features
    and supports both 2D and 3D medical images.
    
    The model returns segmentation predictions by default and can optionally return
    detection bounding boxes and class predictions via the return_aux flag.
    
    Note:
        - Requires image and segmentation label inputs from the dataset
        - Supports dataset-driven initialization with automatic processor fitting
        - Returns standard PyHealth output keys: logit, y_prob, y_true (optional), loss (optional)
        - Optional auxiliary outputs: seg_preds, class_logits, bbox_deltas, detections, boxes
    
    Args:
        dataset (SampleDataset): Dataset with fitted input and output processors.
        feature_key (str): Key for image input in the dataset. Default is "image".
        seg_label_key (str): Key for segmentation labels. Default is "seg".
        box_label_key (str): Key for bounding box targets (optional). Default is "bb_target".
        class_label_key (str): Key for class labels (optional). Default is "roi_labels".
        in_channels (int): Number of input channels. Default is 1.
        num_classes (int): Number of detection classes. Default is 2.
        head_classes (int): Alternative parameter name (mapped to num_classes). Default is None.
        num_seg_classes (int): Number of segmentation classes. Default is 2.
        dim (int): Spatial dimensionality (2 or 3). Default is 2.
        fpn_base_channels (int): Base channels for FPN. Default is 48.
        fpn_out_channels (int): Output channels for FPN features. Default is 192.
        fpn_num_blocks (List[int]): Blocks per FPN stage. Default is None.
        rpn_hidden_channels (int): Hidden channels in detection/segmentation heads. Default is 256.
        norm_type (str): Normalization type ('batch', 'instance', None). Default is None.
        activation (str): Activation function. Default is 'relu'.
        rpn_anchor_ratios (List[float]): Anchor aspect ratios. Default is None.
        rpn_anchor_scales (Dict): Anchor scales per pyramid level. Default is None.
        rpn_anchor_stride (int): Anchor stride. Default is 1.
        pyramid_levels (List[int]): FPN pyramid levels. Default is None.
    
    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> from pyhealth.models import RetinaUNet
        >>> # Create a sample dataset with medical images and segmentation masks
        >>> dataset = SampleDataset(
        ...     samples=[...],  # samples with 'image' and 'seg' keys
        ...     input_schema={"image": "image"},
        ...     output_schema={"seg": "image"},
        ... )
        >>> model = RetinaUNet(dataset=dataset, num_classes=2, dim=2)
        >>> # Forward pass with image and segmentation label
        >>> output = model(image=x, seg=y)
        >>> print(output.keys())  # ['logit', 'y_prob', 'y_true', 'loss']
        >>> # Get auxiliary detection outputs
        >>> aux_output = model(image=x, seg=y, return_aux=True)
        >>> print('detections' in aux_output)  # True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "image",
        seg_label_key: str = "seg",
        box_label_key: Optional[str] = "bb_target",
        class_label_key: Optional[str] = "roi_labels",
        in_channels: int = 1,
        num_classes: int = 2,
        head_classes: Optional[int] = None,
        num_seg_classes: int = 2,
        dim: int = 2,
        fpn_base_channels: int = 48,
        fpn_out_channels: int = 192,
        fpn_num_blocks: Optional[List[int]] = None,
        rpn_hidden_channels: int = 256,
        norm_type: Optional[str] = None,
        activation: str = "relu",
        rpn_anchor_ratios: Optional[List[float]] = None,
        rpn_anchor_scales: Optional[Dict[str, Dict[str, List[float]]]] = None,
        rpn_anchor_stride: int = 1,
        pyramid_levels: Optional[List[int]] = None,
    ):
        super().__init__(dataset=dataset)

        self.feature_key = feature_key
        self.seg_label_key = seg_label_key
        self.box_label_key = box_label_key
        self.class_label_key = class_label_key

        self.label_key = self.seg_label_key
        self.mode = self.dataset.output_schema[self.label_key]

        self.image_processor = ImageProcessor()
        self.tensor_processor = TensorProcessor()

        if rpn_anchor_ratios is None:
            rpn_anchor_ratios = [0.5, 1.0, 2.0]
        if pyramid_levels is None:
            pyramid_levels = [2, 3, 4, 5]
        if head_classes is not None:
            num_classes = head_classes

        self.core = RetinaUNetLayer(
            in_channels=in_channels,
            num_classes=num_classes,
            num_seg_classes=num_seg_classes,
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

    def _prepare_input_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        x_tensor = self.tensor_processor.process([x]) if not isinstance(x, list) else self.tensor_processor.process(x)
        if x_tensor.ndim >= 3 and x_tensor.shape[1] not in (1, 3):
            x_tensor = self.image_processor.process([x]).to(self.device)
        return x_tensor.float().to(self.device)

    def _prepare_seg_target(self, seg_target: Any) -> torch.Tensor:
        if isinstance(seg_target, torch.Tensor):
            target = seg_target.to(self.device)
        elif isinstance(seg_target, np.ndarray):
            target = torch.from_numpy(seg_target).to(self.device)
        else:
            target = self.tensor_processor.process([seg_target]) if not isinstance(seg_target, list) else self.tensor_processor.process(seg_target)
            target = target.to(self.device)

        if target.ndim >= 4 and target.shape[1] == 1:
            target = target[:, 0]
        return target.long()

    def forward(self, **kwargs) -> Dict[str, Any]:
        x = self._prepare_input_tensor(kwargs[self.feature_key])
        outputs = self.core(x)

        seg_logits = outputs["segmentation"]
        if seg_logits.shape[1] > 1:
            y_prob = torch.softmax(seg_logits, dim=1)
        else:
            y_prob = torch.sigmoid(seg_logits)

        results: Dict[str, Any] = {
            "logit": seg_logits,
            "y_prob": y_prob,
        }

        if self.seg_label_key in kwargs:
            y_true = self._prepare_seg_target(kwargs[self.seg_label_key])
            results["y_true"] = y_true

            if seg_logits.shape[1] > 1:
                loss = F.cross_entropy(seg_logits, y_true)
            else:
                y_true_float = y_true.float().unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(seg_logits, y_true_float)
            results["loss"] = loss

        if kwargs.get("return_aux", False):
            results.update(
                {
                    "seg_preds": torch.argmax(seg_logits, dim=1) if seg_logits.shape[1] > 1 else (y_prob > 0.5).long().squeeze(1),
                    "class_logits": outputs["class_logits"],
                    "bbox_deltas": outputs["bbox_deltas"],
                    "detections": outputs["detections"],
                    "boxes": outputs["detections"],
                    "monitor_values": {},
                }
            )

        return results


if __name__ == '__main__':
    # Test the core layer directly
    print("Testing RetinaUNetLayer (core PyTorch module):")
    model_2d = RetinaUNetLayer(in_channels=1, num_classes=2, dim=2)
    x_2d = torch.randn(2, 1, 256, 256)
    output_2d = model_2d(x_2d)
    print("2D Output shapes:")
    print(f"  class_logits: {output_2d['class_logits'].shape}")
    print(f"  bbox_deltas: {output_2d['bbox_deltas'].shape}")
    print(f"  segmentation: {output_2d['segmentation'].shape}")
    print(f"  detections: {output_2d['detections'].shape}")
    
    model_3d = RetinaUNetLayer(in_channels=1, num_classes=2, dim=3)
    x_3d = torch.randn(2, 1, 128, 128, 64)
    output_3d = model_3d(x_3d)
    print("\n3D Output shapes:")
    print(f"  class_logits: {output_3d['class_logits'].shape}")
    print(f"  bbox_deltas: {output_3d['bbox_deltas'].shape}")
    print(f"  segmentation: {output_3d['segmentation'].shape}")
    print(f"  detections: {output_3d['detections'].shape}")
    
    # Test the BaseModel wrapper with a sample dataset
    print("\n" + "="*50)
    print("Testing RetinaUNet (PyHealth BaseModel wrapper):")
    from pyhealth.datasets import SampleDataset
    
    # Create a minimal sample dataset
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "image": torch.randn(1, 64, 64),  # Single channel 64x64 image
            "seg": torch.randint(0, 2, (1, 64, 64)),  # Binary segmentation mask
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "image": torch.randn(1, 64, 64),
            "seg": torch.randint(0, 2, (1, 64, 64)),
        },
    ]
    
    dataset = SampleDataset(samples=samples)
    model = RetinaUNet(dataset=dataset, in_channels=1, num_classes=2, dim=2)
    
    print(f"Model created successfully on device: {model.device}")
    print(f"Feature key: {model.feature_key}")
    print(f"Segmentation label key: {model.seg_label_key}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")