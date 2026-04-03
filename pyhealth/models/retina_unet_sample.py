"""
Clean Retina U-Net Implementation for Medical Image Object Detection

Reference: Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision 
for Medical Object Detection (https://arxiv.org/abs/1811.08661)

This implementation uses only standard PyTorch without custom CUDA operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np



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
        layers.append(
            Conv(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        )
        
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
        self.shortcut =None

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
        
        c5_out_channels = base_channels * 2
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
        self.lateral_c5 = ConvBlock(base_channels * 32, base_channels * 8, 1)
        self.lateral_c4 = ConvBlock(base_channels * 16, base_channels * 8, 1)
        self.lateral_c3 = ConvBlock(base_channels * 8, base_channels * 8, 1)
        self.lateral_c2 = ConvBlock(base_channels * 4, base_channels * 8, 1)
        
        self.smooth_p5 = ConvBlock(base_channels * 8, base_channels * 8, 3, 1, 1, norm_type, activation, dim)
        self.smooth_p4 = ConvBlock(base_channels * 8, base_channels * 8, 3, 1, 1, norm_type, activation, dim)
        self.smooth_p3 = ConvBlock(base_channels * 8, base_channels * 8, 3, 1, 1, norm_type, activation, dim)
        self.smooth_p2 = ConvBlock(base_channels * 8, base_channels * 8, 3, 1, 1, norm_type, activation, dim)
        
        self.lateral_c1 = ConvBlock(base_channels, base_channels * 8, 1)
        self.smooth_p1 = ConvBlock(base_channels * 8, base_channels * 8, 3, 1, 1, norm_type, activation, dim)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, pool=False):
        """Build residual layer."""
        layers = []

        # For the first block
        if pool:
            MaxPool = nn.MaxPool2d if self.dim == 2 else nn.MaxPool3d
            stride = 2 if self.dim == 2 else (2, 2, 1)
            
            layers.append(MaxPool(kernel_size=3, stride=stride, padding=1))
   
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
        Returns: [p2, p3, p4, p5] or [p1, p2, p3, p4, p5] if add_stride1=True
        """
        # Stem
        if self.add_stride1:
            c1 = self.conv0(x)
            x = self.pool0(c1)
        else:
            x = self.pool0(x)
        
        # Backbone
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN top-down path
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = self.smooth_p4(p4)
        
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = self.smooth_p3(p3)
        
        p2 = self.lateral_c2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p2 = self.smooth_p2(p2)
        
        p5 = self.smooth_p5(p5)
        
        if self.add_stride1:
            p1 = self.lateral_c1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
            p1 = self.smooth_p1(p1)
            return [p1, p2, p3, p4, p5]
        
        return [p2, p3, p4, p5]


class AnchorGenerator(nn.Module):
    """Generates anchor boxes for object detection."""
    
    def __init__(
        self,
        sizes: List[float] = None,
        ratios: List[float] = None,
        dim: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.sizes = sizes or [32, 64, 128, 256]  # Per pyramid level
        self.ratios = ratios or [0.5, 1, 2]
    
    def generate_anchors(
        self,
        feature_height: int,
        feature_width: int,
        stride: int,
        feature_depth: Optional[int] = None
    ) -> torch.Tensor:
        """Generate anchor boxes for a feature map."""
        anchors = []
        
        if self.dim == 2:
            for y in range(feature_height):
                for x in range(feature_width):
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    
                    for size in self.sizes:
                        for ratio in self.ratios:
                            w = size * np.sqrt(ratio)
                            h = size / np.sqrt(ratio)
                            
                            anchors.append([cy - h/2, cx - w/2, cy + h/2, cx + w/2])
        else:
            # 3D anchors
            for z in range(feature_depth):
                for y in range(feature_height):
                    for x in range(feature_width):
                        cx = (x + 0.5) * stride
                        cy = (y + 0.5) * stride
                        cz = (z + 0.5) * stride
                        
                        for size in self.sizes:
                            for ratio in self.ratios:
                                w = size * np.sqrt(ratio)
                                h = size / np.sqrt(ratio)
                                d = size
                                
                                anchors.append([cy - h/2, cx - w/2, cy + h/2, cx + w/2, 
                                              cz - d/2, cz + d/2])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, feature_maps: List[torch.Tensor], strides: List[int]) -> torch.Tensor:
        """Generate all anchors from feature maps."""
        all_anchors = []
        
        for fm, stride, size in zip(feature_maps, strides, self.sizes):
            if self.dim == 2:
                _, _, height, width = fm.shape
                self.sizes = [size]  # One size per pyramid level
                anchors = self.generate_anchors(height, width, stride)
            else:
                _, _, height, width, depth = fm.shape
                self.sizes = [size]
                anchors = self.generate_anchors(height, width, stride, depth)
            
            all_anchors.append(anchors)
        
        return torch.cat(all_anchors, dim=0)


class ClassificationHead(nn.Module):
    """Classification head for detecting object presence."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        hidden_channels: int = 256,
        norm_type: str = 'batch',
        activation: str = 'relu',
        dim: int = 2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.dim = dim
        
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.conv3 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.conv4 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.final = Conv(hidden_channels, num_anchors * num_classes, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class predictions."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        
        # Reshape to (B, -1, num_classes)
        batch_size = x.size(0)
        if self.dim == 2:
            x = x.permute(0, 2, 3, 1).contiguous()
        else:
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        x = x.view(batch_size, -1, self.num_classes)
        return x


class BBoxHead(nn.Module):
    """Bounding box regression head."""
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 9,
        hidden_channels: int = 256,
        norm_type: str = 'batch',
        activation: str = 'relu',
        dim: int = 2
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.dim = dim
        output_channels = num_anchors * (dim * 2)
        
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.conv3 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.conv4 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.final = Conv(hidden_channels, output_channels, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning bbox deltas."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        
        # Reshape to (B, -1, dim*2)
        batch_size = x.size(0)
        if self.dim == 2:
            x = x.permute(0, 2, 3, 1).contiguous()
        else:
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        x = x.view(batch_size, -1, self.dim * 2)
        return x


class SegmentationHead(nn.Module):
    """U-Net style decoder for segmentation."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        hidden_channels: int = 256,
        norm_type: str = 'batch',
        activation: str = 'relu',
        dim: int = 2
    ):
        super().__init__()
        self.dim = dim
        
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        
        # Decoder
        self.up_conv1 = ConvBlock(in_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.up_conv2 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.up_conv3 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        self.up_conv4 = ConvBlock(hidden_channels, hidden_channels, 3, 1, 1, norm_type, activation, dim)
        
        self.final = Conv(hidden_channels, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning segmentation mask."""
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = self.up_conv4(x)
        x = self.final(x)
        return torch.sigmoid(x)


class RetinaUNet(nn.Module):
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
        num_anchors: int = 9,
        base_channels: int = 48,
        norm_type: str = None,
        activation: str = 'relu',
        dim: int = 2,
        add_stride1: bool = True,
        num_blocks: List[int] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.dim = dim
        
        # Backbone
        self.fpn = FPN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks or ([3, 4, 6, 3] if dim == 2 else [3, 4, 6, 3]),
            norm_type=norm_type,
            activation=activation,
            dim=dim,
            add_stride1=add_stride1
        )
        
        # Heads
        fpn_out_channels = base_channels * 8
        self.classification_head = ClassificationHead(
            fpn_out_channels, num_classes, num_anchors, 256, norm_type, activation, dim
        )
        self.bbox_head = BBoxHead(
            fpn_out_channels, num_anchors, 256, norm_type, activation, dim
        )
        self.segmentation_head = SegmentationHead(
            fpn_out_channels, 1, 256, norm_type, activation, dim
        )
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(dim=dim)
    
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
                - 'features': List of feature maps from FPN
        """
        # Backbone
        features = self.fpn(x)  # List of multi-scale features
        
        # Heads on lowest resolution (highest semantic)
        class_logits = self.classification_head(features[-1])
        bbox_deltas = self.bbox_head(features[-1])
        
        # Segmentation on higher resolution features
        seg_features = features[0]  # Highest resolution
        segmentation = self.segmentation_head(seg_features)
        
        return {
            'class_logits': class_logits,
            'bbox_deltas': bbox_deltas,
            'segmentation': segmentation,
            'features': features
        }


if __name__ == '__main__':
    # Test 2D
    model_2d = RetinaUNet(in_channels=1, num_classes=2, dim=2)
    x_2d = torch.randn(2, 1, 256, 256)
    output_2d = model_2d(x_2d)
    print("2D Output shapes:")
    print(f"  class_logits: {output_2d['class_logits'].shape}")
    print(f"  bbox_deltas: {output_2d['bbox_deltas'].shape}")
    print(f"  segmentation: {output_2d['segmentation'].shape}")
    
    # Test 3D
    model_3d = RetinaUNet(in_channels=1, num_classes=2, dim=3)
    x_3d = torch.randn(2, 1, 128, 128, 64)
    output_3d = model_3d(x_3d)
    print("\n3D Output shapes:")
    print(f"  class_logits: {output_3d['class_logits'].shape}")
    print(f"  bbox_deltas: {output_3d['bbox_deltas'].shape}")
    print(f"  segmentation: {output_3d['segmentation'].shape}")
