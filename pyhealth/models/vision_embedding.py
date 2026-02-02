# Author: Josh Steier
# Description: Vision embedding model for medical imaging

from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import shutil
from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import ImageProcessor


class Permute(nn.Module):
    """Utility module to permute tensor dimensions in nn.Sequential.

    Args:
        dims: Variable number of integers specifying the desired ordering
            of dimensions.

    Example:
        >>> permute = Permute(0, 2, 1)
        >>> x = torch.randn(32, 256, 49)  # (B, E, spatial)
        >>> out = permute(x)  # (B, spatial, E)
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings using ViT-style projection.

    Splits an image into non-overlapping patches and projects each patch
    to an embedding vector using a convolutional layer.

    Args:
        image_size: Input image size (assumes square images).
        patch_size: Size of each square patch.
        in_channels: Number of input channels.
        embedding_dim: Output embedding dimension for each patch.

    Example:
        >>> patch_embed = PatchEmbedding(224, 16, 3, 256)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> patches = patch_embed(x)  # (4, 196, 256)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, E, H/P, W/P) -> (B, num_patches, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionEmbeddingModel(BaseModel):
    """Vision embedding model for medical image inputs.

    Converts medical images to sequences of patch embeddings suitable for
    attention-based fusion with other modalities (EHR, text).

    Supports multiple backbone types:
        - "patch": ViT-style patch projection (lightweight)
        - "cnn": Small CNN encoder (good inductive bias)
        - "resnet18"/"resnet50": Pretrained backbones

    Output shape: (batch, num_patches, embedding_dim)

    Args:
        dataset: SampleDataset with ImageProcessor fields.
        embedding_dim: Output embedding dimension. Default 128.
        patch_size: Patch size for "patch" backbone. Default 16.
        backbone: One of "patch", "cnn", "resnet18", "resnet50".
        pretrained: Use ImageNet weights for ResNet. Default True.
        freeze_backbone: Freeze pretrained weights. Default False.
        dropout: Dropout rate. Default 0.0.
        use_cls_token: Prepend learnable [CLS] token. Default False.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> model = VisionEmbeddingModel(dataset, embedding_dim=256)
        >>> embeddings = model({"chest_xray": images})
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        patch_size: int = 16,
        backbone: Literal["patch", "cnn", "resnet18", "resnet50"] = "patch",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.backbone_type = backbone
        self.use_cls_token = use_cls_token

        self.embedding_layers = nn.ModuleDict()
        self.pos_embeddings = nn.ParameterDict()
        self.cls_tokens = nn.ParameterDict() if use_cls_token else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._field_info: Dict[str, Dict[str, Any]] = {}

        for field_name, processor in self.dataset.input_processors.items():
            if not isinstance(processor, ImageProcessor):
                continue

            image_size = processor.image_size
            in_channels = self._infer_channels(processor)

            num_patches = self._build_embedding_layer(
                field_name, image_size, in_channels, backbone, pretrained, freeze_backbone
            )

            num_positions = num_patches + 1 if use_cls_token else num_patches
            self.pos_embeddings[field_name] = nn.Parameter(
                torch.randn(1, num_positions, embedding_dim) * 0.02
            )

            if use_cls_token:
                self.cls_tokens[field_name] = nn.Parameter(
                    torch.randn(1, 1, embedding_dim) * 0.02
                )

            self._field_info[field_name] = {
                "num_patches": num_patches,
                "image_size": image_size,
                "in_channels": in_channels,
            }

    def _infer_channels(self, processor: ImageProcessor) -> int:
        """Infer number of input channels from processor mode."""
        mode = getattr(processor, "mode", None)
        if mode == "L":
            return 1
        elif mode == "RGBA":
            return 4
        return 3

    def _build_embedding_layer(
        self,
        field_name: str,
        image_size: int,
        in_channels: int,
        backbone: str,
        pretrained: bool,
        freeze_backbone: bool,
    ) -> int:
        """Build embedding layer and return number of output patches."""
        if backbone == "patch":
            num_patches = (image_size // self.patch_size) ** 2
            self.embedding_layers[field_name] = PatchEmbedding(
                image_size, self.patch_size, in_channels, self.embedding_dim
            )

        elif backbone == "cnn":
            num_patches = 7 * 7
            self.embedding_layers[field_name] = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.embedding_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(2),
                Permute(0, 2, 1),
            )

        elif backbone in ("resnet18", "resnet50"):
            num_patches = 7 * 7
            self.embedding_layers[field_name] = self._build_resnet_backbone(
                backbone, in_channels, pretrained, freeze_backbone
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        return num_patches

    def _build_resnet_backbone(
        self, backbone: str, in_channels: int, pretrained: bool, freeze: bool
    ) -> nn.Module:
        """Build pretrained ResNet backbone with spatial output."""
        try:
            import torchvision.models as models
        except ImportError as e:
            raise ImportError("torchvision required for ResNet backbones") from e

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            feature_dim = 512
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            feature_dim = 2048

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        layers = list(resnet.children())[:-2]
        backbone_net = nn.Sequential(*layers)

        if freeze:
            for param in backbone_net.parameters():
                param.requires_grad = False

        return nn.Sequential(
            backbone_net,
            nn.Conv2d(feature_dim, self.embedding_dim, kernel_size=1),
            nn.Flatten(2),
            Permute(0, 2, 1),
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        output_mask: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            inputs: Dict mapping field names to image tensors (B, C, H, W).
            output_mask: If True, also return attention masks.

        Returns:
            Dict of embeddings (B, num_patches, E), optionally with masks.
        """
        embedded: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {} if output_mask else None

        for field_name, tensor in inputs.items():
            if field_name not in self.embedding_layers:
                embedded[field_name] = tensor
                continue

            tensor = tensor.to(self.device)
            batch_size = tensor.size(0)

            x = self.embedding_layers[field_name](tensor)

            if self.use_cls_token:
                cls = self.cls_tokens[field_name].expand(batch_size, -1, -1)
                x = torch.cat([cls, x], dim=1)

            x = x + self.pos_embeddings[field_name]
            x = self.dropout(x)

            embedded[field_name] = x

            if output_mask:
                masks[field_name] = torch.ones(
                    batch_size, x.size(1), dtype=torch.bool, device=x.device
                )

        return (embedded, masks) if output_mask else embedded

    def get_output_info(self, field_name: str) -> Dict[str, Any]:
        """Get metadata about embedding output for a field."""
        if field_name not in self._field_info:
            raise KeyError(f"Field '{field_name}' not found")

        info = self._field_info[field_name].copy()
        info["embedding_dim"] = self.embedding_dim
        info["has_cls_token"] = self.use_cls_token
        info["num_tokens"] = info["num_patches"] + (1 if self.use_cls_token else 0)
        return info

    def __repr__(self) -> str:
        fields = list(self.embedding_layers.keys())
        return (
            f"VisionEmbeddingModel(backbone={self.backbone_type!r}, "
            f"embedding_dim={self.embedding_dim}, fields={fields})"
        )


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset
    from pyhealth.datasets.utils import get_dataloader
    import tempfile
    import os
    from PIL import Image
    import numpy as np

    # Create synthetic images
    temp_dir = tempfile.mkdtemp()
    samples = []
    for i in range(10):
        img_path = os.path.join(temp_dir, f"img_{i}.png")
        img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L")
        img.save(img_path)
        samples.append({
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "chest_xray": img_path,
            "label": i % 2,
        })

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"chest_xray": ("image", {"image_size": 224, "mode": "L"})},
        output_schema={"label": "binary"},
        dataset_name="test_vision",
    )

    model = VisionEmbeddingModel(
        dataset=dataset,
        embedding_dim=128,
        backbone="cnn",
        use_cls_token=True,
    )

    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    embeddings = model({"chest_xray": batch["chest_xray"]})
    print(f"Input shape: {batch['chest_xray'].shape}")
    print(f"Output shape: {embeddings['chest_xray'].shape}")
    print(f"Output info: {model.get_output_info('chest_xray')}")

    # Cleanup
   
    shutil.rmtree(temp_dir)