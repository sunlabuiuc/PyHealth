from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class AlzheimerCNN(BaseModel):
    """AlzheimerCNN model

    This model uses a CNN architecture to classify MRI brain scans for Alzheimers Disease

    The model expects one input:
        - 2D MRI Brainscan: shape (batch, width, height)

    Args:
        dataset: the dataset to train the model
        init_channels: inital # of channels. Default is 32
        classifier_hidden_dim: # of hidden dimension. Default is 128
        dropout: the dropout rate. Default is 0.5
    """

    def __init__(
        self,
        dataset: SampleDataset,
        init_channels: int = 32,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super(AlzheimerCNN, self).__init__(dataset=dataset)

        self.init_channels = init_channels
        self.classifier_hidden_dim = classifier_hidden_dim

        # CNN Blocks
        # 2D convolution, 2D Instance Normalization, LeakyReLU activation function, 2D Max Pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(init_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                init_channels, init_channels * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(init_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                init_channels * 2, init_channels * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(init_channels * 4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Uses Average Pooling rather than MaxPool
        self.block4 = nn.Sequential(
            nn.Conv2d(
                init_channels * 4, init_channels * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(init_channels * 8),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # ── Classifier: FC(8C → hidden_dim) → LeakyReLU → Dropout → FC(hidden_dim → output) ─
        output_size = self.get_output_size()
        # Classifier Layer using:
        # Linear layer w/ our input hidden_dim
        # LeakyReLU activation function
        # Dropout using input arg
        # Final linear layer determing output (size 4)
        self.classifier = nn.Sequential(
            nn.Linear(init_channels * 8, classifier_hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, output_size),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # Extract inputs
        x = kwargs[self.feature_keys[0]].to(self.device)
        labels = kwargs[self.label_keys[0]].to(self.device)

        # CNN Architecture
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten: (B, 8C, 1, 1) → (B, 8C)
        # Input:  (Batch Size, 8 * init_channels, 1, 1)
        # Output: (Batch Size, 8 * init_channels)
        emb = torch.flatten(x, 1)

        # Classifier Layer
        logits = self.classifier(emb)

        # Loss + metrics via BaseModel helpers
        loss = self.get_loss_function()(logits, labels)
        y_prob = self.prepare_y_prob(logits)

        # Return everything required by BaseModel
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "logit": logits,
        }


# Ablation Number One where we use group normalization
class AlzheimerCNNNormVariant(BaseModel):

    def __init__(
        self,
        dataset,
        norm_type: str = "instance",
        num_groups: int = 8,
        init_channels: int = 32,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super(AlzheimerCNNNormVariant, self).__init__(dataset=dataset)

        if norm_type not in ("instance", "group", "layer"):
            raise ValueError(
                f"norm_type must be 'instance', 'group', or 'layer', got '{norm_type}'"
            )

        self.norm_type = norm_type
        self.num_groups = num_groups
        self.init_channels = init_channels

        def _get_norm(channels: int) -> nn.Module:
            """Build the appropriate normalization layer for a given channel count.

            Args:
                channels: Number of feature map channels to normalise.

            Returns:
                An nn.Module normalisation layer.
            """
            if norm_type == "instance":
                return nn.InstanceNorm2d(channels)
            elif norm_type == "group":
                # Ensure num_groups evenly divides channels
                groups = min(num_groups, channels)
                while channels % groups != 0:
                    groups //= 2
                return nn.GroupNorm(groups, channels)
            else:
                # layer norm: GroupNorm(1, C) is equivalent to
                # LayerNorm over the channel dimension
                return nn.GroupNorm(1, channels)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, stride=1, padding=1),
            _get_norm(init_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                init_channels, init_channels * 2, kernel_size=3, stride=1, padding=1
            ),
            _get_norm(init_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                init_channels * 2, init_channels * 4, kernel_size=3, stride=1, padding=1
            ),
            _get_norm(init_channels * 4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(
                init_channels * 4, init_channels * 8, kernel_size=3, stride=1, padding=1
            ),
            _get_norm(init_channels * 8),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        output_size = self.get_output_size()
        self.classifier = nn.Sequential(
            nn.Linear(init_channels * 8, classifier_hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, output_size),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # Input and Device Management
        x = kwargs[self.feature_keys[0]].to(self.device)
        labels = kwargs[self.label_keys[0]].to(self.device)

        # Convolutions
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten
        emb = torch.flatten(x, 1)

        logits = self.classifier(emb)

        # Metrics
        loss = self.get_loss_function()(logits, labels)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "logit": logits,
        }


# Ablation 2, where we use a vision transformer to try and capture global traits as well
class PatchEmbedding(nn.Module):
    """Converts a CNN feature map into a sequence of patch tokens.

    Splits a (B, C, H, W) feature map into non-overlapping patches of size
    patch_size x patch_size, projects each flattened patch to embed_dim via
    a linear layer, prepends a learnable (CLS) token, and adds learnable
    positional embeddings.

    Args:
        in_channels: Number of channels in the input feature map.
        patch_size: Height and width of each square patch.
        embed_dim: Dimensionality of the projected patch tokens.
        feature_map_size: Spatial height (and width) of the input feature map.
            Assumed to be square.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        feature_map_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        num_patches = (feature_map_size // patch_size) ** 2

        # Each patch is flattened to (C * patch_size * patch_size) dims
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)

        # Learnable (CLS) token: one per batch item
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings: one per patch + one for (CLS)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize a CNN feature map into a patch sequence with CLS token.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Token sequence of shape (B, num_patches + 1, embed_dim).
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # (B, C, H, W) → (B, C, H//p, W//p, p, p)
        x = x.unfold(2, p, p).unfold(3, p, p)
        # (B, C, num_patches, p*p)
        x = x.contiguous().view(B, C, -1, p * p)
        # (B, num_patches, C, p*p)
        x = x.permute(0, 2, 1, 3)
        # (B, num_patches, C*p*p)
        x = x.contiguous().view(B, -1, C * p * p)

        # Project each patch to embed_dim
        x = self.projection(x)  # (B, num_patches, embed_dim)

        # Prepend (CLS) token and add positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embedding

        return x


class AlzheimerCNNViT(BaseModel):
    """CNN + Vision Transformer hybrid for Alzheimer's disease classification.

    Combines a two-block CNN backbone for local feature extraction with a
    Transformer encoder for global context modelling over patch tokens. The
    CNN reduces the 128x128 input to a 32x32 feature map, which is then split
    into 4x4 patches and fed to the Transformer. Classification is performed
    on the output [CLS] token.

    Architecture::

        Input (B, 1, 128, 128)
        block1: Conv2d, InstanceNorm2d, LeakyReLU, MaxPool: Size (B, C, 64, 64)
        block2: Conv2d, InstanceNorm2d, LeakyReLU, MaxPool: Size  (B, 2C, 32, 32)
        PatchEmbedding (patch_size=4) -> (B, 65, embed_dim)
        TransformerEncoder (num_transformer_layers)
        (CLS) token, Linear, LeakyReLU, Dropout, Linear, logits

    Args:
        dataset: The dataset to train the model. Must expose input_schema,
            output_schema, and output_processors to satisfy BaseModel.
        init_channels: Base number of CNN filters. Blocks use C and 2C
            channels. Default is 32.
        embed_dim: Dimensionality of Transformer patch token embeddings.
            Default is 128.
        num_heads: Number of attention heads in the Transformer encoder.
            Must evenly divide embed_dim. Default is 4.
        num_transformer_layers: Number of Transformer encoder layers.
            Default is 4.
        classifier_hidden_dim: Number of units in the hidden FC layer of
            the classifier head. Default is 128.
        dropout: Dropout probability applied in the Transformer and before
            the output linear layer. Default is 0.1.
    """

    def __init__(
        self,
        dataset,
        init_channels: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 4,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super(AlzheimerCNNViT, self).__init__(dataset=dataset)

        self.init_channels = init_channels
        self.embed_dim = embed_dim

        # CNN Layer: local feature extraction
        # Two pooling stages reduce 128×128 -> 32×32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(init_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 → 64
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                init_channels, init_channels * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(init_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 → 32
        )

        # Patch embedding: 32×32 feature map -> 64 patch tokens + CLS
        # patch_size=4 gives (32/4)^2 = 64 patches
        self.patch_embed = PatchEmbedding(
            in_channels=init_channels * 2,
            patch_size=4,
            embed_dim=embed_dim,
            feature_map_size=32,
        )

        # Transformer encoder: global context over patch sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Classifier head: operates on the (CLS) token output
        output_size = self.get_output_size()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(classifier_hidden_dim, output_size),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Keys match the dataset's input_schema and output_schema.
                Expects an image tensor of shape (B, 1, H, W) and an integer
                label per sample.

        Returns:
            A dict with keys:
                - loss: scalar cross-entropy loss tensor.
                - y_prob: predicted probabilities, shape (B, num_classes).
                - y_true: ground-truth label tensor.
                - logit: raw logits, shape (B, num_classes).
        """
        # Extract inputs
        x = kwargs[self.feature_keys[0]].to(self.device)
        labels = kwargs[self.label_keys[0]].to(self.device)

        # CNN local feature extraction
        x = self.block1(x)  # (B, C, 64, 64)
        x = self.block2(x)  # (B, 2C, 32, 32)

        # Patch tokenization
        x = self.patch_embed(x)  # (B, 65, embed_dim)

        # Transformer global context modelling
        x = self.transformer(x)  # (B, 65, embed_dim)

        # Extract (CLS) token (index 0) for classification
        cls_output = x[:, 0, :]  # (B, embed_dim)

        # Classifier Layer
        logits = self.classifier(cls_output)  # (B, num_classes)

        loss = self.get_loss_function()(logits, labels)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "logit": logits,
        }
