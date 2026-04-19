# Author: Rahul Chakraborty
# Description: Wav2Sleep multimodal sleep stage classification model for PyHealth
#
# Paper-faithful implementation of:
#   - Transformer-based fusion for multimodal aggregation
#   - Dilated CNN sequence mixer for temporal modeling
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         INTEGRATION HOOKS                                  ║
# ╠═══════════════════════════════════════════════════════════════════════════╣
# ║  DHRUV'S HOOK (Modality Encoders):  Lines 486-491, 580-588                ║
# ║  NAFIS'S HOOK (Fusion Module):      Lines 493-506, 593-607                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel

# =============================================================================
# Modality Encoders
# =============================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        input_num_channels: int,
        output_num_channels: int,
        activation: str = 'leaky',
        norm: str = 'batch',
        dropout: float = 0.0,
        causal: bool = False,
        eps: float | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.layer1 = ConvLayer(
            input_channels=input_num_channels, 
            output_channels=output_num_channels, 
            activation=activation,
            norm=norm,
            dropout=dropout,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=causal,
            eps=eps,
        )

        self.layer2 = ConvLayer(
            input_channels=output_num_channels, 
            output_channels=output_num_channels, 
            activation=activation,
            norm=norm,
            dropout=dropout,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=causal,
            eps=eps,
        )

        self.layer3 = ConvLayer(
            input_channels=output_num_channels, 
            output_channels=output_num_channels, 
            activation=activation,
            norm=norm,
            dropout=dropout,
            kernel_size=3,
            stride=2,
            padding=1,
            causal=causal,
            eps=eps,
        )        

        self.activation = get_activation(activation)

        if self.residual:
            self.down = nn.Conv1d(input_num_channels, output_num_channels, kernel_size=1, stride=2, padding=0, bias=False)
        else:
            self.register_parameter('down', None)
        
    def forward(self, x: Tensor) -> Tensor:
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)

        if self.residual:
            output = output + self.down(x)
        output = self.activation(output)
        return output

class ConvLayer(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        activation: str = 'relu',
        norm: str | None = 'batch',
        eps: float | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        causal: bool = False,
        groups: int = 1,
        bias: bool = False,
    ) -> None:

        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout
        self.causal = causal
        self.groups = groups
        self.bias = bias

        if causal:
            self.padding = (self.kernel_size - 1) * self.dilation
        
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=bias or norm is None,
            dilation=dilation,
        )

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)

        if norm == 'weight':
            self.norm = nn.Identity()
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        else:
            norm_info = {}
            if eps is not None:
                norm_info = {'norm_eps': eps}
            self.norm = get_norm(norm, num_features=output_channels, causal=causal, **norm_info)


    def forward(self, x: Tensor) -> Tensor:
        output = self.conv(x)

        if self.causal and self.padding > 0:
            if isinstance(self.conv.stride, tuple):
                stride = self.conv.stride[0]
            else:
                stride = self.conv.stride
            trim = max(self.padding - stride + 1, 0)
            if trim > 0:
                output = output[:, :, :-trim]
        
        output = self.norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output
        

            




class SignalEncoder(nn.Module):

    def __init__(
        self,
        input_num_channels: int = 1,
        epoch_embedding_dim: int = 256,
        activation: str = 'gelu',
        samples_per_epoch: int = 1024,
        norm: str = 'instance',
        init_feature_channels: int = 16,
        max_channels: int = 128,
        output_norm: bool = False,
        residual: bool = True,
        causal: bool = False,
        chunk_causal: bool = True,
    ):
        super().__init__()
        self.input_num_channels = input_num_channels
        self.epoch_embedding_dim = epoch_embedding_dim
        self.activation = activation
        self.samples_per_epoch = samples_per_epoch
        self.norm = norm
        self.init_feature_channels = init_feature_channels
        self.max_channels = max_channels
        self.output_norm = output_norm
        self.residual = residual
        self.causal = causal
        self.chunk_causal = chunk_causal
        self.activation = get_activation(activation)

        blocks = []

        if samples_per_epoch & (samples_per_epoch - 1) != 0:
            raise ValueError("samples_per_epoch must be even")
        num_conv_blocks = int(math.log2(samples_per_epoch)) - 2
        num_channels_per_block = [min(init_feature_channels * 2**(i//2), max_channels) for i in range(num_conv_blocks)]
        self.epoch_size = num_channels_per_block[-1] * 4

        in_ch = input_num_channels
        for i, dim in enumerate(num_channels_per_block):
            if norm == "auto":
                norm_type = "instance" if i < 2 else "layer"
            else:
                norm_type = norm
            eps = 1e-2 if norm_type == "instance" else None
            blocks.append(ConvBlock(input_num_channels=in_ch, output_num_channels=dim, activation=activation, norm=norm_type, eps=eps, causal=(causal and not chunk_causal), residual=residual))
            in_ch = dim

        self.cnn = nn.Sequential(*blocks)
        self.linear = nn.Linear(self.epoch_size, epoch_embedding_dim)
        self.output_norm = nn.LayerNorm(epoch_embedding_dim) if output_norm else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) % self.samples_per_epoch:
            raise ValueError(f"Input length must be a multiple of {self.samples_per_epoch}")
        
        batch = x.size(0)
        epochs = x.size(-1) // self.samples_per_epoch #number of epochs in the batch

        if self.causal and self.chunk_causal: #CNN runs separately on each epoch's S samples - no overlap between epochs
            y = x.view(batch, epochs, self.samples_per_epoch) #split the batch into epochs
            y = y.reshape(batch * epochs, 1, self.samples_per_epoch) #reshape the batch into [B*E, 1, S]
            y = self.cnn(y) #run the CNN on the batch
            y = y.transpose(-1,-2).reshape(batch, epochs, self.epoch_size) #reshape the batch back into [B, E, epoch_size]
        else:   #CNN runs on the entire batch [B, 1, T*S] at once - overlap between epochs
            y = x.unsqueeze(1)
            y = self.cnn(y)
            y = y.transpose(-1, -2).reshape(batch, -1, self.epoch_size)

        y = self.output_norm(self.activation(self.linear(y)))
        return y


    
SIGNAL_TO_SAMPLES_PER_EPOCH = {
    'ABD': 256,
    'THX': 256,
    'ECG': 1024,
    'PPG': 1024,
    'EOG_L': 4096,
    'EOG_R': 4096,
}

class SignalEncoders(nn.Module):

    def __init__(
        self,
        signal_encoder_map: dict[str, str],
        feature_dim: int,
        activation: str,
        norm: str = 'instance',
        include_signal: bool = False,
        init_feature_channels: int = 16,
        max_channels: int = 128,
        output_norm: bool = False,
        residual: bool = True,
        causal: bool = False,
        chunk_causal: bool = True,
    ) -> None:
        super().__init__()
        self.signal_encoder_map = signal_encoder_map
        self.feature_dim = feature_dim
        self.activation = activation
        self.norm = norm
        self.include_signal = include_signal
        self.init_feature_channels = init_feature_channels
        self.max_channels = max_channels
        self.output_norm = output_norm
        self.residual = residual
        self.causal = causal
        self.chunk_causal = chunk_causal

        signal_encoders = {}

        for signal, encoder in signal_encoder_map.items():
            if encoder in signal_encoders:
                continue
            if signal not in SIGNAL_TO_SAMPLES_PER_EPOCH:
                raise ValueError(f"Signal {signal} not found in SIGNAL_TO_SAMPLES_PER_EPOCH")
            samples_per_epoch = SIGNAL_TO_SAMPLES_PER_EPOCH[signal]

            signal_encoders[encoder] = SignalEncoder(
                input_num_channels=1,
                epoch_embedding_dim=feature_dim,
                activation=activation,
                samples_per_epoch=samples_per_epoch,
                norm=norm,
                init_feature_channels=init_feature_channels,
                max_channels=max_channels,
                output_norm=output_norm,
                residual=residual,
                causal=causal,
                chunk_causal=chunk_causal,
            )

        if self.include_signal:
            self.embedding = nn.Embedding(num_embeddings=len(signal_encoder_map), embedding_dim=self.feature_dim) #embedding is a lookup table that maps a signal to a feature vector
        else:
            self.register_parameter('embedding', None)

        self.include_signal = include_signal
        self.signal_encoders = nn.ModuleDict(signal_encoders)
        self.signal_to_idx = {signal: i for i, signal in enumerate(sorted(signal_encoder_map.keys()))}


    def __len__(self) -> int:
        return len(self.signal_encoders)

    def get_signal_encoder(self, signal: str) -> 'SignalEncoder':

        if self.signal_encoder_map is not None:
            return self.signal_encoders[self.signal_encoder_map[signal]]
        else:
            return self.signal_encoders[signal]
    
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        out: dict[str, torch.Tensor] = {}

        for signal, x_signal in x.items():
            inf_batch_mask = torch.isinf(x_signal[:,0])
            x_signal = torch.where(torch.isinf(x_signal), 0.0, x_signal) #mask out inf values for batches with no signal
            out_BSF = self.get_signal_encoder(signal)(x_signal) #encode the signal with 1D CNN
            out_BSF = torch.where(inf_batch_mask[:,None,None], float('-inf'), out_BSF) #mask out inf values for batches with no signal

            if self.include_signal:
                embed = self.embedding(
                    torch.tensor([self.signal_to_idx[signal]], device=out_BSF.device, dtype=torch.int64) #embed the signal through embedding lookup table
                )
                out_BSF = out_BSF + embed.view(1,1,-1) #reshape embedding to match the shape of the encoded signal and apply activation function
            out[signal] = out_BSF
        return out

class ConvLayerNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
    
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1,keepdim=True)
        sigma = (x - mean).pow(2).mean(1,keepdim=True)
        x = (x - mean) / torch.sqrt(sigma + self.eps)
        x = self.bias + self.weight * x
        return x

class ConvRMSNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        sigma = x.pow(2).mean(1,keepdim=True)
        x = x / torch.sqrt(sigma + self.eps)
        x = self.weight * x
        return x

class ConvGroupNorm(nn.Module):

    def __init__(self, num_features: int, num_groups: int = 8, eps: float = 1e-5, channels : int | None = None):
        super().__init__()
        
        if channels is not None:
            num_groups = num_features // channels
        if num_features < num_groups:
            num_groups = num_features
        if num_features % num_groups != 0:
            raise ValueError(f"num_features must be divisible by num_groups")
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)

        
def get_activation(name: str, **kwargs):
    """Return an activation function from its name."""
    if name == 'relu':
        return nn.ReLU(**kwargs)
    elif name == 'leaky':
        return nn.LeakyReLU(**kwargs)
    elif name == 'gelu':
        return nn.GELU(**kwargs)
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(**kwargs)
    elif name == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f'{name=} is unsupported.')


def get_norm(name: str | None = 'batch', causal: bool = False, *args, **kwargs) -> nn.Module:
    # Extract norm_eps - only used by instance norm, but may be passed for any norm type
    norm_eps = kwargs.pop('norm_eps', None)

    if name == 'batch':
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == 'layer':
        return ConvLayerNorm(*args, **kwargs)
    elif name == 'rms':
        return ConvRMSNorm(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    elif name == 'instance':
        if norm_eps is not None:
            kwargs['eps'] = norm_eps
        return nn.InstanceNorm1d(*args, **kwargs)
    elif name == 'group':
        return ConvGroupNorm(*args, **kwargs)
    else:
        raise ValueError(f'Normalisation with {name=} and {causal=} unknown.')

# =============================================================================
# Transformer Fusion Module
# =============================================================================
# This file uses the reusable TransformerFusion fusion layer from
# pyhealth.models.fusion to aggregate modality embeddings.
# The transformer fusion module accepts a dict of temporally aligned
# modality tensors of shape [B, T, D] and returns a fused tensor of shape [B, T, D].



"""Transformer-based fusion module for multimodal feature aggregation."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union


class BaseFusionModule(nn.Module, ABC):
    """Abstract base class for multimodal fusion modules."""

    def __init__(self, embed_dim: int, num_modalities: int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self._config = kwargs

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return self._config

    @abstractmethod
    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse multimodal features.

        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: Optional mask indicating present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        pass

    def get_modality_mask(self, modality_features: List[Optional[Tensor]]) -> Tensor:
        """Generate modality mask from feature list."""
        device = next((f.device for f in modality_features if f is not None), None)
        batch_size = next((f.size(0) for f in modality_features if f is not None), 1)
        modality_mask = torch.tensor(
            [feat is not None for feat in modality_features],
            device=device
        ).unsqueeze(0).repeat(batch_size, 1)
        return modality_mask


class AttentionMechanism(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    @abstractmethod
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pass


class MultiHeadAttention(AttentionMechanism):
    """Standard multi-head attention implementation."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = query.size(0)

        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended)
        return output


class CrossModalAttention(BaseFusionModule):
    """Cross-modal attention mechanism for fusing features from different modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 4,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            attended_output: [batch_size, seq_len, embed_dim]
        """
        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Filter present modalities
        present_features = [f for f in modality_features if f is not None]
        if not present_features:
            raise ValueError("At least one modality must be present")

        # Concatenate present features along sequence dimension for cross-attention
        concatenated = torch.cat(present_features, dim=1)  # [batch, total_seq, embed_dim]

        # Use first present feature as query, concatenated as key/value
        query = present_features[0]

        # Create attention mask for missing modalities
        # This is a simplified version - in practice, you'd want more sophisticated masking
        seq_lengths = [f.size(1) for f in present_features]
        total_seq = sum(seq_lengths)

        # For now, use the attention mechanism directly
        output = self.attention(query, concatenated, concatenated)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        attention_type: str = 'self',
    ):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == 'self':
            self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        elif attention_type == 'cross':
            self.attention = CrossModalAttention(embed_dim, num_heads, dropout, num_modalities=1)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.attention_type == 'cross' and context is not None:
            attn_out = self.attention.attention(x, context, context, mask)
        else:
            attn_out = self.attention(x, x, x, mask)

        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerFusion(BaseFusionModule):
    """Transformer-based fusion module for multimodal features with dynamic missing modality handling."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 1000,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.register_buffer('pos_embedding', torch.randn(1, max_seq_len, embed_dim))

        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, 'self')
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def _project_modalities(self, modality_features: List[Optional[Tensor]]) -> List[Tensor]:
        """Project each modality to common embedding space."""
        projected = []
        for i, feat in enumerate(modality_features):
            if feat is not None:
                projected.append(self.modality_projections[i](feat))
            else:
                # Create zero tensor for missing modalities
                batch_size, seq_len = modality_features[0].size(0), modality_features[0].size(1)
                device = modality_features[0].device
                projected.append(torch.zeros(batch_size, seq_len, self.embed_dim, device=device))
        return projected

    def _fuse_features(
        self,
        projected_features: List[Tensor],
        modality_mask: Tensor
    ) -> Tensor:
        """Fuse projected features using weighted average."""
        # Stack features: [num_modalities, batch_size, seq_len, embed_dim]
        stacked = torch.stack(projected_features, dim=0)

        # Apply modality weights and mask
        weights = self.modality_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # [1, num_mod, 1, 1, 1]
        mask_expanded = modality_mask.t().unsqueeze(-1).unsqueeze(-1)  # [num_mod, batch, 1, 1]

        weighted_features = stacked * weights * mask_expanded.float()

        # Average across modalities, accounting for missing ones
        num_present = modality_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        fused = weighted_features.sum(dim=0) / num_present.unsqueeze(-1).unsqueeze(-1)

        return fused

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Project modalities
        projected_features = self._project_modalities(modality_features)

        # Fuse features
        fused = self._fuse_features(projected_features, modality_mask)

        batch_size, seq_len, _ = fused.size()

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            extended_pos = torch.cat([
                self.pos_embedding,
                torch.zeros(1, seq_len - self.max_seq_len, self.embed_dim, device=fused.device)
            ], dim=1)
            fused = fused + extended_pos

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, mask=None)  # Self-attention within fused features

        # Final projection
        output = self.output_projection(fused)

        return output


class FusionTransformer(nn.Module):
    """Transformer-based modality fusion (dict input, missing modality safe)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_modality_token: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_modality_token = use_modality_token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_modality_token:
            self.modal_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        modality_embeddings: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse modality embeddings and return [B, T, D].

        Args:
            modality_embeddings: dict of modality->Tensor [B, T, D]
            attention_mask: optional boolean mask [B, M] or [B, T, M], True=present

        Returns:
            Tensor [B, T, D]
        """
        if not modality_embeddings:
            raise ValueError("modality_embeddings must have at least one modality")

        # ensure deterministic key order
        modality_keys = list(modality_embeddings.keys())
        embeddings = [modality_embeddings[k] for k in modality_keys]

        B, T, D = embeddings[0].shape
        for x in embeddings:
            if x.shape != (B, T, D):
                raise ValueError("All modality tensors must share shape [B, T, D]")

        x = torch.stack(embeddings, dim=2)  # [B, T, M, D]

        if self.use_modality_token:
            modality_token = self.modal_token.expand(B, T, -1, -1)  # [B, T, 1, D]
            x = torch.cat([x, modality_token], dim=2)

        B, T, M, D = x.shape

        x = x.reshape(B * T, M, D)

        src_key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # [B, M]
                mask_2d = attention_mask
                if self.use_modality_token:
                    token_mask = torch.ones(B, 1, dtype=mask_2d.dtype, device=mask_2d.device)
                    mask_2d = torch.cat([mask_2d, token_mask], dim=1)
                src_key_padding_mask = ~mask_2d.repeat_interleave(T, dim=0)
            elif attention_mask.ndim == 3:
                # [B, T, M]
                if self.use_modality_token:
                    token_mask = torch.ones(B, T, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, token_mask], dim=2)
                src_key_padding_mask = ~(attention_mask.reshape(B * T, -1))
            else:
                raise ValueError("attention_mask must be 2D or 3D")

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        x = x.mean(dim=1)
        x = x.reshape(B, T, D)

        return x


class FusionFactory:
    """Factory class for creating fusion modules."""

    @staticmethod
    def create_fusion(
        fusion_type: str,
        embed_dim: int,
        num_modalities: int,
        **kwargs
    ) -> BaseFusionModule:
        """Create a fusion module based on type."""
        if fusion_type == 'transformer':
            return TransformerFusion(embed_dim, num_modalities, **kwargs)
        elif fusion_type == 'attention':
            return CrossModalAttention(embed_dim, num_modalities, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")


class DynamicModalityFusion(BaseFusionModule):
    """Dynamic fusion that adapts based on available modalities using factory pattern."""

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 4,
        fusion_type: str = 'transformer',
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, fusion_type=fusion_type, **kwargs)

        # Create fusion module using factory
        self.fusion_module = FusionFactory.create_fusion(
            fusion_type, embed_dim, num_modalities, **kwargs
        )

        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: Optional mask indicating present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        # Delegate to the fusion module
        return self.fusion_module(modality_features, modality_mask)
    """Cross-modal attention mechanism for fusing features from different modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 4,
    ):
        super().__init__(embed_dim, num_modalities)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Multi-head attention for cross-modal interactions
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query: [batch_size, seq_len_q, embed_dim]
            key_value: [batch_size, seq_len_kv, embed_dim]
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            attended_output: [batch_size, seq_len_q, embed_dim]
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply modality mask if provided
        if modality_mask is not None:
            # Expand mask to match attention dimensions
            # modality_mask: [batch_size, num_modalities]
            # We need to mask out attention to missing modalities
            # Assuming key_value is concatenated from modalities
            # For simplicity, if any modality is missing, we can apply a global mask
            # But for more precise control, we'd need per-modality masking
            mask_expanded = modality_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_modalities]
            # This is a simplified version; in practice, need to align with sequence positions
            attn_weights = attn_weights.masked_fill(~mask_expanded.bool(), float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended)
        return output


class TransformerFusionLayer(BaseFusionModule):
    """Single transformer layer for multimodal fusion."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
    ):
        super().__init__(embed_dim, num_modalities)
        self.cross_attn = CrossModalAttention(embed_dim, num_heads, dropout, num_modalities)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            modality_mask: [batch_size, num_modalities]

        Returns:
            fused_output: [batch_size, seq_len, embed_dim]
        """
        # Self-attention with cross-modal interaction
        attn_out = self.cross_attn(x, x, modality_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TransformerFusion(BaseFusionModule):
    """Transformer-based fusion module for multimodal features with dynamic missing modality handling."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 1000,
    ):
        super().__init__(embed_dim, num_modalities)
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerFusionLayer(embed_dim, num_heads, ff_dim, dropout, num_modalities)
            for _ in range(num_layers)
        ])

        # Modality-specific projections (optional, if features have different dims)
        self.modality_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        modality_features: List[Tensor],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors, each [batch_size, seq_len, feature_dim]
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        batch_size = modality_features[0].size(0)
        seq_len = modality_features[0].size(1)

        # Project each modality to common embedding space
        projected_features = []
        for i, feat in enumerate(modality_features):
            if feat is not None:
                proj_feat = self.modality_projs[i](feat)
                projected_features.append(proj_feat)
            else:
                # For missing modalities, use zero tensor
                zero_feat = torch.zeros(batch_size, seq_len, self.embed_dim, device=feat.device if feat is not None else None)
                projected_features.append(zero_feat)

        # Concatenate along modality dimension or average
        # For simplicity, we'll average the features, but mask out missing ones
        if modality_mask is not None:
            # Stack features: [num_modalities, batch_size, seq_len, embed_dim]
            stacked = torch.stack(projected_features, dim=0)
            # modality_mask: [batch_size, num_modalities] -> [num_modalities, batch_size, 1, 1]
            mask_expanded = modality_mask.t().unsqueeze(-1).unsqueeze(-1)
            # Mask missing modalities
            masked_features = stacked * mask_expanded.float()
            # Average across modalities, accounting for missing ones
            num_present = modality_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            fused = masked_features.sum(dim=0) / num_present.unsqueeze(-1).unsqueeze(-1)
        else:
            # Simple average if no mask
            stacked = torch.stack(projected_features, dim=0)
            fused = stacked.mean(dim=0)

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        else:
            # If sequence is longer, use learned positional encoding up to max_len
            pos_embed = torch.cat([
                self.pos_embedding,
                torch.zeros(1, seq_len - self.max_seq_len, self.embed_dim, device=fused.device)
            ], dim=1)
            fused = fused + pos_embed

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, modality_mask)

        # Final projection
        output = self.output_proj(fused)

        return output


class DynamicModalityFusion(BaseFusionModule):
    """Dynamic fusion that adapts based on available modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 4,
        fusion_type: str = 'transformer',
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities)
        self.fusion_type = fusion_type

        if fusion_type == 'transformer':
            self.fusion = TransformerFusion(embed_dim=embed_dim, num_modalities=num_modalities, **kwargs)
        elif fusion_type == 'attention':
            self.fusion = CrossModalAttention(embed_dim=embed_dim, num_modalities=num_modalities, **kwargs)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: Optional mask indicating present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        # Create mask if not provided
        if modality_mask is None:
            batch_size = modality_features[0].size(0) if modality_features[0] is not None else 1
            modality_mask = torch.tensor([
                feat is not None for feat in modality_features
            ], device=modality_features[0].device if modality_features[0] is not None else None).unsqueeze(0).repeat(batch_size, 1)

        # Filter out None features and corresponding mask
        present_features = [feat for feat in modality_features if feat is not None]
        present_mask = modality_mask

        if len(present_features) == 0:
            raise ValueError("At least one modality must be present")

        # Apply fusion
        fused = self.fusion(present_features, present_mask)

        return fused
    """Cross-modal attention mechanism for fusing features from different modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Multi-head attention for cross-modal interactions
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: [batch_size, seq_len_q, embed_dim]
            key_value: [batch_size, seq_len_kv, embed_dim]
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            attended_output: [batch_size, seq_len_q, embed_dim]
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply modality mask if provided
        if modality_mask is not None:
            # Expand mask to match attention dimensions
            # modality_mask: [batch_size, num_modalities]
            # We need to mask out attention to missing modalities
            # Assuming key_value is concatenated from modalities
            # For simplicity, if any modality is missing, we can apply a global mask
            # But for more precise control, we'd need per-modality masking
            mask_expanded = modality_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_modalities]
            # This is a simplified version; in practice, need to align with sequence positions
            attn_weights = attn_weights.masked_fill(~mask_expanded.bool(), float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended)
        return output


class TransformerFusionLayer(nn.Module):
    """Single transformer layer for multimodal fusion."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
    ):
        super().__init__()
        self.cross_attn = CrossModalAttention(embed_dim, num_heads, dropout, num_modalities)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            modality_mask: [batch_size, num_modalities]

        Returns:
            fused_output: [batch_size, seq_len, embed_dim]
        """
        # Self-attention with cross-modal interaction
        attn_out = self.cross_attn(x, x, modality_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TransformerFusion(BaseFusionModule):
    """Transformer-based fusion module for multimodal features with dynamic missing modality handling."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 1000,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerFusionLayer(embed_dim, num_heads, ff_dim, dropout, num_modalities)
            for _ in range(num_layers)
        ])

        # Modality-specific projections (optional, if features have different dims)
        self.modality_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None, each [batch_size, seq_len, feature_dim]
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        batch_size = modality_features[0].size(0)
        seq_len = modality_features[0].size(1)

        # Project each modality to common embedding space
        projected_features = []
        for i, feat in enumerate(modality_features):
            if feat is not None:
                proj_feat = self.modality_projs[i](feat)
                projected_features.append(proj_feat)
            else:
                # For missing modalities, use zero tensor
                zero_feat = torch.zeros(batch_size, seq_len, self.embed_dim, device=feat.device if feat is not None else None)
                projected_features.append(zero_feat)

        # Concatenate along modality dimension or average
        # For simplicity, we'll average the features, but mask out missing ones
        if modality_mask is not None:
            # Stack features: [num_modalities, batch_size, seq_len, embed_dim]
            stacked = torch.stack(projected_features, dim=0)
            # modality_mask: [batch_size, num_modalities] -> [num_modalities, batch_size, 1, 1]
            mask_expanded = modality_mask.t().unsqueeze(-1).unsqueeze(-1)
            # Mask missing modalities
            masked_features = stacked * mask_expanded.float()
            # Average across modalities, accounting for missing ones
            num_present = modality_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            fused = masked_features.sum(dim=0) / num_present.unsqueeze(-1).unsqueeze(-1)
        else:
            # Simple average if no mask
            stacked = torch.stack(projected_features, dim=0)
            fused = stacked.mean(dim=0)

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        else:
            # If sequence is longer, use learned positional encoding up to max_len
            pos_embed = torch.cat([
                self.pos_embedding,
                torch.zeros(1, seq_len - self.max_seq_len, self.embed_dim, device=fused.device)
            ], dim=1)
            fused = fused + pos_embed

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, modality_mask)

        # Final projection
        output = self.output_proj(fused)

        return output


class DynamicModalityFusion(nn.Module):
    """Dynamic fusion that adapts based on available modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 4,
        fusion_type: str = 'transformer',
        **kwargs
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type

        if fusion_type == 'transformer':
            self.fusion = TransformerFusion(embed_dim=embed_dim, num_modalities=num_modalities, **kwargs)
        elif fusion_type == 'attention':
            self.fusion = CrossModalAttention(embed_dim=embed_dim, num_modalities=num_modalities, **kwargs)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def forward(
        self,
        modality_features: list[Tensor | None],
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: Optional mask indicating present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        # Create mask if not provided
        if modality_mask is None:
            batch_size = modality_features[0].size(0) if modality_features[0] is not None else 1
            modality_mask = torch.tensor([
                feat is not None for feat in modality_features
            ], device=modality_features[0].device if modality_features[0] is not None else None).unsqueeze(0).repeat(batch_size, 1)

        # Filter out None features and corresponding mask
        present_features = [feat for feat in modality_features if feat is not None]
        present_mask = modality_mask

        if len(present_features) == 0:
            raise ValueError("At least one modality must be present")

        # Apply fusion
        fused = self.fusion(present_features, present_mask)
        return fused


# =============================================================================
# Dilated CNN Sequence Mixer (Paper-Faithful)
# =============================================================================

class DilatedConvBlock(nn.Module):
    """Single dilated convolution block with residual connection.
    
    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super(DilatedConvBlock, self).__init__()
        
        # Padding to preserve sequence length with dilation
        # For causal convolution: padding = (kernel_size - 1) * dilation
        # For same convolution: padding = ((kernel_size - 1) * dilation) // 2
        self.padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv1d(
            channels, 
            channels, 
            kernel_size, 
            padding=self.padding,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape [B, C, T] (channels-first)
            
        Returns:
            Output tensor of shape [B, C, T]
        """
        residual = x
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual  # Residual connection
        return out


class DilatedCNNSequenceMixer(nn.Module):
    """Dilated CNN sequence mixer for temporal modeling (paper-faithful).
    
    This module implements the dilated temporal convolution block from wav2sleep:
    - Uses stacked dilated convolutions with exponentially increasing dilation
    - Preserves sequence length throughout
    - Captures long-range sleep-stage dependencies through large receptive field
    
    Why dilated CNN instead of standard temporal CNN?
    1. **Exponentially growing receptive field**: Each layer doubles the receptive field,
       allowing the model to capture dependencies across many sleep epochs (e.g., 
       capturing full sleep cycles of 90 minutes)
    2. **Computational efficiency**: Fewer parameters than fully-connected or 
       standard convolutions with same receptive field
    3. **No pooling required**: Maintains temporal resolution while seeing long context
    4. **Sleep cycle awareness**: Dilations of [1, 2, 4, 8, 16] capture patterns from
       individual epochs to multi-hour sleep cycles
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden feature dimension  
        kernel_size: Convolution kernel size
        num_layers: Number of dilated conv layers
        dilations: List of dilation factors (default: exponential [1, 2, 4, 8, 16])
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        num_layers: int = 5,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super(DilatedCNNSequenceMixer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Default: exponentially increasing dilations
        if dilations is None:
            dilations = [2**i for i in range(num_layers)]  # [1, 2, 4, 8, 16]
        
        self.dilations = dilations
        
        # Input projection if dimensions don't match
        self.input_proj = None
        if input_dim != hidden_dim:
            self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Stacked dilated convolution blocks
        self.dilated_blocks = nn.ModuleList([
            DilatedConvBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Calculate receptive field
        self._receptive_field = self._compute_receptive_field(kernel_size, dilations)
    
    def _compute_receptive_field(self, kernel_size: int, dilations: List[int]) -> int:
        """Compute the total receptive field of the dilated CNN."""
        rf = 1
        for d in dilations:
            rf += (kernel_size - 1) * d
        return rf
    
    @property
    def receptive_field(self) -> int:
        """Return the receptive field in number of time steps."""
        return self._receptive_field
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dilated CNN sequence mixer.
        
        Args:
            x: Input tensor of shape [B, T, D]
            
        Returns:
            Output tensor of shape [B, T, hidden_dim]
        """
        # Convert to channels-first format for Conv1d: [B, D, T]
        x = x.transpose(1, 2)
        
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Apply stacked dilated convolutions
        for block in self.dilated_blocks:
            x = block(x)
        
        # Convert back to [B, T, D] format
        x = x.transpose(1, 2)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x


# =============================================================================
# Standard Temporal Conv Block (Simplified Version - for comparison)
# =============================================================================

class TemporalConvBlock(nn.Module):
    """Standard temporal convolution block (simplified, non-dilated).
    
    This is the SIMPLIFIED version that does NOT match the paper.
    Use DilatedCNNSequenceMixer for paper-faithful implementation.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden feature dimension
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super(TemporalConvBlock, self).__init__()
        
        padding = kernel_size // 2  # Same padding to preserve sequence length
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Residual connection projection if dimensions don't match
        self.residual_proj = None
        if input_dim != hidden_dim:
            self.residual_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [B, T, D]
            
        Returns:
            Output tensor of shape [B, T, hidden_dim]
        """
        # Convert to channels-first format for Conv1d: [B, D, T]
        x_transposed = x.transpose(1, 2)
        
        # Apply convolutions
        out = self.conv1(x_transposed)
        out = self.conv2(out)
        
        # Residual connection
        residual = x_transposed
        if self.residual_proj is not None:
            residual = self.residual_proj(x_transposed)
        
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        
        # Convert back to [B, T, D] format
        return out.transpose(1, 2)


class BaseFusionModule(nn.Module, ABC):
    """Abstract base class for multimodal fusion modules."""

    def __init__(self, embed_dim: int, num_modalities: int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self._config = kwargs

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return self._config

    @abstractmethod
    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse multimodal features.

        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: Optional mask indicating present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        pass

    def get_modality_mask(self, modality_features: Union[List[Optional[Tensor]], Dict[str, Optional[Tensor]]]) -> Tensor:
        """Generate modality mask from a feature list or dict."""
        if isinstance(modality_features, dict):
            modality_features = list(modality_features.values())

        device = next((f.device for f in modality_features if f is not None and hasattr(f, 'device')), None)
        batch_size = next((f.size(0) for f in modality_features if f is not None and hasattr(f, 'size')), 1)
        modality_mask = torch.tensor(
            [feat is not None and hasattr(feat, 'device') for feat in modality_features],
            device=device
        ).unsqueeze(0).repeat(batch_size, 1)
        return modality_mask


class AttentionMechanism(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    @abstractmethod
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pass


class MultiHeadAttention(AttentionMechanism):
    """Standard multi-head attention implementation."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = query.size(0)

        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended)
        return output


class CrossModalAttention(BaseFusionModule):
    """Cross-modal attention mechanism for fusing features from different modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 4,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            attended_output: [batch_size, seq_len, embed_dim]
        """
        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Filter present modalities
        present_features = [f for f in modality_features if f is not None]
        if not present_features:
            raise ValueError("At least one modality must be present")

        # Concatenate present features along sequence dimension for cross-attention
        concatenated = torch.cat(present_features, dim=1)  # [batch, total_seq, embed_dim]

        # Use first present feature as query, concatenated as key/value
        query = present_features[0]

        # Create attention mask for missing modalities
        # This is a simplified version - in practice, you'd want more sophisticated masking
        seq_lengths = [f.size(1) for f in present_features]
        total_seq = sum(seq_lengths)

        # For now, use the attention mechanism directly
        output = self.attention(query, concatenated, concatenated)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        attention_type: str = 'self',
    ):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == 'self':
            self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        elif attention_type == 'cross':
            self.attention = CrossModalAttention(embed_dim, num_heads, dropout, num_modalities=1)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.attention_type == 'cross' and context is not None:
            attn_out = self.attention.attention(x, context, context, mask)
        else:
            attn_out = self.attention(x, x, x, mask)

        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerFusion(BaseFusionModule):
    """Transformer-based fusion module for multimodal features with dynamic missing modality handling."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 1000,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.register_buffer('pos_embedding', torch.randn(1, max_seq_len, embed_dim))

        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, 'self')
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def _project_modalities(self, modality_features: List[Optional[Tensor]]) -> List[Tensor]:
        """Project each modality to common embedding space."""
        projected = []
        for i, feat in enumerate(modality_features):
            if feat is not None:
                projected.append(self.modality_projections[i](feat))
            else:
                # Create zero tensor for missing modalities
                batch_size, seq_len = modality_features[0].size(0), modality_features[0].size(1)
                device = modality_features[0].device
                projected.append(torch.zeros(batch_size, seq_len, self.embed_dim, device=device))
        return projected

    def _fuse_features(
        self,
        projected_features: List[Tensor],
        modality_mask: Tensor
    ) -> Tensor:
        """Fuse projected features using weighted average."""
        # Stack features: [num_modalities, batch_size, seq_len, embed_dim]
        stacked = torch.stack(projected_features, dim=0)

        # Apply modality weights and mask
        # weights shape: [num_mod, 1, 1, 1] to broadcast over [num_mod, batch, seq_len, embed_dim]
        weights = self.modality_weights.view(-1, 1, 1, 1)
        # mask_expanded shape: [num_mod, batch, 1, 1]
        mask_expanded = modality_mask.t().unsqueeze(-1).unsqueeze(-1)

        weighted_features = stacked * weights * mask_expanded.float()

        # Average across modalities, accounting for missing ones
        num_present = modality_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [batch, 1]
        fused = weighted_features.sum(dim=0) / num_present.unsqueeze(-1)  # [batch, seq_len, embed_dim]
        return fused

    def forward(
        self,
        modality_features: Union[List[Optional[Tensor]], Dict[str, Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None, or dict of modality->Tensor
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        if isinstance(modality_features, dict):
            modality_features = list(modality_features.values())

        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Project modalities
        projected_features = self._project_modalities(modality_features)

        # Fuse features
        fused = self._fuse_features(projected_features, modality_mask)

        batch_size, seq_len, _ = fused.size()

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            extended_pos = torch.cat([
                self.pos_embedding,
                torch.zeros(1, seq_len - self.max_seq_len, self.embed_dim, device=fused.device)
            ], dim=1)
            fused = fused + extended_pos

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, mask=None)  # Self-attention within fused features

        # Final projection
        output = self.output_projection(fused)

        return output


# =============================================================================
# Main Wav2Sleep Model
# =============================================================================

class Wav2Sleep(BaseModel):
    """Wav2Sleep multimodal sleep stage classification model.
    
    This is a PAPER-FAITHFUL implementation that includes:
    - Transformer-based fusion for multimodal aggregation
    - Dilated CNN sequence mixer for temporal modeling
    
    Architecture:
    1. Modality-specific encoders (ECG, PPG, Respiratory) -> [B, T, D] each
    2. Transformer-based fusion -> [B, T, D]
    3. Dilated CNN sequence mixer -> [B, T, D]
    4. Classification head -> [B, T, 5] (Wake, N1, N2, N3, REM)
    
    Args:
        dataset (SampleDataset): Dataset with fitted input and output processors
        embedding_dim (int): Embedding dimension for input features
        hidden_dim (int): Hidden dimension for temporal modeling
        num_classes (int): Number of sleep stages (default: 5)
        num_fusion_heads (int): Number of attention heads in fusion transformer
        num_fusion_layers (int): Number of transformer layers in fusion
        num_temporal_layers (int): Number of dilated conv layers
        temporal_kernel_size (int): Kernel size for temporal convolutions
        dilations (List[int]): Dilation factors for temporal CNN (default: [1,2,4,8,16])
        dropout (float): Dropout rate
        use_paper_faithful (bool): If True, use paper-faithful components; 
                                   if False, use simplified versions
        
    Note:
        The model expects input dictionary with keys: 'ecg', 'ppg', 'resp'
        Missing modalities are handled gracefully - at least one modality must be present.
    """
    
    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_fusion_heads: int = 4,
        num_fusion_layers: int = 2,
        num_temporal_layers: int = 5,
        temporal_kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_paper_faithful: bool = True,
    ):
        super(Wav2Sleep, self).__init__(dataset=dataset)
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_paper_faithful = use_paper_faithful
        
        # Ensure single label key
        assert len(self.label_keys) == 1, "Wav2Sleep only supports single label key"
        self.label_key = self.label_keys[0]
        
        # Expected modality keys
        self.modality_keys = ['ecg', 'ppg']
        self.modality_to_signal = {'ecg': 'ECG', 'ppg': 'PPG'}
        
        # Check which modalities are available in the dataset
        self.available_modalities = [key for key in self.modality_keys if key in self.feature_keys]
        if not self.available_modalities:
            raise ValueError("At least one modality (ecg, ppg) must be present in dataset")
        
        # Embedding model for initial feature processing
        # self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        
        # ╔═════════════════════════════════════════════════════════════════╗
        # ║  ██████╗ ██╗  ██╗██████╗ ██╗   ██╗██╗   ██╗                     ║
        # ║  ██╔══██╗██║  ██║██╔══██╗██║   ██║██║   ██║                     ║
        # ║  ██║  ██║███████║██████╔╝██║   ██║██║   ██║                     ║
        # ║  ██║  ██║██╔══██║██╔══██╗██║   ██║╚██╗ ██╔╝                     ║
        # ║  ██████╔╝██║  ██║██║  ██║╚██████╔╝ ╚████╔╝                      ║
        # ║  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═══╝                       ║
        # ║                                                                 ║
        # ║  INTEGRATION HOOK: Modality-Specific CNN Encoders               ║
        # ║                                                                 ║
        # ║  Replace nn.Identity() with your CNN encoder classes:           ║
        # ║    - ECGEncoder(input_dim, output_dim)                          ║
        # ║    - PPGEncoder(input_dim, output_dim)                          ║
        # ║    - RespEncoder(input_dim, output_dim)                         ║
        # ║                                                                 ║
        # ║  Expected interface:                                            ║
        # ║    Input:  [B, T, embedding_dim]  (from EmbeddingModel)         ║
        # ║    Output: [B, T, embedding_dim]  (encoded features)            ║
        # ║                                                                 ║
        # ║  See INTEGRATION_HOOKS.md for detailed instructions.            ║
        # ╚═════════════════════════════════════════════════════════════════╝


        # self.modality_encoders = nn.ModuleDict()
        # for modality in self.available_modalities:
        #     # ┌─────────────────────────────────────────────────────────────┐
        #     # │ TODO [DHRUV]: Replace nn.Identity() with your CNN encoder   │
        #     # │                                                             │
        #     # │ Example:                                                    │
        #     # │   self.modality_encoders[modality] = ECGEncoder(            │
        #     # │       input_dim=embedding_dim,                              │
        #     # │       output_dim=embedding_dim,                             │
        #     # │       dropout=dropout,                                      │
        #     # │   )                                                         │
        #     # └─────────────────────────────────────────────────────────────┘
        #     self.modality_encoders[modality] = nn.Identity()  # ← REPLACE THIS

        self.signal_encoders = SignalEncoders(
                                    signal_encoder_map={
                                        self.modality_to_signal[m]: self.modality_to_signal[m] for m in self.available_modalities
                                    },
                                    feature_dim=embedding_dim,
                                    activation='relu',
                                    norm='instance',
                                    include_signal=False,
                                )
        
        
        # ╔═════════════════════════════════════════════════════════════════╗
        # ║  ███╗   ██╗ █████╗ ███████╗██╗███████╗                          ║
        # ║  ████╗  ██║██╔══██╗██╔════╝██║██╔════╝                          ║
        # ║  ██╔██╗ ██║███████║█████╗  ██║███████╗                          ║
        # ║  ██║╚██╗██║██╔══██║██╔══╝  ██║╚════██║                          ║
        # ║  ██║ ╚████║██║  ██║██║     ██║███████║                          ║
        # ║  ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝                          ║
        # ║                                                                 ║
        # ║  INTEGRATION HOOK: Multimodal Fusion Module                     ║
        # ║                                                                 ║
        # ║  Option A: Use existing TransformerFusion (default)          ║
        # ║  Option B: Replace with your custom fusion module             ║
        # ║                                                                 ║
        # ║  Expected interface:                                            ║
        # ║    Input:  Dict[str, Tensor[B, T, D]] - modality embeddings     ║
        # ║    Output: Tensor[B, T, D] - fused representation               ║
        # ║                                                                 ║
        # ║  Must handle missing modalities (1, 2, or 3 inputs)!            ║
        # ║  See INTEGRATION_HOOKS.md for detailed instructions.            ║
        # ╚═════════════════════════════════════════════════════════════════╝
        if use_paper_faithful:
            # ┌─────────────────────────────────────────────────────────────┐
            # │ TODO [NAFIS]: Keep this OR replace with custom fusion       │
            # │                                                             │
            # │ Current: Paper-faithful TransformerFusion-based fusion      │
            # │ To replace:                                                 │
            # │   self.fusion_module = YourFusionModule(                    │
            # │       hidden_dim=embedding_dim,                             │
            # │       ...                                                   │
            # │   )                                                         │
            # └─────────────────────────────────────────────────────────────┘
            self.fusion_module = TransformerFusion(
                embed_dim=embedding_dim,
                num_heads=num_fusion_heads,
                num_layers=num_fusion_layers,
                dropout=dropout,
                num_modalities=len(self.available_modalities),
            )
        else:
            # Simplified: Identity (will use first modality or simple aggregation)
            self.fusion_module = None
        
        # =================================================================
        # PAPER-FAITHFUL: Dilated CNN Sequence Mixer
        # =================================================================
        if use_paper_faithful:
            self.temporal_layer = DilatedCNNSequenceMixer(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                kernel_size=temporal_kernel_size,
                num_layers=num_temporal_layers,
                dilations=dilations,
                dropout=dropout,
            )
        else:
            # Simplified: Standard temporal conv block
            self.temporal_layer = TemporalConvBlock(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                kernel_size=temporal_kernel_size,
                dropout=dropout,
            )
        
        # Classification head for sleep stage prediction
        # Maps [B, T, hidden_dim] -> [B, T, num_classes]
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def _validate_modality_shapes(self, modality_embeddings: Dict[str, torch.Tensor]) -> None:
        """Validate that all modality embeddings have compatible shapes.
        
        Args:
            modality_embeddings: Dict mapping modality names to tensors of shape [B, T, D]
        """
        if not modality_embeddings:
            raise ValueError("No modality embeddings provided")
        
        # Get reference shape from first modality
        reference_shape = next(iter(modality_embeddings.values())).shape
        batch_size, seq_len, embed_dim = reference_shape
        
        # Validate all modalities have the same shape
        for modality, embedding in modality_embeddings.items():
            if embedding.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch in modality '{modality}': expected {reference_shape}, "
                    f"got {embedding.shape}"
                )
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the Wav2Sleep model.
        
        Args:
            **kwargs: Input features containing modality data and labels
                Expected keys: any subset of ['ecg', 'ppg', 'resp'] plus label_key
                
        Returns:
            Dict containing:
                - loss: scalar tensor representing the loss
                - y_prob: tensor of shape [B*T, num_classes] with predicted probabilities  
                - y_true: tensor of shape [B*T] with true labels
                - logit: tensor of shape [B*T, num_classes] with predicted logits
        """
        # Step 1: Extract and validate available modalities from input
        available_inputs = {}
        for modality in self.available_modalities:
            if modality in kwargs:
                available_inputs[modality] = kwargs[modality]
        
        if not available_inputs:
            raise ValueError("At least one modality must be provided in input")
        
        # Step 2: Process through embedding model
        # embedded_inputs = self.embedding_model(available_inputs)
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │ Step 3: Apply modality-specific encoders                        │
        # │         ══════════════════════════════════                      │
        # │         DHRUV'S ENCODERS ARE CALLED HERE                        │
        # │         Each encoder: [B, T, D] → [B, T, D]                     │
        # └─────────────────────────────────────────────────────────────────┘
        # modality_embeddings = {}
        # for modality, embedded_data in embedded_inputs.items():
        #     # Ensure data is on correct device
        #     embedded_data = embedded_data.to(self.device)
            
            # ═══════════════════════════════════════════════════════════════
            # DHRUV'S ENCODER CALLED HERE: self.modality_encoders[modality]
            # ═══════════════════════════════════════════════════════════════
            # encoded = self.modality_encoders[modality](embedded_data)
            # modality_embeddings[modality] = encoded

        wave_inputs = {}
        for modality in self.available_modalities:
            if modality not in kwargs:
                continue
            sig = self.modality_to_signal[modality]  # ecg -> ECG, ppg -> PPG
            wave_inputs[sig] = kwargs[modality].to(self.device).float()
        z_signal = self.signal_encoders(wave_inputs)

        modality_embeddings = {
            modality: z_signal[self.modality_to_signal[modality]]
            for modality in self.available_modalities
            if modality in kwargs
        }
        
        # Step 4: Validate shapes before fusion
        self._validate_modality_shapes(modality_embeddings)

        # ┌─────────────────────────────────────────────────────────────────┐
        # │ Step 5: Multimodal fusion                                       │
        # │         ═════════════════════                                   │
        # │         NAFIS'S FUSION MODULE IS CALLED HERE                    │
        # │         Input: Dict[str, [B,T,D]] → Output: [B,T,D]             │
        # └─────────────────────────────────────────────────────────────────┘
        if self.use_paper_faithful and self.fusion_module is not None:
            # ═══════════════════════════════════════════════════════════════
            # NAFIS'S FUSION CALLED HERE: self.fusion_module(...)
            # ═══════════════════════════════════════════════════════════════
            fused_features = self.fusion_module(modality_embeddings)
        else:
            # SIMPLIFIED: Use first available modality or mean
            if len(modality_embeddings) == 1:
                fused_features = next(iter(modality_embeddings.values()))
            else:
                # Simple mean fusion as fallback
                stacked = torch.stack(list(modality_embeddings.values()), dim=0)
                fused_features = stacked.mean(dim=0)
        
        # Step 6: Temporal sequence modeling
        # PAPER-FAITHFUL: Dilated CNN sequence mixer
        # SIMPLIFIED: Standard temporal conv block
        temporal_features = self.temporal_layer(fused_features)  # [B, T, hidden_dim]
        
        # Step 7: Classification
        logits = self.classifier(temporal_features)  # [B, T, num_classes]
        
        # Step 8: Prepare output for loss computation
        # Flatten for sequence classification: [B, T, C] -> [B*T, C]
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)  # [B*T, num_classes]
        
        # Extract true labels and flatten
        y_true = kwargs[self.label_key].to(self.device)  # [B, T]
        y_true_flat = y_true.view(-1)  # [B*T]
        
        # Step 9: Compute loss and probabilities
        loss = self.get_loss_function()(logits_flat, y_true_flat)
        y_prob = self.prepare_y_prob(logits_flat)  # [B*T, num_classes]
        
        # Step 10: Return structured output
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true_flat,
            "logit": logits_flat,
        }
        
        # Optional: include embeddings if requested
        if kwargs.get("embed", False):
            results["embed"] = temporal_features
            
        return results
    
    def get_reproduction_fidelity_report(self) -> Dict[str, str]:
        """Return a report on reproduction fidelity to the wav2sleep paper.
        
        Returns:
            Dict with component names and their fidelity status
        """
        report = {
            "overall": "paper_faithful" if self.use_paper_faithful else "simplified",
            "fusion_module": (
                "TransformerFusion (paper-faithful)" 
                if self.use_paper_faithful 
                else "Mean pooling (simplified)"
            ),
            "temporal_layer": (
                f"Dilated CNN with receptive field {self.temporal_layer.receptive_field} epochs (paper-faithful)"
                if self.use_paper_faithful and hasattr(self.temporal_layer, 'receptive_field')
                else "Standard temporal CNN (simplified)"
            ),
            "modality_encoders": "Placeholder (needs Dhruv's CNN encoders)",
            "classification_head": "Linear layer (paper-faithful)",
        }
        return report


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    # This section can be used for basic testing
    print("Wav2Sleep model loaded successfully!")
    print("\nPaper-faithful components:")
    print("  - TransformerFusion: Transformer-based multimodal aggregation")
    print("  - DilatedCNNSequenceMixer: Dilated temporal convolutions")
    print("\nSimplified components (for comparison):")
    print("  - TemporalConvBlock: Standard temporal CNN")
