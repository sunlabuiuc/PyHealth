"""
Wav2Sleep: A Unified Multi-Modal Approach to Sleep Stage Classification.

This module implements the Wav2Sleep model for sleep stage classification
from physiological signals, supporting variable sets of signals and joint
training across heterogeneous datasets.

The architecture consists of three main stages:
1. Modality-specific CNN Encoders: Extracts features from 1D physiological signals.
2. Epoch Mixer: A Transformer-based fusion module using a [CLS] token.
3. Sequence Mixer: A dilated CNN capturing long-range temporal dependencies.

Reference:
    Carter, J. F., & Tarassenko, L. (2024). wav2sleep: A Unified Multi-Modal
    Approach to Sleep Stage Classification from Physiological Signals.
    arXiv:2411.04644
"""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


# ---------------------------------------------------------------------------
# 1. Internal Building Blocks: Signal Encoding
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Residual convolutional block for 1D physiological signal processing.

    This block implements a standard residual connection with two stages
    of Conv1d, Instance Normalization, and GELU activation.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int): Size of the 1D convolution kernel. Defaults to 3.
        stride (int): Stride of the first convolution. Defaults to 1.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1
    ):
        super(_ResBlock, self).__init__()

        # Standard padding to maintain temporal length before pooling
        padding = kernel_size // 2

        # Main path
        self.conv_path = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm1d(out_channels)
        )

        # Skip connection path (residual)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.InstanceNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.pooling = nn.MaxPool1d(2)
        self.final_activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (N, C_in, L_in)
        Returns:
            torch.Tensor: Output of shape (N, C_out, L_in // 2)
        """
        residual = self.shortcut(x)
        out = self.conv_path(x)
        out = self.final_activation(out + residual)
        return self.pooling(out)


class _SignalEncoder(nn.Module):
    """CNN encoder for modality-specific feature extraction.

    As per Section 3.1 of the paper, the architecture depth is dynamically
    adjusted based on the sampling frequency to ensure fixed-size embeddings.

    Args:
        sampling_rate (int): Number of samples per epoch (e.g., 30s @ 100Hz = 3000).
        feature_dim (int): Final output dimension for the epoch embedding.
    """

    def __init__(
            self,
            sampling_rate: int,
            feature_dim: int = 128
    ):
        super(_SignalEncoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.feature_dim = feature_dim

        # Determine depth: High frequency signals (ECG) need more pooling stages
        if sampling_rate >= 512:
            self.channel_cfg = [16, 16, 32, 32, 64, 64, 128, 128]
        else:
            self.channel_cfg = [16, 32, 64, 64, 128, 128]

        # Build residual stages
        layers = []
        curr_in = 1
        for curr_out in self.channel_cfg:
            layers.append(_ResBlock(curr_in, curr_out))
            curr_in = curr_out

        self.backbone = nn.Sequential(*layers)

        # Calculate latent feature length after pooling
        # Each ResBlock has a MaxPool1d(2)
        reduced_len = sampling_rate // (2 ** len(self.channel_cfg))
        self.flatten_dim = self.channel_cfg[-1] * max(1, reduced_len)

        # Final projection to shared latent space
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Signal of shape (B, 1, T * sampling_rate)
        Returns:
            torch.Tensor: Epoch embeddings of shape (B, T, feature_dim)
        """
        batch_size, _, total_len = x.shape
        num_epochs = total_len // self.sampling_rate

        # 1. Reshape to process all epochs across batch in parallel
        # (B, 1, T*L) -> (B*T, 1, L)
        x_epochs = x.view(batch_size * num_epochs, 1, self.sampling_rate)

        # 2. Extract features through CNN backbone
        # (B*T, 1, L) -> (B*T, C_final, L_reduced)
        feat = self.backbone(x_epochs)

        # 3. Flatten and project to common embedding dimension
        # (B*T, C_final * L_reduced) -> (B*T, feature_dim)
        feat = feat.view(feat.size(0), -1)
        out = self.projection(feat)

        # 4. Reshape back to temporal sequence
        # (B*T, feature_dim) -> (B, T, feature_dim)
        return out.view(batch_size, num_epochs, -1)


# ---------------------------------------------------------------------------
# 2. Internal Building Blocks: Fusion & Temporal Modeling
# ---------------------------------------------------------------------------

class _EpochMixer(nn.Module):
    """Transformer Mixer for cross-modal signal fusion via [CLS] token.

    Attributes:
        cls_token: A learnable parameter prepended to modality features.
    """

    def __init__(
            self,
            feature_dim: int,
            num_layers: int,
            nhead: int,
            dropout: float
    ):
        super(_EpochMixer, self).__init__()
        self.feature_dim = feature_dim

        # Learnable [CLS] token representing the fused state of the epoch
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: List of (B, T, D) tensors from different signals.
        Returns:
            Fused epoch sequence: (B, T, D)
        """
        batch_size = modality_features[0].shape[0]
        num_epochs = modality_features[0].shape[1]

        # Reshape to treat each epoch (across batch) as a sequence for Transformer
        # (B, T, D) -> (B*T, 1, D)
        stacked_features = [f.view(batch_size * num_epochs, 1, -1)
                            for f in modality_features]

        # Concatenate features from all modalities: (B*T, Num_Modalities, D)
        x = torch.cat(stacked_features, dim=1)

        # Prepend [CLS] token: (B*T, Num_Modalities + 1, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Perform cross-modal attention
        x = self.transformer(x)
        x = self.layer_norm(x)

        # Extract the fused [CLS] representation and restore batch/time dimensions
        # (B*T, D) -> (B, T, D)
        fused_out = x[:, 0, :].view(batch_size, num_epochs, -1)
        return fused_out


class _SequenceMixer(nn.Module):
    """Temporal Sequence Mixer using Dilated Convolutions.

    Designed to capture long-range dependencies (sleep stage transitions)
    across several hours of sleep recording.
    """

    def __init__(
            self,
            feature_dim: int,
            num_classes: int,
            dropout: float
    ):
        super(_SequenceMixer, self).__init__()

        # Use exponentially increasing dilations to increase receptive field
        # Kernel 7 with dilations [1, 2, 4, 8, 16, 32] covers a large temporal window
        dilations = [1, 2, 4, 8, 16, 32]

        self.blocks = nn.ModuleList()
        for d in dilations:
            padding = (7 - 1) * d // 2  # Maintain length
            self.blocks.append(nn.Sequential(
                nn.Conv1d(
                    feature_dim,
                    feature_dim,
                    kernel_size=7,
                    dilation=d,
                    padding=padding),
                nn.InstanceNorm1d(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Fused sequence (B, T, D)
        Returns:
            torch.Tensor: Prediction logits (B, T, num_classes)
        """
        # Conv1d expects (Batch, Channels, Time)
        x = x.transpose(1, 2)

        # Apply sequential dilated blocks
        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 3. PyHealth BaseModel Wrapper
# ---------------------------------------------------------------------------

class Wav2Sleep(BaseModel):
    """Wav2Sleep: Unified Multi-Modal Sleep Stage Classification Model.

    This model integrates various physiological signals into a unified latent
    space using modality-specific CNNs, fuses them with a Transformer Epoch
    Mixer, and captures temporal context with a Dilated CNN Sequence Mixer.

    Paper:
        Carter, J. F., & Tarassenko, L. (2024). wav2sleep: A Unified Multi-Modal
        Approach to Sleep Stage Classification from Physiological Signals.
        arXiv:2411.04644

    Args:
        dataset (SampleDataset): The dataset instance for schema inference.
        modalities (Dict[str, int]): Map of signal keys to their sampling rates
            per epoch (e.g., {"ecg": 3000, "resp": 750}).
        label_key (str): The key for sleep stage labels in the dataset.
        mode (str): Task mode, "multiclass" for sleep staging.
        embedding_dim (int): Hidden dimension size. Defaults to 128.
        nhead (int): Number of heads in Transformer. Defaults to 8.
        num_layers (int): Number of Transformer layers. Defaults to 2.
        mask_prob (Dict[str, float], optional): Modality-specific stochastic
            drop probabilities for robust learning.
        dropout (float): Dropout probability. Defaults to 0.1.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Wav2Sleep
        >>> import torch
        >>> samples = [
        ...     {
        ...         "patient_id": "p1",
        ...         "ecg": torch.randn(5, 3000).tolist(),
        ...         "resp": torch.randn(5, 750).tolist(),
        ...         "label": [0, 1, 2, 1, 0],
        ...     }
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ecg": "tensor", "resp": "tensor", "label": "tensor"},
        ...     output_schema={},
        ... )
        >>> loader = get_dataloader(dataset, batch_size=1)
        >>> model = Wav2Sleep(
        ...     dataset=dataset,
        ...     modalities={"ecg": 3000, "resp": 750},
        ...     label_key="label",
        ...     mode="multiclass",
        ...     num_classes=5,
        ... )
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> output["y_prob"].shape
        torch.Size([1, 5, 5])
    """

    def __init__(
            self,
            dataset: SampleDataset,
            modalities: Dict[str, int],
            label_key: str,
            mode: str,
            embedding_dim: int = 128,
            nhead: int = 8,
            num_layers: int = 2,
            mask_prob: Optional[Dict[str, float]] = None,
            dropout: float = 0.1,
            **kwargs
    ):
        num_classes_from_kwargs = kwargs.pop("num_classes", None)
        super(Wav2Sleep, self).__init__(dataset, **kwargs)

        self.modalities = modalities
        self.label_key = label_key
        self.mode = mode

        # 1. Initialize Signal Encoders for each modality
        self.encoders = nn.ModuleDict({
            name: _SignalEncoder(rate, embedding_dim)
            for name, rate in modalities.items()
        })

        # 2. Initialize Fusion and Sequence Mixers
        self.epoch_mixer = _EpochMixer(embedding_dim, num_layers, nhead, dropout)

        # Resolve output size (number of sleep stages)
        try:
            self.num_classes = self.get_output_size(dataset)
        except Exception:
            self.num_classes = num_classes_from_kwargs or 5
        self.sequence_mixer = _SequenceMixer(embedding_dim, self.num_classes, dropout)

        # Stochastic Masking probabilities (Paper Section 3.2)
        self.mask_probs = mask_prob or {k: 0.5 for k in modalities.keys()}

    def _check_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Validates and prepares input tensors from the PyHealth batch."""
        prepared = {}
        for name in self.modalities.keys():
            if name not in kwargs:
                continue

            val = kwargs[name]
            # Ensure tensor format and correct device
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, device=self.device)
            else:
                val = val.to(self.device)

            # Standardize shape to (Batch, 1, Total_Length)
            if val.dim() == 2:
                val = val.unsqueeze(1)

            prepared[name] = val.float()
        return prepared

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for training and inference.

        Returns:
            Dict containing 'logit', 'y_prob', 'loss', and 'y_true'.
        """
        # 1. Preprocess and validate inputs
        inputs = self._check_inputs(kwargs)
        if not inputs:
            raise ValueError(f"None of the required modalities "
                             f"{list(self.modalities.keys())} found.")

        # 2. Extract modality features and apply Stochastic Masking
        modality_embeddings = []
        for name, tensor in inputs.items():
            if tensor.dim() == 3 and name in self.modalities:
                B, T, L = tensor.shape
                if L == self.modalities[name]:
                    tensor = tensor.view(B, 1, -1)

            z = self.encoders[name](tensor)  # (B, T, D)

            # Stochastic masking during training to improve robustness
            if self.training:
                p = self.mask_probs.get(name.lower(), 0.5)
                mask = (torch.rand(z.shape[0], 1, 1, device=z.device) > p).float()
                z = z * mask

            modality_embeddings.append(z)

        # 3. Cross-modal Fusion (Epoch Mixer)
        fused_epochs = self.epoch_mixer(modality_embeddings)  # (B, T, D)

        # 4. Temporal Transition Modeling (Sequence Mixer)
        logits = self.sequence_mixer(fused_epochs)  # (B, T, num_classes)

        # 5. Package results for PyHealth
        y_prob = torch.softmax(logits, dim=-1)
        results = {"logit": logits, "y_prob": y_prob}

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).long()
            # Flatten B and T dimensions for standard cross-entropy loss
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.num_classes),
                y_true.view(-1)
            )
            results["loss"] = loss
            results["y_true"] = y_true

        return results
