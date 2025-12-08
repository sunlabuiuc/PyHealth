"""
Wav2Sleep: Multi-Modal Sleep Stage Classification Model

Author: Meredith McClain (mmcclan2)
Paper: wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification 
       from Physiological Signals
Link: https://arxiv.org/abs/2411.04644
Description: Unified model for sleep stage classification that operates on 
             variable sets of physiological signals (ECG, PPG, ABD, THX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ResidualBlock(nn.Module):
    """Residual convolutional block for signal encoding.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolutional kernel size
        stride: Convolutional stride
        
    Example:
        >>> block = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3)
        >>> x = torch.randn(8, 32, 1024)
        >>> out = block(x)  # Shape: (8, 64, 512)
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.bn1 = nn.InstanceNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding
        )
        self.bn2 = nn.InstanceNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding
        )
        self.bn3 = nn.InstanceNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.InstanceNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.pool = nn.MaxPool1d(2)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Output tensor of shape (batch, out_channels, length//2)
        """
        identity = self.shortcut(x)
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out = self.pool(out + identity)
        out = self.activation(out)
        
        return out


class SignalEncoder(nn.Module):
    """CNN encoder for individual physiological signals.
    
    Encodes raw time-series signal into sequence of feature vectors,
    one per sleep epoch (30-second window).
    
    Args:
        sampling_rate: Number of samples per 30-second epoch (k)
        feature_dim: Output feature dimension
        channels: List of channel sizes for residual blocks
        
    Example:
        >>> encoder = SignalEncoder(sampling_rate=1024, feature_dim=128)
        >>> x = torch.randn(8, 1, 1200*1024)  # 8 samples, 1200 epochs
        >>> z = encoder(x)  # Shape: (8, 1200, 128)
    """
    
    def __init__(
        self,
        sampling_rate: int,
        feature_dim: int = 128,
        channels: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.sampling_rate = sampling_rate
        self.feature_dim = feature_dim
        
        # Default channel progression based on sampling rate
        if channels is None:
            if sampling_rate == 256:  # Low freq (respiratory)
                channels = [16, 32, 64, 64, 128, 128]
            else:  # High freq (ECG/PPG)
                channels = [16, 16, 32, 32, 64, 64, 128, 128]
        
        # Build residual blocks
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.append(ResidualBlock(in_ch, out_ch, kernel_size=3))
            in_ch = out_ch
            
        self.encoder = nn.Sequential(*layers)
        
        # Calculate output length after pooling
        self.output_length = sampling_rate // (2 ** len(channels))
        
        # Dense layer to produce feature vectors
        self.dense = nn.Linear(channels[-1] * self.output_length, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal into feature sequence.
        
        Args:
            x: Input signal of shape (batch, 1, seq_len)
               where seq_len = T * sampling_rate
               
        Returns:
            Feature sequence of shape (batch, T, feature_dim)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        T = seq_len // self.sampling_rate
        
        # Reshape to process each epoch
        # (batch, 1, T*k) -> (batch*T, 1, k)
        x = x.view(batch_size * T, 1, self.sampling_rate)
        
        # Encode through CNN
        z = self.encoder(x)  # (batch*T, channels, output_length)
        
        # Flatten spatial dimension
        z = z.view(batch_size * T, -1)
        
        # Apply dense layer
        z = self.dense(z)  # (batch*T, feature_dim)
        
        # Reshape back to sequence
        z = z.view(batch_size, T, self.feature_dim)
        
        return z


class EpochMixer(nn.Module):
    """Transformer encoder for cross-modal fusion.
    
    Fuses information from multiple signal modalities for each epoch
    using a transformer with CLS token.
    
    Args:
        feature_dim: Feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feedforward network
        dropout: Dropout probability
        
    Example:
        >>> mixer = EpochMixer(feature_dim=128)
        >>> # Multiple modalities for 1200 epochs
        >>> z_ecg = torch.randn(8, 1200, 128)
        >>> z_ppg = torch.randn(8, 1200, 128)
        >>> z_fused = mixer([z_ecg, z_ppg])  # Shape: (8, 1200, 128)
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(
        self, 
        features: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse multi-modal features.
        
        Args:
            features: List of feature tensors, each of shape (batch, T, feature_dim)
            mask: Optional attention mask for missing modalities
            
        Returns:
            Fused features of shape (batch, T, feature_dim)
        """
        batch_size = features[0].shape[0]
        T = features[0].shape[1]
        
        # Process each timestep
        fused_features = []
        
        for t in range(T):
            # Gather features for this epoch from all modalities
            epoch_features = [f[:, t:t+1, :] for f in features]
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            epoch_input = torch.cat([cls_tokens] + epoch_features, dim=1)
            
            # Apply transformer
            epoch_output = self.transformer(epoch_input, src_key_padding_mask=mask)
            
            # Extract CLS token output
            fused_features.append(epoch_output[:, 0:1, :])
        
        # Concatenate all epochs
        fused = torch.cat(fused_features, dim=1)
        
        return fused


class SequenceMixer(nn.Module):
    """Dilated CNN for temporal sequence modeling.
    
    Models long-range temporal dependencies in sleep stage sequences
    using dilated convolutions.
    
    Args:
        feature_dim: Feature dimension
        num_blocks: Number of dilated blocks
        num_classes: Number of sleep stage classes
        kernel_size: Convolutional kernel size
        dropout: Dropout probability
        
    Example:
        >>> mixer = SequenceMixer(feature_dim=128, num_classes=5)
        >>> z = torch.randn(8, 1200, 128)
        >>> logits = mixer(z)  # Shape: (8, 1200, 5)
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_blocks: int = 2,
        num_classes: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Dilated convolutional blocks
        dilations = [1, 2, 4, 8, 16, 32]
        
        blocks = []
        for _ in range(num_blocks):
            for dilation in dilations:
                padding = (kernel_size - 1) * dilation // 2
                blocks.extend([
                    nn.Conv1d(
                        feature_dim, feature_dim, kernel_size,
                        dilation=dilation, padding=padding
                    ),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
        
        self.dilated_conv = nn.Sequential(*blocks)
        
        # Output projection
        self.output = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence to predict sleep stages.
        
        Args:
            x: Input features of shape (batch, T, feature_dim)
            
        Returns:
            Logits of shape (batch, T, num_classes)
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # (batch, feature_dim, T)
        
        # Apply dilated convolutions
        x = self.dilated_conv(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch, T, feature_dim)
        
        # Project to classes
        logits = self.output(x)
        
        return logits


class Wav2Sleep(nn.Module):
    """Wav2sleep: Unified multi-modal sleep stage classification model.
    
    Operates on variable sets of physiological signals (ECG, PPG, ABD, THX)
    to classify sleep stages. Supports joint training on heterogeneous datasets
    and inference with any subset of signals.
    
    Architecture:
        1. Signal Encoders: Separate CNNs for each modality
        2. Epoch Mixer: Transformer for cross-modal fusion
        3. Sequence Mixer: Dilated CNN for temporal modeling
    
    Args:
        modalities: Dict mapping modality names to sampling rates
                   e.g. {"ecg": 1024, "ppg": 1024, "abd": 256, "thx": 256}
        num_classes: Number of sleep stage classes (default: 5)
        feature_dim: Feature dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        
    Example:
        >>> modalities = {"ecg": 1024, "ppg": 1024, "thx": 256}
        >>> model = Wav2Sleep(modalities=modalities, num_classes=5)
        >>> 
        >>> # Training with all modalities
        >>> inputs = {
        ...     "ecg": torch.randn(8, 1, 1200*1024),
        ...     "ppg": torch.randn(8, 1, 1200*1024),
        ...     "thx": torch.randn(8, 1, 1200*256)
        ... }
        >>> logits = model(inputs)  # Shape: (8, 1200, 5)
        >>> 
        >>> # Inference with subset of modalities
        >>> inputs_subset = {"ecg": torch.randn(8, 1, 1200*1024)}
        >>> logits = model(inputs_subset)  # Shape: (8, 1200, 5)
    """
    
    def __init__(
        self,
        modalities: Dict[str, int],
        num_classes: int = 5,
        feature_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.modalities = modalities
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Create signal encoders for each modality
        self.encoders = nn.ModuleDict({
            name: SignalEncoder(
                sampling_rate=rate,
                feature_dim=feature_dim
            )
            for name, rate in modalities.items()
        })
        
        # Epoch mixer for cross-modal fusion
        self.epoch_mixer = EpochMixer(
            feature_dim=feature_dim,
            num_layers=2,
            num_heads=8,
            hidden_dim=512,
            dropout=dropout
        )
        
        # Sequence mixer for temporal modeling
        self.sequence_mixer = SequenceMixer(
            feature_dim=feature_dim,
            num_blocks=2,
            num_classes=num_classes,
            kernel_size=7,
            dropout=dropout
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through wav2sleep model.
        
        Args:
            inputs: Dictionary of input signals, each of shape (batch, 1, seq_len)
            labels: Optional ground truth labels of shape (batch, T)
            
        Returns:
            Dictionary containing:
                - logits: Predicted logits of shape (batch, T, num_classes)
                - loss: Cross-entropy loss (if labels provided)
                - predictions: Predicted sleep stages (if labels provided)
        """
        # Encode each available modality
        features = []
        for name, signal in inputs.items():
            if name in self.encoders:
                z = self.encoders[name](signal)
                features.append(z)
        
        # Fuse cross-modal information
        fused = self.epoch_mixer(features)
        
        # Model temporal dependencies
        logits = self.sequence_mixer(fused)
        
        # Prepare output
        output = {"logits": logits}
        
        if labels is not None:
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_classes),
                labels.reshape(-1)
            )
            output["loss"] = loss
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            output["predictions"] = predictions
        
        return output
    
    def predict_proba(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get predicted probabilities for sleep stages.
        
        Args:
            inputs: Dictionary of input signals
            
        Returns:
            Probability distributions of shape (batch, T, num_classes)
        """
        with torch.no_grad():
            output = self.forward(inputs)
            probs = F.softmax(output["logits"], dim=-1)
        return probs


def main():
    """Example usage of Wav2Sleep model."""
    
    print("Wav2Sleep Model Example")
    print("=" * 50)
    
    # Define modalities
    modalities = {
        "ecg": 1024,   # 34 Hz * 30 sec
        "ppg": 1024,   # 34 Hz * 30 sec
        "abd": 256,    # 8 Hz * 30 sec
        "thx": 256     # 8 Hz * 30 sec
    }
    
    # Create model
    model = Wav2Sleep(
        modalities=modalities,
        num_classes=5,  # Wake, N1, N2, N3, REM
        feature_dim=128,
        dropout=0.1
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example 1: All modalities
    print("\n--- Example 1: Training with all modalities ---")
    batch_size = 4
    T = 1200  # 10 hours
    
    inputs_all = {
        "ecg": torch.randn(batch_size, 1, T * 1024),
        "ppg": torch.randn(batch_size, 1, T * 1024),
        "abd": torch.randn(batch_size, 1, T * 256),
        "thx": torch.randn(batch_size, 1, T * 256)
    }
    labels = torch.randint(0, 5, (batch_size, T))
    
    output = model(inputs_all, labels)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Predictions shape: {output['predictions'].shape}")
    
    # Example 2: Subset of modalities
    print("\n--- Example 2: Inference with ECG only ---")
    inputs_ecg = {
        "ecg": torch.randn(batch_size, 1, T * 1024)
    }
    
    probs = model.predict_proba(inputs_ecg)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Example probabilities for first epoch:\n{probs[0, 0]}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
