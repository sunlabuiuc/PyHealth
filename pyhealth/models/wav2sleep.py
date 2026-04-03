# Author: Rahul Chakraborty
# Description: Wav2Sleep multimodal sleep stage classification model for PyHealth
#
# Paper-faithful implementation of:
#   - CLS-token Transformer fusion for multimodal aggregation
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

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


# =============================================================================
# CLS-Token Transformer Fusion Module (Paper-Faithful)
# =============================================================================

class CLSTokenTransformerFusion(nn.Module):
    """CLS-token Transformer fusion module for multimodal aggregation.
    
    This module implements the paper-faithful fusion strategy from wav2sleep:
    - Takes modality embeddings of shape [B, T, D] from available modalities
    - For each epoch, stacks available modality embeddings
    - Prepends a learnable CLS token
    - Applies a Transformer encoder
    - Returns the CLS output as the fused epoch representation
    
    Why CLS-token fusion instead of mean pooling or concatenation?
    1. **Learnable aggregation**: CLS token learns an optimal weighting of modalities
    2. **Missing modality robustness**: CLS can attend to available modalities only,
       naturally handling variable numbers of inputs
    3. **Fixed output dimension**: Output is always [B, T, D] regardless of how many
       modalities are present
    4. **Cross-modal attention**: Transformer allows modalities to attend to each other,
       capturing inter-modal relationships
    
    Args:
        embed_dim: Embedding dimension (D)
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        dropout: Dropout rate
        max_modalities: Maximum number of modalities (default: 3 for ECG, PPG, Resp)
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_modalities: int = 3,
    ):
        super(CLSTokenTransformerFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_modalities = max_modalities
        
        # Learnable CLS token: [1, 1, D]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Modality-type embeddings to distinguish different modalities
        # +1 for CLS token position
        self.modality_embeddings = nn.Embedding(max_modalities + 1, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input shape: [B, seq_len, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize CLS token and embeddings."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embeddings.weight, std=0.02)
    
    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        modality_order: List[str] = ['ecg', 'ppg', 'resp'],
    ) -> torch.Tensor:
        """Forward pass for CLS-token transformer fusion.
        
        Args:
            modality_embeddings: Dict mapping modality names to tensors of shape [B, T, D]
            modality_order: List defining the order/index of modalities for embeddings
            
        Returns:
            Fused representation of shape [B, T, D]
        """
        if not modality_embeddings:
            raise ValueError("At least one modality must be provided for fusion")
        
        # Get batch size and sequence length from first available modality
        first_modality = next(iter(modality_embeddings.values()))
        batch_size, seq_len, embed_dim = first_modality.shape
        device = first_modality.device
        
        # Process each epoch independently
        # Reshape to process all epochs as a batch: [B*T, num_modalities, D]
        fused_epochs = []
        
        for t in range(seq_len):
            # Collect available modality features for this epoch: [B, num_avail, D]
            epoch_features = []
            modality_indices = []
            
            for mod_name, mod_tensor in modality_embeddings.items():
                # Get epoch t features: [B, D]
                epoch_feat = mod_tensor[:, t, :]  
                epoch_features.append(epoch_feat.unsqueeze(1))  # [B, 1, D]
                
                # Get modality index (1-indexed, 0 is CLS)
                mod_idx = modality_order.index(mod_name) + 1 if mod_name in modality_order else 1
                modality_indices.append(mod_idx)
            
            # Stack modality features: [B, num_avail_modalities, D]
            stacked_features = torch.cat(epoch_features, dim=1)
            num_modalities = stacked_features.shape[1]
            
            # Expand CLS token for batch: [B, 1, D]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            
            # Prepend CLS token: [B, 1 + num_modalities, D]
            sequence = torch.cat([cls_tokens, stacked_features], dim=1)
            
            # Add modality-type embeddings
            # CLS gets index 0, modalities get their respective indices
            modality_ids = torch.tensor([0] + modality_indices, device=device)
            modality_ids = modality_ids.unsqueeze(0).expand(batch_size, -1)  # [B, 1+num_mod]
            modality_emb = self.modality_embeddings(modality_ids)  # [B, 1+num_mod, D]
            
            sequence = sequence + modality_emb
            
            # Apply transformer encoder: [B, 1 + num_modalities, D]
            encoded = self.transformer_encoder(sequence)
            
            # Extract CLS token output as fused representation: [B, D]
            cls_output = encoded[:, 0, :]
            cls_output = self.layer_norm(cls_output)
            
            fused_epochs.append(cls_output.unsqueeze(1))  # [B, 1, D]
        
        # Stack all epochs: [B, T, D]
        fused_output = torch.cat(fused_epochs, dim=1)
        
        return fused_output


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


# =============================================================================
# Main Wav2Sleep Model
# =============================================================================

class Wav2Sleep(BaseModel):
    """Wav2Sleep multimodal sleep stage classification model.
    
    This is a PAPER-FAITHFUL implementation that includes:
    - CLS-token Transformer fusion for multimodal aggregation
    - Dilated CNN sequence mixer for temporal modeling
    
    Architecture:
    1. Modality-specific encoders (ECG, PPG, Respiratory) -> [B, T, D] each
    2. CLS-token Transformer fusion -> [B, T, D]
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
        self.modality_keys = ['ecg', 'ppg', 'resp']
        
        # Check which modalities are available in the dataset
        self.available_modalities = [key for key in self.modality_keys if key in self.feature_keys]
        if not self.available_modalities:
            raise ValueError("At least one modality (ecg, ppg, resp) must be present in dataset")
        
        # Embedding model for initial feature processing
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        
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
        self.modality_encoders = nn.ModuleDict()
        for modality in self.available_modalities:
            # ┌─────────────────────────────────────────────────────────────┐
            # │ TODO [DHRUV]: Replace nn.Identity() with your CNN encoder   │
            # │                                                             │
            # │ Example:                                                    │
            # │   self.modality_encoders[modality] = ECGEncoder(            │
            # │       input_dim=embedding_dim,                              │
            # │       output_dim=embedding_dim,                             │
            # │       dropout=dropout,                                      │
            # │   )                                                         │
            # └─────────────────────────────────────────────────────────────┘
            self.modality_encoders[modality] = nn.Identity()  # ← REPLACE THIS
        
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
        # ║  Option A: Use existing CLSTokenTransformerFusion (default)     ║
        # ║  Option B: Replace with your custom fusion module               ║
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
            # │ Current: Paper-faithful CLS-token Transformer fusion        │
            # │ To replace:                                                 │
            # │   self.fusion_module = YourFusionModule(                    │
            # │       embed_dim=embedding_dim,                              │
            # │       ...                                                   │
            # │   )                                                         │
            # └─────────────────────────────────────────────────────────────┘
            self.fusion_module = CLSTokenTransformerFusion(
                embed_dim=embedding_dim,
                num_heads=num_fusion_heads,
                num_layers=num_fusion_layers,
                dropout=dropout,
                max_modalities=len(self.modality_keys),
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
        embedded_inputs = self.embedding_model(available_inputs)
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │ Step 3: Apply modality-specific encoders                        │
        # │         ══════════════════════════════════                      │
        # │         DHRUV'S ENCODERS ARE CALLED HERE                        │
        # │         Each encoder: [B, T, D] → [B, T, D]                     │
        # └─────────────────────────────────────────────────────────────────┘
        modality_embeddings = {}
        for modality, embedded_data in embedded_inputs.items():
            # Ensure data is on correct device
            embedded_data = embedded_data.to(self.device)
            
            # ═══════════════════════════════════════════════════════════════
            # DHRUV'S ENCODER CALLED HERE: self.modality_encoders[modality]
            # ═══════════════════════════════════════════════════════════════
            encoded = self.modality_encoders[modality](embedded_data)
            modality_embeddings[modality] = encoded
        
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
            fused_features = self.fusion_module(
                modality_embeddings, 
                modality_order=self.modality_keys
            )
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
                "CLS-token Transformer (paper-faithful)" 
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
    print("  - CLSTokenTransformerFusion: CLS-token based multimodal aggregation")
    print("  - DilatedCNNSequenceMixer: Dilated temporal convolutions")
    print("\nSimplified components (for comparison):")
    print("  - TemporalConvBlock: Standard temporal CNN")