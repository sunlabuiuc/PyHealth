"""Temporal Pointwise Convolutional Networks (TPC) for PyHealth.

Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://arxiv.org/abs/2101.10043 (CHIL 2021)
Authors: Emma Rocheteau, Catherine Schwarz, Ari Ercole, Pietro Liò, Stephanie Hyland
GitHub: https://github.com/EmmaRocheteau/TPC-LoS-prediction

Implementation Authors: Zakaria Coulibaly
NetID: zakaria5

Description:
    State-of-the-art implementation of TPC for healthcare time series prediction.
    TPC naturally handles irregular time sampling through temporal convolutions
    combined with pointwise (1x1) convolutions, making it ideal for ICU data and
    other irregularly-sampled medical time series.

Key Features:
    - Temporal convolutions with increasing dilation for multi-scale patterns
    - Pointwise convolutions for feature interactions
    - Dense skip connections within each layer
    - Support for irregular time series and variable-length sequences
    - Configurable loss functions (PyHealth standard MSE recommended for stability)

Architecture:
    Each TPC layer consists of:
    1. Temporal convolution (grouped, one per input feature)
    2. Pointwise (1x1) convolution for feature mixing
    3. Dense skip connections combining: [input, temporal_out, pointwise_out]

    Layers are stacked with each layer's output feeding into the next,
    creating a progressively deeper representation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models import EmbeddingModel


class TPCBlock(nn.Module):
    """Single Temporal Pointwise Convolutional Block.

    Implements one TPC layer as described in Section 2 and Figure 3 of the paper.
    Each block consists of:
        1. Temporal (grouped) convolution - captures time-series patterns
        2. Pointwise (1x1) convolution - enables feature interactions
        3. Dense skip connections - concatenates [input, temp_out, point_out]

    The temporal convolution uses grouped convolutions where each feature
    has its own convolutional kernel, preserving feature independence while
    capturing temporal patterns.

    Args:
        input_channels (int): Number of input channels/features.
        num_filters (int): Number of filters per temporal convolution.
            Default is 8 as in the paper.
        kernel_size (int): Temporal convolution kernel size. Default is 4
            as in the paper.
        dilation (int): Dilation rate for temporal convolution. Increases
            with layer depth for multi-scale pattern capture. Default is 1.
        pointwise_channels (int): Output channels for pointwise convolution.
            Default is 128 as in the paper.
        dropout (float): Dropout probability applied after each convolution.
            Default is 0.3 as in the paper.

    Attributes:
        temporal_conv (nn.Conv1d): Grouped 1D convolution for temporal patterns.
        pointwise_conv (nn.Linear): Linear layer implementing 1x1 convolution.
        bn_temporal (nn.BatchNorm1d): Batch normalization for temporal features.
        bn_pointwise (nn.BatchNorm1d): Batch normalization for pointwise features.
        output_channels (int): Total output channels after skip connections.

    Shape:
        - Input: (batch_size, input_channels, sequence_length)
        - Output: (batch_size, output_channels, sequence_length)

        where output_channels = input_channels + (input_channels * num_filters)
                                + pointwise_channels

    Examples:
        >>> import torch
        >>> block = TPCBlock(input_channels=128, num_filters=8, kernel_size=4)
        >>> x = torch.randn(32, 128, 50)  # batch=32, channels=128, seq_len=50
        >>> output = block(x)
        >>> output.shape
        torch.Size([32, 384, 50])
    """

    def __init__(
            self,
            input_channels: int,
            num_filters: int = 8,
            kernel_size: int = 4,
            dilation: int = 1,
            pointwise_channels: int = 128,
            dropout: float = 0.3,
    ):
        """Initializes TPCBlock.

        Args:
            input_channels: Number of input channels.
            num_filters: Number of filters per temporal convolution.
            kernel_size: Temporal convolution kernel size.
            dilation: Dilation rate for temporal convolution.
            pointwise_channels: Output channels for pointwise convolution.
            dropout: Dropout probability.
        """
        super(TPCBlock, self).__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pointwise_channels = pointwise_channels

        # Temporal convolution (grouped - one filter per input channel)
        # This preserves feature independence as described in Section 2.1
        self.temporal_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels * num_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=input_channels,  # Key: each feature has independent weights
            padding=0,  # Use causal padding manually
            bias=True,
        )

        # Batch normalization for temporal features
        self.bn_temporal = nn.BatchNorm1d(
            input_channels * num_filters, track_running_stats=True
        )

        # Pointwise (1x1) convolution for feature mixing
        # Implemented as Linear layer as in paper's reference code
        self.pointwise_conv = nn.Linear(input_channels, pointwise_channels)

        # Batch normalization for pointwise features
        self.bn_pointwise = nn.BatchNorm1d(
            pointwise_channels, track_running_stats=True
        )

        # Dropout layers
        self.dropout_temporal = nn.Dropout(dropout)
        self.dropout_pointwise = nn.Dropout(dropout)

        # Output dimension after skip connections (concatenation)
        self.output_channels = (
                input_channels + (input_channels * num_filters) + pointwise_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TPC block.

        Applies temporal convolution, pointwise convolution, and combines
        them with skip connections as described in Section 2.1 of the paper.

        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with skip connections, shape
                (batch_size, output_channels, sequence_length).
        """
        batch_size, channels, seq_length = x.shape

        # === Temporal Convolution Branch ===
        # Apply causal padding to preserve temporal ordering
        # Padding amount ensures output length matches input length
        padding_amount = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (padding_amount, 0), mode="constant", value=0)

        # Apply temporal convolution
        temporal_out = self.temporal_conv(x_padded)

        # Batch normalization - handle edge case of single timestep
        if self.training and seq_length == 1:
            # Switch to eval mode temporarily to avoid batch norm error
            self.bn_temporal.eval()
            temporal_out = self.bn_temporal(temporal_out)
            self.bn_temporal.train()
        else:
            temporal_out = self.bn_temporal(temporal_out)

        temporal_out = self.dropout_temporal(temporal_out)

        # === Pointwise Convolution Branch ===
        # Reshape for pointwise: (batch_size, sequence_length, input_channels)
        x_transposed = x.permute(0, 2, 1)

        # Apply pointwise convolution (1x1 conv implemented as Linear)
        pointwise_out = self.pointwise_conv(x_transposed)
        pointwise_out = pointwise_out.permute(0, 2, 1)  # Back to (B, C, T)

        # Batch normalization - handle single timestep edge case
        if self.training and seq_length == 1:
            self.bn_pointwise.eval()
            pointwise_out = self.bn_pointwise(pointwise_out)
            self.bn_pointwise.train()
        else:
            pointwise_out = self.bn_pointwise(pointwise_out)

        pointwise_out = self.dropout_pointwise(pointwise_out)

        # === Dense Skip Connections ===
        # Concatenate [original_input, temporal_features, pointwise_features]
        # This is the key innovation: preserving information through layers
        output = torch.cat([x, temporal_out, pointwise_out], dim=1)

        # Apply ReLU activation
        output = F.relu(output)

        return output


class TPC(BaseModel):
    """Temporal Pointwise Convolutional Network for healthcare prediction.

    Full implementation of the TPC architecture from:
    "Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit" (Rocheteau et al., CHIL 2021)

    The model processes irregularly-sampled time series data through stacked
    TPC blocks, each combining temporal and pointwise convolutions with dense
    skip connections. The final representation is obtained by taking the last
    valid timestep for each sequence.

    Key Advantages:
        - Handles irregular time sampling naturally
        - Captures multi-scale temporal patterns via dilation
        - Enables rich feature interactions via pointwise convolutions
        - Efficient on long sequences due to causal convolutions
        - Supports variable-length sequences with masking

    Architecture Details:
        1. Embedding layer: Converts categorical features to dense vectors
        2. TPC blocks: Stacked layers with increasing dilation rates
        3. Temporal pooling: Extract last valid timestep representation
        4. Classification/Regression head: Final prediction layer

    Loss Function:
        Uses PyHealth's standard losses:
        - MSE for regression tasks
        - Binary cross-entropy for binary classification
        - Cross-entropy for multiclass classification

        Note: The original paper uses masked MSLE loss, but that approach is
        designed for sequence-to-sequence prediction. This implementation performs
        sequence-to-one prediction (single LoS value per patient), which already
        handles variable-length sequences by extracting the last valid timestep.

    Args:
        dataset (SampleDataset): PyHealth dataset with input/output schemas.
            Must contain at least one feature key and one label key.
        embedding_dim (int): Dimension for embedding categorical features.
            Default is 128.
        num_layers (int): Number of TPC blocks to stack. Default is 3 as in paper.
        num_filters (int): Number of filters per temporal convolution. Default
            is 8 as in paper.
        pointwise_channels (int): Channels for pointwise convolutions. Default
            is 128 as in paper.
        kernel_size (int): Temporal convolution kernel size. Default is 4 as
            in paper.
        dropout (float): Dropout probability. Default is 0.3 as in paper.
        **kwargs: Additional arguments passed to BaseModel.

    Attributes:
        embedding_model (EmbeddingModel): Handles feature embeddings.
        tpc_blocks (nn.ModuleList): Stack of TPC blocks.
        fc (nn.Linear): Final classification/regression layer.
        mode (str): Task mode - "regression", "binary", or "multiclass".

    Raises:
        ValueError: If num_layers < 1, embedding_dim < 1, or num_filters < 1.
        AssertionError: If dataset does not have exactly one label key.

    Examples:
        >>> # Standard usage
        >>> samples = [{"patient_id": "p1", "conditions": [...], "label": 3.5}]
        >>> dataset = SampleDataset(samples=samples, ...)
        >>> model = TPC(dataset=dataset)
        >>>
        >>> # Custom hyperparameters
        >>> model = TPC(
        ...     dataset=dataset,
        ...     embedding_dim=256,
        ...     num_layers=5,
        ...     num_filters=12,
        ...     dropout=0.4
        ... )

    """

    def __init__(
            self,
            dataset: SampleDataset,
            embedding_dim: int = 128,
            num_layers: int = 3,
            num_filters: int = 8,
            pointwise_channels: int = 128,
            kernel_size: int = 4,
            dropout: float = 0.3,
            **kwargs
    ):
        """Initializes TPC model.

        Args:
            dataset: PyHealth dataset with input/output schemas.
            embedding_dim: Dimension for embedding categorical features.
            num_layers: Number of TPC blocks to stack.
            num_filters: Number of filters per temporal convolution.
            pointwise_channels: Channels for pointwise convolutions.
            kernel_size: Temporal convolution kernel size.
            dropout: Dropout probability.
            **kwargs: Additional arguments for BaseModel.

        Raises:
            ValueError: If hyperparameters are invalid.
        """
        super(TPC, self).__init__(dataset=dataset, **kwargs)

        # Validate configuration
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if num_filters < 1:
            raise ValueError(f"num_filters must be >= 1, got {num_filters}")

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.pointwise_channels = pointwise_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Get label key from dataset
        assert len(self.label_keys) == 1, "TPC supports only one label key"
        self.label_key = self.label_keys[0]

        # Use PyHealth's EmbeddingModel for handling embeddings
        self.embedding_model = EmbeddingModel(
            dataset=dataset, embedding_dim=embedding_dim
        )

        # Build TPC blocks with increasing dilation
        self.tpc_blocks = nn.ModuleList()

        # Initial input dimension: num_features * embedding_dim
        current_channels = len(self.feature_keys) * embedding_dim

        for layer_idx in range(num_layers):
            # Dilation increases with depth for multi-scale patterns
            # As per paper Section 2.1: d = 1 + n(k-1) where n is layer index
            dilation = 1 + layer_idx * (kernel_size - 1)

            block = TPCBlock(
                input_channels=current_channels,
                num_filters=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                pointwise_channels=pointwise_channels,
                dropout=dropout,
            )

            self.tpc_blocks.append(block)

            # Update channels for next layer (due to skip connections)
            current_channels = block.output_channels

        # Final classification/regression layer
        output_size = self.get_output_size()
        self.fc = nn.Linear(current_channels, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation through TPC model.

        Processes input features through embeddings, TPC blocks, temporal pooling,
        and final prediction layer. Handles variable-length sequences via masking.

        Args:
            **kwargs: Dictionary containing:
                - Feature keys (from self.feature_keys): Input feature tensors
                - Label key (from self.label_key): Ground truth labels (optional)
                - "embed" (bool): If True, return patient embeddings (optional)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "y_prob": Predicted probabilities/values, shape (batch_size, output_size)
                - "logit": Raw model outputs before activation, shape (batch_size, output_size)
                - "loss": Loss value (scalar), only if labels provided
                - "y_true": Ground truth labels, only if labels provided
                - "embed": Patient embeddings, only if embed=True in kwargs

        Examples:
            >>> # Forward pass during training
            >>> batch = {
            ...     "conditions": tensor([[1, 2, 3], [4, 5, 0]]),
            ...     "procedures": tensor([[10, 11], [12, 13]]),
            ...     "label": tensor([3.5, 7.2])
            ... }
            >>> outputs = model(**batch)
            >>> loss = outputs["loss"]
            >>> predictions = outputs["y_prob"]
        """
        # Get batch size and device from first feature
        batch_size = kwargs[self.feature_keys[0]].shape[0]
        device = kwargs[self.feature_keys[0]].device

        # === Step 1: Embed all features using EmbeddingModel ===
        embedded_dict = self.embedding_model(kwargs)

        # === Step 2: Find maximum sequence length across all features ===
        max_seq_len = 0
        for feature_key in self.feature_keys:
            seq_len = embedded_dict[feature_key].shape[1]
            max_seq_len = max(max_seq_len, seq_len)

        # === Step 3: Pad sequences to same length and create masks ===
        padded_features = []
        padded_masks = []

        for feature_key in self.feature_keys:
            embedded = embedded_dict[feature_key]  # (batch, seq_len, embedding_dim)
            current_len = embedded.shape[1]

            # Pad if necessary
            if current_len < max_seq_len:
                pad_len = max_seq_len - current_len
                embedded = F.pad(embedded, (0, 0, 0, pad_len), value=0)

            padded_features.append(embedded)

            # Create mask (1 for valid tokens, 0 for padding)
            mask = torch.ones(batch_size, max_seq_len, device=device)
            if current_len < max_seq_len:
                mask[:, current_len:] = 0

            # Also check for actual padding in original data
            # Sum over embedding dimension: if all zeros, it's padding
            original_mask = (embedded_dict[feature_key].sum(dim=-1) != 0).float()
            mask[:, :current_len] = original_mask

            padded_masks.append(mask)

        # === Step 4: Concatenate features ===
        # Shape: (batch, max_seq_len, num_features * embedding_dim)
        x = torch.cat(padded_features, dim=2)

        # Transpose to (batch, channels, seq_length) for convolutions
        x = x.permute(0, 2, 1)

        # Combined mask: valid if ANY feature is valid at that timestep
        combined_mask = torch.stack(padded_masks, dim=1).max(dim=1)[0]

        # === Step 5: Process through TPC blocks ===
        for block in self.tpc_blocks:
            x = block(x)

        # === Step 6: Global temporal pooling ===
        # x shape: (batch, final_channels, seq_length)
        x = x.permute(0, 2, 1)  # (batch, seq_length, final_channels)

        # Get indices of last valid timestep for each sequence
        seq_lengths = combined_mask.sum(dim=1).long()  # (batch,)
        last_indices = (seq_lengths - 1).clamp(min=0)

        # Extract last valid representation for each sequence
        patient_embedding = x[torch.arange(batch_size, device=device), last_indices]
        # Shape: (batch, final_channels)

        # === Step 7: Final prediction ===
        logits = self.fc(patient_embedding)

        # === Step 8: Compute loss and predictions ===
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]

            # Handle label shapes - ensure proper dimensions
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)

            # Compute loss based on task mode and loss function choice
            if self.mode == "regression":
                # Use PyHealth's standard MSE loss
                loss = F.mse_loss(logits, y_true)
            else:
                # Classification: use standard cross-entropy from PyHealth
                loss_fn = self.get_loss_function()
                if self.mode == "binary":
                    loss = loss_fn(logits, y_true.float())
                else:
                    loss = loss_fn(logits, y_true.squeeze(1).long())
        else:
            loss = None
            y_true = None

        # Prepare predictions (convert logits to probabilities based on mode)
        y_prob = self.prepare_y_prob(logits)

        # Build output dictionary
        results = {"y_prob": y_prob, "logit": logits}

        if loss is not None:
            results["loss"] = loss
        if y_true is not None:
            results["y_true"] = y_true

        # Optionally return embeddings for analysis
        if kwargs.get("embed", False):
            results["embed"] = patient_embedding

        return results


if __name__ == "__main__":
    """Test TPC model with dummy data.

    This main function demonstrates how to use the TPC model with PyHealth,
    showing both standard PyHealth loss and the paper's masked MSLE loss.
    """
    # Create dummy dataset
    from pyhealth.datasets import SampleDataset
    from pyhealth.processors import SequenceProcessor
    from pyhealth.datasets import SampleDataset, split_by_patient, get_dataloader

    samples = [
        {
            "patient_id": f"patient-{i}",
            "visit_id": f"visit-{i}",
            "conditions": [f"ICD{j}" for j in range(3)],
            "procedures": [f"PROC{j}" for j in range(2)],
            "label": float(i % 10 + 1),  # LoS between 1-10 days
        }
        for i in range(100)
    ]

    # Create PyHealth SampleDataset with proper schemas
    dataset = SampleDataset(
        samples=samples,
        dataset_name="test_tpc",
        task_name="length_of_stay",
        input_schema={
            "conditions": SequenceProcessor,
            "procedures": SequenceProcessor,
        },
        output_schema={"label": "regression"},
    )

    # Initialize the model
    print("=" * 80)
    print("TPC Model Test - Standard PyHealth Loss")
    print("=" * 80)

    # Standard PyHealth loss (default)
    model_standard = TPC(
        dataset=dataset,
        embedding_dim=128,         # Embedding dimension for medical codes
        num_layers=3,              # Number of TPC blocks (default from paper)
        num_filters=8,             # Filters per temporal convolution (paper default)
        pointwise_channels=128,    # Channels for pointwise convolutions (paper default)
        kernel_size=4,             # Temporal convolution kernel size (paper default)
        dropout=0.3,               # Dropout rate (paper default)
    )

    print(f"Model initialized successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
    print(f"Feature keys: {model_standard.feature_keys}")
    print(f"Label key: {model_standard.label_key}")
    print(f"Mode: {model_standard.mode}")
    print(f"Output size: {model_standard.get_output_size()}")
    print(f"Loss function: PyHealth standard MSE")

    # Spliting the data train/val/test
    train_dataset, val_dataset, test_dataset = split_by_patient(
        dataset, ratios=[0.7, 0.15, 0.15], seed=42
    )

    print("\nDataset Split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create dataloader for the train dataset
    train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True)

    # Get a real batch from the dataloader
    batch = next(iter(train_loader))

    # Forward pass
    print("\n" + "=" * 80)
    print("Running forward pass...")
    print("=" * 80)
    outputs = model_standard(**batch)
    print(f"\nLoss (Standard MSE): {outputs['loss'].item():.4f}")
    print(f"Predictions shape: {outputs['y_prob'].shape}")
    print(f"Sample predictions: {outputs['y_prob'][:3].flatten().tolist()}")
    print(f"Sample ground truth: {outputs['y_true'][:3].flatten().tolist()}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)



















