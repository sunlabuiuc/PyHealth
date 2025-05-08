"""
Jacob Lizarraga
jdl8
MedFuse: Multi-modal Fusion with Clinical Time-series Data and Chest X-ray Images
https://arxiv.org/abs/2207.07027

This is an implementation of the MedFuse model, which uses an LSTM-based fusion approach 
to integrate clinical time-series data and chest X-ray images for healthcare prediction tasks.
The model can accommodate both uni-modal and multi-modal inputs, making it suitable for
partially paired healthcare data where not all samples have both modalities.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class MedFuseModel(BaseModel):
    """MedFuse model for multi-modal fusion of clinical time-series data and medical images.
    
    MedFuse treats multi-modal representation as a sequence of uni-modal representations,
    which allows it to handle missing modalities naturally through LSTM's capability
    to process variable-length sequences.
    
    Args:
        dataset (SampleDataset): The dataset to train the model.
        ehr_dim (int): Dimension of the clinical time-series features after initial processing.
        img_dim (int): Dimension of the image features after initial processing.
        hidden_dim (int): Hidden dimension size for the LSTM layers.
        dropout (float): Dropout rate for regularization.
        num_lstm_layers (int): Number of stacked LSTM layers.
        use_attention (bool): Whether to use attention mechanism for fusion.
    """
    
    def __init__(
        self,
        dataset: SampleDataset,
        ehr_dim: int = 76,
        img_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_lstm_layers: int = 2,
        use_attention: bool = False,
    ):
        super(MedFuseModel, self).__init__(
            dataset=dataset,
        )
        
        self.ehr_dim = ehr_dim
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention
        
        # Verify we have two feature keys (ehr and image)
        assert len(self.feature_keys) == 2, "MedFuse requires exactly two feature keys (ehr and image)"
        self.ehr_key = self.feature_keys[0]
        self.img_key = self.feature_keys[1]
        
        # Verify we have exactly one label key
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        
        # Get output mode and size
        self.mode = self.dataset.output_schema[self.label_key]
        self.output_size = self.get_output_size()
        
        # Define EHR encoder
        self.ehr_encoder = nn.LSTM(
            input_size=ehr_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Define image encoder (assuming pre-processed image features)
        self.img_projection = nn.Linear(img_dim, hidden_dim)
        
        # Define fusion module
        if use_attention:
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            
            # Modality importance weighting
            self.modality_attn = nn.Sequential(
                nn.Linear(hidden_dim*2, 2),
                nn.Softmax(dim=1)
            )
        
        # Fusion LSTM
        self.fusion_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, self.output_size)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation for MedFuse model.
        
        Processes clinical time-series data and medical images, performs fusion,
        and outputs predictions.
        
        Args:
            **kwargs: Dictionary containing input features and labels.
                Must include the keys specified in self.feature_keys and self.label_keys.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss: The loss value
                - y_prob: The predicted probabilities
                - y_true: The ground truth labels
                - modality_weights: (Optional) Attention weights between modalities
        """
        # Get inputs
        ehr_data = kwargs[self.ehr_key].to(self.device)  # [batch_size, seq_len, ehr_dim]
        img_data = kwargs[self.img_key].to(self.device)  # [batch_size, img_dim]
        y_true = kwargs[self.label_key].to(self.device)  # [batch_size, output_size]
        
        # Get modality availability mask (1 if image available, 0 if not)
        # Assuming img_data has zeros if not available
        has_image = torch.sum(torch.abs(img_data), dim=1) > 0
        
        # Process EHR data
        ehr_lengths = kwargs.get("ehr_lengths", torch.tensor([ehr_data.size(1)] * ehr_data.size(0)))
        ehr_lengths = ehr_lengths.to(self.device)
        
        # Pack the padded sequences
        packed_ehr = nn.utils.rnn.pack_padded_sequence(
            ehr_data, 
            ehr_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process with LSTM
        _, (ehr_hidden, _) = self.ehr_encoder(packed_ehr)
        
        # Get the last layer's hidden state
        ehr_features = ehr_hidden[-1]  # [batch_size, hidden_dim]
        
        # Process image data
        img_features = self.img_projection(img_data)  # [batch_size, hidden_dim]
        
        # Zero out img_features where image is not available
        img_features = img_features * has_image.unsqueeze(1).float()
        
        # Create modality sequence
        batch_size = ehr_data.size(0)
        
        # Different fusion strategies
        if self.use_attention:
            # Stack features for attention
            stacked_features = torch.stack([ehr_features, img_features], dim=1)  # [batch_size, 2, hidden_dim]
            
            # Calculate modality attention weights
            modality_weights = self.modality_attn(
                torch.cat([ehr_features, img_features], dim=1)
            )  # [batch_size, 2]
            
            # Apply attention
            attn_output, _ = self.attention(
                stacked_features, 
                stacked_features, 
                stacked_features,
                key_padding_mask=~has_image.unsqueeze(1).repeat(1, 2)
            )
            
            # Use the attended features
            fused_features = attn_output[:, 0, :]  # [batch_size, hidden_dim]
            
        else:
            # Use LSTM fusion
            # Prepare sequences and lengths for LSTM
            seq_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
            seq_lengths[has_image] = 2  # Two elements in sequence if image is available
            
            # Create sequences - always start with EHR features
            sequences = []
            for i in range(batch_size):
                if has_image[i]:
                    # Both modalities available
                    seq = torch.stack([ehr_features[i], img_features[i]], dim=0)  # [2, hidden_dim]
                else:
                    # Only EHR available
                    seq = ehr_features[i].unsqueeze(0)  # [1, hidden_dim]
                sequences.append(seq)
            
            # Pad and pack sequences
            padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            packed_seqs = nn.utils.rnn.pack_padded_sequence(
                padded_seqs, 
                seq_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            # Process with fusion LSTM
            _, (fusion_hidden, _) = self.fusion_lstm(packed_seqs)
            
            # Get the final hidden state
            fused_features = fusion_hidden.squeeze(0)  # [batch_size, hidden_dim]
            
            # Set default modality weights for compatibility
            modality_weights = torch.ones(batch_size, 2, device=self.device) * 0.5
        
        # Apply dropout for regularization
        fused_features = self.dropout_layer(fused_features)
        
        # Output layer
        logits = self.output_layer(fused_features)
        
        # Compute loss
        loss = self.get_loss_function()(logits, y_true)
        
        # Prepare output probabilities
        y_prob = self.prepare_y_prob(logits)
        
        # Return results
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }
        
        if self.use_attention:
            results["modality_weights"] = modality_weights
            
        return results

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from pyhealth.datasets import SampleDataset

    class DummyProcessor:
        def __init__(self, size_value=1):
            self.size_value = size_value
        
        def size(self):
            return self.size_value
        
        def process(self, value):
            # Simple pass-through for testing
            return value
        
        def fit(self, samples, key):
            # No-op for testing
            pass
    
    # Create a dummy dataset with EHR and image features
    class DummyDataset(SampleDataset):
        def __init__(self):
            self.input_schema = {
                "ehr_data": "sequence",
                "img_data": "flat"
            }
            self.output_schema = {
                "label": "binary"
            }
            
            self.input_processors = {}
            self.output_processors = {
                "label": DummyProcessor(size_value=1)
            }
            
        def size(self):
            return 1
        
        def __len__(self):
            return 1
    
    # Create dummy dataset
    dataset = DummyDataset()
    
    # Create the model
    model = MedFuseModel(
        dataset=dataset,
        ehr_dim=76,
        img_dim=512,
        hidden_dim=256,
        dropout=0.3,
        num_lstm_layers=2,
        use_attention=True
    )
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    ehr_data = torch.randn(batch_size, seq_len, 76)
    img_data = torch.randn(batch_size, 512)
    # Make one sample not have image
    img_data[2] = torch.zeros(512)
    
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    ehr_lengths = torch.tensor([seq_len] * batch_size)
    
    # Forward pass
    output = model(
        ehr_data=ehr_data,
        img_data=img_data,
        label=labels,
        ehr_lengths=ehr_lengths
    )
    
    print("Loss:", output["loss"].item())
    print("Predictions shape:", output["y_prob"].shape)
    print("Modality weights:\n", output["modality_weights"])
    
    # Test backward pass
    output["loss"].backward()
    print("Backward pass successful")