import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pyhealth.models import BaseModel

class ClinicalTSFTransformer(BaseModel):
    """Clinical Time-Series Forecasting Transformer.

    This model handles multi-task learning by performing clinical 
    feature forecasting and classification (e.g., sepsis prediction) 
    simultaneously.

    Args:
        dataset: The PyHealth dataset object.
        feature_size: Number of input clinical features (default: 131).
        d_model: Internal embedding dimension (must be divisible by nhead).
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dataset: Any,
        feature_size: int = 131,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super(ClinicalTSFTransformer, self).__init__(dataset=dataset, **kwargs)

        self.feature_size = feature_size
        self.d_model = d_model

        # Projection layer to ensure d_model is divisible by nhead
        self.embedding = nn.Linear(feature_size, d_model)
        
        # Positional Encoding (Learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, 200, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-task heads
        self.forecasting_head = nn.Linear(d_model, feature_size)
        self.classification_head = nn.Linear(d_model, 1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            **kwargs: Dictionary containing 'x' [batch, time, features] 
                      and 'y' [batch] labels.
        """
        x = kwargs["x"]
        y_true = kwargs["y"]

        # 1. Embedding and Positional Encoding
        # Project 131 -> d_model (128)
        x_in = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        
        # 2. Transformer Encoder
        h = self.transformer(x_in)
        
        # 3. Multi-task Outputs
        # Map back to 131 for reconstruction
        recon = self.forecasting_head(h)
        # Classification based on the last hidden state
        logits = self.classification_head(h[:, -1, :])
        y_prob = torch.sigmoid(logits)

        # 4. Loss Calculation
        loss_cls = nn.BCEWithLogitsLoss()(logits.view(-1), y_true.float())
        loss_recon = nn.MSELoss()(recon, x)
        
        # Combined MTL loss (weighted)
        total_loss = loss_cls + (0.1 * loss_recon)

        return {
            "loss": total_loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "reconstruction": recon
        }