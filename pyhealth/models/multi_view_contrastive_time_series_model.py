"""Multi-view contrastive time-series model for PyHealth datasets."""

import math
from typing import Any, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * 
                             (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, hidden_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiViewContrastiveTimeSeriesModel(BaseModel):
    """Multi-view contrastive model for time-series tensors.

    This model follows the multi-view contrastive learning setup used for
    time-domain, derivative, and frequency-domain views. Each view is projected,
    encoded with a Transformer encoder, fused with cross-view attention, and
    used for either contrastive pretraining or downstream classification.

    Args:
        dataset (SampleDataset): Dataset with ``xt``, ``xd``, and ``xf`` tensor
            inputs and one output label.
        training_stage (str): Training stage, either ``"pretrain"`` for
            contrastive representation learning or ``"finetune"`` for
            classification. Default is ``"pretrain"``.
        num_classes (int): Number of classes used by the classification head.
            Default is 3.
        **kwargs: Additional keyword arguments kept for PyHealth model API
            compatibility.

    Attributes:
        hidden_dim: Hidden dimension used by projections, encoders, and fusion.
        lambda_cl: Weight for the contrastive penalty during finetuning.
        tau: Temperature used by the NT-Xent contrastive loss.

    Examples:
        >>> from pyhealth.models import MultiViewContrastiveTimeSeriesModel
        >>> model = MultiViewContrastiveTimeSeriesModel(
        ...     dataset=sample_dataset,
        ...     training_stage="pretrain",
        ...     num_classes=5,
        ... )
        >>> output = model(xt=xt, xd=xd, xf=xf)
        >>> sorted(output.keys())
        ['loss', 'z_d', 'z_f', 'z_t']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        training_stage: str = "pretrain",
        num_classes: int = 3,
        **kwargs: Any
    ):
        super().__init__(dataset=dataset)
        self.hidden_dim = 128
        self.training_stage = training_stage
        self.lambda_cl = 0.1
        self.tau = 0.07
        self.num_classes = num_classes

        self.temporal_projection = nn.Linear(1, self.hidden_dim)
        self.derivative_projection = nn.Linear(1, self.hidden_dim)
        self.frequency_projection = nn.Linear(1, self.hidden_dim)

        self.pos_encoder = PositionalEncoding(self.hidden_dim, dropout=0.1)

        def make_encoder() -> nn.TransformerEncoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=4, batch_first=True, dropout=0.2
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=3)
            
        self.encoder_t: nn.TransformerEncoder = make_encoder()
        self.encoder_d: nn.TransformerEncoder = make_encoder()
        self.encoder_f: nn.TransformerEncoder = make_encoder()

        self.fusion_mha: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=4, batch_first=True
        )
        self.fusion_layer_norm: nn.LayerNorm = nn.LayerNorm(self.hidden_dim)

        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            
        self.temporal_feature_projector: nn.Sequential = projector()
        self.derivative_feature_projector: nn.Sequential = projector()
        self.frequency_feature_projector: nn.Sequential = projector()

        self.classifier_mha = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=1, batch_first=True
        )
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_classes)

    def augment_time(self, x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Time-domain jitter augmentation"""
        noise = torch.randn_like(x) * std
        return x + noise
        
    def augment_freq(self, sample: torch.Tensor, perturb_ratio: float = 0.05) -> torch.Tensor:
        """Frequency-domain augmentation (remove and add frequencies)"""
        removed_frequency = self.remove_frequency(sample, perturb_ratio)
        added_frequency = self.add_frequency(sample, perturb_ratio)
        return removed_frequency + added_frequency

    def remove_frequency(self, x: torch.Tensor, perturb_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > perturb_ratio
        return x * mask

    def add_frequency(self, x: torch.Tensor, perturb_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > (1 - perturb_ratio)
        max_amplitude = x.max()
        random_amplitude = torch.rand(mask.shape, device=x.device) * (max_amplitude * 0.1)
        perturbation = mask * random_amplitude
        return x + perturbation
        
    def ntxent_loss(self, zis: torch.Tensor, zjs: torch.Tensor, tau: float) -> torch.Tensor:
        """2N x 2N NTXentLoss aligned with the TFC implementation."""
        batch_size = zis.size(0)
        
        # Normalize the representations
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        # Concatenate into 2N
        representations = torch.cat([zjs, zis], dim=0) # [2N, hidden_dim]
        
        # Compute 2Nx2N cosine similarity matrix
        similarity_matrix = torch.mm(representations, representations.T)
        
        # Extract the positive pairs (offset by batch_size)
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        
        # Create a mask to remove self-similarity (the diagonal)
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=zis.device))
        
        # Extract negatives (everything except the diagonal)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)
        
        # Concatenate logits: [positives, negatives]
        logits = torch.cat((positives, negatives), dim=1)
        logits /= tau
        
        # The positive sample is always at index 0 for each row
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=zis.device)
        
        # PyTorch CrossEntropy applies the log-softmax calculation
        loss = F.cross_entropy(logits, labels, reduction="sum")
        
        return loss / (2 * batch_size)
    
    def _forward_features(self, x_t: torch.Tensor, x_d: torch.Tensor, x_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = self.temporal_projection(x_t)
        x_d = self.derivative_projection(x_d)
        x_f = self.frequency_projection(x_f)

        x_t = self.pos_encoder(x_t)
        x_d = self.pos_encoder(x_d)
        x_f = self.pos_encoder(x_f)

        h_t = self.encoder_t(x_t)
        h_d = self.encoder_d(x_d)
        h_f = self.encoder_f(x_f)

        batch_size, seq_length, _ = h_t.shape
        view_sequence = torch.stack([h_t, h_d, h_f], dim=2)
        flattened_views = (
            view_sequence
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * 3, seq_length, self.hidden_dim)
        )

        attention_output, _ = self.fusion_mha(
            flattened_views, flattened_views, flattened_views
        )
        fused_views = self.fusion_layer_norm(attention_output + flattened_views)

        fused_views = fused_views.view(batch_size, 3, seq_length, self.hidden_dim).permute(0, 2, 1, 3)
        h_t_star, h_d_star, h_f_star = (
            fused_views[:, :, 0, :],
            fused_views[:, :, 1, :],
            fused_views[:, :, 2, :],
        )
        
        # Pool across sequence length and concatenate with pre-interaction features
        h_t_pool = torch.cat([h_t.mean(dim=1), h_t_star.mean(dim=1)], dim=-1)
        h_d_pool = torch.cat([h_d.mean(dim=1), h_d_star.mean(dim=1)], dim=-1)
        h_f_pool = torch.cat([h_f.mean(dim=1), h_f_star.mean(dim=1)], dim=-1)
        
        z_t = self.temporal_feature_projector(h_t_pool)
        z_d = self.derivative_feature_projector(h_d_pool)
        z_f = self.frequency_feature_projector(h_f_pool)
        
        return z_t, z_d, z_f

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        temporal_tensor = self._prepare_tensor(kwargs.get("xt"))  # [N, L, 1]
        derivative_tensor = self._prepare_tensor(kwargs.get("xd"))  # [N, L, 1]
        frequency_tensor = self._prepare_tensor(kwargs.get("xf"))  # [N, L, 1]

        if self.training_stage == "pretrain":
            x_t_aug = self.augment_time(temporal_tensor)
            x_d_aug = self.augment_time(derivative_tensor)
            x_f_aug = self.augment_freq(frequency_tensor)

            z_t, z_d, z_f = self._forward_features(temporal_tensor, derivative_tensor, frequency_tensor)
            z_t_aug, z_d_aug, z_f_aug = self._forward_features(x_t_aug, x_d_aug, x_f_aug)
            
            loss = self.ntxent_loss(z_t, z_t_aug, self.tau) + \
                   self.ntxent_loss(z_d, z_d_aug, self.tau) + \
                   self.ntxent_loss(z_f, z_f_aug, self.tau)
                   
            # Dict strictly containing torch.Tensor
            return {
                "loss": loss, 
                "z_t": z_t, 
                "z_d": z_d, 
                "z_f": z_f
            }
            
        elif self.training_stage == "finetune":
            z_t, z_d, z_f = self._forward_features(temporal_tensor, derivative_tensor, frequency_tensor)
            
            # Cross-view attention for classification
            stacked_emb = torch.stack([z_t, z_d, z_f], dim=1) # [batch_size, 3, hidden_dim]
            attention_output, _ = self.classifier_mha(stacked_emb, stacked_emb, stacked_emb)
            emb = attention_output + stacked_emb # Residual connection
            
            z_combined = emb.reshape(emb.size(0), -1) # Flatten to [batch_size, 3 * hidden_dim]
            logits = self.classifier(z_combined)
            
            # Use PyHealth's automatic label parsing
            label_key = self.label_keys[0]
            y_true = cast(torch.Tensor, kwargs[label_key])
            y_true = y_true.to(logits.device)
            
            # Use PyHealth's automatic loss function mapping
            criterion = self.get_loss_function()
            # Cross entropy expects raw logits, not argmax class indices.
            loss_ce = criterion(logits, y_true)
            
            # Contrastive penalty during finetuning
            x_t_aug = self.augment_time(temporal_tensor)
            x_d_aug = self.augment_time(derivative_tensor)
            x_f_aug = self.augment_freq(frequency_tensor)

            z_t_aug, z_d_aug, z_f_aug = self._forward_features(x_t_aug, x_d_aug, x_f_aug)
            loss_cl = self.ntxent_loss(z_t, z_t_aug, self.tau) + \
                      self.ntxent_loss(z_d, z_d_aug, self.tau) + \
                      self.ntxent_loss(z_f, z_f_aug, self.tau)
            
            total_loss = (self.lambda_cl * loss_cl) + loss_ce
            
            # Return PyHealth's expected dictionary schema
            return {
                "loss": total_loss,
                "logit": logits,
                "y_prob": self.prepare_y_prob(logits), # Autocast to prob
                "y_true": y_true
            }
        return {}
        
    def _prepare_tensor(self, x) -> torch.Tensor:
        """Converts lists to batched tensors, enforces float32, and moves to device."""
        if isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                x = torch.stack(x)
            else:
                x = torch.from_numpy(np.stack(x))
                
        # Enforce standard float precision and push to GPU
        return x.float().to(self.device)