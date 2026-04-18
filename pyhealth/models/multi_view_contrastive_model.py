import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.models import BaseModel
from typing import cast

class MultiViewContrastiveModel(BaseModel):
    """A simple multi-view contrastive model for demonstration purposes."""

    def __init__(self, dataset, training_stage="pretrain", num_classes=3, **kwargs):
        super().__init__(dataset=dataset)
        self.hidden_dim = 128
        seq_length = 256
        self.training_stage = training_stage
        self.lambda_cl = 0.001
        self.tau = 0.07
        self.num_classes = num_classes

        self.proj_t = nn.Linear(1, self.hidden_dim)
        self.proj_d = nn.Linear(1, self.hidden_dim)
        self.proj_f = nn.Linear(1, self.hidden_dim)

        def make_encoder():
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=4, batch_first=True, dropout=0.2
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder_t = make_encoder()
        self.encoder_d = make_encoder()
        self.encoder_f = make_encoder()

        # Now we need MHA
        self.fusion_mha = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.fusion_layer_norm = nn.LayerNorm(self.hidden_dim)

        # Feature-specific projectors
        def projector():
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.F_t = projector()
        self.F_d = projector()
        self.F_f = projector()

        # self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 , 1024),
            nn.ReLU(),
            nn.Linear(1024 , 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def augment(self, x, std=0.1):
        # Placeholder for time-series augmentation (e.g., adding Gaussian noise)
        noise = torch.randn_like(x) * std
        return x + noise
        
    def data_transform_fd(self, sample: torch.Tensor, pertub_ratio: float = 0.05) -> torch.Tensor:
        aug_1 = self.remove_frequency(sample, pertub_ratio)
        aug_2 = self.add_frequency(sample, pertub_ratio)
        return aug_1 + aug_2

    def remove_frequency(self, x: torch.Tensor, pertub_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > pertub_ratio
        return x * mask

    def add_frequency(self, x: torch.Tensor, pertub_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > (1 - pertub_ratio)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape, device=x.device) * (max_amplitude * 0.1)
        pertub_matrix = mask * random_am
        return x + pertub_matrix
        
    def info_nce_loss(self, z_i, z_j, tau, symmetric=True):
        # Compute cosine similarity
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        sim_ij = torch.mm(z_i, z_j.T) / tau
        # Positive pairs are on the diagonal
        labels = torch.arange(sim_ij.size(0), device=sim_ij.device)
        loss_ij = F.cross_entropy(sim_ij, labels)
        
        if symmetric:
            sim_ji = torch.mm(z_j, z_i.T) / tau
            loss_ji = F.cross_entropy(sim_ji, labels)
            return (loss_ij + loss_ji) / 2
            
        return loss_ij
    
    def _forward_features(self, x_t, x_d, x_f):
        x_t = self.proj_t(x_t)
        x_d = self.proj_d(x_d)
        x_f = self.proj_f(x_f)

        h_t = self.encoder_t(x_t)
        h_d = self.encoder_d(x_d)
        h_f = self.encoder_f(x_f)

        batch_size, seq_length, _ = h_t.shape
        H = torch.stack([h_t, h_d, h_f], dim=2)
        H_flat = H.view(-1, 3, self.hidden_dim)

        MHA_out, _ = self.fusion_mha(H_flat, H_flat, H_flat)
        H_out = self.fusion_layer_norm(MHA_out + H_flat)

        H_out = H_out.view(batch_size, seq_length, 3, self.hidden_dim)
        h_t_star, h_d_star, h_f_star = H_out[:, :, 0, :], H_out[:, :, 1, :], H_out[:, :, 2, :]
        z_t = self.F_t(h_t_star).mean(dim=1)
        z_d = self.F_d(h_d_star).mean(dim=1)
        z_f = self.F_f(h_f_star).mean(dim=1)
        return z_t, z_d, z_f

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        temporal_tensor = self._prepare_tensor(kwargs.get("xt"))  # [N, L, 1]
        derivative_tensor = self._prepare_tensor(kwargs.get("xd"))  # [N, L, 1]
        frequency_tensor = self._prepare_tensor(kwargs.get("xf"))  # [N, L, 1]

        # --- Stage Routing ---
        if self.training_stage == "pretrain":
            x_t_aug = self.augment(temporal_tensor)
            x_d_aug = self.augment(derivative_tensor)
            x_f_aug = self.augment(frequency_tensor)

            # x_f_aug = self.data_transform_fd(frequency_tensor, 0.05)


            z_t, z_d, z_f = self._forward_features(temporal_tensor, derivative_tensor, frequency_tensor)
            z_t_aug, z_d_aug, z_f_aug = self._forward_features(x_t_aug, x_d_aug, x_f_aug)
            loss = self.info_nce_loss(z_t, z_t_aug, self.tau) + \
                   self.info_nce_loss(z_d, z_d_aug, self.tau) + \
                   self.info_nce_loss(z_f, z_f_aug, self.tau)
            # print (f"Pretrain Loss: {loss.item():.4f}")
            return {"loss": loss, 
                    "zs":[z_t, z_d, z_f]
                    } # Return the embeddings for each view
            
        elif self.training_stage == "finetune":
            z_t, z_d, z_f = self._forward_features(temporal_tensor, derivative_tensor, frequency_tensor)
            z_combined = torch.cat([z_t, z_d, z_f], dim=1)
            z_combined = z_combined.view(z_combined.size(0), -1) 
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
            x_t_aug = self.augment(temporal_tensor)
            x_d_aug = self.augment(derivative_tensor)
            x_f_aug = self.augment(frequency_tensor)

            # x_f_aug = self.data_transform_fd(frequency_tensor, 0.05)

            z_t_aug, z_d_aug, z_f_aug = self._forward_features(x_t_aug, x_d_aug, x_f_aug)
            loss_cl = self.info_nce_loss(z_t, z_t_aug, self.tau) + \
                      self.info_nce_loss(z_d, z_d_aug, self.tau) + \
                      self.info_nce_loss(z_f, z_f_aug, self.tau)
            
            total_loss = (self.lambda_cl * loss_cl) + loss_ce
            
            # Return PyHealth's expected dictionary schema
            return {
                "loss": total_loss,
                "logit": logits,
                "y_prob": self.prepare_y_prob(logits), # Autocast to prob
                "y_true": y_true
            }
        return {}
        
    def _prepare_tensor(self, x):
        """Converts lists to batched tensors, enforces float32, and moves to device."""
        if isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                x = torch.stack(x)
            else:
                import numpy as np
                x = torch.from_numpy(np.stack(x))
                
        # Enforce standard float precision and push to GPU
        return x.float().to(self.device)