import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel
from typing import Callable, Dict, Optional, cast

class MultiViewContrastiveModel(BaseModel):
    """A generic, plug-and-play Multi-View Contrastive Learning (MVCL) model.
    This model supports an arbitrary number of views and modalities by dynamically 
    constructing its architecture based on the provided dictionaries of encoders, 
    projectors, and augmentations. It implements hierarchical cross-view fusion 
    via Multi-Head Attention and uses a 2N x 2N NT-Xent (InfoNCE) contrastive loss, 
    aligning with the conceptual framework of Oh and Bui (2025) and TFC-pretraining.

    Args:
        dataset (SampleDataset): The PyHealth dataset object.
        encoders (nn.ModuleDict): A dictionary mapping view names (e.g., "xt", "xf") 
            to their respective PyTorch encoder modules (e.g., Transformer, CNN).
        projectors (nn.ModuleDict): A dictionary mapping view names to their initial 
            feature projection layers (e.g., mapping raw inputs to `hidden_dim`).
        augmentations (Dict[str, Callable]): A dictionary mapping view names to 
            their specific data augmentation functions (e.g., jittering, frequency masking).
        pos_encoders (nn.ModuleDict, optional): A dictionary mapping view names to 
            their positional encoding modules. If a view is not included, it defaults 
            to `nn.Identity()`. Useful for sequence models. Defaults to an empty dict.
        hidden_dim (int, optional): The hidden dimension size for the embeddings, 
            MHA fusion, and projections. Defaults to 128.
        training_stage (str, optional): The current stage of the model. Accepts 
            "pretrain" (contrastive representation learning) or "finetune" 
            (downstream classification). Defaults to "pretrain".
        num_classes (int, optional): The number of target classes for the downstream 
            classification task. Defaults to 3.
        lambda_cl (float, optional): The weight/penalty hyperparameter for the contrastive 
            loss during the fine-tuning stage. Defaults to 0.1.
        tau (float, optional): The temperature parameter for the NT-Xent loss function. 
            Defaults to 0.07.

    Outputs (dict):
        During "pretrain":
            - loss (torch.Tensor): The aggregated 2N x 2N NT-Xent contrastive loss across all views.
            - z_{view} (torch.Tensor): The final fused embeddings for each provided view.
        During "finetune":
            - loss (torch.Tensor): The combined cross-entropy loss and contrastive penalty.
            - logit (torch.Tensor): The raw classification logits.
            - y_prob (torch.Tensor): The predicted class probabilities.
            - y_true (torch.Tensor): The ground truth labels.
            
    Example:
        >>> encoders = nn.ModuleDict({"v1": CNNEncoder(), "v2": TextEncoder()})
        >>> projectors = nn.ModuleDict({"v1": nn.Linear(3, 128), "v2": nn.Linear(768, 128)})
        >>> augs = {"v1": image_jitter, "v2": text_mask}
        >>> model = MultiViewContrastiveModel(dataset, encoders, projectors, augs)
    """
    def __init__(
        self, 
        dataset, 
        encoders: nn.ModuleDict, 
        projectors: nn.ModuleDict,
        augmentations: Dict[str, Callable],
        pos_encoders: Optional[nn.ModuleDict] = None,
        hidden_dim: int = 128,
        training_stage: str = "pretrain", 
        num_classes: int = 3,
        lambda_cl: float = 0.1,
        tau: float = 0.07,
        **kwargs
    ):
        super().__init__(dataset=dataset)
        self.training_stage = training_stage
        self.mode = ""
        
        # Disable inference metrics during pre-training
        if self.training_stage == "pretrain":
            self.mode = None 
            
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lambda_cl = lambda_cl
        self.tau = tau
        self.view_names = list(encoders.keys())

        self.encoders = encoders
        self.projectors = projectors
        self.augmentations = augmentations
        self.pos_encoders = pos_encoders if pos_encoders is not None else nn.ModuleDict({
            view: nn.Identity() for view in self.view_names
        })
        
        # Generic Fusion: Applies attention across the V different views
        self.fusion_mha = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.fusion_layer_norm = nn.LayerNorm(self.hidden_dim)

        # Dynamic feature-specific projectors (F_k)
        self.F_projectors = nn.ModuleDict({
            view: nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            ) for view in self.view_names
        })

        # Classifier dynamically sizes itself based on the number of views
        self.classifier_mha = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.view_names), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def ntxent_loss(self, zis: torch.Tensor, zjs: torch.Tensor, tau: float) -> torch.Tensor:
        """2N x 2N NTXentLoss."""
        batch_size = zis.size(0)
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = torch.mm(representations, representations.T)
        
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=zis.device))
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)
        
        logits = torch.cat((positives, negatives), dim=1) / tau
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=zis.device)
        
        return F.cross_entropy(logits, labels, reduction="sum") / (2 * batch_size)

    def _forward_features(self, views_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded_views = {}
        
        # 1. Project and Encode (keep sequence length for MHA)
        for view in self.view_names:
            x = views_data[view]
            x = self.projectors[view](x)
            if view in self.pos_encoders:
                x = self.pos_encoders[view](x)
            h = self.encoders[view](x)
            encoded_views[view] = h  # Shape: [N, L, hidden_dim]

        # 2. Cross-View Fusion
        batch_size = encoded_views[self.view_names[0]].shape[0]
        seq_length = encoded_views[self.view_names[0]].shape[1]
        num_views = len(self.view_names)

        # Stack into [N, num_views, L, hidden_dim]
        H = torch.stack([encoded_views[v] for v in self.view_names], dim=1)
        
        H_permuted = H.permute(0, 2, 1, 3).contiguous()
        
        # Flatten Batch and Sequence length together: [N * L, num_views, hidden_dim]
        H_flat = H_permuted.view(batch_size * seq_length, num_views, self.hidden_dim)

        MHA_out, _ = self.fusion_mha(H_flat, H_flat, H_flat)
        H_out = self.fusion_layer_norm(MHA_out + H_flat)

        # Reshape back to [N, L, num_views, hidden_dim]
        H_out = H_out.view(batch_size, seq_length, num_views, self.hidden_dim)
        
        # 3. Concatenate and Project 
        final_zs = {}
        for i, view in enumerate(self.view_names):
            h_pre = encoded_views[view].mean(dim=1)   # Pre-interaction
            
            # Extract from dimension 2 (num_views dimension)
            h_post = H_out[:, :, i, :].mean(dim=1)    # Post-interaction
            
            # Concatenation of pre and post features!
            h_pool = torch.cat([h_pre, h_post], dim=-1) # Shape: [N, hidden_dim * 2]
            
            final_zs[view] = self.F_projectors[view](h_pool)
            
        return final_zs

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        views_data = {view: self._prepare_tensor(kwargs.get(view)) for view in self.view_names}

        if self.training_stage == "pretrain":
            # Augment
            augmented_views = {
                view: self.augmentations[view](views_data[view]) for view in self.view_names
            }

            # Encode
            zs = self._forward_features(views_data)
            zs_aug = self._forward_features(augmented_views)
            
            # Dynamic Loss Calculation
            loss = torch.tensor(0.0, device=self.device)
            for view in self.view_names:
                loss += self.ntxent_loss(zs[view], zs_aug[view], self.tau)
                   
            result = {"loss": loss}
            result.update({f"z_{v}": zs[v] for v in self.view_names}) # Add embeddings
            return result
            
        elif self.training_stage == "finetune":
            zs = self._forward_features(views_data)
            
            # Stack and fuse for classification
            stacked_emb = torch.stack([zs[v] for v in self.view_names], dim=1)
            attn_out, _ = self.classifier_mha(stacked_emb, stacked_emb, stacked_emb)
            emb = attn_out + stacked_emb
            
            z_combined = emb.reshape(emb.size(0), -1) 
            logits = self.classifier(z_combined)
            
            label_key = self.label_keys[0]
            y_true = cast(torch.Tensor, kwargs[label_key]).to(logits.device)
            loss_ce = self.get_loss_function()(logits, y_true)
            
            # Contrastive Penalty
            augmented_views = {v: self.augmentations[v](views_data[v]) for v in self.view_names}
            zs_aug = self._forward_features(augmented_views)
            
            loss_cl = torch.tensor(0.0, device=self.device)
            for view in self.view_names:
                loss_cl += self.ntxent_loss(zs[view], zs_aug[view], self.tau)
            
            total_loss = (self.lambda_cl * loss_cl) + loss_ce
            
            return {
                "loss": total_loss,
                "logit": logits,
                "y_prob": self.prepare_y_prob(logits), 
                "y_true": y_true
            }
        return {}
        
    def _prepare_tensor(self, x) -> torch.Tensor:
        if isinstance(x, list):
            import numpy as np
            x = torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.from_numpy(np.stack(x))
        return x.float().to(self.device)