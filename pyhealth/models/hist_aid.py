import torch
import torch.nn as nn
from typing import Dict
from transformers import ViTModel, BertModel
from pyhealth.models import BaseModel


class HistAID(BaseModel):
    """HIST-AID: Dual-Stream Transformer with Transformer-based Fusion.

    This model implements the HIST-AID architecture using Hugging Face
    backbones and a transformer layer to fuse vision and text tokens.
    """

    def __init__(
        self,
        dataset,
        vision_model: str = "google/vit-base-patch16-224-in21k",
        text_model: str = "bert-base-uncased",
        fusion_dim: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset, **kwargs)

        # 1. Vision Stream
        self.vision_encoder = ViTModel.from_pretrained(vision_model)
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, fusion_dim)

        # 2. Text/Temporal Stream
        self.text_encoder = BertModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=8, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=2)

        # 3. Transformer Fusion Layer
        self.fusion_transformer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, batch_first=True
        )

        self.classifier = nn.Linear(fusion_dim, self.num_labels)

    def forward(
        self,
        image: torch.Tensor,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal fusion."""
        # Vision Embedding: Extract [CLS] token
        v_out = self.vision_encoder(pixel_values=image).last_hidden_state[:, 0, :]
        v_token = self.vision_proj(v_out).unsqueeze(1)  # [batch, 1, fusion_dim]

        # Text/Temporal Path: Flatten batch/seq for BERT
        b, s, l = history_input_ids.shape
        t_out = self.text_encoder(
            input_ids=history_input_ids.view(-1, l),
            attention_mask=history_attention_mask.view(-1, l),
        )
        t_feat = t_out.last_hidden_state[:, 0, :].view(b, s, -1)
        t_feat = self.temporal_transformer(t_feat).mean(dim=1)
        t_token = self.text_proj(t_feat).unsqueeze(1)  # [batch, 1, fusion_dim]

        # Transformer Fusion: Treat Image and History as tokens in a sequence
        fusion_input = torch.cat([v_token, t_token], dim=1)  # [batch, 2, fusion_dim]
        fused_seq = self.fusion_transformer(fusion_input)
        fused_feat = fused_seq.mean(dim=1)  # Aggregate cross-modal representation

        logits = self.classifier(fused_feat)
        y_prob = torch.sigmoid(logits)
        loss = nn.BCEWithLogitsLoss()(logits, label.float())

        return {"loss": loss, "y_prob": y_prob, "y_true": label}