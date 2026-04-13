"""Event-Based Contrastive Learning (EBCL) model.

This module implements a tensor-first PyHealth model for contrastive pretraining on
paired pre-event and post-event medical time-series windows.

The model is intentionally lightweight and self-contained so it is easy to test with
synthetic data and easy to plug into a downstream example script.

Expected tensor shapes:
    left_x:  [batch_size, seq_len, input_dim]
    right_x: [batch_size, seq_len, input_dim]

Optional masks:
    left_mask:  [batch_size, seq_len] with 1/True for valid tokens
    right_mask: [batch_size, seq_len] with 1/True for valid tokens

Forward return:
    {
        "left_emb": ...,
        "right_emb": ...,
        "patient_emb": ...,
        "logit": ... (if classifier head is enabled),
        "y_prob": ... (if classifier head is enabled),
        "loss": ...,
        "contrastive_loss": ...,
        "supervised_loss": ... (if labels are provided),
        "y_true": ... (if labels are provided),
    }
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class AttentionPooling(nn.Module):
    """Attention pooling over a sequence.

    Args:
        hidden_dim: Dimensionality of the sequence embeddings.

    Shape:
        x: [batch_size, seq_len, hidden_dim]
        mask: [batch_size, seq_len] where True means valid token
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool a sequence into a single vector.

        Args:
            x: Sequence embeddings of shape [batch_size, seq_len, hidden_dim].
            mask: Optional validity mask of shape [batch_size, seq_len].

        Returns:
            Pooled tensor of shape [batch_size, hidden_dim].
        """
        attn_logits = self.score(x).squeeze(-1)  # [B, T]

        if mask is not None:
            mask = mask.bool()
            attn_logits = attn_logits.masked_fill(~mask, -1e9)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, T]
        pooled = torch.einsum("bt,bth->bh", attn_weights, x)
        return pooled


class EBCL(BaseModel):
    """Event-Based Contrastive Learning model for medical time series.

    This model uses a shared encoder for left/pre-event and right/post-event windows,
    then applies a symmetric CLIP/InfoNCE-style contrastive loss over the batch.

    It also optionally supports a lightweight supervised prediction head on top of the
    concatenated pair embedding for downstream tasks.

    Args:
        dataset: Optional PyHealth dataset. Can be None for synthetic tests.
        input_dim: Feature dimension of each time step.
        hidden_dim: Transformer hidden dimension.
        projection_dim: Dimension of the final contrastive embedding.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward dimension inside the Transformer encoder.
        dropout: Dropout probability.
        temperature: Temperature for contrastive loss.
        classifier_out_dim: Output dimension for optional supervised head.
            Use 1 for binary classification, >1 for multiclass, or 0 to disable.
        max_seq_len: Maximum supported sequence length for positional embeddings.

    Example:
        >>> model = EBCL(
        ...     dataset=None,
        ...     input_dim=16,
        ...     hidden_dim=32,
        ...     projection_dim=32,
        ... )
        >>> left_x = torch.randn(4, 8, 16)
        >>> right_x = torch.randn(4, 8, 16)
        >>> out = model(left_x=left_x, right_x=right_x)
        >>> sorted(out.keys())
        ['contrastive_loss', 'left_emb', 'loss', 'patient_emb', 'right_emb']
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset],
        input_dim: int,
        hidden_dim: int = 32,
        projection_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.07,
        classifier_out_dim: int = 0,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__(dataset=dataset)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_prob = dropout
        self.temperature = temperature
        self.classifier_out_dim = classifier_out_dim
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.pooler = AttentionPooling(hidden_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.classifier: Optional[nn.Module]
        if classifier_out_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(2 * projection_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, classifier_out_dim),
            )
        else:
            self.classifier = None

        # Match PyHealth convention where possible.
        if classifier_out_dim == 1:
            self.mode = "binary"
        elif classifier_out_dim > 1:
            self.mode = "multiclass"
        else:
            self.mode = None

        self.binary_loss_fn = nn.BCEWithLogitsLoss()
        self.multiclass_loss_fn = nn.CrossEntropyLoss()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        for module in self.projection_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        if self.classifier is not None:
            for module in self.classifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_input_tensor(name: str, x: torch.Tensor) -> None:
        """Validate a sequence input tensor.

        Args:
            name: Name of the tensor for error messages.
            x: Tensor expected to have shape [batch_size, seq_len, input_dim].
        """
        if x.dim() != 3:
            raise ValueError(
                f"{name} must have shape [batch_size, seq_len, input_dim], "
                f"but got shape {tuple(x.shape)}."
            )

    @staticmethod
    def _normalize_mask(
        mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize a mask to boolean [batch_size, seq_len].

        If mask is None, all positions are treated as valid.

        Args:
            mask: Optional mask tensor.
            x: Input sequence tensor used to infer shape.

        Returns:
            Boolean mask of shape [batch_size, seq_len].
        """
        if mask is None:
            return torch.ones(
                x.size(0),
                x.size(1),
                dtype=torch.bool,
                device=x.device,
            )

        if mask.dim() != 2:
            raise ValueError(
                "Mask must have shape [batch_size, seq_len], "
                f"but got {tuple(mask.shape)}."
            )

        mask = mask.to(device=x.device).bool()

        # Ensure each sample has at least one valid token.
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True

        return mask

    def _encode_sequence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a sequence into a single normalized embedding.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            mask: Optional boolean/0-1 mask of shape [batch_size, seq_len].

        Returns:
            Normalized embedding tensor of shape [batch_size, projection_dim].
        """
        self._validate_input_tensor("x", x)
        mask = self._normalize_mask(mask, x)

        if x.size(1) > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
            mask = mask[:, : self.max_seq_len]

        h = self.input_proj(x)
        h = h + self.pos_embedding[:, : h.size(1), :]
        h = self.input_dropout(h)

        # PyTorch transformer uses True for padding in src_key_padding_mask.
        src_key_padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        pooled = self.pooler(h, mask=mask)
        emb = self.projection_head(pooled)
        emb = F.normalize(emb, dim=-1)

        return emb

    def contrastive_loss(
        self,
        left_emb: torch.Tensor,
        right_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric batch contrastive loss.

        Diagonal pairs are treated as positives. Off-diagonal pairs are negatives.

        Args:
            left_emb: Left embeddings [batch_size, projection_dim].
            right_emb: Right embeddings [batch_size, projection_dim].

        Returns:
            Scalar contrastive loss.
        """
        if left_emb.shape != right_emb.shape:
            raise ValueError(
                "left_emb and right_emb must have the same shape, got "
                f"{tuple(left_emb.shape)} vs {tuple(right_emb.shape)}."
            )

        logits = torch.matmul(left_emb, right_emb.transpose(0, 1))
        logits = logits / self.temperature

        targets = torch.arange(left_emb.size(0), device=left_emb.device)

        loss_left_to_right = F.cross_entropy(logits, targets)
        loss_right_to_left = F.cross_entropy(logits.transpose(0, 1), targets)

        return 0.5 * (loss_left_to_right + loss_right_to_left)

    def _compute_supervised_outputs(
        self,
        left_emb: torch.Tensor,
        right_emb: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute optional supervised outputs from pair embeddings.

        Args:
            left_emb: Left embeddings [batch_size, projection_dim].
            right_emb: Right embeddings [batch_size, projection_dim].
            y: Optional labels.

        Returns:
            Dictionary containing supervised outputs if classifier is enabled.
        """
        outputs: dict[str, torch.Tensor] = {}

        if self.classifier is None:
            return outputs

        patient_emb = torch.cat([left_emb, right_emb], dim=-1)
        logit = self.classifier(patient_emb)

        outputs["patient_emb"] = patient_emb
        outputs["logit"] = logit

        if self.classifier_out_dim == 1:
            y_prob = torch.sigmoid(logit)
            outputs["y_prob"] = y_prob
            if y is not None:
                y = y.float().view(-1, 1)
                outputs["supervised_loss"] = self.binary_loss_fn(logit, y)
                outputs["y_true"] = y
        else:
            y_prob = torch.softmax(logit, dim=-1)
            outputs["y_prob"] = y_prob
            if y is not None:
                y = y.long().view(-1)
                outputs["supervised_loss"] = self.multiclass_loss_fn(logit, y)
                outputs["y_true"] = y

        return outputs

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Public helper to encode a single sequence.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim].
            mask: Optional mask [batch_size, seq_len].

        Returns:
            Embedding tensor [batch_size, projection_dim].
        """
        return self._encode_sequence(x, mask)

    def forward(
        self,
        left_x: torch.Tensor,
        right_x: torch.Tensor,
        left_mask: Optional[torch.Tensor] = None,
        right_mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        supervised_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for contrastive pretraining and optional supervision.

        Args:
            left_x: Pre-event window tensor [batch_size, seq_len, input_dim].
            right_x: Post-event window tensor [batch_size, seq_len, input_dim].
            left_mask: Optional mask for left_x.
            right_mask: Optional mask for right_x.
            y: Optional labels for downstream prediction.
            supervised_weight: Weight applied to supervised loss if enabled.

        Returns:
            A dictionary containing embeddings and losses.
        """
        self._validate_input_tensor("left_x", left_x)
        self._validate_input_tensor("right_x", right_x)

        left_emb = self._encode_sequence(left_x, left_mask)
        right_emb = self._encode_sequence(right_x, right_mask)

        contrastive = self.contrastive_loss(left_emb, right_emb)

        outputs: dict[str, torch.Tensor] = {
            "left_emb": left_emb,
            "right_emb": right_emb,
            "contrastive_loss": contrastive,
            "loss": contrastive,
        }

        supervised_outputs = self._compute_supervised_outputs(
            left_emb=left_emb,
            right_emb=right_emb,
            y=y,
        )
        outputs.update(supervised_outputs)

        if "patient_emb" not in outputs:
            outputs["patient_emb"] = torch.cat([left_emb, right_emb], dim=-1)

        if "supervised_loss" in outputs:
            outputs["loss"] = contrastive + supervised_weight * outputs[
                "supervised_loss"
            ]

        return outputs

    def forward_from_embedding(
        self,
        left_x: torch.Tensor,
        right_x: torch.Tensor,
        left_mask: Optional[torch.Tensor] = None,
        right_mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        supervised_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compatibility wrapper for PyHealth interpretability APIs.

        Since this model already accepts dense tensors directly, this method simply
        calls ``forward``.

        Args:
            left_x: Embedded pre-event input.
            right_x: Embedded post-event input.
            left_mask: Optional mask for left_x.
            right_mask: Optional mask for right_x.
            y: Optional labels.
            supervised_weight: Weight for supervised loss.

        Returns:
            Same dictionary as ``forward``.
        """
        return self.forward(
            left_x=left_x,
            right_x=right_x,
            left_mask=left_mask,
            right_mask=right_mask,
            y=y,
            supervised_weight=supervised_weight,
        )