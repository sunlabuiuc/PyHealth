import math
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

"""
Labrador (Bellamy, Kumar, Wang, & Beam, 2024)
---------------------------------------------

This module provides a PyTorch reimplementation of *Labrador*, a Transformer-based
architecture designed for self-supervised and supervised modeling of clinical 
laboratory trajectories.

Labrador was introduced in:

    Bellamy, Kumar, Wang, and Beam. 
    "Labrador: Exploring the Limits of Masked Language Modeling for Laboratory Data." 
    ICML 2024.

The original model was implemented in TensorFlow/Keras and proposed a masked
language modeling (MLM) objective that jointly predicts laboratory *codes* 
(categorical) and *values* (continuous) from irregular clinical time series. 
Its key architectural ideas include:

- **Dual embedding pathways** for categorical lab codes and continuous lab values,
  combined with special embeddings for mask and null tokens.
- **Transformer encoder blocks** applied over the lab sequence to contextualize
  each measurement.
- **Masked mean pooling** for downstream classification tasks.
- **Optional static (non-time-series) feature integration**, as used in the
  downstream “fine-tuning wrapper” in the original work.

This PyTorch port is adapted for the PyHealth framework and supports downstream
supervised learning tasks (binary, multiclass, multilabel, regression). It does 
*not* implement the MLM pretraining objective, but faithfully reconstructs the 
backbone encoder and downstream fine-tuning architecture used for clinical prediction.

Model outputs follow the PyHealth `BaseModel` interface and are compatible with
PyHealth's Trainer and dataloaders.

"""



class ContinuousEmbeddingTorch(nn.Module):
    """
    PyTorch port of the ContinuousEmbedding 
    It:
      - projects continuous lab values into an embedding space,
      - replaces values equal to mask_token / null_token with learned embeddings,
      - adds the categorical embeddings,
      - runs an MLP + LayerNorm.

    Args:
        embedding_dim: dimension of the (categorical + continuous) embedding.
        hidden_dim: hidden dimension inside the MLP.
        mask_token: special scalar used in continuous_input to indicate masked values.
        null_token: special scalar used to indicate null / missing values.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        mask_token: float = -2.0,
        null_token: float = -1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.mask_token = mask_token
        self.null_token = null_token

        # project scalar value -> embedding_dim
        self.value_proj = nn.Linear(1, embedding_dim)

        # learned embeddings for mask and null
        self.mask_embedding = nn.Parameter(torch.zeros(embedding_dim))
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))

        # small MLP and LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        continuous_input: torch.Tensor,      # [B, T]
        categorical_embeddings: torch.Tensor # [B, T, D]
    ) -> torch.Tensor:
        # continuous_input: [B, T] of standardized lab values, with special tokens
        # categorical_embeddings: [B, T, D]

        # project values -> embeddings
        vals = continuous_input.unsqueeze(-1)  # [B, T, 1]
        val_emb = self.value_proj(vals)        # [B, T, D]

        # identify mask / null positions
        mask_mask = continuous_input.eq(self.mask_token)   # [B, T]
        null_mask = continuous_input.eq(self.null_token)   # [B, T]

        if mask_mask.any():
            # broadcast mask_embedding over masked positions
            mask_emb = self.mask_embedding.view(1, 1, -1)
            val_emb = torch.where(
                mask_mask.unsqueeze(-1),
                mask_emb.expand_as(val_emb),
                val_emb,
            )

        if null_mask.any():
            null_emb = self.null_embedding.view(1, 1, -1)
            val_emb = torch.where(
                null_mask.unsqueeze(-1),
                null_emb.expand_as(val_emb),
                val_emb,
            )

        # combine with categorical embeddings
        x = val_emb + categorical_embeddings

        # MLP + LayerNorm
        x = self.mlp(x)
        x = self.layer_norm(x)

        return x


class TransformerBlockTorch(nn.Module):
    """
    Standard Transformer encoder block (MultiheadAttention + FFN + LayerNorm),
    implemented with batch_first=True so we can use [B, T, D] tensors.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T, D]
        key_padding_mask: torch.Tensor,  # [B, T], True for PAD positions
    ) -> torch.Tensor:
        # Multi-head self-attention
        attn_output, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,  # positions to ignore
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x


class LabradorEncoder(nn.Module):
    """
    Labrador backbone encoder (no MLM head).

    Inputs:
        categorical_input: [B, T] integer codes (0 = pad_token).
        continuous_input: [B, T] continuous lab values (with mask/null tokens).

    Outputs:
        hidden_states: [B, T, D]
        padding_mask: [B, T] boolean mask (True where positions are PAD).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        transformer_heads: int,
        transformer_blocks: int,
        transformer_ff_dim: int,
        pad_token: int = 0,
        mask_token: float = -2.0,
        null_token: float = -1.0,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token = pad_token

        # +2 to allow potential mask/null codes like in original Keras model
        self.categorical_embedding = nn.Embedding(
            num_embeddings=vocab_size + 2,
            embedding_dim=embedding_dim,
            padding_idx=pad_token,
        )

        self.continuous_embedding = ContinuousEmbeddingTorch(
            embedding_dim=embedding_dim,
            hidden_dim=transformer_ff_dim,
            mask_token=mask_token,
            null_token=null_token,
        )

        self.projection = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockTorch(
                    d_model=embedding_dim,
                    num_heads=transformer_heads,
                    ff_dim=transformer_ff_dim,
                    dropout=dropout_rate,
                )
                for _ in range(transformer_blocks)
            ]
        )

    def forward(
        self,
        categorical_input: torch.Tensor,  # [B, T]
        continuous_input: torch.Tensor,   # [B, T]
    ) -> (torch.Tensor, torch.Tensor):
        # embeddings
        cat_emb = self.categorical_embedding(categorical_input)  # [B, T, D]
        x = self.continuous_embedding(continuous_input, cat_emb) # [B, T, D]
        x = self.projection(x)
        x = self.dropout(x)

        # key_padding_mask: True at PAD positions
        padding_mask = categorical_input.eq(self.pad_token)  # [B, T]

        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)

        return x, padding_mask


class Labrador(BaseModel):
    """
    PyTorch Labrador model adapted for PyHealth.

    This class wraps the Labrador encoder (Transformer over lab trajectories)
    plus a downstream head similar to `LabradorFinetuneWrapper`:

        - masked mean pooling over time
        - optional non-MIMIC static features
        - optional extra dense layer
        - final classification/regression head

    Args:
        dataset: a SampleDataset (PyHealth) used to infer schemas.
        feature_keys: list of input feature keys, e.g.
            ["categorical_input", "continuous_input", "non_mimic_features?"].
            The first two are required; the third is optional.
        label_key: the output label key, e.g. "label".
        mode: "binary", "multiclass", "multilabel", or "regression".
        max_seq_len: maximum sequence length (truncation like original wrapper).
        vocab_size: size of the lab code vocabulary.
        embedding_dim: embedding dimension for codes/values.
        transformer_heads: number of attention heads.
        transformer_blocks: number of transformer layers.
        transformer_ff_dim: hidden size in feed-forward nets.
        dropout_rate: dropout rate.
        add_extra_dense_layer: if True, add a big Dense layer before output (like 1038).
        non_mimic_dim: output dimension of non-MIMIC dense block (default: 14).
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        max_seq_len: int,
        vocab_size: int,
        embedding_dim: int = 128,
        transformer_heads: int = 8,
        transformer_blocks: int = 4,
        transformer_ff_dim: int = 256,
        dropout_rate: float = 0.1,
        add_extra_dense_layer: bool = False,
        non_mimic_dim: int = 14,
        pad_token: int = 0,
        mask_token: float = -2.0,
        null_token: float = -1.0,
        **kwargs: Any,
    ):
        super().__init__(dataset=dataset)
        assert len(feature_keys) >= 2, "Need at least categorical + continuous inputs"
        self.feature_keys = feature_keys
        self.label_keys = [label_key]
        self.label_key = label_key
        self.mode = mode

        self.max_seq_len = max_seq_len
        self.add_extra_dense_layer = add_extra_dense_layer

        # Backbone encoder
        self.encoder = LabradorEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            transformer_heads=transformer_heads,
            transformer_blocks=transformer_blocks,
            transformer_ff_dim=transformer_ff_dim,
            pad_token=pad_token,
            mask_token=mask_token,
            null_token=null_token,
            dropout_rate=dropout_rate,
        )

        # pooling: masked mean over time
        self.non_mimic_key: Optional[str] = (
            feature_keys[2] if len(feature_keys) > 2 else None
        )

        # static / non-MIMIC dense
        if self.non_mimic_key is not None:
            self.non_mimic_dense = nn.Linear(
                in_features=self._infer_feature_dim(dataset, self.non_mimic_key),
                out_features=non_mimic_dim,
            )
            head_input_dim = embedding_dim + non_mimic_dim
        else:
            head_input_dim = embedding_dim

        # optional extra dense layer
        if add_extra_dense_layer:
            # original wrapper uses 1038 units; you can adjust if needed
            self.extra_dense = nn.Linear(head_input_dim, 1038)
            head_input_dim = 1038

        self.dropout = nn.Dropout(dropout_rate)

        # output head size based on mode
        if mode == "multiclass":
            output_size = self.get_output_size()
        else:
            output_size = 1  # binary / multilabel / regression

        self.output_layer = nn.Linear(head_input_dim, output_size)

    def _infer_feature_dim(self, dataset, key: str) -> int:
        """
        Best-effort helper to infer the dim of a static feature from dataset.
        If not available, you can hard-code it in model init instead.
        """
        # Often, input_processors[key].size() gives dim, but this depends on processor.
        processor = dataset.input_processors.get(key, None)
        if processor is not None and hasattr(processor, "size"):
            try:
                return processor.size()
            except Exception:
                pass
        # Fallback: require user to pass explicit dim via kwargs (not implemented here)
        raise ValueError(
            f"Cannot infer input dim for non-mimic feature '{key}'. "
            "Please adjust Labrador.__init__ to pass it explicitly."
        )

    def masked_mean_pool(
        self,
        x: torch.Tensor,         # [B, T, D]
        mask: torch.Tensor,      # [B, T], True for PAD positions
    ) -> torch.Tensor:
        # invert mask: 1 for valid, 0 for pad
        valid = (~mask).float()  # [B, T]
        # avoid division by zero
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        # [B, T, D] * [B, T, 1]
        x_masked = x * valid.unsqueeze(-1)
        return x_masked.sum(dim=1) / denom  # [B, D]

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PyHealth Trainer.

        kwargs must contain:
            - all feature_keys (categorical_input, continuous_input, [non_mimic_features])
            - label_key

        Returns:
            dict with keys: "loss", "y_prob", "y_true", "logit".
        """
        device = self.device

        # 1. Extract labels
        y_true = kwargs[self.label_key].to(device)

        # 2. Extract main sequence inputs
        cat_key = self.feature_keys[0]
        cont_key = self.feature_keys[1]
        categorical_input = kwargs[cat_key][:, : self.max_seq_len].to(device).long()
        continuous_input = kwargs[cont_key][:, : self.max_seq_len].to(device).float()

        # 3. Encode sequence
        hidden, padding_mask = self.encoder(categorical_input, continuous_input)
        # 4. Masked mean pooling over time -> [B, D]
        pooled = self.masked_mean_pool(hidden, padding_mask)

        # 5. Optional non-mimic static features
        if self.non_mimic_key is not None and self.non_mimic_key in kwargs:
            static = kwargs[self.non_mimic_key].to(device).float()
            static_emb = F.relu(self.non_mimic_dense(static))
            pooled = torch.cat([pooled, static_emb], dim=-1)

        x = pooled

        # 6. Optional extra dense layer (1038 units as in original wrapper)
        if self.add_extra_dense_layer:
            x = F.relu(self.extra_dense(x))

        # 7. Dropout + final head
        x = self.dropout(x)
        logits = self.output_layer(x)

        # 8. Compute loss
        loss_fn = self.get_loss_function()
        label_key = self.label_keys[0]
        mode = self._resolve_mode(self.dataset.output_schema[label_key])

        if mode == "multiclass":
            loss = loss_fn(logits, y_true.long())
        elif mode in ["binary", "multilabel"]:
            loss = loss_fn(logits, y_true.float())
        elif mode == "regression":
            # ensure matching shapes: [B, 1] vs [B] or [B, 1]
            loss = loss_fn(logits.view_as(y_true.float()), y_true.float())
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 9. Convert logits -> probabilities
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
