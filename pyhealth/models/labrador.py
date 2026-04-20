"""PyHealth adaptation of Labrador for laboratory data.

Paper: Bellamy et al., "Labrador: Exploring the Limits of Masked Language
Modeling for Laboratory Data", ML4H 2024.
https://arxiv.org/abs/2312.11502

This implementation supports:
- joint modeling of categorical lab codes and continuous lab values
- optional dual-head masked language modeling (MLM)
- optional downstream classifier head for supervised tasks
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class LabradorValueEmbedding(nn.Module):
    """Embeds lab values and fuses them with code embeddings.

    The module projects each scalar value to ``hidden_dim`` and replaces
    designated special tokens (mask/null) with learned embeddings.

    Args:
        hidden_dim: Embedding size.
        mask_value_token: Sentinel float used for masked values.
        null_value_token: Sentinel float used for null values.
    """

    def __init__(
        self,
        hidden_dim: int,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
    ) -> None:
        super().__init__()
        self.mask_value_token = mask_value_token
        self.null_value_token = null_value_token

        self.value_projection = nn.Linear(1, hidden_dim)
        self.special_value_embedding = nn.Embedding(2, hidden_dim)
        self.fusion_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, lab_values: torch.Tensor, code_emb: torch.Tensor) -> torch.Tensor:
        """Returns fused value-code embeddings with shape ``(B, L, H)``."""
        value_emb = self.value_projection(lab_values.unsqueeze(-1).float())

        mask_positions = lab_values.eq(self.mask_value_token)
        null_positions = lab_values.eq(self.null_value_token)

        if mask_positions.any():
            mask_vec = self.special_value_embedding.weight[0].view(1, 1, -1)
            value_emb = torch.where(mask_positions.unsqueeze(-1), mask_vec, value_emb)
        if null_positions.any():
            null_vec = self.special_value_embedding.weight[1].view(1, 1, -1)
            value_emb = torch.where(null_positions.unsqueeze(-1), null_vec, value_emb)

        x = value_emb + code_emb
        x = self.fusion_activation(self.fusion_mlp(x))
        return self.layer_norm(x)


class LabradorEmbedding(nn.Module):
    """Joint embedding of lab code and lab value.

    Args:
        vocab_size: Number of lab codes (excluding 2 extra special IDs).
        hidden_dim: Embedding size.
        mask_code_token: ID used to represent masked code tokens.
        pad_code_token: ID used for padding.
        padding_idx: ``nn.Embedding`` padding index.
        mask_value_token: Sentinel float used for masked values.
        null_value_token: Sentinel float used for null values.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        mask_code_token: int = 1,
        pad_code_token: int = 0,
        padding_idx: int = 0,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
    ) -> None:
        super().__init__()
        self.mask_code_token = mask_code_token
        self.pad_code_token = pad_code_token

        self.code_embedding = nn.Embedding(
            num_embeddings=vocab_size + 2,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )
        self.value_embedding = LabradorValueEmbedding(
            hidden_dim=hidden_dim,
            mask_value_token=mask_value_token,
            null_value_token=null_value_token,
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lab_codes: torch.Tensor, lab_values: torch.Tensor) -> torch.Tensor:
        """Returns joint embeddings with shape ``(B, L, H)``."""
        code_emb = self.code_embedding(lab_codes)
        x = self.value_embedding(lab_values=lab_values, code_emb=code_emb)
        return self.output_projection(x)


class LabradorMLMHead(nn.Module):
    """Two-head masked-language-modeling module.

    Output heads:
      1) categorical logits/probabilities over lab code vocabulary
      2) continuous value regression for masked lab values

    Args:
        hidden_dim: Hidden size from the transformer encoder.
        vocab_size: Number of categorical lab codes.
        continuous_head_activation: ``"sigmoid"`` or ``"linear"``.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        continuous_head_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.categorical_dense = nn.Linear(hidden_dim, hidden_dim)
        self.categorical_activation = nn.ReLU()
        self.categorical_head = nn.Linear(hidden_dim, vocab_size)

        self.continuous_dense = nn.Linear(hidden_dim + vocab_size, hidden_dim + vocab_size)
        self.continuous_activation = nn.ReLU()
        self.continuous_head = nn.Linear(hidden_dim + vocab_size, 1)

        if continuous_head_activation == "sigmoid":
            self.continuous_output_activation = nn.Sigmoid()
        elif continuous_head_activation == "linear":
            self.continuous_output_activation = nn.Identity()
        else:
            raise ValueError(
                "continuous_head_activation must be 'sigmoid' or 'linear', "
                f"got {continuous_head_activation}"
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns MLM predictions for each token position."""
        cat_hidden = self.categorical_activation(self.categorical_dense(x))
        categorical_logits = self.categorical_head(cat_hidden)
        categorical_output = torch.softmax(categorical_logits, dim=-1)

        continuous_input = torch.cat([x, categorical_output], dim=-1)
        cont_hidden = self.continuous_activation(self.continuous_dense(continuous_input))
        continuous_output = self.continuous_output_activation(self.continuous_head(cont_hidden))

        return {
            "categorical_logits": categorical_logits,
            "categorical_output": categorical_output,
            "continuous_output": continuous_output,
            # Backward-compatible aliases.
            "mlm_code_logit": categorical_logits,
            "mlm_value_pred": continuous_output.squeeze(-1),
        }


class LabradorModel(BaseModel):
    """Labrador model with optional classifier and MLM heads.

    Args:
        dataset: ``SampleDataset`` from ``dataset.set_task(...)``.
        vocab_size: Number of lab codes.
        hidden_dim: Transformer hidden size.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        feedforward_dim: Feed-forward layer size in encoder blocks.
        dropout: Dropout probability.
        padding_idx: Padding index for code embedding.
        mask_code_token: Special ID for masked code.
        pad_code_token: Special ID for padded code.
        mask_value_token: Special float token for masked value.
        null_value_token: Special float token for null value.
        include_mlm_head: Whether to enable MLM outputs/losses.
        include_classifier_head: Whether to enable supervised classifier output.
        continuous_head_activation: Activation for continuous MLM head.
        mlm_ignore_index: Ignore value for masked targets not used in loss.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import LabradorModel
        >>> samples = [{
        ...     "patient_id": "p0", "visit_id": "v0",
        ...     "lab_codes": [1, 2, 3], "lab_values": [0.2, 0.7, 0.4],
        ...     "label": 1,
        ... }]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"lab_codes": "sequence", "lab_values": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="demo",
        ... )
        >>> model = LabradorModel(dataset=dataset, vocab_size=32)
    """

    def __init__(
        self,
        dataset,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
        mask_code_token: int = 1,
        pad_code_token: int = 0,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
        include_mlm_head: bool = True,
        include_classifier_head: bool = True,
        continuous_head_activation: str = "sigmoid",
        mlm_ignore_index: int = -1,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

        super().__init__(dataset)

        self.pad_code_token = pad_code_token
        self.include_mlm_head = include_mlm_head
        self.include_classifier_head = include_classifier_head
        self.mlm_ignore_index = mlm_ignore_index

        self.embedding = LabradorEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            mask_code_token=mask_code_token,
            pad_code_token=pad_code_token,
            padding_idx=padding_idx,
            mask_value_token=mask_value_token,
            null_value_token=null_value_token,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        self.classifier = (
            nn.Linear(hidden_dim, self.get_output_size()) if include_classifier_head else None
        )
        self.mlm_head = (
            LabradorMLMHead(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                continuous_head_activation=continuous_head_activation,
            )
            if include_mlm_head
            else None
        )

    def categorical_mlm_loss(
        self,
        categorical_logits: torch.Tensor,
        masked_lab_codes: torch.Tensor,
        mlm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy over valid categorical masked positions."""
        if mlm_mask is not None:
            masked_lab_codes = torch.where(
                mlm_mask,
                masked_lab_codes,
                torch.full_like(masked_lab_codes, self.mlm_ignore_index),
            )
        active = masked_lab_codes.ne(self.mlm_ignore_index)
        if not active.any():
            return torch.zeros((), device=categorical_logits.device)
        labels = masked_lab_codes[active].long() - 1
        return F.cross_entropy(categorical_logits[active], labels, reduction="mean")

    def continuous_mlm_loss(
        self,
        continuous_output: torch.Tensor,
        masked_lab_values: torch.Tensor,
        mlm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE over valid continuous masked positions."""
        if mlm_mask is not None:
            masked_lab_values = torch.where(
                mlm_mask,
                masked_lab_values,
                torch.full_like(masked_lab_values, float(self.mlm_ignore_index)),
            )
        continuous_output = continuous_output.squeeze(-1)
        active = masked_lab_values.ne(float(self.mlm_ignore_index))
        if not active.any():
            return torch.zeros((), device=continuous_output.device)
        return F.mse_loss(
            continuous_output[active],
            masked_lab_values[active].float(),
            reduction="mean",
        )

    def forward(
        self,
        lab_codes: torch.Tensor,
        lab_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        masked_lab_codes: Optional[torch.Tensor] = None,
        masked_lab_values: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        mlm_target_codes: Optional[torch.Tensor] = None,
        mlm_target_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Computes forward outputs for classifier and/or MLM heads."""
        if padding_mask is None:
            padding_mask = lab_codes.eq(self.pad_code_token)

        x = self.embedding(lab_codes=lab_codes.long(), lab_values=lab_values)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-8)

        output: Dict[str, torch.Tensor] = {}

        if self.include_classifier_head:
            logit = self.classifier(pooled)
            y_prob = self.prepare_y_prob(logit)
            output.update({"logit": logit, "y_prob": y_prob})

        if self.include_mlm_head:
            mlm_outputs = self.mlm_head(x)
            output.update(
                {
                    "categorical_output": mlm_outputs["categorical_output"],
                    "continuous_output": mlm_outputs["continuous_output"],
                    "mlm_code_logit": mlm_outputs["mlm_code_logit"],
                    "mlm_value_pred": mlm_outputs["mlm_value_pred"],
                }
            )

            masked_codes = masked_lab_codes if masked_lab_codes is not None else mlm_target_codes
            masked_values = masked_lab_values if masked_lab_values is not None else mlm_target_values
            if mlm_mask is not None and masked_codes is not None:
                masked_codes = torch.where(
                    mlm_mask,
                    masked_codes,
                    torch.full_like(masked_codes, self.mlm_ignore_index),
                )
            if mlm_mask is not None and masked_values is not None:
                masked_values = torch.where(
                    mlm_mask,
                    masked_values.float(),
                    torch.full_like(masked_values.float(), float(self.mlm_ignore_index)),
                )

            if masked_codes is not None:
                output["categorical_mlm_loss"] = self.categorical_mlm_loss(
                    mlm_outputs["categorical_logits"], masked_codes.long()
                )
            if masked_values is not None:
                output["continuous_mlm_loss"] = self.continuous_mlm_loss(
                    mlm_outputs["continuous_output"], masked_values
                )
            if "categorical_mlm_loss" in output and "continuous_mlm_loss" in output:
                output["mlm_loss"] = output["categorical_mlm_loss"] + output["continuous_mlm_loss"]

        if label is not None:
            if not self.include_classifier_head:
                raise ValueError("label is provided but include_classifier_head=False")
            if self.mode in ["binary", "multilabel", "regression"]:
                label = label.float().view_as(logit)
            elif self.mode == "multiclass":
                label = label.long().view(-1)

            output["loss"] = self.get_loss_function()(logit, label)
            output["y_true"] = label

        return output
