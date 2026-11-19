"""UDC: Unveiling Discrete Clues for Rare Disease Prediction.

Paper: Chuang Zhao, Hui Tang, Jiheng Zhang, Xiaomeng Li. Unveiling Discrete
Clues: Superior Healthcare Predictions for Rare Diseases. The Web Conference
(WWW 2025). https://arxiv.org/abs/2501.16373

In Electronic Health Records (EHR), predicting rare diseases is severely
hindered by extreme sample scarcity, long-tail code distributions, and
weak disease-code co-occurrence patterns. Standard continuous representation
learning models tend to be dominated by frequent common conditions, causing
clues indicating rare conditions to be smoothed out or lost.

UDC overcomes this by introducing condition-aware discrete clues via
Residual Vector Quantization (RVQ). By mapping continuous medical code and
visit representations to a discrete codebook hierarchy, UDC establishes
compact, distinguishable prototypes for subtle clinical clues.

The PyHealth UDC architecture comprises:
    1. Input Feature Embedding: Projects medical codes (conditions,
       procedures, drugs, etc.) into continuous representations using
       :class:`~pyhealth.models.EmbeddingModel`. Nested visit sequences
       are pooled into visit-level representations via configurable
       intra-visit pooling (sum, mean, max, or attention).
    2. Discrete Clue Encoding: Employs :class:`ResidualVectorQuantizer`
       across multiple residual codebook stages to extract discrete clue
       prototypes, fusing continuous and discrete representations via
       residual connection and layer normalization.
    3. Longitudinal Visit Modeling: A temporal sequence encoder (GRU,
       LSTM, or Transformer) models the patient's multi-visit trajectory
       over time.
    4. Prediction Head & Joint Loss: Patient representations from the
       last valid visit are mapped to outcome predictions. The training
       loss jointly optimizes task prediction and discrete vector
       quantization commitment.

Supports binary, multiclass, and multilabel classification tasks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.utils import get_last_visit


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ) with straight-through gradient estimation.

    Decomposes an input continuous vector into a sum of discrete codebook
    prototypes across multiple codebook stages. Each stage quantizes the
    residual error left by the preceding stages, enabling a rich discrete
    representation with compact individual codebooks.

    Args:
        n_codebooks: Number of residual codebook stages. Default is 2.
        codebook_size: Number of prototype embeddings per codebook.
            Default is 128.
        embedding_dim: Dimensionality of each prototype vector.
            Default is 128.
        commitment_cost: Commitment loss scaling factor beta. Default is 0.25.

    Examples:
        >>> import torch
        >>> from pyhealth.models.udc import ResidualVectorQuantizer
        >>> rvq = ResidualVectorQuantizer(
        ...     n_codebooks=2, codebook_size=32, embedding_dim=16
        ... )
        >>> x = torch.randn(3, 5, 16)
        >>> quantized, vq_loss, indices = rvq(x)
        >>> quantized.shape
        torch.Size([3, 5, 16])
        >>> indices.shape
        torch.Size([3, 5, 2])
        >>> bool(vq_loss.detach() >= 0.0)
        True
    """

    def __init__(
        self,
        n_codebooks: int = 2,
        codebook_size: int = 128,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        if n_codebooks <= 0:
            raise ValueError(f"n_codebooks must be positive, got {n_codebooks}")
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if commitment_cost < 0:
            raise ValueError(
                f"commitment_cost cannot be negative, got {commitment_cost}"
            )

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, embedding_dim) for _ in range(n_codebooks)]
        )
        for codebook in self.codebooks:
            nn.init.uniform_(
                codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantizes the continuous input tensor across residual codebooks.

        Args:
            x: Continuous input tensor of shape ``(..., embedding_dim)``.
            mask: Optional boolean mask of shape ``(...)`` indicating valid visits.

        Returns:
            Tuple of:
                - quantized: Quantized tensor with straight-through gradient,
                  shape ``(..., embedding_dim)``.
                - vq_loss: Scalar commitment and codebook loss.
                - indices: Selected codebook prototype indices, shape
                  ``(..., n_codebooks)``.
        """
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.embedding_dim)
        residual = flat_x
        flat_mask = mask.reshape(-1).bool() if mask is not None else None

        quantized_out = torch.zeros_like(flat_x)
        all_indices: list[torch.Tensor] = []
        vq_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for codebook in self.codebooks:
            cb_w = codebook.weight  # (codebook_size, embedding_dim)
            # Pairwise squared Euclidean distance
            dist = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * (residual @ cb_w.t())
                + cb_w.pow(2).sum(dim=-1, keepdim=True).t()
            )
            indices = torch.argmin(dist, dim=-1)
            all_indices.append(indices)

            quantized_i = codebook(indices)
            if flat_mask is not None and flat_mask.any():
                q_valid = quantized_i[flat_mask]
                res_valid = residual[flat_mask]
                loss_i = F.mse_loss(
                    q_valid, res_valid.detach()
                ) + self.commitment_cost * F.mse_loss(res_valid, q_valid.detach())
            else:
                loss_i = F.mse_loss(
                    quantized_i, residual.detach()
                ) + self.commitment_cost * F.mse_loss(residual, quantized_i.detach())
            vq_loss = vq_loss + loss_i

            quantized_out = quantized_out + quantized_i
            residual = residual - quantized_i.detach()

        # Straight-through gradient estimator
        quantized_st = flat_x + (quantized_out - flat_x).detach()
        quantized_st = quantized_st.view(orig_shape)

        stacked_indices = torch.stack(all_indices, dim=-1).view(
            *orig_shape[:-1], self.n_codebooks
        )
        mean_vq_loss = vq_loss / self.n_codebooks
        return quantized_st, mean_vq_loss, stacked_indices


class DiscreteClueEncoder(nn.Module):
    """Encodes continuous visit representations with condition-aware discrete clues.

    In UDC, continuous embeddings for rare diseases and infrequent clinical events
    suffer from noisy, sparse gradients. The Discrete Clue Encoder projects
    continuous features into discrete prototype spaces using Residual Vector
    Quantization (RVQ), discretizing clues into compact, distinguishable
    representations that prevent frequent conditions from dominating.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension for discrete clues.
        n_codebooks: Number of residual codebooks. Default is 2.
        codebook_size: Number of prototype embeddings per codebook.
            Default is 128.
        commitment_cost: Weight beta of the commitment loss term.
            Default is 0.25.
        dropout: Dropout rate. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.udc import DiscreteClueEncoder
        >>> encoder = DiscreteClueEncoder(
        ...     input_dim=16, hidden_dim=16, n_codebooks=2, codebook_size=32
        ... )
        >>> x = torch.randn(2, 4, 16)
        >>> fused, vq_loss, indices = encoder(x)
        >>> fused.shape
        torch.Size([2, 4, 16])
        >>> indices.shape
        torch.Size([2, 4, 2])
        >>> bool(vq_loss.detach() >= 0.0)
        True
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_codebooks: int = 2,
        codebook_size: int = 128,
        commitment_cost: float = 0.25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj: nn.Module = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.rvq = ResidualVectorQuantizer(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            embedding_dim=hidden_dim,
            commitment_cost=commitment_cost,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects continuous representations and fuses residual discrete clues.

        Args:
            x: Input continuous tensor of shape ``(..., input_dim)``.
            mask: Optional boolean mask of shape ``(...)`` indicating valid visits.

        Returns:
            Tuple of:
                - fused: Discrete-clue-enhanced representation of shape
                  ``(..., hidden_dim)``.
                - vq_loss: Residual vector quantization loss scalar.
                - indices: Selected discrete clue indices of shape
                  ``(..., n_codebooks)``.
        """
        projected = self.proj(x)
        quantized, vq_loss, indices = self.rvq(projected, mask=mask)
        fused = self.norm(projected + quantized)
        fused = self.dropout(fused)
        if mask is not None:
            fused = fused.masked_fill(~mask.unsqueeze(-1), 0.0)
        return fused, vq_loss, indices


class UDC(BaseModel):
    """UDC: Unveiling Discrete Clues for Rare Disease Prediction.

    Paper: Chuang Zhao, Hui Tang, Jiheng Zhang, Xiaomeng Li. Unveiling Discrete
    Clues: Superior Healthcare Predictions for Rare Diseases. The Web
    Conference (WWW 2025). https://arxiv.org/abs/2501.16373

    In Electronic Health Records (EHR), predicting rare diseases is severely
    challenged by extreme sample sparsity, long-tail class distributions, and
    weak co-occurrence patterns between rare conditions and clinical events.
    Standard continuous deep learning models are overwhelmed by frequent
    dominant diagnoses, obscuring subtle clinical clues essential for rare
    disease detection.

    UDC resolves this by establishing condition-aware discrete clues through
    Residual Vector Quantization (RVQ). Hierarchical discrete codebooks
    quantize continuous visit embeddings into sharp, distinct clue prototypes,
    preventing rare clinical signals from vanishing.

    Args:
        dataset: The dataset to train the model.
        feature_keys: Names of input feature keys to use from dataset. If None,
            all feature keys in ``dataset.input_schema`` are used.
        label_key: Name of the label key from dataset. If None, the single
            label key from ``dataset.output_schema`` is used.
        embedding_dim: Dimension of input code embeddings. Default is 128.
        hidden_dim: Dimension of hidden representations. Default is 128.
        n_codebooks: Number of residual codebooks for discrete clues.
            Default is 2.
        codebook_size: Number of prototype entries per codebook.
            Default is 128.
        commitment_cost: Weight beta of commitment loss. Default is 0.25.
        vq_loss_weight: Weight lambda of discrete quantization loss in the
            total training loss. Default is 0.1.
        intra_visit_pooling: Method to pool codes within a visit ('sum',
            'mean', 'max', 'attention'). Default is 'sum'.
        temporal_model: Type of temporal sequence encoder ('gru', 'lstm',
            'transformer'). Default is 'gru'.
        num_layers: Number of temporal recurrent or transformer layers.
            Default is 1.
        dropout: Dropout probability. Default is 0.1.
        **kwargs: Additional keyword arguments.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import UDC
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": [["c1", "c2"], ["c3"]],
        ...         "procedures": [["p1"], ["p2"]],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "conditions": [["c2"], ["c4", "c5"]],
        ...         "procedures": [["p3"], ["p1"]],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={
        ...         "conditions": "nested_sequence",
        ...         "procedures": "nested_sequence",
        ...     },
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = UDC(
        ...     dataset=dataset,
        ...     embedding_dim=16,
        ...     hidden_dim=16,
        ...     codebook_size=32,
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> batch = next(iter(loader))
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['embed', 'logit', 'loss', 'vq_indices', 'vq_loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: list[str] | None = None,
        label_key: str | None = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_codebooks: int = 2,
        codebook_size: int = 128,
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 0.1,
        intra_visit_pooling: str = "sum",
        temporal_model: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset=dataset)

        if intra_visit_pooling not in ("sum", "mean", "max", "attention"):
            raise ValueError(
                f"Unsupported intra_visit_pooling: {intra_visit_pooling!r}. "
                "Must be one of 'sum', 'mean', 'max', 'attention'."
            )
        if temporal_model not in ("gru", "lstm", "transformer"):
            raise ValueError(
                f"Unsupported temporal_model: {temporal_model!r}. "
                "Must be one of 'gru', 'lstm', 'transformer'."
            )
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_codebooks <= 0:
            raise ValueError(f"n_codebooks must be positive, got {n_codebooks}")
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if vq_loss_weight < 0:
            raise ValueError(
                f"vq_loss_weight cannot be negative, got {vq_loss_weight}"
            )

        if feature_keys is not None:
            self.feature_keys = feature_keys
        if label_key is not None:
            self.label_key = label_key
        else:
            self.label_key = self.label_keys[0]

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.intra_visit_pooling = intra_visit_pooling
        self.temporal_model_name = temporal_model
        self.vq_loss_weight = vq_loss_weight

        # Unified embedding model for input modalities
        self.embedding_model = EmbeddingModel(
            dataset=dataset, embedding_dim=embedding_dim
        )

        # Intra-visit attention pooling projection (if attention pooling is selected)
        if self.intra_visit_pooling == "attention":
            self.intra_attn = nn.Linear(embedding_dim, 1)

        # Multi-feature fusion layer
        num_features = len(self.feature_keys)
        fused_dim = num_features * embedding_dim
        self.feat_proj = nn.Linear(fused_dim, hidden_dim)

        # Discrete Clue Encoder
        self.clue_encoder = DiscreteClueEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
            dropout=dropout,
        )

        # Temporal sequence model across longitudinal visits
        if temporal_model == "gru":
            self.temporal_encoder = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif temporal_model == "lstm":
            self.temporal_encoder = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # transformer
            nhead = 2 if hidden_dim % 2 == 0 else 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

        # Classification prediction head
        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(hidden_dim, output_size)

    def _pool_intra_visit(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pools events/codes within each visit along dimension 2.

        Args:
            x: Tensor of shape ``(B, T, K, D)``.
            mask: Optional boolean mask of shape ``(B, T, K)``.

        Returns:
            Pooled visit representations of shape ``(B, T, D)``.
        """
        if self.intra_visit_pooling == "sum":
            return torch.sum(x, dim=2)
        elif self.intra_visit_pooling == "mean":
            if mask is not None:
                expanded_mask = mask.unsqueeze(-1).float()
                sum_x = torch.sum(x * expanded_mask, dim=2)
                count = expanded_mask.sum(dim=2).clamp(min=1.0)
                return sum_x / count
            return torch.mean(x, dim=2)
        elif self.intra_visit_pooling == "max":
            if mask is not None:
                expanded_mask = mask.unsqueeze(-1)
                x_masked = x.masked_fill(~expanded_mask, -1e9)
                pooled = torch.max(x_masked, dim=2)[0]
                empty_visits = ~mask.any(dim=-1, keepdim=True)
                return pooled.masked_fill(empty_visits, 0.0)
            return torch.max(x, dim=2)[0]
        else:  # attention
            weights = self.intra_attn(x)  # (B, T, K, 1)
            if mask is not None:
                weights = weights.masked_fill(~mask.unsqueeze(-1), -1e9)
            attn = torch.softmax(weights, dim=2)
            attn = torch.nan_to_num(attn, nan=0.0)
            pooled = torch.sum(attn * x, dim=2)
            if mask is not None:
                empty_visits = ~mask.any(dim=-1, keepdim=True)
                pooled = pooled.masked_fill(empty_visits, 0.0)
            return pooled

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward propagation for UDC.

        Args:
            **kwargs: Keyword arguments containing input feature tensors and
                optionally the label tensor.

        Returns:
            Dictionary containing:
                - y_prob: Predicted outcome probabilities.
                - logit: Raw output logits.
                - embed: Patient-level latent representations from the last visit.
                - vq_loss: Residual vector quantization loss (always returned).
                - vq_indices: Discrete prototype indices from vector quantization.
                - loss: Total training loss (when labels provided).
                - y_true: Ground truth outcome labels (when labels provided).
        """
        # Unpack tuple inputs if any
        cleaned_kwargs: dict[str, Any] = {}
        raw_masks: dict[str, torch.Tensor | None] = {}
        for key, val in kwargs.items():
            if isinstance(val, tuple):
                schema = (
                    self.dataset.input_processors[key].schema()
                    if (
                        hasattr(self.dataset, "input_processors")
                        and key in self.dataset.input_processors
                    )
                    else None
                )
                if (
                    schema is not None
                    and "time" in schema
                    and "value" in schema
                ):
                    value = val[schema.index("value")]
                    mask = val[schema.index("mask")] if "mask" in schema else None
                elif len(val) == 2:
                    first, second = val[0], val[1]
                    if (
                        isinstance(second, torch.Tensor)
                        and second.dtype == torch.bool
                    ):
                        value = first
                        mask = second
                    elif (
                        isinstance(first, torch.Tensor)
                        and first.is_floating_point()
                        and isinstance(second, torch.Tensor)
                        and not second.is_floating_point()
                    ):
                        # (time, value)
                        value = second
                        mask = None
                    else:
                        value = first
                        mask = None
                elif len(val) >= 3:
                    value = val[1]
                    mask = (
                        val[2]
                        if isinstance(val[2], torch.Tensor)
                        and val[2].dtype == torch.bool
                        else None
                    )
                elif schema is not None and "value" in schema:
                    value = val[schema.index("value")]
                    mask = val[schema.index("mask")] if "mask" in schema else None
                else:
                    value = val[0]
                    mask = None

                cleaned_kwargs[key] = value
                raw_masks[key] = mask if isinstance(mask, torch.Tensor) else None
            else:
                cleaned_kwargs[key] = val
                raw_masks[key] = None

        embedded = self.embedding_model(cleaned_kwargs)

        feat_visits: list[torch.Tensor] = []
        visit_masks: list[torch.Tensor] = []

        for feature_key in self.feature_keys:
            feat = embedded[feature_key]
            if feat.dim() > 4:
                raise ValueError(
                    f"Unsupported feature dimension {feat.dim()} for "
                    f"feature '{feature_key}'. Only 2D sequences (B, D), "
                    "3D sequences (B, T, D), and 4D nested sequences "
                    "(B, T, K, D) are supported."
                )
            v_mask = None
            # Nested sequences: (B, T, K, D)
            if feat.dim() == 4:
                mask = raw_masks.get(feature_key)
                if mask is not None and mask.dim() == 3:
                    code_mask = mask != 0
                else:
                    code_mask = (feat != 0).any(dim=-1)
                v_mask = code_mask.any(dim=-1)
                feat = self._pool_intra_visit(feat, code_mask)

            # Flat sequences: (B, D) -> (B, 1, D)
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)

            # Visit mask: step is valid if code_mask had valid codes or non-zero embedding
            if v_mask is None:
                mask = raw_masks.get(feature_key)
                if mask is not None and mask.dim() == feat.dim() - 1:
                    v_mask = mask != 0
                else:
                    v_mask = (feat != 0).any(dim=-1)
            else:
                feat = feat.masked_fill(~v_mask.unsqueeze(-1), 0.0)

            feat_visits.append(feat)
            visit_masks.append(v_mask)

        # Concatenate features along embedding dimension and project
        if len(feat_visits) > 1:
            # Ensure equal sequence lengths across features
            max_t = max(f.shape[1] for f in feat_visits)
            aligned_feats = []
            for f in feat_visits:
                if f.shape[1] < max_t:
                    pad_len = max_t - f.shape[1]
                    f = F.pad(f, (0, 0, 0, pad_len))
                aligned_feats.append(f)
            combined = torch.cat(aligned_feats, dim=-1)

            # Pad every mask to max_t before combining
            aligned_masks = []
            for m in visit_masks:
                if m.shape[1] < max_t:
                    m = F.pad(m, (0, max_t - m.shape[1]), value=False)
                aligned_masks.append(m)

            overall_mask = aligned_masks[0]
            for m in aligned_masks[1:]:
                overall_mask = overall_mask | m
        else:
            combined = feat_visits[0]
            overall_mask = visit_masks[0]

        visit_repr = self.feat_proj(combined)  # (B, T, hidden_dim)
        visit_repr = visit_repr.masked_fill(~overall_mask.unsqueeze(-1), 0.0)

        # Extract and fuse discrete clues
        fused_visits, vq_loss, vq_indices = self.clue_encoder(
            visit_repr, mask=overall_mask
        )

        # Temporal sequence modeling
        if self.temporal_model_name in ("gru", "lstm"):
            temporal_out, _ = self.temporal_encoder(fused_visits)
        else:  # transformer
            if overall_mask is not None:
                all_pad_patients = (~overall_mask).all(dim=-1)
                padding_mask = ~overall_mask
                # Unmask position 0 for all-pad patients so softmax is finite
                if all_pad_patients.any():
                    padding_mask = padding_mask.clone()
                    padding_mask[all_pad_patients, 0] = False
            else:
                padding_mask = None
                all_pad_patients = None

            temporal_out = self.temporal_encoder(
                fused_visits, src_key_padding_mask=padding_mask
            )
            if all_pad_patients is not None and all_pad_patients.any():
                temporal_out = temporal_out.masked_fill(
                    all_pad_patients.unsqueeze(-1).unsqueeze(-1), 0.0
                )

        # Extract patient representation from the last valid visit
        patient_emb = get_last_visit(temporal_out, overall_mask)

        # Predict logits and outcome probabilities
        logits = self.fc(patient_emb)
        y_prob = self.label_processor.activation(logits)

        results: dict[str, torch.Tensor] = {
            "y_prob": y_prob,
            "logit": logits,
            "embed": patient_emb,
            "vq_loss": vq_loss,
            "vq_indices": vq_indices,
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            if isinstance(y_true, tuple):
                y_true = y_true[0]
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.as_tensor(y_true, device=logits.device)
            task_loss = self.label_processor.loss(logits, y_true)
            total_loss = task_loss + self.vq_loss_weight * vq_loss
            results["loss"] = total_loss
            results["y_true"] = y_true

        return results
