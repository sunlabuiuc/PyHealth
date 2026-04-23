"""TransEHR: Transformer-Based Model for Clinical Time Series Data.

This module implements a simplified, supervised version of the TransEHR
architecture from Xu et al. (2023) within the PyHealth framework.

Unlike the existing :class:`pyhealth.models.Transformer` model, which treats
each patient record as a *flat* sequence of codes, ``TransEHR`` preserves the
*hierarchical* structure of EHR data:

    patient → visits (temporal) → codes within each visit

This design matches the clinical reality of longitudinal patient records and is
the core architectural contribution of the original paper.

Reference:
    Xu, Y., Xu, S., Ramprassad, M., Tumanov, A., & Zhang, C. (2023).
    TransEHR: Self-Supervised Transformer for Clinical Time Series Data.
    *Proceedings of Machine Learning Research*.
    https://proceedings.mlr.press/v209/xu23a.html
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for visit sequences.

    Adds position information to visit embeddings using the standard
    sinusoidal formulation from Vaswani et al. (2017).  The encoding is
    computed once and cached as a buffer so it is never trained.

    Args:
        embedding_dim: Dimension of the embedding vectors. Must be even.
        max_len: Maximum sequence length that can be encoded. Default: 512.
        dropout: Dropout rate applied after adding the encoding. Default: 0.1.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(1, max_len, embedding_dim)  # (1, L, D)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Args:
            x: Embedding tensor of shape ``(batch, seq_len, embedding_dim)``.

        Returns:
            torch.Tensor: Positionally-encoded tensor of the same shape.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


class TransEHR(BaseModel):
    """Transformer-based model for multi-visit clinical EHR time series.

    ``TransEHR`` is a supervised transformer encoder that models the *temporal
    sequence of hospital visits* in a patient record, inspired by the
    architecture described in Xu et al. (2023).

    **Key design choices:**

    * Accepts ``nested_sequence`` inputs — each sample is a list of visits,
      and each visit is a list of medical codes (diagnoses, procedures, etc.).
      This is in contrast to :class:`~pyhealth.models.Transformer`, which
      flattens all codes into a single sequence and loses visit-level temporal
      structure.
    * Within each visit, code embeddings are **mean-pooled** into a single
      visit-level representation.
    * A sinusoidal positional encoding is added to encode the temporal order
      of visits.
    * A standard transformer encoder with multi-head self-attention models
      cross-visit dependencies.
    * Multiple feature streams (e.g., conditions + procedures) are processed
      independently and **concatenated** before the classification head,
      allowing the model to jointly reason over heterogeneous event types.

    Paper:
        TransEHR: Self-Supervised Transformer for Clinical Time Series Data
        (Xu et al., PMLR 2023). https://proceedings.mlr.press/v209/xu23a.html

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` whose input
            schema uses ``"nested_sequence"`` for EHR features and a
            classification label in the output schema.
        embedding_dim: Dimension of the code and visit embeddings. Default: 128.
        num_heads: Number of attention heads per transformer layer. Must evenly
            divide ``embedding_dim``. Default: 4.
        num_layers: Number of stacked transformer encoder layers. Default: 2.
        dropout: Dropout probability applied in attention, feed-forward, and
            after positional encoding. Default: 0.1.
        feedforward_dim: Inner dimension of the position-wise feed-forward
            network. Default: 256.
        max_visits: Maximum number of visits a patient sequence can have (used
            for positional encoding buffer). Default: 512.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": [["I10", "E11"], ["J45"]],
        ...         "procedures": [["4A023N6"], ["0BJ08ZZ"]],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v0",
        ...         "conditions": [["K21"], ["N18", "I50"]],
        ...         "procedures": [["5A1935Z"]],
        ...         "label": 0,
        ...     },
        ... ]
        >>> input_schema = {
        ...     "conditions": "nested_sequence",
        ...     "procedures": "nested_sequence",
        ... }
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema, output_schema, dataset_name="demo"
        ... )
        >>> model = TransEHR(dataset=dataset, embedding_dim=64, num_heads=2)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> sorted(output.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
        >>> output["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 256,
        max_visits: int = 512,
    ) -> None:
        super().__init__(dataset=dataset)

        assert (
            len(self.label_keys) == 1
        ), "TransEHR supports exactly one label key."

        self.label_key: str = self.label_keys[0]
        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.dropout_rate: float = dropout
        self.feedforward_dim: int = feedforward_dim
        self.max_visits: int = max_visits

        # Shared code-level embedding table across all feature streams.
        # EmbeddingModel handles different processor types automatically.
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Per-feature-stream visit-level positional encodings.
        self.pos_encodings: nn.ModuleDict = nn.ModuleDict(
            {
                key: _SinusoidalPositionalEncoding(
                    embedding_dim=embedding_dim,
                    max_len=max_visits,
                    dropout=dropout,
                )
                for key in self.feature_keys
            }
        )

        # Per-feature-stream transformer encoders.
        encoder_layer = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformers: nn.ModuleDict = nn.ModuleDict(
            {key: nn.TransformerEncoder(encoder_layer(), num_layers=num_layers)
             for key in self.feature_keys}
        )

        # Classification head: concatenated per-stream representations → output.
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return the device this model currently lives on."""
        return self._dummy_param.device

    @staticmethod
    def _pool_visits(
        embedded: torch.Tensor,
        raw_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool code embeddings within each visit.

        Args:
            embedded: Code embeddings of shape
                ``(batch, visits, codes, embedding_dim)``.
            raw_ids: Raw integer code indices of shape
                ``(batch, visits, codes)`` — padding positions are 0.

        Returns:
            visit_emb: Visit-level embeddings of shape
                ``(batch, visits, embedding_dim)``.
            visit_mask: Boolean mask of shape ``(batch, visits)`` — ``True``
                for *valid* (non-padding) visits.
        """
        # Mask over individual codes: padding index == 0
        code_mask = (raw_ids != 0).float()  # (B, V, C)
        num_valid_codes = code_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, V, 1)

        # Mean pool: sum embeddings where codes are valid, divide by count
        visit_emb = (embedded * code_mask.unsqueeze(-1)).sum(dim=2) / num_valid_codes
        # visit_emb: (B, V, D)

        # A visit is valid if it has at least one non-padding code
        visit_mask = code_mask.sum(dim=-1) > 0  # (B, V)
        return visit_emb, visit_mask

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, **kwargs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the TransEHR model.

        Processes each feature stream through:

        1. Code embedding lookup (shared :class:`EmbeddingModel`).
        2. Within-visit mean pooling → visit-level representations.
        3. Sinusoidal positional encoding over the visit sequence.
        4. Transformer encoder over visits with padding masking.
        5. Valid-visit mean pooling → patient-level representation.

        The per-stream patient representations are concatenated and fed to a
        linear classification head.

        Args:
            **kwargs: Batch dictionary.  Each feature key maps to a
                ``torch.Tensor`` of shape
                ``(batch_size, num_visits, num_codes)`` (nested_sequence).
                The label key maps to a ``torch.Tensor`` of shape
                ``(batch_size,)`` for binary/multiclass tasks.

        Returns:
            dict with keys:

            * ``"logit"``: Raw logits of shape ``(batch, output_size)``.
            * ``"y_prob"``: Predicted probabilities of the same shape.
            * ``"loss"`` *(only when label key is present)*: Scalar loss.
            * ``"y_true"`` *(only when label key is present)*: Ground-truth
              labels of shape ``(batch,)`` or ``(batch, num_classes)``.
        """
        patient_embs: List[torch.Tensor] = []

        for feature_key in self.feature_keys:
            raw_ids: torch.Tensor = kwargs[feature_key].to(self.device)
            # raw_ids: (B, V, C)  — integer code indices, 0 = padding

            # 1. Embed codes: (B, V, C) → (B, V, C, D)
            embedded = self.embedding_model({feature_key: raw_ids})[feature_key]

            # 2. Pool codes within visits: (B, V, C, D) → (B, V, D)
            visit_emb, visit_mask = self._pool_visits(embedded, raw_ids)

            # 3. Add positional encoding
            visit_emb = self.pos_encodings[feature_key](visit_emb)  # (B, V, D)

            # 4. Transformer encoder over the visit sequence
            # src_key_padding_mask: True where padding (invalid visits)
            padding_mask = ~visit_mask  # (B, V)
            encoded = self.transformers[feature_key](
                visit_emb, src_key_padding_mask=padding_mask
            )  # (B, V, D)

            # 5. Mean-pool over valid visits → patient representation
            valid = visit_mask.float().unsqueeze(-1)  # (B, V, 1)
            patient_rep = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            # patient_rep: (B, D)

            patient_embs.append(patient_rep)

        # Concatenate per-stream patient representations
        patient_emb = torch.cat(patient_embs, dim=-1)  # (B, num_features * D)

        logits = self.fc(patient_emb)  # (B, output_size)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results
