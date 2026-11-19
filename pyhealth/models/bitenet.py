# Contributor: Ibrahim Rabiu (GitHub: @alhibb, alhibb01@gmail.com)
# Paper: Chen, Xueping, et al. "BiteNet: Bidirectional Temporal Encoder Network
# to Predict Medical Outcomes."
# IEEE International Conference on Data Mining (ICDM 2020): 412-421.
# Paper link: https://ieeexplore.ieee.org/document/9378278

import math
from typing import Any, cast

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.utils import get_last_visit


class AttentionPooling(nn.Module):
    """Attention pooling layer.

    Paper: Chen et al. "BiteNet: Bidirectional Temporal Encoder Network to
    Predict Medical Outcomes." IEEE ICDM 2020.

    Computes an attention-weighted sum over a sequence of feature representations:

    .. math::
        s = W_2 \\text{ReLU}(W_1 x)
        \\alpha = \\text{softmax}(s)
        u = \\sum_{t} \\alpha_t x_t

    Args:
        embedding_dim: Dimensionality of inputs and projected representations.
        dropout: Dropout rate applied to intermediate activations. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.bitenet import AttentionPooling
        >>> pool = AttentionPooling(embedding_dim=32)
        >>> x = torch.randn(2, 6, 32)
        >>> mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        >>> out = pool(x, mask=mask)
        >>> out.shape
        torch.Size([2, 32])
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for attention pooling.

        Args:
            x: Input tensor of shape ``[batch_size, seq_len, embedding_dim]``.
            mask: Optional boolean mask of shape ``[batch_size, seq_len]`` where
                ``True`` indicates valid (unpadded) positions.

        Returns:
            Pooled representation tensor of shape ``[batch_size, embedding_dim]``.
        """
        scores = self.net(x)  # [batch_size, seq_len, 1]
        if mask is not None:
            # Guard against all-padded sequences to prevent NaNs in softmax
            all_pad = ~mask.any(dim=1)
            valid_mask = mask
            if all_pad.any():
                valid_mask = mask.clone()
                valid_mask[all_pad, 0] = True

            scores = scores.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
            weights = torch.softmax(scores, dim=1)
            weights = weights * mask.unsqueeze(-1).float()
            denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-9)
            weights = weights / denom
            pooled = torch.sum(weights * x, dim=1)
            if all_pad.any():
                pooled[all_pad] = 0.0
            return pooled

        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * x, dim=1)


class BiteNetTemporalEncoding(nn.Module):
    """Temporal interval and position encoding for BiteNet.

    Paper: Chen et al. "BiteNet: Bidirectional Temporal Encoder Network to
    Predict Medical Outcomes." IEEE ICDM 2020.

    Encodes inter-visit time intervals or sequence positions using sinusoidal
    encodings or learned interval embeddings.

    Args:
        embedding_dim: Dimensionality of embeddings.
        pos_encoding: Encoding strategy: ``"sinusoidal"``, ``"embedding"``,
            or ``"none"``. Default is ``"sinusoidal"``.
        max_interval: Maximum supported time interval index when
            ``pos_encoding="embedding"``. Default is 365.
        dropout: Dropout rate applied to temporal encodings. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.bitenet import BiteNetTemporalEncoding
        >>> enc = BiteNetTemporalEncoding(embedding_dim=32, pos_encoding="sinusoidal")
        >>> intervals = torch.tensor([[0, 15, 30], [0, 7, 14]])
        >>> out = enc(seq_len=3, interval_ids=intervals)
        >>> out.shape
        torch.Size([2, 3, 32])
    """

    def __init__(
        self,
        embedding_dim: int,
        pos_encoding: str = "sinusoidal",
        max_interval: int = 365,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pos_encoding not in ("sinusoidal", "embedding", "none"):
            raise ValueError(
                f"pos_encoding must be 'sinusoidal', 'embedding', or 'none', "
                f"got {pos_encoding!r}"
            )
        self.embedding_dim = embedding_dim
        self.pos_encoding = pos_encoding
        self.max_interval = max_interval
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == "embedding":
            self.interval_embeddings = nn.Embedding(max_interval, embedding_dim)
        else:
            self.interval_embeddings = None

    def _sinusoidal_encoding(
        self, positions: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Computes sinusoidal encoding handling both even and odd dimensions."""
        pos = positions.unsqueeze(-1).float()
        num_terms = (dim + 1) // 2
        div_term = torch.exp(
            torch.arange(0, num_terms, dtype=torch.float, device=positions.device)
            * (-math.log(10000.0) / dim)
        )
        sin_term = torch.sin(pos * div_term)
        cos_term = torch.cos(pos * div_term)
        emb = torch.zeros(*positions.shape, dim, device=positions.device)
        emb[..., 0::2] = sin_term[..., : emb[..., 0::2].shape[-1]]
        emb[..., 1::2] = cos_term[..., : emb[..., 1::2].shape[-1]]
        return emb

    @staticmethod
    def align_temporal_ids(
        ids: torch.Tensor, target_len: int, max_val: int | None = None
    ) -> torch.Tensor:
        """Pads or truncates temporal sequence IDs to match target sequence length."""
        if ids.dim() == 1:
            ids = ids.unsqueeze(1).expand(-1, target_len)
        elif ids.size(1) > target_len:
            ids = ids[:, :target_len]
        elif ids.size(1) < target_len:
            pad_len = target_len - ids.size(1)
            last_val = (
                ids[:, -1:]
                if ids.size(1) > 0
                else torch.zeros((ids.size(0), 1), dtype=ids.dtype, device=ids.device)
            )
            pad = last_val.expand(-1, pad_len)
            ids = torch.cat([ids, pad], dim=1)
        if max_val is not None:
            ids = torch.clamp(ids, 0, max_val - 1)
        return ids

    def forward(
        self,
        seq_len: int,
        batch_size: int = 1,
        interval_ids: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Forward pass generating temporal interval or position encodings.

        Args:
            seq_len: Sequence length of the target visit sequence.
            batch_size: Batch size of input features. Default is 1.
            interval_ids: Optional tensor of inter-visit days or interval indices
                of shape ``[batch_size, seq_len]`` or ``[batch_size]``.
            device: Target device for generated tensors.

        Returns:
            Temporal encoding tensor of shape ``[batch_size, seq_len, embedding_dim]``.
        """
        if self.pos_encoding == "none":
            dev = device if device is not None else torch.device("cpu")
            return torch.zeros((batch_size, seq_len, self.embedding_dim), device=dev)

        if interval_ids is not None:
            target_device = device if device is not None else interval_ids.device
            max_val = (
                self.max_interval if self.pos_encoding == "embedding" else None
            )
            aligned = self.align_temporal_ids(
                interval_ids.to(target_device), seq_len, max_val=max_val
            )
            if self.pos_encoding == "embedding":
                assert self.interval_embeddings is not None
                enc = self.interval_embeddings(aligned.long())
            else:
                enc = self._sinusoidal_encoding(aligned, self.embedding_dim)
        else:
            dev = device if device is not None else torch.device("cpu")
            positions = torch.arange(seq_len, dtype=torch.float, device=dev)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            if self.pos_encoding == "embedding":
                assert self.interval_embeddings is not None
                clamped_pos = torch.clamp(positions.long(), 0, self.max_interval - 1)
                enc = self.interval_embeddings(clamped_pos)
            else:
                enc = self._sinusoidal_encoding(positions, self.embedding_dim)

        return self.dropout(enc)


class BiteNetEncoderLayer(nn.Module):
    """Single direction-aware Transformer encoder layer for BiteNet.

    Args:
        d_model: Feature dimensionality.
        heads: Number of attention heads.
        dropout: Dropout rate. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.bitenet import BiteNetEncoderLayer
        >>> layer = BiteNetEncoderLayer(d_model=32, heads=2)
        >>> x = torch.randn(2, 5, 32)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, d_model: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for encoder layer."""
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class BiteNetEncoder(nn.Module):
    """Directional Transformer encoder stack for BiteNet.

    Paper: Chen et al. "BiteNet: Bidirectional Temporal Encoder Network to
    Predict Medical Outcomes." IEEE ICDM 2020.

    Applies multi-head self-attention with directional temporal causal masking:
        - ``"forward"``: Causal attention (each visit attends to past visits :math:`j \\le i`).
        - ``"backward"``: Anti-causal attention (each visit attends to future visits :math:`j \\ge i`).
        - ``"none"``: Bidirectional attention (unconstrained attention across all visits).

    Args:
        d_model: Dimensionality of inputs and outputs.
        heads: Number of attention heads. Default is 4.
        num_layers: Number of stacked encoder layers. Default is 1.
        direction: Temporal masking direction: ``"forward"``, ``"backward"``,
            or ``"none"``. Default is ``"forward"``.
        dropout: Dropout rate. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.bitenet import BiteNetEncoder
        >>> encoder = BiteNetEncoder(d_model=32, heads=2, num_layers=1, direction="forward")
        >>> x = torch.randn(2, 5, 32)
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([2, 5, 32])
    """

    def __init__(
        self,
        d_model: int,
        heads: int = 4,
        num_layers: int = 1,
        direction: str = "forward",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if direction not in ("forward", "backward", "none"):
            raise ValueError(
                f"direction must be 'forward', 'backward', or 'none', got {direction!r}"
            )
        self.d_model = d_model
        self.heads = heads
        self.num_layers = num_layers
        self.direction = direction
        self.layers = nn.ModuleList(
            [
                BiteNetEncoderLayer(d_model, heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for directional encoder stack.

        Args:
            x: Input tensor of shape ``[batch_size, seq_len, d_model]``.
            mask: Optional boolean mask of shape ``[batch_size, seq_len]`` where
                ``True`` indicates valid (unpadded) sequence tokens.

        Returns:
            Encoded representation tensor of shape ``[batch_size, seq_len, d_model]``.
        """
        _, seq_len, _ = x.shape
        device = x.device

        if self.direction == "forward":
            directional = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
        elif self.direction == "backward":
            directional = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=-1,
            )
        else:
            directional = torch.zeros(
                seq_len, seq_len, dtype=torch.bool, device=device
            )

        if mask is not None:
            # Combine directional mask and key padding mask
            combined = directional.unsqueeze(0) | (~mask).unsqueeze(1)
            # Guard against all-masked rows to prevent NaNs in softmax forward and backward
            all_masked = combined.all(dim=-1)
            if all_masked.any():
                combined = combined.clone()
                combined[all_masked, 0] = False
            attn_mask = combined.repeat_interleave(self.heads, dim=0)
        else:
            attn_mask = directional

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x


class BiteNet(BaseModel):
    """BiteNet model for electronic health records.

    Paper: Chen, Xueping, et al. "BiteNet: Bidirectional Temporal Encoder Network
    to Predict Medical Outcomes."
    IEEE International Conference on Data Mining (ICDM 2020): 412-421.
    Paper link: https://ieeexplore.ieee.org/document/9378278

    BiteNet models longitudinal patient EHR journeys through a bidirectional
    temporal architecture:
        1. Intra-visit representation: Encodes medical codes within each visit
           into a visit representation using attention pooling.
        2. Temporal interval encoding: Injects inter-visit time intervals
           or sequence positions.
        3. Inter-visit bidirectional encoding: Evaluates forward (past-to-present)
           and backward (future-to-present) temporal dependencies.
        4. Bidirectional aggregation: Combines forward and backward representations
           via attention pooling and a feed-forward projection.

    Temporal interval feature sources:
        - ``interval`` or ``time``: Inter-visit time intervals in days or integer
          timestamps (tensor of shape ``[batch_size, seq_len]`` or ``[batch_size]``)
          passed directly in batch kwargs (e.g. ``kwargs["interval"] = torch.tensor(...)``).
          Supplying raw integer tensors allows exact day intervals to directly index
          learned interval embeddings or sinusoidal encodings. If declared in
          ``dataset.input_schema`` with a sequence processor, intervals are converted
          to token vocabulary indices. If omitted from inputs, BiteNet operates as
          a position-aware bidirectional encoder using sequence positional encodings.

    Sequence pooling:
        - ``"attention"`` (default): Uses attention pooling over forward and backward
          encoder hidden representations.
        - ``"mean"``: Computes the average of hidden representations across valid tokens.
        - ``"cls"``: Uses the last valid token representation for forward (which sees
          all preceding visits) and index 0 for backward (which sees all succeeding visits).

    Args:
        dataset: PyHealth dataset containing features and labels.
        embedding_dim: Dimensionality of embeddings. Default is 128.
        heads: Number of attention heads. Default is 4.
        dropout: Dropout rate. Default is 0.1.
        num_layers: Number of transformer encoder layers per direction. Default is 1.
        pos_encoding: Temporal interval encoding mode: ``"sinusoidal"``,
            ``"embedding"``, or ``"none"``. Default is ``"sinusoidal"``.
        max_interval: Maximum inter-visit interval index. Default is 365.
        pooling: Sequence pooling method: ``"attention"``, ``"mean"``, or
            ``"cls"``. Default is ``"attention"``.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import BiteNet
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["c1", "c2", "c3"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v0",
        ...         "conditions": ["c2", "c4"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={"conditions": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="demo_bitenet",
        ... )
        >>> model = BiteNet(dataset=dataset, embedding_dim=64, heads=2, num_layers=1)
        >>> batch = dataset[0]
        >>> for k in batch:
        ...     if isinstance(batch[k], torch.Tensor):
        ...         batch[k] = batch[k].unsqueeze(0)
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 1,
        pos_encoding: str = "sinusoidal",
        max_interval: int = 365,
        pooling: str = "attention",
    ) -> None:
        super().__init__(dataset=dataset)
        if pooling not in ("attention", "mean", "cls"):
            raise ValueError(
                f"pooling must be 'attention', 'mean', or 'cls', got {pooling!r}"
            )
        if embedding_dim % heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by heads ({heads})."
            )

        assert len(self.label_keys) == 1, "BiteNet only supports a single label key"
        self.label_key = self.label_keys[0]

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.pos_encoding = pos_encoding
        self.max_interval = max_interval
        self.pooling = pooling

        # Protect auxiliary temporal inputs from being treated as code tokens
        self.code_feature_keys = [
            k for k in self.feature_keys if k not in ("interval", "time")
        ]
        if not self.code_feature_keys:
            raise ValueError(
                "BiteNet requires at least one code feature stream in dataset "
                "input_schema."
            )

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Temporal position / interval encoding
        self.temporal_encoding = BiteNetTemporalEncoding(
            embedding_dim=embedding_dim,
            pos_encoding=pos_encoding,
            max_interval=max_interval,
            dropout=dropout,
        )

        # Intra-visit pooling for hierarchical code inputs
        self.intra_visit_pool = AttentionPooling(embedding_dim, dropout=dropout)

        # Inter-visit bidirectional encoders
        self.forward_encoder = BiteNetEncoder(
            d_model=embedding_dim,
            heads=heads,
            num_layers=num_layers,
            direction="forward",
            dropout=dropout,
        )
        self.backward_encoder = BiteNetEncoder(
            d_model=embedding_dim,
            heads=heads,
            num_layers=num_layers,
            direction="backward",
            dropout=dropout,
        )

        # Attention pooling for inter-visit forward and backward representations
        self.forward_pool = AttentionPooling(embedding_dim, dropout=dropout)
        self.backward_pool = AttentionPooling(embedding_dim, dropout=dropout)

        # Bidirectional fusion feed-forward network
        self.fusion = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(len(self.code_feature_keys) * embedding_dim, output_size)

    def _pool_embedding(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Flattens or pools hierarchical code sequences into ``[batch, seq_len, hidden]``."""
        if x.dim() == 4:
            batch_size, num_visits, num_codes, dim = x.shape
            x_flat = x.view(batch_size * num_visits, num_codes, dim)
            code_mask = None
            if mask is not None and mask.dim() >= 3:
                code_mask = mask.view(batch_size * num_visits, num_codes)
            elif mask is None:
                code_mask = (x != 0).any(dim=-1).view(
                    batch_size * num_visits, num_codes
                )
            pooled = self.intra_visit_pool(x_flat, mask=code_mask)
            return pooled.view(batch_size, num_visits, dim)
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def _mask_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Infers sequence padding mask from non-zero embedding vectors."""
        return (x != 0).any(dim=-1)

    def forward_from_embedding(self, **kwargs) -> dict[str, Any]:
        """Forward propagation using pre-computed embeddings.

        Args:
            **kwargs: Embedded features, true label tensors, and optional
                ``interval`` or ``time`` tensors.

        Returns:
            Dictionary containing ``"logit"``, ``"y_prob"``, and optionally
            ``"loss"``, ``"y_true"``, and ``"embed"``.
        """
        interval_ids = kwargs.get("interval", kwargs.get("time", None))
        if isinstance(interval_ids, tuple):
            interval_ids = interval_ids[0]
        patient_embeddings: list[torch.Tensor] = []

        for feature_key in self.code_feature_keys:
            processor = self.dataset.input_processors[feature_key]
            schema = processor.schema()
            feature = kwargs[feature_key]

            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if len(feature) == len(schema) + 1 and mask is None:
                mask = feature[-1]

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' in the schema."
                )

            value = value.to(self.device)
            if mask is not None:
                mask = mask.to(self.device).bool()

            value = self._pool_embedding(value, mask=mask)

            batch_size, seq_len, _ = value.shape

            if mask is not None:
                while mask.dim() > 2:
                    mask = mask.any(dim=-1)
                if mask.size(1) != seq_len:
                    if mask.size(1) > seq_len:
                        mask = mask[:, :seq_len]
                    else:
                        pad = torch.zeros(
                            (mask.size(0), seq_len - mask.size(1)),
                            dtype=torch.bool,
                            device=mask.device,
                        )
                        mask = torch.cat([mask, pad], dim=1)
            else:
                mask = self._mask_from_embeddings(value).to(self.device)

            # Temporal interval / position encoding
            temporal_enc = self.temporal_encoding(
                seq_len=seq_len,
                batch_size=batch_size,
                interval_ids=interval_ids,
                device=self.device,
            )
            v = value + temporal_enc

            # Inter-visit bidirectional encoding
            o_fw = self.forward_encoder(v, mask=mask)
            o_bw = self.backward_encoder(v, mask=mask)

            # Sequence pooling
            if self.pooling == "attention":
                u_fw = self.forward_pool(o_fw, mask=mask)
                u_bw = self.backward_pool(o_bw, mask=mask)
            elif self.pooling == "mean":
                mask_expanded = mask.unsqueeze(-1).float()
                denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                u_fw = torch.sum(o_fw * mask_expanded, dim=1) / denom
                u_bw = torch.sum(o_bw * mask_expanded, dim=1) / denom
            else:  # cls
                u_fw = get_last_visit(o_fw, mask)
                u_bw = o_bw[:, 0, :]

            # Bidirectional concatenation and projection
            u_bi = torch.cat([u_fw, u_bw], dim=-1)
            s = self.fusion(u_bi)
            patient_embeddings.append(s)

        patient_emb = torch.cat(patient_embeddings, dim=-1)
        logits = self.fc(patient_emb)
        y_prob = self.label_processor.activation(logits)

        results: dict[str, Any] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if kwargs.get("embed", False):
            results["embed"] = patient_emb

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            if isinstance(y_true, tuple):
                y_true = y_true[0]
            y_true = cast(torch.Tensor, y_true).to(self.device)
            loss = self.label_processor.loss(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def forward(self, **kwargs) -> dict[str, Any]:
        """Main forward pass accepting batch dictionary.

        Args:
            **kwargs: Inputs matching dataset processor schemas and optional
                ``interval`` or ``time`` tensors.

        Returns:
            Dictionary containing ``"logit"``, ``"y_prob"``, and optionally
            ``"loss"``, ``"y_true"``, and ``"embed"``.
        """
        forward_kwargs = dict(kwargs)

        # Protect auxiliary inputs from being overwritten by EmbeddingModel
        raw_interval = kwargs.get("interval", None)
        raw_time = kwargs.get("time", None)

        for feature_key in self.code_feature_keys:
            feature = kwargs[feature_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            processor = self.dataset.input_processors[feature_key]
            schema = processor.schema()

            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' in the schema."
                )
            value = value.to(self.device)

            if mask is not None:
                mask = mask.to(self.device)
                embedded_val = self.embedding_model(
                    {feature_key: value}, masks={feature_key: mask}
                )[feature_key]
            else:
                embedded_val = self.embedding_model({feature_key: value})[feature_key]

            idx = schema.index("value")
            forward_kwargs[feature_key] = (
                feature[:idx] + (embedded_val,) + feature[idx + 1 :]
            )

        if raw_interval is not None:
            forward_kwargs["interval"] = (
                raw_interval[0] if isinstance(raw_interval, tuple) else raw_interval
            )
        if raw_time is not None:
            forward_kwargs["time"] = (
                raw_time[0] if isinstance(raw_time, tuple) else raw_time
            )

        return self.forward_from_embedding(**forward_kwargs)

