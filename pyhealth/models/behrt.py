# Contributor: Ibrahim Rabiu (GitHub: @alhibb, alhibb01@gmail.com)
# Paper: Li, Yikuan, et al. "BEHRT: Transformer for Electronic Health Records."
# Scientific Reports 10.1 (2020): 7155.
# Paper link: https://www.nature.com/articles/s41598-020-62922-y

import math
from typing import Any, cast

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class BEHRTEmbedding(nn.Module):
    """BEHRT composite embedding layer.

    Paper: Li, Yikuan, et al. "BEHRT: Transformer for Electronic Health Records."
    Scientific Reports 10.1 (2020): 7155.

    Computes token representations as the element-wise sum of:
        - Concept / code embedding (from the base sequence representations)
        - Chronological age embedding
        - Visit segment embedding (distinguishing visit boundaries)
        - Sequence position embedding

    .. math::
        \\mathbf{E} = \\mathbf{E}_{\\text{code}} + \\mathbf{E}_{\\text{age}} +
        \\mathbf{E}_{\\text{seg}} + \\mathbf{E}_{\\text{pos}}

    Args:
        embedding_dim: Dimensionality of embeddings.
        max_age: Maximum age index supported. Default is 120.
        max_segments: Maximum visit segments supported. Default is 64.
        max_position: Maximum sequence length. Default is 512.
        dropout: Dropout rate applied after layer normalization. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.behrt import BEHRTEmbedding
        >>> emb_layer = BEHRTEmbedding(embedding_dim=64)
        >>> code_embs = torch.randn(2, 8, 64)
        >>> out = emb_layer(code_embs)
        >>> out.shape
        torch.Size([2, 8, 64])
    """

    def __init__(
        self,
        embedding_dim: int,
        max_age: int = 120,
        max_segments: int = 64,
        max_position: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_age = max_age
        self.max_segments = max_segments
        self.max_position = max_position

        self.age_embeddings = nn.Embedding(max_age, embedding_dim)
        self.segment_embeddings = nn.Embedding(max_segments, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position, embedding_dim)

        self._init_position_weights()

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def _init_position_weights(self) -> None:
        """Initializes positional embeddings with sinusoidal encodings."""
        weight = torch.zeros(self.max_position, self.embedding_dim)
        position = torch.arange(0, self.max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / self.embedding_dim)
        )
        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        weight[:, 0::2] = sin_term[:, : weight[:, 0::2].size(1)]
        weight[:, 1::2] = cos_term[:, : weight[:, 1::2].size(1)]
        self.position_embeddings.weight.data.copy_(weight)

    @staticmethod
    def _align_temporal_ids(
        ids: torch.Tensor, target_len: int, max_val: int
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
        return torch.clamp(ids, 0, max_val - 1)

    def forward(
        self,
        code_embeddings: torch.Tensor,
        age_ids: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combines code, age, segment, and positional embeddings.

        Args:
            code_embeddings: Tensor of shape ``[batch_size, seq_len, embedding_dim]``.
            age_ids: Optional tensor of patient ages of shape ``[batch_size, seq_len]``
                or ``[batch_size]``. If shorter than ``seq_len``, the last observed age
                is forward-filled.
            segment_ids: Optional tensor of visit segment indicators of shape
                ``[batch_size, seq_len]`` or ``[batch_size]``. If shorter than
                ``seq_len``, the last segment is forward-filled.
            position_ids: Optional tensor of explicit positions of shape
                ``[batch_size, seq_len]``.

        Returns:
            Normalized and dropout-regularized combined tensor of shape
            ``[batch_size, seq_len, embedding_dim]``.
        """
        batch_size, seq_len, _ = code_embeddings.size()
        device = code_embeddings.device

        embeddings = code_embeddings

        if age_ids is not None:
            age_tensor = self._align_temporal_ids(
                age_ids.to(device).long(), seq_len, self.max_age
            )
            embeddings = embeddings + self.age_embeddings(age_tensor)

        if segment_ids is not None:
            seg_tensor = self._align_temporal_ids(
                segment_ids.to(device).long(), seq_len, self.max_segments
            )
            embeddings = embeddings + self.segment_embeddings(seg_tensor)

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(device).long()
            if position_ids.size(1) != seq_len:
                position_ids = self._align_temporal_ids(
                    position_ids, seq_len, self.max_position
                )

        position_ids = torch.clamp(position_ids, 0, self.max_position - 1)
        embeddings = embeddings + self.position_embeddings(position_ids)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BEHRTLayer(nn.Module):
    """BEHRT Transformer Encoder stack.

    Applies multi-head self-attention and position-wise feedforward networks
    with residual connections, layer normalization, and GELU activation.

    Args:
        hidden_size: Embedding and hidden dimension.
        heads: Number of attention heads. Default is 4.
        dim_feedforward: Feedforward network dimension. Default is ``4 * hidden_size``.
        dropout: Dropout rate. Default is 0.1.
        num_layers: Number of transformer encoder layers. Default is 2.

    Examples:
        >>> import torch
        >>> from pyhealth.models.behrt import BEHRTLayer
        >>> layer = BEHRTLayer(hidden_size=64, heads=2, num_layers=2)
        >>> x = torch.randn(2, 10, 64)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encodes sequence representations with padding awareness.

        Args:
            x: Tensor of shape ``[batch_size, seq_len, hidden_size]``.
            mask: Optional boolean tensor of shape ``[batch_size, seq_len]`` where
                True indicates a valid token and False indicates padding.

        Returns:
            Encoded sequence tensor of shape ``[batch_size, seq_len, hidden_size]``.
        """
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
            # Guard against entirely masked rows causing NaN attention weights
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class BEHRT(BaseModel):
    """BEHRT: Transformer for Electronic Health Records.

    Paper: Li, Yikuan, et al. "BEHRT: Transformer for Electronic Health Records."
    Scientific Reports 10.1 (2020): 7155.
    Paper link: https://www.nature.com/articles/s41598-020-62922-y

    BEHRT extends the BERT architecture to handle sequential electronic health records
    by combining four specialized embedding modalities:
        1. Concept / medical code embeddings
        2. Patient chronological age embeddings
        3. Visit segment embeddings
        4. Token sequence position embeddings

    Temporal feature sources:
        - ``age``: Patient chronological age (integer or float tensor of shape
          ``[batch_size, seq_len]`` or ``[batch_size]``) representing age at each
          clinical encounter, with values in ``[0, max_age - 1]``. Users can supply
          integer age tensors directly in batch kwargs (e.g. ``kwargs["age"]``).
          Note that categorical sequence processors like ``SequenceProcessor`` map
          unique tokens to arbitrary vocabulary indices rather than biological ages;
          to preserve true chronological ages, provide pre-binned integer tensors.
          If omitted from inputs, age embeddings are skipped.
        - ``segment``: Visit segment indicators (integer tensor of shape
          ``[batch_size, seq_len]`` or ``[batch_size]``) distinguishing consecutive
          visits (e.g. ``[0, 0, 1, 1, 2]``). If omitted, segment embeddings are skipped.
        When neither ``age`` nor ``segment`` is provided, BEHRT functions as a
        position-aware Transformer baseline over medical code sequences.

    Sequence pooling:
        - ``"cls"`` (default): Uses the hidden representation of the first code token
          (index 0) in the sequence as the patient summary embedding.
        - ``"mean"``: Computes the average of hidden representations across all valid
          tokens according to the padding mask.

    Args:
        dataset: PyHealth dataset containing features and labels.
        embedding_dim: Dimensionality of embeddings. Default is 128.
        heads: Number of attention heads. Default is 4.
        dropout: Dropout rate. Default is 0.1.
        num_layers: Number of transformer encoder layers. Default is 2.
        max_age: Maximum patient age index. Default is 120.
        max_segments: Maximum visit segments. Default is 64.
        max_seq_len: Maximum token sequence length. Default is 512.
        pooling: Sequence pooling method: ``"cls"`` (first token) or
            ``"mean"`` (average over valid tokens). Default is ``"cls"``.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import BEHRT
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
        ...     dataset_name="demo",
        ... )
        >>> model = BEHRT(dataset=dataset, embedding_dim=64, heads=2, num_layers=1)
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
        num_layers: int = 2,
        max_age: int = 120,
        max_segments: int = 64,
        max_seq_len: int = 512,
        pooling: str = "cls",
    ) -> None:
        super().__init__(dataset=dataset)
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        if embedding_dim % heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by heads ({heads})."
            )

        assert len(self.label_keys) == 1, "BEHRT only supports a single label key"
        self.label_key = self.label_keys[0]

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.max_age = max_age
        self.max_segments = max_segments
        self.max_seq_len = max_seq_len
        self.pooling = pooling

        # Isolate code feature keys to protect temporal auxiliary inputs
        self.code_feature_keys = [
            k for k in self.feature_keys if k not in ("age", "segment")
        ]
        if not self.code_feature_keys:
            raise ValueError(
                "BEHRT requires at least one code feature stream in dataset input_schema."
            )

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.behrt_embeddings: nn.ModuleDict = nn.ModuleDict()
        self.behrt_layers: nn.ModuleDict = nn.ModuleDict()

        for feature_key in self.code_feature_keys:
            self.behrt_embeddings[feature_key] = BEHRTEmbedding(
                embedding_dim=embedding_dim,
                max_age=max_age,
                max_segments=max_segments,
                max_position=max_seq_len,
                dropout=dropout,
            )
            self.behrt_layers[feature_key] = BEHRTLayer(
                hidden_size=embedding_dim,
                heads=heads,
                dropout=dropout,
                num_layers=num_layers,
            )

        self.pooler = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )

        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(len(self.code_feature_keys) * embedding_dim, output_size)

    def _pool_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens nested code sequences into ``[batch, seq_len, hidden]``."""
        while x.dim() > 3:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
        return x

    @staticmethod
    def _mask_from_embeddings(x: torch.Tensor) -> torch.Tensor:
        """Derives a boolean token validity mask from embedded representations."""
        mask = torch.any(torch.abs(x) > 0, dim=-1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return mask.bool()

    def forward_from_embedding(
        self,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass from pre-embedded feature tensors.

        Args:
            **kwargs: Feature and target tensors. May include ``"age"`` and
                ``"segment"`` tensors.

        Returns:
            Dictionary containing ``"logit"``, ``"y_prob"``, and optionally
            ``"loss"``, ``"y_true"``, and ``"embed"``.
        """
        patient_embeddings: list[torch.Tensor] = []
        age_ids = kwargs.get("age", None)
        segment_ids = kwargs.get("segment", None)

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
                    f"Feature '{feature_key}' must contain 'value' in schema."
                )

            value = value.to(self.device)
            value = self._pool_embedding(value)

            if mask is not None:
                mask = mask.to(self.device).bool()
                while mask.dim() > 2:
                    mask = mask.any(dim=-1)
                if mask.size(1) > self.max_seq_len:
                    mask = mask[:, : self.max_seq_len]
            else:
                mask = self._mask_from_embeddings(value).to(self.device)

            combined_embs = self.behrt_embeddings[feature_key](
                code_embeddings=value,
                age_ids=age_ids,
                segment_ids=segment_ids,
            )

            encoded = self.behrt_layers[feature_key](combined_embs, mask=mask)

            if self.pooling == "cls":
                # First code token (index 0) is used as patient summary representation
                pooled = encoded[:, 0, :]
            else:
                mask_expanded = mask.unsqueeze(-1).float()
                denom = mask_expanded.sum(dim=1).clamp(min=1.0)
                pooled = (encoded * mask_expanded).sum(dim=1) / denom

            pooled = self.pooler(pooled)
            patient_embeddings.append(pooled)

        patient_rep = torch.cat(patient_embeddings, dim=1)
        logits = self.fc(patient_rep)
        y_prob = self.label_processor.activation(logits)

        results: dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            loss = self.label_processor.loss(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        if kwargs.get("embed", False):
            results["embed"] = patient_rep

        return results

    def forward(
        self,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Encodes raw patient events through BEHRT and predicts output label.

        Args:
            **kwargs: Raw sample dictionary or batch of features, labels, and
                optional ``"age"`` and ``"segment"`` identifiers.

        Returns:
            Dictionary containing ``"logit"``, ``"y_prob"``, and optionally
            ``"loss"``, ``"y_true"``, and ``"embed"``.
        """
        forward_kwargs = dict(kwargs)

        # Protect raw age and segment inputs from being overwritten by EmbeddingModel
        raw_age = kwargs.get("age", None)
        raw_segment = kwargs.get("segment", None)

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

        if raw_age is not None:
            forward_kwargs["age"] = raw_age
        if raw_segment is not None:
            forward_kwargs["segment"] = raw_segment

        return self.forward_from_embedding(**forward_kwargs)
