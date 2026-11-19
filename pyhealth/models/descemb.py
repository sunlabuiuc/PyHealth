"""DescEmb: Description-based Medical Code Embeddings for EHR.

Paper: Sengmin Yoon, Taeuk Kim, Jinyoung Yeo, Dongha Lee, Hwanjo Yu.
Description-based Medical Code Embeddings for Electronic Health Records.
Machine Learning for Healthcare (MLHC 2022). Proceedings of Machine
Learning Research (PMLR) 182:700-721. https://arxiv.org/abs/2208.06648

Traditional Electronic Health Record (EHR) predictive architectures assign
arbitrary integer indices to medical concepts (e.g., ICD-9, ICD-10, OMOP)
and train code embeddings from scratch. This introduces two severe
limitations:
    1. Zero Cross-System Transferability: Embeddings trained on one coding
       system (e.g., ICD-9 on MIMIC-III) cannot transfer to another coding
       system (e.g., ICD-10 or OMOP on eICU / MIMIC-IV).
    2. Semantic Ignorance: Random index embeddings discard rich lexical and
       clinical semantics available in official code descriptions (e.g.,
       "Essential (primary) hypertension").

DescEmb overcomes these limitations by initializing code embeddings from
the standardized textual descriptions of medical codes using a text encoder
(such as a bidirectional GRU). When initialized with a user-supplied
pretrained text encoder, this injects learned clinical text semantics into the
code embedding layer prior to longitudinal sequence modeling; with the default
un-pretrained encoder, initialization acts as a pseudo-random projection.

The PyHealth DescEmb architecture comprises:
    1. Description Text Encoding (:class:`DescEmbCodeEncoder`): Embeds
       medical code descriptions using word-level tokenization and a
       bidirectional recurrent neural network (BiGRU). The forward and
       backward endpoints are pooled and projected to form description-based
       code vectors.
    2. Input Feature Embedding & Intra-Visit Pooling: Projects code
       sequences into continuous representations via
       :class:`~pyhealth.models.EmbeddingModel`. Nested visit sequences
       are pooled along the event dimension using configurable intra-visit
       pooling (sum, mean, max, or attention).
    3. Value Conditioning (Optional): Fuses continuous numerical measurements
       (e.g., lab values, vitals) with code embeddings.
    4. Longitudinal Visit Modeling: A temporal sequence model (GRU,
       LSTM, or Transformer) models the patient's multi-visit trajectory
       over time.
    5. Outcome Prediction: Patient representations from the last valid
       visit are mapped to outcome predictions across binary, multiclass,
       and multilabel tasks.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.utils import get_last_visit


def simple_word_tokenize(
    text: str, max_tokens: int = 16
) -> list[str]:
    """Tokenizes a clinical description string into lowercase alphanumeric words.

    Args:
        text: Clinical description string.
        max_tokens: Maximum number of tokens to keep. Default is 16.

    Returns:
        List of lowercase string tokens.

    Examples:
        >>> simple_word_tokenize("Essential (primary) hypertension")
        ['essential', 'primary', 'hypertension']
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if t]
    return tokens[:max_tokens]


simple_subword_tokenize = simple_word_tokenize


class DescEmbCodeEncoder(nn.Module):
    """Encodes medical code text descriptions into semantic code embeddings.

    Uses an embedding layer over description word tokens followed by a
    bidirectional GRU. Forward and backward representations are pooled via
    endpoint concatenation (forward last state and backward first state)
    and projected to the target embedding dimension.

    Args:
        vocab_size: Size of description word vocabulary. Default is 5000.
        subword_dim: Dimensionality of word embeddings. Default is 64.
        hidden_dim: Hidden dimension of bidirectional GRU. Default is 64.
        output_dim: Target dimension of output code embeddings. Default is 128.
        dropout: Dropout rate. Default is 0.1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.descemb import DescEmbCodeEncoder
        >>> encoder = DescEmbCodeEncoder(
        ...     vocab_size=100, subword_dim=16, hidden_dim=16, output_dim=32
        ... )
        >>> token_ids = torch.randint(1, 90, (4, 8))
        >>> code_embs = encoder(token_ids)
        >>> code_embs.shape
        torch.Size([4, 32])
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        subword_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if subword_dim <= 0:
            raise ValueError(f"subword_dim must be positive, got {subword_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.vocab_size = vocab_size
        self.subword_dim = subword_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.subword_embedding = nn.Embedding(
            vocab_size, subword_dim, padding_idx=0
        )
        self.gru = nn.GRU(
            subword_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass encoding description token IDs to code representations.

        Args:
            input_ids: Tensor of token IDs of shape ``(num_codes, max_len)``
                or ``(batch, seq_len, max_len)``.

        Returns:
            Encoded code embeddings of shape ``(..., output_dim)``.
        """
        orig_shape = input_ids.shape
        flat_ids = input_ids.reshape(-1, orig_shape[-1])
        batch_size, max_len = flat_ids.shape

        mask = flat_ids != 0
        lengths = mask.sum(dim=-1).clamp(min=1).cpu()

        x = self.dropout(self.subword_embedding(flat_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=max_len
        )

        # Endpoint pooling: forward last token and backward first token
        batch_idx = torch.arange(batch_size, device=out.device)
        last_step_idx = (lengths - 1).to(out.device)
        f_last = out[batch_idx, last_step_idx, : self.hidden_dim]
        b_first = out[batch_idx, 0, self.hidden_dim :]
        pooled = torch.cat([f_last, b_first], dim=-1)

        projected = self.proj(pooled)
        return projected.view(*orig_shape[:-1], self.output_dim)


class DescEmb(BaseModel):
    """DescEmb: Description-based Medical Code Embeddings for EHR.

    Paper: Sengmin Yoon, Taeuk Kim, Jinyoung Yeo, Dongha Lee, Hwanjo Yu.
    Description-based Medical Code Embeddings for Electronic Health Records.
    Machine Learning for Healthcare (MLHC 2022). https://arxiv.org/abs/2208.06648

    DescEmb addresses the cross-system transfer and semantic gap of standard
    code embeddings by encoding medical concepts through their textual
    descriptions. It uses :class:`DescEmbCodeEncoder` to initialize medical
    concept embeddings from clinical text descriptions (via a BiGRU over
    word tokens). When provided with a pretrained text encoder, this injects
    learned clinical text semantics into the embedding layer prior to
    longitudinal sequence modeling; with an un-pretrained encoder,
    initialization serves as a pseudo-random projection. During training, these
    description-initialized embeddings are fine-tuned as part of the predictive
    model.

    Args:
        dataset: The dataset to train the model.
        feature_keys: Names of input feature keys to use from dataset.
        label_key: Name of the label key from dataset.
        embedding_dim: Dimension of code and visit representations.
            Default is 128.
        text_hidden_dim: Hidden dimension of text encoder. Default is 64.
        subword_vocab_size: Vocabulary size for description word tokenizer.
            Default is 5000.
        temporal_hidden_dim: Hidden dimension of patient sequence model.
            Default is 128.
        temporal_model: Type of sequence model ('gru', 'lstm', 'transformer').
            Default is 'gru'.
        num_layers: Number of temporal recurrent or transformer layers.
            Default is 1.
        intra_visit_pooling: Method to pool codes within a visit ('sum',
            'mean', 'max', 'attention'). Default is 'sum'.
        dropout: Dropout probability. Default is 0.1.
        code_descriptions: Optional dictionary mapping code strings to
            clinical text descriptions (e.g., ``{"401.9": "hypertension"}``).
            Missing codes fall back to their raw string representation.
        pretrained_text_encoder: Optional pretrained module for encoding
            description text. If None, a newly initialized
            :class:`DescEmbCodeEncoder` is created.
        text_vocab: Optional pre-defined vocabulary dictionary mapping word
            tokens to integer IDs, matching ``pretrained_text_encoder``. If
            not provided, token IDs are assigned dynamically in first-seen
            order starting from 2 (with 0 reserved for ``<pad>`` and 1 for
            ``<unk>``).
        **kwargs: Additional keyword arguments.

    Note:
        When using randomly initialized :class:`DescEmbCodeEncoder` (the
        default when `pretrained_text_encoder` is None), description-based
        initialization functions as a pseudo-random projection (dependent on
        the initialization seed) of description tokens into embedding space.
        Learned clinical semantics are only introduced when users provide a
        pretrained module via `pretrained_text_encoder` and matching `text_vocab`,
        or load weights into `self.code_text_encoder`. Codes not found in
        `code_descriptions` fall back to using the code string itself as the
        description.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import DescEmb
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": [["c1", "c2"], ["c3"]],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "conditions": [["c2"], ["c3"]],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = DescEmb(
        ...     dataset=dataset,
        ...     embedding_dim=16,
        ...     text_hidden_dim=16,
        ...     temporal_hidden_dim=16,
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> batch = next(iter(loader))
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['embed', 'logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: list[str] | None = None,
        label_key: str | None = None,
        embedding_dim: int = 128,
        text_hidden_dim: int = 64,
        subword_vocab_size: int = 5000,
        temporal_hidden_dim: int = 128,
        temporal_model: str = "gru",
        num_layers: int = 1,
        intra_visit_pooling: str = "sum",
        dropout: float = 0.1,
        code_descriptions: dict[str, str] | None = None,
        pretrained_text_encoder: nn.Module | None = None,
        text_vocab: dict[str, int] | None = None,
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
        if text_hidden_dim <= 0:
            raise ValueError(
                f"text_hidden_dim must be positive, got {text_hidden_dim}"
            )
        if temporal_hidden_dim <= 0:
            raise ValueError(
                f"temporal_hidden_dim must be positive, got {temporal_hidden_dim}"
            )

        if feature_keys is not None:
            self.feature_keys = feature_keys
        if label_key is not None:
            self.label_key = label_key
        else:
            self.label_key = self.label_keys[0]

        self.embedding_dim = embedding_dim
        self.text_hidden_dim = text_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.intra_visit_pooling = intra_visit_pooling
        self.temporal_model_name = temporal_model

        # Description text encoder
        if pretrained_text_encoder is not None:
            self.code_text_encoder = pretrained_text_encoder
        else:
            self.code_text_encoder = DescEmbCodeEncoder(
                vocab_size=subword_vocab_size,
                subword_dim=min(64, embedding_dim),
                hidden_dim=text_hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout,
            )
        # Freeze text encoder parameters since used only for code initialization
        for param in self.code_text_encoder.parameters():
            param.requires_grad = False

        # Underlying embedding model
        self.embedding_model = EmbeddingModel(
            dataset=dataset, embedding_dim=embedding_dim
        )

        # Word vocabulary index map for descriptions
        if text_vocab is not None:
            self.subword_to_id: dict[str, int] = dict(text_vocab)
            self.text_vocab_fixed = True
        else:
            self.subword_to_id: dict[str, int] = {"<pad>": 0, "<unk>": 1}
            self.next_subword_id = 2
            self.text_vocab_fixed = False

        # Initialize code embeddings with descriptions if provided
        if code_descriptions is not None:
            self.initialize_with_descriptions(code_descriptions)

        # Intra-visit attention pooling projection (if enabled)
        if self.intra_visit_pooling == "attention":
            self.intra_attn = nn.Linear(embedding_dim, 1)

        # Multi-feature fusion projection
        num_features = len(self.feature_keys)
        fused_dim = num_features * embedding_dim
        self.feat_proj = nn.Linear(fused_dim, temporal_hidden_dim)

        # Optional value conditioning layer (for continuous values)
        self.val_proj = nn.Linear(1, embedding_dim)
        self.post_val_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        # Temporal sequence model
        if temporal_model == "gru":
            self.temporal_encoder = nn.GRU(
                temporal_hidden_dim,
                temporal_hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif temporal_model == "lstm":
            self.temporal_encoder = nn.LSTM(
                temporal_hidden_dim,
                temporal_hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # transformer
            nhead = 2 if temporal_hidden_dim % 2 == 0 else 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=temporal_hidden_dim,
                nhead=nhead,
                dim_feedforward=temporal_hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

        # Outcome classification head
        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(temporal_hidden_dim, output_size)

    def _tokenize_text(self, text: str, max_len: int = 16) -> list[int]:
        """Converts a description string to word token IDs."""
        tokens = simple_word_tokenize(text, max_tokens=max_len)
        ids = []
        for t in tokens:
            if self.text_vocab_fixed:
                unk_id = self.subword_to_id.get("<unk>", 1)
                ids.append(self.subword_to_id.get(t, unk_id))
            else:
                if t not in self.subword_to_id:
                    if self.next_subword_id < self.code_text_encoder.vocab_size:
                        self.subword_to_id[t] = self.next_subword_id
                        self.next_subword_id += 1
                    else:
                        self.subword_to_id[t] = 1  # <unk>
                ids.append(self.subword_to_id[t])
        if len(ids) < max_len:
            pad_id = self.subword_to_id.get("<pad>", 0)
            ids = ids + [pad_id] * (max_len - len(ids))
        return ids[:max_len]

    def initialize_with_descriptions(
        self, code_descriptions: dict[str, str], max_desc_len: int = 16
    ) -> None:
        """Initializes code embeddings using their textual descriptions.

        Args:
            code_descriptions: Dictionary mapping code strings to descriptions.
                Codes missing from the dictionary fall back to the raw code string.
            max_desc_len: Maximum tokens per description. Default is 16.
        """
        was_training = self.code_text_encoder.training
        self.code_text_encoder.eval()
        try:
            device = self._dummy_param.device
            for feat_key in self.feature_keys:
                if feat_key not in self.embedding_model.embedding_layers:
                    continue
                emb_layer = self.embedding_model.embedding_layers[feat_key]
                if not isinstance(emb_layer, nn.Embedding):
                    continue

                processor = self.dataset.input_processors.get(feat_key)
                if not hasattr(processor, "code_vocab"):
                    continue

                vocab = processor.code_vocab
                token_ids_list: list[list[int]] = []
                for code in vocab:
                    desc = code_descriptions.get(str(code), str(code))
                    token_ids_list.append(self._tokenize_text(desc, max_desc_len))

                token_tensor = torch.tensor(
                    token_ids_list, dtype=torch.long, device=device
                )
                with torch.no_grad():
                    desc_embs = self.code_text_encoder(token_tensor)
                    emb_layer.weight.data.copy_(desc_embs)
                    emb_layer.weight.data[0].zero_()
        finally:
            if was_training:
                self.code_text_encoder.train()

    def _pool_intra_visit(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pools events/codes within each visit along dimension 2."""
        if self.intra_visit_pooling == "sum":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
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
            weights = self.intra_attn(x)
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
        """Forward propagation for DescEmb.

        Args:
            **kwargs: Keyword arguments containing input feature tensors and
                optionally the label tensor and numeric values.

        Returns:
            Dictionary containing:
                - loss: Training loss (if labels provided).
                - y_prob: Predicted outcome probabilities.
                - y_true: Ground truth outcome labels (if provided).
                - logit: Raw output logits.
                - embed: Patient-level representations from the last valid visit.
        """
        cleaned_kwargs: dict[str, Any] = {}
        raw_masks: dict[str, torch.Tensor | None] = {}
        values_dict: dict[str, torch.Tensor | None] = {}
        input_masks: dict[str, torch.Tensor | None] = {}

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

                time_tensor = None
                numeric_val = None
                mask = None
                value = None

                if (
                    schema is not None
                    and "time" in schema
                    and "value" in schema
                ):
                    value = val[schema.index("value")]
                    time_tensor = val[schema.index("time")]
                    if "mask" in schema:
                        mask = val[schema.index("mask")]
                elif len(val) == 2:
                    first, second = val[0], val[1]
                    if (
                        isinstance(second, torch.Tensor)
                        and second.dtype == torch.bool
                    ):
                        # (value, mask)
                        value = first
                        mask = second
                    elif (
                        isinstance(first, torch.Tensor)
                        and first.is_floating_point()
                        and isinstance(second, torch.Tensor)
                        and not second.is_floating_point()
                    ):
                        # (time, value_codes)
                        time_tensor = first
                        value = second
                    elif (
                        isinstance(first, torch.Tensor)
                        and not first.is_floating_point()
                        and isinstance(second, torch.Tensor)
                        and second.is_floating_point()
                    ):
                        # (value_codes, numeric_val)
                        value = first
                        numeric_val = second
                    else:
                        value = first
                        mask = (
                            second if isinstance(second, torch.Tensor)
                            else None
                        )
                elif len(val) >= 3:
                    time_tensor = val[0]
                    value = val[1]
                    mask = val[2]
                elif schema is not None and "value" in schema:
                    value = val[schema.index("value")]
                else:
                    value = val[0]

                if numeric_val is None:
                    for elem in val:
                        if (
                            isinstance(elem, torch.Tensor)
                            and elem.is_floating_point()
                            and elem is not time_tensor
                            and elem is not value
                        ):
                            numeric_val = elem
                            break

                values_dict[key] = numeric_val
                cleaned_kwargs[key] = value
                raw_masks[key] = mask if isinstance(mask, torch.Tensor) else None
            else:
                cleaned_kwargs[key] = val
                raw_masks[key] = None
                values_dict[key] = None

            if (
                isinstance(cleaned_kwargs[key], torch.Tensor)
                and not cleaned_kwargs[key].is_floating_point()
            ):
                input_masks[key] = cleaned_kwargs[key] != 0
            else:
                input_masks[key] = None

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

            # Optional value conditioning: fuse numerical measurement
            val = values_dict.get(feature_key)
            if val is not None and val.shape == feat.shape[:-1]:
                val_emb = self.val_proj(val.unsqueeze(-1).float())
                feat = self.post_val_proj(torch.cat([feat, val_emb], dim=-1))

            v_mask = None
            # Nested sequences: (B, T, K, D)
            if feat.dim() == 4:
                mask = raw_masks.get(feature_key)
                if mask is None:
                    mask = input_masks.get(feature_key)
                if mask is not None and mask.dim() == 3:
                    code_mask = mask != 0
                else:
                    code_mask = (feat != 0).any(dim=-1)
                v_mask = code_mask.any(dim=-1)
                feat = feat.masked_fill(~code_mask.unsqueeze(-1), 0.0)
                feat = self._pool_intra_visit(feat, code_mask)

            # Flat sequences: (B, D) -> (B, 1, D)
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)

            if v_mask is None:
                mask = raw_masks.get(feature_key)
                if mask is None:
                    mask = input_masks.get(feature_key)
                if mask is not None and mask.dim() == feat.dim() - 1:
                    v_mask = mask != 0
                else:
                    v_mask = (feat != 0).any(dim=-1)
            feat = feat.masked_fill(~v_mask.unsqueeze(-1), 0.0)

            feat_visits.append(feat)
            visit_masks.append(v_mask)

        # Concatenate features along embedding dimension
        if len(feat_visits) > 1:
            max_t = max(f.shape[1] for f in feat_visits)
            aligned_feats = []
            for f in feat_visits:
                if f.shape[1] < max_t:
                    pad_len = max_t - f.shape[1]
                    f = F.pad(f, (0, 0, 0, pad_len))
                aligned_feats.append(f)
            combined = torch.cat(aligned_feats, dim=-1)

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

        visit_repr = self.feat_proj(combined)
        if overall_mask is not None:
            visit_repr = visit_repr.masked_fill(~overall_mask.unsqueeze(-1), 0.0)

        # Temporal sequence modeling
        if self.temporal_model_name in ("gru", "lstm"):
            temporal_out, _ = self.temporal_encoder(visit_repr)
        else:
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
                visit_repr, src_key_padding_mask=padding_mask
            )
            if all_pad_patients is not None and all_pad_patients.any():
                temporal_out = temporal_out.masked_fill(
                    all_pad_patients.unsqueeze(-1).unsqueeze(-1), 0.0
                )

        # Extract patient representation from the last valid visit
        patient_emb = get_last_visit(temporal_out, overall_mask)

        logits = self.fc(patient_emb)
        y_prob = self.label_processor.activation(logits)

        results: dict[str, torch.Tensor] = {
            "y_prob": y_prob,
            "logit": logits,
            "embed": patient_emb,
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            if isinstance(y_true, tuple):
                y_true = y_true[0]
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.as_tensor(y_true, device=logits.device)
            loss = self.label_processor.loss(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results
