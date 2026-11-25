"""Globally Aligned Medical Embeddings (GAME) for Multi-institutional EHR Data.

Paper: Tong Han et al. "Representation learning to advance multi-institutional
studies with electronic health record data from US and France."
Nature Communications (2025). https://doi.org/10.1038/s41467-025-56837-1
Official repository: https://github.com/celehs/GAME
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


def normalize_game_code(code: str) -> str:
    """Normalizes a clinical code for GAME embedding lookup.

    Strips common vocabulary prefix tags (e.g., 'ICD10CM:', 'CCS:', 'LOINC:',
    'RxNorm:'), removes leading/trailing whitespace, and converts to lowercase.

    Args:
        code: The raw clinical code string.

    Returns:
        The normalized code string.

    Examples:
        >>> from pyhealth.models.game import normalize_game_code
        >>> normalize_game_code("ICD10CM:I10")
        'i10'
        >>> normalize_game_code("CCS:1")
        '1'
        >>> normalize_game_code(" LOINC:15069-8 ")
        '15069-8'
        >>> normalize_game_code("401.9")
        '401.9'
    """
    code_str = str(code).strip()
    if not code_str:
        return ""
    # Strip prefix before colon (e.g., 'ICD10CM:', 'CCS:', 'LOINC:')
    if ":" in code_str:
        code_str = code_str.split(":", 1)[1].strip()
    return code_str.lower()


def load_game_embeddings(
    source: dict[str, torch.Tensor | list[float]] | str | torch.Tensor,
    embedding_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Loads GAME code embeddings from a dictionary, file path, or tensor.

    Supported file formats:
    - PyTorch saved object (``.pt`` or ``.pth``): dict of tensors or single tensor.
    - NumPy file (``.npy`` or ``.npz``): ndarray or archive of arrays.
    - Delimited text/CSV file (``.csv``, ``.tsv``, ``.txt``): code followed by floats.

    Args:
        source: Dictionary mapping code to vectors, a file path, or a 2D tensor.
        embedding_dim: Expected embedding dimension. If None, inferred from source.

    Returns:
        Dictionary mapping normalized code strings to 1D float tensors.

    Examples:
        >>> import torch
        >>> from pyhealth.models.game import load_game_embeddings
        >>> raw = {"i10": torch.randn(16), "401.9": torch.randn(16)}
        >>> embs = load_game_embeddings(raw)
        >>> "i10" in embs
        True
        >>> embs["i10"].shape
        torch.Size([16])
    """
    if isinstance(source, dict):
        result: dict[str, torch.Tensor] = {}
        for k, v in source.items():
            norm_k = normalize_game_code(k)
            if not norm_k:
                continue
            if isinstance(v, torch.Tensor):
                t = v.detach().float().view(-1)
            else:
                t = torch.tensor(v, dtype=torch.float32).view(-1)
            if embedding_dim is not None and t.shape[0] != embedding_dim:
                raise ValueError(
                    f"Vector for '{k}' has dimension {t.shape[0]}, "
                    f"expected {embedding_dim}."
                )
            result[norm_k] = t
        return result

    if isinstance(source, torch.Tensor):
        if source.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {source.shape}.")
        return {
            str(i): source[i].detach().float().view(-1) for i in range(source.shape[0])
        }

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"GAME embedding file not found: {source}")

        ext = os.path.splitext(source)[1].lower()
        if ext in (".pt", ".pth"):
            loaded = torch.load(source, map_location="cpu", weights_only=True)
            return load_game_embeddings(loaded, embedding_dim=embedding_dim)

        if ext == ".npy":
            arr = np.load(source)
            t = torch.from_numpy(arr).float()
            return load_game_embeddings(t, embedding_dim=embedding_dim)

        if ext == ".npz":
            archive = np.load(source)
            if "embeddings" in archive:
                arr = archive["embeddings"]
                codes = archive.get("codes", None)
            else:
                first_key = archive.files[0]
                arr = archive[first_key]
                codes = None
            if codes is not None:
                d = {
                    str(c): torch.from_numpy(arr[i]).float()
                    for i, c in enumerate(codes)
                }
                return load_game_embeddings(d, embedding_dim=embedding_dim)
            return load_game_embeddings(
                torch.from_numpy(arr).float(), embedding_dim=embedding_dim
            )

        # Plain text or CSV
        result = {}
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[\s,]+", line)
                if len(parts) < 2:
                    continue
                code = normalize_game_code(parts[0])
                if not code:
                    continue
                try:
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                vec = torch.tensor(vals, dtype=torch.float32)
                if embedding_dim is not None and vec.shape[0] != embedding_dim:
                    continue
                result[code] = vec
        return result

    raise TypeError(f"Unsupported source type: {type(source)}.")


def init_embedding_with_game(
    embedding: nn.Embedding,
    code_vocab: dict[Any, int],
    game_embeddings: dict[str, torch.Tensor | list[float]] | str | torch.Tensor,
    code_mapping: dict[str, str] | None = None,
    freeze: bool = False,
    normalize: bool = False,
    padding_idx: int = 0,
    pad_token: str = "<pad>",
) -> int:
    """Initializes an ``nn.Embedding`` layer with GAME harmonized embeddings.

    Tokens in ``code_vocab`` matching entries in ``game_embeddings`` have their
    embedding row replaced with the pretrained vector. Tokens without matching
    pretrained vectors retain their existing initialization. The padding row
    at ``padding_idx`` is explicitly zeroed.

    Args:
        embedding: Target ``nn.Embedding`` layer to initialize.
        code_vocab: Dictionary mapping code tokens to integer indices.
        game_embeddings: Pretrained GAME embeddings (dict, file path, or tensor).
        code_mapping: Optional dict mapping dataset codes to GAME code identifiers.
        freeze: Whether to freeze embedding weights (requires_grad=False).
            Default is False.
        normalize: Whether to L2-normalize loaded vectors. Default is False.
        padding_idx: Index of the padding token. Default is 0.
        pad_token: String representation of the padding token. Default is '<pad>'.

    Returns:
        The number of vocabulary tokens successfully initialized from GAME.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.models.game import init_embedding_with_game
        >>> vocab = {"<pad>": 0, "c1": 1, "c2": 2, "c3": 3}
        >>> embs = {"c1": torch.ones(8), "c2": torch.full((8,), 2.0)}
        >>> layer = nn.Embedding(len(vocab), 8, padding_idx=0)
        >>> matched = init_embedding_with_game(layer, vocab, embs)
        >>> matched
        2
        >>> (layer.weight[0] == 0.0).all().item()
        True
        >>> (layer.weight[1] == 1.0).all().item()
        True
    """
    dim = embedding.embedding_dim
    vectors = load_game_embeddings(game_embeddings, embedding_dim=dim)

    matched_count = 0
    with torch.no_grad():
        for token, idx in code_vocab.items():
            if str(token) == pad_token or idx == padding_idx:
                embedding.weight[idx].zero_()
                continue

            query_code = normalize_game_code(str(token))
            if code_mapping is not None and str(token) in code_mapping:
                query_code = normalize_game_code(code_mapping[str(token)])

            vec = vectors.get(query_code, None)
            if vec is None:
                continue

            if vec.shape[0] != dim:
                continue

            if normalize:
                norm = vec.norm(p=2).clamp_min(1e-12)
                vec = vec / norm

            embedding.weight[idx].copy_(vec)
            matched_count += 1

        if padding_idx is not None and 0 <= padding_idx < embedding.num_embeddings:
            embedding.weight[padding_idx].zero_()

    if freeze:
        embedding.weight.requires_grad = False

    return matched_count


class GAMEEmbedding(nn.Module):
    """Drop-in embedding layer initialized with GAME harmonized medical vectors.

    Wraps an ``nn.Embedding`` layer with GAME initialization and provides
    optional dimension projection, L2-normalization, and weight freezing.

    Args:
        vocab_size: Size of the code vocabulary (number of embeddings).
        embedding_dim: Dimensionality of the base code embeddings.
        code_vocab: Optional code vocabulary mapping token strings to indices.
        game_embeddings: Optional GAME pretrained embeddings (dict, path, tensor).
        code_mapping: Optional dictionary mapping dataset tokens to GAME codes.
        projection_dim: Optional target dimension. If specified and different
            from ``embedding_dim``, a linear projection is applied.
        freeze: Whether to freeze embedding weights. Default is False.
        normalize: Whether to L2-normalize embedding vectors. Default is False.
        padding_idx: Vocabulary index for padding tokens. Default is 0.

    Examples:
        >>> import torch
        >>> from pyhealth.models.game import GAMEEmbedding
        >>> vocab = {"<pad>": 0, "I10": 1, "E11": 2}
        >>> raw_embs = {"i10": torch.ones(16), "e11": torch.full((16,), 3.0)}
        >>> layer = GAMEEmbedding(
        ...     vocab_size=3,
        ...     embedding_dim=16,
        ...     code_vocab=vocab,
        ...     game_embeddings=raw_embs,
        ...     projection_dim=32,
        ... )
        >>> x = torch.tensor([[1, 2], [2, 0]])
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 2, 32])
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        code_vocab: dict[Any, int] | None = None,
        game_embeddings: (
            dict[str, torch.Tensor | list[float]] | str | torch.Tensor | None
        ) = None,
        code_mapping: dict[str, str] | None = None,
        projection_dim: int | None = None,
        freeze: bool = False,
        normalize: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim or embedding_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.matched_tokens = 0
        if game_embeddings is not None and code_vocab is not None:
            self.matched_tokens = init_embedding_with_game(
                embedding=self.embedding,
                code_vocab=code_vocab,
                game_embeddings=game_embeddings,
                code_mapping=code_mapping,
                freeze=freeze,
                normalize=normalize,
                padding_idx=padding_idx,
            )
        elif freeze:
            self.embedding.weight.requires_grad = False

        if self.projection_dim != self.embedding_dim:
            self.projection: nn.Module = nn.Sequential(
                nn.Linear(self.embedding_dim, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds input indices and applies optional projection.

        Args:
            x: Input tensor of token indices with shape ``(B,)``, ``(B, L)``,
               or ``(B, L, C)``.

        Returns:
            Embedded tensor with shape ``(*x.shape, output_dim)``.
        """
        embs = self.embedding(x)
        return self.projection(embs)


class GAME(BaseModel):
    """Globally Aligned Medical Embeddings (GAME) model for EHR event prediction.

    Paper: Tong Han et al. "Representation learning to advance multi-institutional
    studies with electronic health record data from US and France."
    Nature Communications (2025). https://doi.org/10.1038/s41467-025-56837-1

    Integrates GAME harmonized medical code embeddings as drop-in pretrained
    representations for clinical codes (MIMIC-IV, OMOP, ICD, CCS, LOINC, RxNorm),
    encodes longitudinal visit histories using a temporal recurrent or Transformer
    backbone, and predicts patient health outcomes.

    Args:
        dataset: PyHealth SampleDataset containing patient samples and schemas.
        feature_keys: Optional list of feature keys to use from dataset.
            Defaults to all sequential feature keys in the dataset.
        label_key: Optional label key to predict. Defaults to the first label key.
        embedding_dim: Dimensionality of GAME code embeddings. Default is 256.
        game_embeddings: Pretrained GAME embeddings as a dictionary, file path,
            or dictionary mapping feature keys to embeddings. If None, initialized
            with standard random embeddings.
        code_mapping: Optional dictionary mapping dataset tokens to GAME codes.
        freeze_embeddings: Whether to freeze embedding weights during training.
            Default is False.
        normalize_embeddings: Whether to L2-normalize loaded GAME embeddings.
            Default is False.
        hidden_dim: Hidden dimension of temporal model. Default is 128.
        temporal_model: Type of temporal sequence backbone (``"GRU"``, ``"LSTM"``,
            or ``"Transformer"``). Default is ``"GRU"``.
        nhead: Number of attention heads for Transformer backbone. If None,
            automatically picked to divide the input dimension. Default is None.
        pooling_mode: Strategy to aggregate visit representations across time
            (``"attention"``, ``"mean"``, ``"max"``, or ``"last"``).
            Default is ``"attention"``.
        dropout: Dropout probability applied to representations. Default is 0.1.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import GAME
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "conditions": [["c1", "c2"], ["c3"]], "label": 1},
        ...     {"patient_id": "p1", "visit_id": "v1",
        ...      "conditions": [["c2"], ["c3", "c4"]], "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = GAME(
        ...     dataset=dataset,
        ...     embedding_dim=16,
        ...     hidden_dim=32,
        ...     temporal_model="GRU",
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
        embedding_dim: int = 256,
        game_embeddings: (
            dict[str, Any] | str | torch.Tensor | dict[str, dict[str, Any]] | None
        ) = None,
        code_mapping: dict[str, str] | None = None,
        freeze_embeddings: bool = False,
        normalize_embeddings: bool = False,
        hidden_dim: int = 128,
        temporal_model: str = "GRU",
        nhead: int | None = None,
        pooling_mode: str = "attention",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(dataset=dataset)

        if feature_keys is not None:
            self.feature_keys = feature_keys
        if label_key is not None:
            self.label_key = label_key
        else:
            self.label_key = self.label_keys[0]

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temporal_model not in ("GRU", "LSTM", "Transformer"):
            raise ValueError(
                f"Unsupported temporal_model: {temporal_model}. "
                "Choose from 'GRU', 'LSTM', 'Transformer'."
            )
        if pooling_mode not in ("attention", "mean", "max", "last"):
            raise ValueError(
                f"Unsupported pooling_mode: {pooling_mode}. "
                "Choose from 'attention', 'mean', 'max', 'last'."
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.temporal_model_type = temporal_model
        self.pooling_mode = pooling_mode
        self.dropout_rate = dropout

        # Build GAMEEmbedding layers for each sequential feature
        self.game_embeddings = nn.ModuleDict()
        for f_key in self.feature_keys:
            proc = self.dataset.input_processors[f_key]
            vocab = getattr(proc, "code_vocab", None)
            if vocab is None:
                raise ValueError(
                    f"Feature '{f_key}' processor does not expose code_vocab."
                )

            # Determine embedding source for this feature
            f_source = game_embeddings
            if isinstance(game_embeddings, dict) and f_key in game_embeddings:
                f_source = game_embeddings[f_key]

            self.game_embeddings[f_key] = GAMEEmbedding(
                vocab_size=len(vocab),
                embedding_dim=embedding_dim,
                code_vocab=vocab,
                game_embeddings=f_source,
                code_mapping=code_mapping,
                freeze=freeze_embeddings,
                normalize=normalize_embeddings,
                padding_idx=0,
            )

        total_input_dim = embedding_dim * len(self.feature_keys)

        # Build temporal sequence model
        if temporal_model == "GRU":
            self.temporal_model = nn.GRU(
                input_size=total_input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
            )
            self.temporal_dim = hidden_dim
        elif temporal_model == "LSTM":
            self.temporal_model = nn.LSTM(
                input_size=total_input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
            )
            self.temporal_dim = hidden_dim
        else:
            if nhead is not None:
                if nhead <= 0 or total_input_dim % nhead != 0:
                    raise ValueError(
                        f"nhead ({nhead}) must be positive and divide total_input_dim "
                        f"({total_input_dim})."
                    )
                chosen_nhead = nhead
            else:
                chosen_nhead = 1
                for candidate in (8, 4, 2):
                    if total_input_dim % candidate == 0:
                        chosen_nhead = candidate
                        break

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=total_input_dim,
                nhead=chosen_nhead,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_model = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=2,
            )
            self.temporal_dim = total_input_dim

        # Visit pooling across time
        if pooling_mode == "attention":
            self.attention_query = nn.Parameter(torch.empty(self.temporal_dim, 1))
            nn.init.xavier_uniform_(self.attention_query)

        self.dropout = nn.Dropout(dropout)
        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(self.temporal_dim, output_size)

    def _extract_feature(self, batch_data: Any) -> torch.Tensor:
        """Extracts token indices from raw batch feature tensors."""
        if isinstance(batch_data, tuple):
            for elem in batch_data:
                if elem is None:
                    continue
                if isinstance(elem, torch.Tensor):
                    if not elem.is_floating_point():
                        return elem
                elif isinstance(elem, (list, np.ndarray)):
                    t = torch.as_tensor(elem)
                    if not t.is_floating_point():
                        return t
            val = batch_data[0]
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            return val
        if not isinstance(batch_data, torch.Tensor):
            return torch.as_tensor(batch_data)
        return batch_data

    def _pool_visits(
        self,
        seq_reps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates visit representations across time for each patient."""
        # seq_reps: (B, L, H), mask: (B, L)
        if self.pooling_mode == "mean":
            mask_expanded = mask.unsqueeze(-1).float()
            sum_reps = (seq_reps * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp_min(1.0)
            return sum_reps / counts

        if self.pooling_mode == "max":
            masked = seq_reps.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = masked.max(dim=1)[0]
            # Zero out patients with no valid visits
            has_visits = mask.any(dim=1, keepdim=True)
            return torch.where(has_visits, pooled, torch.zeros_like(pooled))

        if self.pooling_mode == "last":
            batch_size = seq_reps.shape[0]
            pooled_list = []
            for b in range(batch_size):
                valid_idx = torch.where(mask[b])[0]
                if len(valid_idx) == 0:
                    pooled_list.append(
                        torch.zeros(self.temporal_dim, device=seq_reps.device)
                    )
                else:
                    pooled_list.append(seq_reps[b, valid_idx[-1]])
            return torch.stack(pooled_list, dim=0)

        # Attention pooling
        # query: (H, 1) -> scores: (B, L, 1) -> (B, L)
        scores = torch.matmul(seq_reps, self.attention_query).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        # In case all visits are padded:
        has_visits = mask.any(dim=1, keepdim=True)
        weights = torch.where(has_visits, weights, torch.zeros_like(weights))
        pooled = torch.sum(seq_reps * weights.unsqueeze(-1), dim=1)
        return pooled

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass for GAME model.

        Args:
            **kwargs: Batch of feature tensors and optional ground truth label.

        Returns:
            Dictionary containing 'loss', 'logit', 'y_prob', 'y_true', and 'embed'.
        """
        visit_emb_list: list[torch.Tensor] = []
        visit_masks: list[torch.Tensor] = []

        for f_key in self.feature_keys:
            raw_feat = self._extract_feature(kwargs[f_key])
            if raw_feat.dim() > 3:
                raise ValueError(
                    f"Feature '{f_key}' has dimension {raw_feat.dim()} > 3, "
                    "which is not supported."
                )
            if raw_feat.dim() == 2:
                # 2D flat sequence: (B, L) -> expand to (B, L, 1)
                raw_feat = raw_feat.unsqueeze(-1)

            # raw_feat shape: (B, L, C)
            mask = (raw_feat != 0).any(dim=-1)  # (B, L)
            visit_masks.append(mask)

            # Embed codes: (B, L, C, D)
            code_embs = self.game_embeddings[f_key](raw_feat)

            # Pool over codes in visit: max-pooling over non-padding codes
            code_mask = (raw_feat != 0).unsqueeze(-1)  # (B, L, C, 1)
            masked_code_embs = code_embs.masked_fill(~code_mask, -1e9)
            visit_embs = masked_code_embs.max(dim=2)[0]  # (B, L, D)

            # Empty visits without valid codes should be strictly zeros
            has_codes = code_mask.any(dim=2)  # (B, L, 1)
            visit_embs = torch.where(
                has_codes, visit_embs, torch.zeros_like(visit_embs)
            )
            visit_emb_list.append(visit_embs)

        # Concatenate visit representations across features: (B, L, total_input_dim)
        combined_seq = torch.cat(visit_emb_list, dim=-1)

        # Patient visit mask is valid if any feature is valid at that visit
        patient_mask = torch.stack(visit_masks, dim=0).any(dim=0)  # (B, L)

        # Pass through temporal model
        if self.temporal_model_type in ("GRU", "LSTM"):
            temporal_out, _ = self.temporal_model(combined_seq)
        else:
            # TransformerEncoder: MultiheadAttention produces NaNs if all positions
            # are masked. For completely padded patients, un-mask the first position
            # during attention, then zero out representations afterwards.
            all_padded = ~patient_mask.any(dim=-1)
            safe_patient_mask = patient_mask.clone()
            if all_padded.any():
                safe_patient_mask[all_padded, 0] = True
            padding_mask = ~safe_patient_mask
            temporal_out = self.temporal_model(
                combined_seq, src_key_padding_mask=padding_mask
            )
            if all_padded.any():
                temporal_out = torch.where(
                    all_padded.unsqueeze(-1).unsqueeze(-1),
                    torch.zeros_like(temporal_out),
                    temporal_out,
                )

        # Pool visits across longitudinal history
        patient_emb = self._pool_visits(temporal_out, patient_mask)
        patient_emb = self.dropout(patient_emb)

        # Classification logits
        logits = self.fc(patient_emb)

        # Format output using label processor
        y_prob = self.label_processor.activation(logits)
        results: dict[str, torch.Tensor] = {
            "embed": patient_emb,
            "logit": logits,
            "y_prob": y_prob,
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
