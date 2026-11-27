"""Context-aware Health Event Prediction via Transition Functions on Dynamic
Disease Graphs (CHET).

Paper: Chang Lu, Tian Han, Yue Ning.
Context-aware Health Event Prediction via Transition Functions on
Dynamic Disease Graphs. Thirty-Sixth AAAI Conference on Artificial
Intelligence (AAAI 2022). https://arxiv.org/abs/2112.05195
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


def _extract_codes(val: Any) -> torch.Tensor:
    """Extracts code indices tensor safely from a tensor, array, or tuple.

    For tuple inputs such as ``(codes, mask)`` or ``(time, codes)``, extracts the first
    non-floating-point tensor or array, skipping ``None`` and floating-point elements
    (e.g., timestamps).
    """
    if isinstance(val, tuple):
        for elem in val:
            if elem is None:
                continue
            if isinstance(elem, torch.Tensor):
                if not elem.is_floating_point():
                    return elem
            elif isinstance(elem, (list, np.ndarray)):
                t = torch.as_tensor(elem)
                if not t.is_floating_point():
                    return t
        for elem in val:
            if elem is not None:
                return torch.as_tensor(elem)
        raise ValueError("Cannot extract code tensor: all tuple elements are None.")
    if not isinstance(val, torch.Tensor):
        val = torch.as_tensor(val)
    return val


def compute_cooccurrence_adj(
    dataset: SampleDataset,
    feature_key: str,
    self_loop: bool = True,
) -> torch.Tensor:
    """Computes a normalized co-occurrence adjacency matrix from a dataset.

    Note:
        To prevent data leakage, compute this adjacency matrix using only the
        training dataset split rather than the combined dataset. Unlike
        ``ehr_cooccurrence_adjacency`` in ``pyhealth.models.utils`` which builds
        binary matrices over label sets for recommendation models, this helper
        operates on input code features, supports self-loops, and row-normalizes
        transition probabilities for disease graph convolutions.

    Args:
        dataset: PyHealth SampleDataset containing clinical samples.
        feature_key: The input feature key (e.g., 'conditions') with code vocabulary.
        self_loop: Whether to add self-loops along the diagonal. Default is True.

    Returns:
        Row-normalized float tensor of shape ``(num_codes, num_codes)``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "conditions": [["c1", "c2"]], "label": 1},
        ...     {"patient_id": "p1", "visit_id": "v1",
        ...      "conditions": [["c2", "c3"]], "label": 0},
        ... ]
        >>> ds = create_sample_dataset(
        ...     samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> adj = compute_cooccurrence_adj(ds, "conditions")
        >>> adj.dim()
        2
        >>> adj.shape[0] == adj.shape[1]
        True
    """
    proc = dataset.input_processors[feature_key]
    vocab = getattr(proc, "code_vocab", None)
    if vocab is None:
        raise ValueError(
            f"Processor for feature '{feature_key}' does not have a code_vocab."
        )

    num_codes = len(vocab)
    adj = torch.zeros((num_codes, num_codes), dtype=torch.float32)

    for sample in dataset:
        feat = _extract_codes(sample[feature_key])
        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat)

        if feat.dim() == 1:
            visits = [feat]
        elif feat.dim() == 2:
            visits = [feat[t] for t in range(feat.shape[0])]
        elif feat.dim() == 3:
            visits = [
                feat[t, k] for t in range(feat.shape[0]) for k in range(feat.shape[1])
            ]
        else:
            continue

        for v in visits:
            valid = v[v != 0].unique()
            if self_loop:
                for c in valid:
                    c_idx = int(c.item())
                    if 0 <= c_idx < num_codes:
                        adj[c_idx, c_idx] += 1.0
            valid_list = [
                int(c.item()) for c in valid if 0 <= int(c.item()) < num_codes
            ]
            for i in range(len(valid_list)):
                for j in range(i + 1, len(valid_list)):
                    c1, c2 = valid_list[i], valid_list[j]
                    adj[c1, c2] += 1.0
                    adj[c2, c1] += 1.0

    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
    adj_norm = adj / deg
    # Ensure pad index 0 is strictly disconnected
    adj_norm[0].zero_()
    adj_norm[:, 0].zero_()
    return adj_norm


class SingleHeadAttentionLayer(nn.Module):
    """Single-head attention layer for transition state updates.

    Args:
        query_size: Feature dimension of query vectors.
        key_size: Feature dimension of key vectors.
        value_size: Output feature dimension after value projection.
        attention_size: Dimension of attention projection space.
        value_input_size: Optional input feature dimension of value vectors.
            If None, defaults to ``value_size``.

    Examples:
        >>> import torch
        >>> from pyhealth.models import SingleHeadAttentionLayer
        >>> layer = SingleHeadAttentionLayer(
        ...     query_size=16,
        ...     key_size=24,
        ...     value_size=32,
        ...     attention_size=16,
        ... )
        >>> q = torch.randn(4, 16)
        >>> k = torch.randn(5, 24)
        >>> v = torch.randn(5, 32)
        >>> out = layer(q, k, v)
        >>> out.shape
        torch.Size([4, 32])
    """

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        attention_size: int,
        value_input_size: int | None = None,
    ) -> None:
        super().__init__()
        self.attention_size = attention_size
        v_in = value_input_size if value_input_size is not None else value_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(v_in, value_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(
            torch.matmul(query, key.transpose(-1, -2)),
            math.sqrt(self.attention_size),
        )
        score = torch.softmax(g, dim=-1)
        output = torch.matmul(score, value)
        return output


class DotProductAttention(nn.Module):
    """Dot-product attention over patient visit representations.

    Args:
        value_size: Input feature dimension of visit representation vectors.
        attention_size: Dimension of intermediate attention projection space.
            Default is 32.

    Examples:
        >>> import torch
        >>> from pyhealth.models import DotProductAttention
        >>> att = DotProductAttention(value_size=32, attention_size=16)
        >>> x = torch.randn(5, 32)
        >>> mask = torch.tensor([True, True, True, False, False])
        >>> out = att(x, mask=mask)
        >>> out.shape
        torch.Size([32])
    """

    def __init__(self, value_size: int, attention_size: int = 32) -> None:
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(
            data=nn.init.xavier_uniform_(torch.empty(attention_size, 1))
        )
        self.dense = nn.Linear(value_size, attention_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros(self.dense.in_features, device=x.device, dtype=x.dtype)
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze(-1)
        if mask is not None and mask.any():
            vu = vu.masked_fill(~mask, -1e9)
        score = torch.softmax(vu, dim=-1)
        score = torch.nan_to_num(score, nan=0.0)
        output = torch.sum(x * score.unsqueeze(-1), dim=0)
        return output


class CHET(BaseModel):
    """Context-aware Health Event Prediction via Transition Functions.

    Paper: Chang Lu, Tian Han, Yue Ning.
    Context-aware Health Event Prediction via Transition Functions on
    Dynamic Disease Graphs. Thirty-Sixth AAAI Conference on Artificial
    Intelligence (AAAI 2022). https://arxiv.org/abs/2112.05195

    CHET models longitudinal patient trajectories by constructing dynamic
    disease graphs for each clinical visit. It explicitly distinguishes between
    persistent recurrent conditions, emergent neighbor conditions, and novel
    unrelated conditions across visits using context-aware transition functions.

    Note:
        Per-visit dense N x N adjacency multiplications scale quadratically with
        vocabulary size N, so large vocabularies (e.g. MIMIC-sized) may require
        pruned or filtered code sets.

    Args:
        dataset: PyHealth SampleDataset containing patient samples and schemas.
        feature_keys: Optional list of feature keys to use from dataset.
            CHET operates on a single sequential disease code feature. If multiple
            keys are supplied, only the primary key (feature_keys[0]) is used and
            a warning is emitted. Defaults to the first feature key in dataset.
        label_key: Optional label key to predict. Defaults to the first label key.
            If the dataset defines multiple labels, specify the target label_key.
        code_size: Dimensionality of center and neighbor code embeddings.
            Default is 48.
        graph_size: Hidden dimension of graph representation layer.
            Default is 32.
        hidden_dim: Hidden dimension of transition GRU and patient representation.
            Default is 128.
        attention_size: Dimension of single-head transition attention.
            Default is 32.
        adj: Optional precomputed adjacency matrix of shape ``(num_codes, num_codes)``.
            If None, computed automatically from dataset co-occurrences. Only the
            training split should be used to construct this matrix or passed when
            adj is None to prevent evaluation leakage. When user-supplied, `adj` is
            expected to be row-normalized with the padding row and column (index 0)
            zeroed.
        dropout: Dropout rate applied to patient representations.
            Default is 0.1.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import CHET, compute_cooccurrence_adj
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
        >>> # To avoid leakage, precompute adjacency on the training split:
        >>> train_adj = compute_cooccurrence_adj(dataset, "conditions")
        >>> model = CHET(
        ...     dataset=dataset,
        ...     code_size=16,
        ...     graph_size=16,
        ...     hidden_dim=32,
        ...     adj=train_adj,
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
        code_size: int = 48,
        graph_size: int = 32,
        hidden_dim: int = 128,
        attention_size: int = 32,
        adj: torch.Tensor | np.ndarray | None = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset=dataset)

        if code_size <= 0:
            raise ValueError(f"code_size must be positive, got {code_size}")
        if graph_size <= 0:
            raise ValueError(f"graph_size must be positive, got {graph_size}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if attention_size <= 0:
            raise ValueError(f"attention_size must be positive, got {attention_size}")

        if feature_keys is not None:
            self.feature_keys = feature_keys
            if len(self.feature_keys) > 1:
                warnings.warn(
                    f"CHET operates on a single disease code feature key. "
                    f"Multiple feature keys were provided ({self.feature_keys}); "
                    f"only the primary key '{self.feature_keys[0]}' will be used.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.feature_keys = [self.feature_keys[0]]

        if label_key is not None:
            self.label_key = label_key
        else:
            self.label_key = self.label_keys[0]

        if self.label_key not in self.dataset.output_processors:
            raise KeyError(
                f"label_key '{self.label_key}' not found in dataset output processors."
            )

        self.primary_feature_key = self.feature_keys[0]
        proc = self.dataset.input_processors[self.primary_feature_key]
        vocab = getattr(proc, "code_vocab", None)
        if vocab is None:
            raise ValueError(
                f"Feature '{self.primary_feature_key}' processor "
                "does not expose code_vocab."
            )

        self.num_codes = len(vocab)
        self.code_size = code_size
        self.graph_size = graph_size
        self.hidden_dim = hidden_dim

        # Compute or validate adjacency matrix
        if adj is None:
            adj_tensor = compute_cooccurrence_adj(dataset, self.primary_feature_key)
        elif isinstance(adj, np.ndarray):
            adj_tensor = torch.from_numpy(adj).float()
        elif isinstance(adj, torch.Tensor):
            adj_tensor = adj.float()
        else:
            raise TypeError(f"Unsupported type for adj: {type(adj)}")

        if adj_tensor.shape != (self.num_codes, self.num_codes):
            raise ValueError(
                f"adj shape must be ({self.num_codes}, {self.num_codes}), "
                f"got {adj_tensor.shape}"
            )
        self.register_buffer("adj", adj_tensor)

        # Code embedding parameters: center, neighbor, and unrelated baseline
        self.c_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_codes, code_size))
        )
        self.n_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_codes, code_size))
        )
        self.u_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_codes, graph_size))
        )

        # Dual graph convolution projection layer
        self.graph_dense = nn.Linear(code_size, graph_size)
        self.graph_act = nn.LeakyReLU()

        # Context-aware transition dynamics
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_dim)
        self.transition_attn = SingleHeadAttentionLayer(
            query_size=graph_size,
            key_size=graph_size,
            value_size=hidden_dim,
            attention_size=attention_size,
            value_input_size=graph_size,
        )

        # Dot-product attention across visits
        self.dot_product_attention = DotProductAttention(
            value_size=hidden_dim, attention_size=attention_size
        )

        self.dropout = nn.Dropout(dropout)

        # Output prediction head
        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(hidden_dim, output_size)

    @property
    def label_processor(self) -> Any:
        """The label processor for the target label_key."""
        if hasattr(self, "label_key") and hasattr(self, "dataset"):
            proc = self.dataset.output_processors.get(self.label_key)
            if proc is not None:
                return proc
        return super().label_processor

    def _extract_codes(self, val: Any) -> torch.Tensor:
        """Extracts code indices tensor safely from tensor or tuple."""
        return _extract_codes(val)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass for CHET.

        Args:
            **kwargs: Input tensors containing patient features and optional labels.

        Returns:
            Dictionary containing:
                - loss: Training loss (returned whenever label key is in kwargs).
                - y_prob: Predicted probabilities.
                - y_true: Ground truth labels (if provided).
                - logit: Raw output logits.
                - embed: Patient-level representations.
        """
        raw_feat = kwargs.get(self.primary_feature_key)
        if raw_feat is None:
            raise KeyError(f"Missing required feature key '{self.primary_feature_key}'")

        feat = self._extract_codes(raw_feat)
        if feat.dim() > 3:
            raise ValueError(
                f"Unsupported feature dimension {feat.dim()} for "
                f"feature '{self.primary_feature_key}'. Supported dimensions "
                "are 2D (B, K) and 3D (B, T, K)."
            )

        if feat.dim() == 2:
            feat = feat.unsqueeze(1)  # (B, 1, K)

        batch_size, max_visits, _ = feat.shape
        device = feat.device
        adj = self.adj

        patient_embeddings: list[torch.Tensor] = []

        for b in range(batch_size):
            h_prev: torch.Tensor | None = None
            prev_no_embs: torch.Tensor | None = None
            prev_center_mask: torch.Tensor | None = None
            prev_neighbor_mask: torch.Tensor | None = None

            visit_outputs: list[torch.Tensor] = []
            valid_visits: list[bool] = []

            for t in range(max_visits):
                visit_codes = feat[b, t]
                valid_code_indices = visit_codes[visit_codes != 0].unique()

                if len(valid_code_indices) == 0:
                    visit_outputs.append(torch.zeros(self.hidden_dim, device=device))
                    valid_visits.append(False)
                    continue

                center_mask = torch.zeros(
                    self.num_codes, device=device, dtype=torch.float32
                )
                center_mask[valid_code_indices] = 1.0
                center_mask[0] = 0.0

                # 1-hop neighbor codes in graph excluding current center codes
                neighbor_mask = (torch.matmul(adj, center_mask) > 0).float() * (
                    1.0 - center_mask
                )
                neighbor_mask[0] = 0.0

                center_codes = center_mask.unsqueeze(-1)
                neighbor_codes = neighbor_mask.unsqueeze(-1)

                center_embs = center_codes * self.c_embeddings
                neighbor_embs = neighbor_codes * self.n_embeddings

                cc = center_codes * torch.matmul(adj, center_embs)
                cn = center_codes * torch.matmul(adj, neighbor_embs)
                nn_embs = neighbor_codes * torch.matmul(adj, neighbor_embs)
                nc = neighbor_codes * torch.matmul(adj, center_embs)

                co_embs = self.graph_act(self.graph_dense(center_embs + cc + cn))
                no_embs = self.graph_act(self.graph_dense(neighbor_embs + nn_embs + nc))

                h_t = torch.zeros(self.num_codes, self.hidden_dim, device=device)

                if t == 0 or prev_center_mask is None:
                    m1_idx = torch.where(center_mask > 0)[0]
                    h_m1 = self.gru(co_embs[m1_idx], None)
                    h_t[m1_idx] = h_m1
                    v_out = torch.max(h_m1, dim=0)[0]
                else:
                    m1_mask = center_mask * prev_center_mask
                    m2_mask = (
                        center_mask * (1.0 - prev_center_mask) * prev_neighbor_mask
                    )
                    m3_mask = (
                        center_mask
                        * (1.0 - prev_center_mask)
                        * (1.0 - prev_neighbor_mask)
                    )

                    m1_idx = torch.where(m1_mask > 0)[0]
                    m2_idx = torch.where(m2_mask > 0)[0]
                    m3_idx = torch.where(m3_mask > 0)[0]

                    out_m1 = None
                    if len(m1_idx) > 0:
                        h_prev_sub = h_prev[m1_idx] if h_prev is not None else None
                        h_m1 = self.gru(co_embs[m1_idx], h_prev_sub)
                        h_t[m1_idx] = h_m1
                        out_m1 = torch.max(h_m1, dim=0)[0]

                    out_m23 = None
                    q_list, v_list = [], []
                    if len(m2_idx) > 0 and prev_no_embs is not None:
                        q_list.append(prev_no_embs[m2_idx])
                        v_list.append(co_embs[m2_idx])
                    if len(m3_idx) > 0:
                        q_list.append(self.u_embeddings[m3_idx])
                        v_list.append(co_embs[m3_idx])

                    if q_list:
                        q = torch.cat(q_list, dim=0)
                        v = torch.cat(v_list, dim=0)
                        h_m23 = torch.tanh(self.transition_attn(q, q, v))
                        offset = 0
                        if len(m2_idx) > 0 and prev_no_embs is not None:
                            h_t[m2_idx] = h_m23[: len(m2_idx)]
                            offset = len(m2_idx)
                        if len(m3_idx) > 0:
                            h_t[m3_idx] = h_m23[offset:]
                        out_m23 = torch.max(h_m23, dim=0)[0]

                    if out_m1 is not None and out_m23 is not None:
                        v_out = torch.max(torch.stack([out_m1, out_m23], dim=0), dim=0)[
                            0
                        ]
                    elif out_m1 is not None:
                        v_out = out_m1
                    elif out_m23 is not None:
                        v_out = out_m23
                    else:
                        v_out = torch.zeros(self.hidden_dim, device=device)

                h_prev = h_t
                prev_no_embs = no_embs
                prev_center_mask = center_mask
                prev_neighbor_mask = neighbor_mask

                visit_outputs.append(v_out)
                valid_visits.append(True)

            if any(valid_visits):
                stacked_v = torch.stack(visit_outputs, dim=0)
                mask_t = torch.tensor(valid_visits, device=device, dtype=torch.bool)
                patient_emb = self.dot_product_attention(stacked_v, mask=mask_t)
            else:
                patient_emb = torch.zeros(self.hidden_dim, device=device)

            patient_embeddings.append(patient_emb)

        embed = torch.stack(patient_embeddings, dim=0)
        embed = self.dropout(embed)
        logits = self.fc(embed)

        out = {
            "logit": logits,
            "embed": embed,
            "y_prob": self.label_processor.activation(logits),
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            out["y_true"] = y_true
            out["loss"] = self.label_processor.loss(logits, y_true)

        return out
