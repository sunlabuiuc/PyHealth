"""Event-Based Contrastive Learning (EBCL) for paired pre/post clinical sequences.

Implements a symmetric InfoNCE objective between embeddings of a pre-index-event
window and a post-index-event window, following the spirit of
`Event-Based Contrastive Learning for Medical Time Series` (Oufattole et al., MLHC 2024).

The model expects exactly two sequence inputs (e.g. ``conditions_pre`` and
``conditions_post``). A shared :class:`~pyhealth.models.embedding.EmbeddingModel`
and shared :class:`~pyhealth.models.rnn.RNNLayer` encode both windows; optional
linear probe uses concatenated pooled states when ``supervised_weight > 0``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.processors import SequenceProcessor

from .base_model import BaseModel
from .embedding import EmbeddingModel
from .rnn import RNNLayer


class EBCL(BaseModel):
    """Event-based contrastive encoder for paired pre/post sequences.

    Use two sequence feature keys representing clinical codes (or other discrete
    tokens) before and after an index event. The contrastive loss aligns the
    pre-window embedding with the post-window embedding for the same patient-event
    and separates mismatched pairs in the batch.

    Args:
        dataset: :class:`~pyhealth.datasets.SampleDataset` with exactly two input
            sequence features and optionally one supervised label for probing.
        embedding_dim: Code embedding size.
        hidden_dim: RNN hidden size.
        projection_dim: Dimension of L2-normalized contrastive vectors.
        temperature: Temperature for InfoNCE logits.
        rnn_type: ``"GRU"``, ``"LSTM"``, or ``"RNN"``.
        supervised_weight: If > 0 and labels are present, adds this multiple of the
            supervised loss (BCE or CE via :meth:`get_loss_function`) to the
            contrastive loss.
        pre_key: Name of the pre-event sequence feature; default ``conditions_pre``
            if present, else the first input key.
        post_key: Name of the post-event sequence feature; default ``conditions_post``
            if present, else the second input key.
        **kwargs: Passed to :class:`~pyhealth.models.rnn.RNNLayer` (e.g.
            ``num_layers``, ``dropout``, ``bidirectional``).

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.1,
        rnn_type: str = "GRU",
        supervised_weight: float = 0.0,
        pre_key: Optional[str] = None,
        post_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 2:
            raise ValueError(
                "EBCL requires exactly two sequence input features (pre/post windows)."
            )

        keys = list(self.feature_keys)
        if pre_key is None:
            if "conditions_pre" in keys and "conditions_post" in keys:
                self.pre_key = "conditions_pre"
                self.post_key = "conditions_post"
            else:
                self.pre_key, self.post_key = keys[0], keys[1]
        else:
            self.pre_key = pre_key
            self.post_key = post_key or keys[1 - keys.index(pre_key)]

        if self.pre_key not in keys or self.post_key not in keys:
            raise ValueError("pre_key and post_key must match dataset input_schema keys.")
        if self.pre_key == self.post_key:
            raise ValueError("pre_key and post_key must differ.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.supervised_weight = supervised_weight

        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self._maybe_tie_sequence_embeddings()

        self.rnn = RNNLayer(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            rnn_type=rnn_type,
            **kwargs,
        )
        self.proj = nn.Linear(hidden_dim, projection_dim)

        self.label_key: Optional[str] = None
        self.fc_probe: Optional[nn.Linear] = None
        if len(self.label_keys) == 1:
            self.label_key = self.label_keys[0]
            out_dim = self.get_output_size()
            self.fc_probe = nn.Linear(hidden_dim * 2, out_dim)

    def _maybe_tie_sequence_embeddings(self) -> None:
        """Share pre/post embeddings when both fields use identical code vocabularies."""
        pre_p = self.dataset.input_processors[self.pre_key]
        post_p = self.dataset.input_processors[self.post_key]
        if not isinstance(pre_p, SequenceProcessor) or not isinstance(
            post_p, SequenceProcessor
        ):
            return
        if pre_p.code_vocab != post_p.code_vocab:
            return
        layers = self.embedding_model.embedding_layers
        if self.pre_key in layers and self.post_key in layers:
            layers[self.post_key].weight = layers[self.pre_key].weight

    def _embed_branch(
        self,
        feature_key: str,
        kwargs: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """Return last RNN hidden state for one branch (same layout as :class:`RNN`)."""
        feature = kwargs[feature_key]
        if isinstance(feature, torch.Tensor):
            feature = (feature,)

        schema = self.dataset.input_processors[feature_key].schema()
        value = feature[schema.index("value")] if "value" in schema else None
        mask = feature[schema.index("mask")] if "mask" in schema else None

        if value is None:
            raise ValueError(f"Feature '{feature_key}' must contain 'value' in the schema.")

        inputs = {feature_key: value}
        masks: Dict[str, torch.Tensor] = {}
        if mask is not None:
            masks[feature_key] = mask

        embedded = self.embedding_model(inputs, masks=masks or None)
        x = embedded[feature_key]

        x_dim_orig = x.dim()
        if x_dim_orig == 4:
            x = x.sum(dim=2)
            if feature_key in masks:
                m = (masks[feature_key].to(self.device).sum(dim=-1) > 0).int()
            else:
                m = (torch.abs(x).sum(dim=-1) != 0).int()
        elif x_dim_orig == 2:
            x = x.unsqueeze(1)
            m = None
        else:
            if feature_key in masks:
                m = masks[feature_key].to(self.device).int()
                if m.dim() == 3:
                    m = (m.sum(dim=-1) > 0).int()
            else:
                m = (torch.abs(x).sum(dim=-1) != 0).int()

        _, last = self.rnn(x, m)
        return last

    @staticmethod
    def _info_nce(z_pre: torch.Tensor, z_post: torch.Tensor, temperature: float) -> torch.Tensor:
        z_pre = F.normalize(z_pre, dim=-1)
        z_post = F.normalize(z_post, dim=-1)
        logits = torch.matmul(z_pre, z_post.transpose(0, 1)) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_pre = F.cross_entropy(logits, targets)
        loss_post = F.cross_entropy(logits.transpose(0, 1), targets)
        return (loss_pre + loss_post) * 0.5

    def forward(
        self,
        **kwargs: torch.Tensor | Tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute CLIP-style contrastive loss on pre/post embeddings.

        Args:
            **kwargs: Batch tensors keyed by ``input_schema`` and ``output_schema``
                field names (including the label key when present).

        Returns:
            Dictionary with at least ``loss``, ``embed_pre``, ``embed_post``,
            ``z_pre``, and ``z_post``. May include ``logit``, ``y_prob``,
            ``y_true`` when a label is provided. Optional key ``embed`` if
            ``kwargs['embed'] is True``.
        """
        h_pre = self._embed_branch(self.pre_key, kwargs)
        h_post = self._embed_branch(self.post_key, kwargs)

        z_pre = self.proj(h_pre)
        z_post = self.proj(h_post)
        contrastive_loss = self._info_nce(z_pre, z_post, self.temperature)

        results: Dict[str, torch.Tensor] = {
            "loss": contrastive_loss,
            "embed_pre": h_pre,
            "embed_post": h_post,
            "z_pre": z_pre,
            "z_post": z_post,
        }

        if (
            self.supervised_weight > 0
            and self.label_key is not None
            and self.fc_probe is not None
            and self.label_key in kwargs
        ):
            h_cat = torch.cat([h_pre, h_post], dim=-1)
            logits = self.fc_probe(h_cat)
            y_true = kwargs[self.label_key].to(self.device)
            sup_loss = self.get_loss_function()(logits, y_true)
            results["loss"] = contrastive_loss + self.supervised_weight * sup_loss
            results["logit"] = logits
            results["y_true"] = y_true
            results["y_prob"] = self.prepare_y_prob(logits)
        elif self.label_key is not None and self.label_key in kwargs and self.fc_probe is not None:
            # Eval / logging: still return predictions without adding to loss
            h_cat = torch.cat([h_pre, h_post], dim=-1)
            logits = self.fc_probe(h_cat)
            y_true = kwargs[self.label_key].to(self.device)
            results["logit"] = logits
            results["y_true"] = y_true
            results["y_prob"] = self.prepare_y_prob(logits)

        if kwargs.get("embed", False):
            results["embed"] = torch.cat([h_pre, h_post], dim=-1)

        return results
