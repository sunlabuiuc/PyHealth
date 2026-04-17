"""Event-Based Contrastive Learning model for PyHealth.

This module implements a PyHealth-compatible adaptation of Event-Based
Contrastive Learning (EBCL) for structured medical time series. It supports
three common usage modes:

1. Supervised fine-tuning on pre-event features only.
2. Contrastive pretraining on matched pre-event and post-event feature pairs.
3. Joint supervised and contrastive training when both labels and post-event
   features are available.

The implementation follows the multi-stream style used by existing PyHealth
models: each feature stream is embedded independently, encoded with a
Transformer encoder, pooled into a feature representation, concatenated, and
fused into a patient representation.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


class FusionSelfAttention(nn.Module):
    """Attention-weighted pooling over a sequence.

    This layer more closely follows the EBCL paper than masked mean pooling.
    After the Transformer encoder, it computes attention scores over valid time
    steps, masks padded positions, normalizes scores with softmax, and returns
    the weighted average of the token embeddings.

    Args:
        input_dim: Dimensionality of each token embedding.
        dropout: Dropout applied before attention scoring.
    """

    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pools a sequence into a single vector.

        Args:
            x: Sequence tensor of shape ``[batch, seq_len, input_dim]``.
            mask: Boolean mask of shape ``[batch, seq_len]`` where ``True``
                indicates a valid token.
            return_attention: Whether to also return normalized attention
                weights.

        Returns:
            The pooled tensor of shape ``[batch, input_dim]``. If
            ``return_attention`` is ``True``, also returns the attention weights
            of shape ``[batch, seq_len]``.
        """
        scores = self.score(self.dropout(x)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Guard against pathological rows that somehow remain fully invalid.
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            scores = scores.clone()
            scores[invalid_rows, 0] = 0.0

        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * attention.unsqueeze(-1), dim=1)
        if return_attention:
            return pooled, attention
        return pooled


class EBCL(BaseModel):
    """Event-Based Contrastive Learning model for structured medical data.

    This implementation adapts the core EBCL idea to the standard PyHealth
    model interface.

    Fine-tuning mode uses the pre-event feature streams already present in a
    :class:`~pyhealth.datasets.SampleDataset`. Contrastive mode is enabled when
    the dataset also includes matching ``post_*`` feature streams, such as
    ``conditions`` and ``post_conditions``. In that case the model learns
    paired representations around an index event using a symmetric InfoNCE /
    CLIP-style loss.

    The design intentionally follows PyHealth's existing multi-stream EHR
    models. Each feature stream is embedded independently, encoded with a
    Transformer encoder, pooled to a feature representation, concatenated, and
    fused into a patient representation. This keeps the model easy to test and
    reuse with existing PyHealth tasks while preserving the event-centered
    contrastive objective.

    Args:
        dataset: Processed sample dataset containing one label key and one or
            more input feature keys.
        embedding_dim: Shared embedding dimension for input feature streams.
        hidden_dim: Hidden dimension of the fused patient representation.
        projection_dim: Output dimension of the contrastive projection heads.
        heads: Number of attention heads in each feature Transformer.
        num_layers: Number of Transformer layers per feature stream.
        dropout: Dropout applied in Transformer, pooling, and head layers.
        temperature: Initial contrastive temperature.
        contrastive_weight: Weight applied to contrastive loss when combined
            with supervised loss.
        post_prefix: Prefix used to identify post-event paired features.
        max_seq_len: Maximum number of steps kept per feature stream.
        feedforward_dim: Feed-forward dimension inside the Transformer encoder.
        supervised_uses_post: Whether the supervised classifier should consume
            concatenated pre-event and post-event embeddings when post-event
            features are available.
        return_attention_weights: Whether to return per-feature pooling
            attention weights from the forward pass.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["A", "B", "C"],
        ...         "labs": [1.0, 2.0, 3.0],
        ...         "post_conditions": ["B", "C"],
        ...         "post_labs": [2.0, 3.0, 4.0],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["D", "E"],
        ...         "labs": [0.5, 1.5, 2.5],
        ...         "post_conditions": ["E"],
        ...         "post_labs": [1.0, 1.5, 2.0],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "sequence",
        ...         "labs": "tensor",
        ...         "post_conditions": "sequence",
        ...         "post_labs": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ebcl_demo",
        ... )
        >>> model = EBCL(dataset=dataset)
        >>> batch = next(
        ...     iter(get_dataloader(dataset, batch_size=2, shuffle=False))
        ... )
        >>> output = model(**batch)
        >>> "contrastive_loss" in output and "supervised_loss" in output
        True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 32,
        hidden_dim: int = 32,
        projection_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        post_prefix: str = "post_",
        max_seq_len: int = 512,
        feedforward_dim: int = 128,
        supervised_uses_post: bool = False,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_weight = contrastive_weight
        self.post_prefix = post_prefix
        self.max_seq_len = max_seq_len
        self.feedforward_dim = feedforward_dim
        self.supervised_uses_post = supervised_uses_post
        self.return_attention_weights = return_attention_weights

        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")

        if len(self.label_keys) != 1:
            raise ValueError("EBCL supports exactly one label key.")
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        raw_feature_keys = list(self.feature_keys)
        self.raw_feature_keys = raw_feature_keys
        self.post_feature_keys: Dict[str, str] = {}
        self.base_feature_keys: list[str] = []
        for feature_key in raw_feature_keys:
            if feature_key.startswith(self.post_prefix):
                base_key = feature_key[len(self.post_prefix) :]
                self.post_feature_keys[base_key] = feature_key
            else:
                self.base_feature_keys.append(feature_key)
        self.feature_keys = self.base_feature_keys

        if not self.base_feature_keys:
            raise ValueError("EBCL requires at least one pre-event feature key.")

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.feature_encoder = nn.ModuleDict()
        self.feature_pooler = nn.ModuleDict()
        self.feature_projector = nn.ModuleDict()

        def build_encoder_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        for feature_key in self.base_feature_keys:
            self.feature_encoder[feature_key] = nn.TransformerEncoder(
                build_encoder_layer(),
                num_layers=num_layers,
                norm=nn.LayerNorm(embedding_dim),
            )
            self.feature_pooler[feature_key] = FusionSelfAttention(
                embedding_dim,
                dropout=dropout,
            )
            self.feature_projector[feature_key] = nn.Linear(
                embedding_dim,
                hidden_dim,
            )

        fused_dim = len(self.base_feature_keys) * hidden_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Use separate pre-event and post-event projection heads, matching the
        # paper's unshared projection design.
        self.pre_projection_head = nn.Linear(hidden_dim, projection_dim)
        self.post_projection_head = nn.Linear(hidden_dim, projection_dim)

        classifier_input_dim = hidden_dim * 2 if supervised_uses_post else hidden_dim
        self.fc = nn.Linear(classifier_input_dim, self.get_output_size())

        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(1.0 / max(temperature, 1e-6)),
                dtype=torch.float32,
            )
        )

    def _get_branch_feature_map(self, use_post: bool = False) -> Dict[str, str]:
        """Builds the feature map for a branch.

        Args:
            use_post: Whether to return the post-event feature mapping.

        Returns:
            A mapping from base feature keys to actual keys used in ``kwargs``.
        """
        if not use_post:
            return {
                feature_key: feature_key for feature_key in self.base_feature_keys
            }
        return {
            feature_key: self.post_feature_keys[feature_key]
            for feature_key in self.base_feature_keys
            if feature_key in self.post_feature_keys
        }

    def _prepare_branch_inputs(
        self,
        feature_map: Dict[str, str],
        kwargs: Dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extracts and embeds the tensors for one branch.

        Args:
            feature_map: Mapping from base feature names to concrete keys in
                the batch.
            kwargs: Batch tensors passed to ``forward``.

        Returns:
            A tuple of embedded feature tensors and their masks.
        """
        inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for actual_key in feature_map.values():
            feature = kwargs[actual_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            schema = self.dataset.input_processors[actual_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{actual_key}' must contain 'value' in the schema."
                )

            inputs[actual_key] = value
            if mask is not None:
                masks[actual_key] = mask

        embedded, out_masks = self.embedding_model(
            inputs,
            masks=masks if masks else None,
            output_mask=True,
        )
        return embedded, out_masks

    def _collapse_sequence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalizes feature tensors to ``[batch, seq_len, dim]`` format.

        Args:
            x: Embedded feature tensor.
            mask: Optional feature mask.

        Returns:
            A tuple of normalized sequence tensor and boolean mask.
        """
        original_x_dim = x.dim()
        while x.dim() > 3:
            x = x.sum(dim=2)
            if mask is not None and mask.dim() > 2:
                mask = mask.any(dim=-1)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if mask is None:
            mask = torch.any(torch.abs(x) > 0, dim=-1)
        else:
            mask = mask.to(self.device)
            while mask.dim() > 2:
                mask = mask.any(dim=-1)
            if mask.dim() == x.dim():
                mask = mask.any(dim=-1)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            elif original_x_dim == 2 and mask.dim() == 2:
                # Dense tensor features are a single event with multiple
                # dimensions, but EmbeddingModel emits a per-dimension mask.
                mask = mask.any(dim=-1, keepdim=True)
            elif mask.size(1) != x.size(1):
                # After pooling nested axes, reduce any stale mask axis to the
                # active sequence dimension.
                mask = mask.any(dim=-1, keepdim=True)

        if x.size(1) > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
            mask = mask[:, : self.max_seq_len]

        mask = mask.bool()
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return x, mask

    def _encode_feature(
        self,
        feature_key: str,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encodes a single feature stream.

        Args:
            feature_key: Base feature name.
            x: Embedded feature tensor of shape ``[batch, seq_len, dim]``.
            mask: Boolean mask of shape ``[batch, seq_len]``.
            return_attention: Whether to return pooling attention weights.

        Returns:
            The projected feature embedding. If ``return_attention`` is
            ``True``, also returns the attention weights.
        """
        positions = torch.arange(x.size(1), device=self.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        encoded = self.feature_encoder[feature_key](
            x,
            src_key_padding_mask=~mask,
        )

        if return_attention:
            pooled, attention = self.feature_pooler[feature_key](
                encoded,
                mask,
                return_attention=True,
            )
            projected = self.feature_projector[feature_key](pooled)
            return projected, attention

        pooled = self.feature_pooler[feature_key](
            encoded,
            mask,
            return_attention=False,
        )
        return self.feature_projector[feature_key](pooled)

    def _encode_branch(
        self,
        feature_map: Dict[str, str],
        kwargs: Dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encodes all features for one branch.

        Args:
            feature_map: Mapping from base feature names to concrete batch keys.
            kwargs: Batch tensors passed to ``forward``.
            return_attention: Whether to collect per-feature attention weights.

        Returns:
            The fused patient embedding. If ``return_attention`` is ``True``,
            also returns a dictionary of attention weights keyed by concrete
            feature name.
        """
        embedded, masks = self._prepare_branch_inputs(feature_map, kwargs)
        feature_embeddings = []
        feature_attention: Dict[str, torch.Tensor] = {}

        for base_key, actual_key in feature_map.items():
            x = embedded[actual_key]
            mask = masks.get(actual_key)
            x, mask = self._collapse_sequence(x, mask)

            if return_attention:
                encoded_feature, attention = self._encode_feature(
                    base_key,
                    x,
                    mask,
                    return_attention=True,
                )
                feature_attention[actual_key] = attention
                feature_embeddings.append(encoded_feature)
            else:
                feature_embeddings.append(
                    self._encode_feature(base_key, x, mask)
                )

        patient_embedding = torch.cat(feature_embeddings, dim=-1)
        patient_embedding = self.fusion(patient_embedding)

        if return_attention:
            return patient_embedding, feature_attention
        return patient_embedding

    def _compute_contrastive_loss(
        self,
        pre_projection: torch.Tensor,
        post_projection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes symmetric contrastive loss between pre and post views.

        Args:
            pre_projection: Normalized pre-event projection tensor.
            post_projection: Normalized post-event projection tensor.

        Returns:
            A tuple of contrastive loss scalar and similarity logits matrix.
        """
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * pre_projection @ post_projection.transpose(0, 1)
        labels = torch.arange(logits.size(0), device=self.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )
        return loss, logits

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Runs a forward pass for supervised, contrastive, or joint training.

        Args:
            **kwargs: Batch tensors produced by a PyHealth dataloader. The batch
                may contain labels and may optionally contain matching
                post-event features prefixed by ``post_``.

        Returns:
            A dictionary containing model outputs. Depending on the available
            inputs, this includes logits, probabilities, loss terms,
            embeddings, and optional attention weights.
        """
        pre_feature_map = self._get_branch_feature_map(use_post=False)
        if self.return_attention_weights:
            patient_embedding, pre_attention = self._encode_branch(
                pre_feature_map,
                kwargs,
                return_attention=True,
            )
        else:
            patient_embedding = self._encode_branch(pre_feature_map, kwargs)
            pre_attention = None

        pre_projection = F.normalize(
            self.pre_projection_head(patient_embedding),
            dim=-1,
        )

        results: Dict[str, torch.Tensor] = {}
        if pre_attention is not None:
            for key, value in pre_attention.items():
                results[f"pre_attention_{key}"] = value

        has_any_post = any(key in kwargs for key in self.post_feature_keys.values())
        has_all_post = (
            all(key in kwargs for key in self.post_feature_keys.values())
            and len(self.post_feature_keys) == len(self.base_feature_keys)
        )

        if has_any_post and not has_all_post:
            missing = sorted(
                key for key in self.post_feature_keys.values() if key not in kwargs
            )
            raise ValueError(
                "EBCL received an incomplete post-event batch. Missing keys: "
                + ", ".join(missing)
            )

        post_embedding = None
        post_projection = None
        if has_all_post:
            post_feature_map = self._get_branch_feature_map(use_post=True)
            if self.return_attention_weights:
                post_embedding, post_attention = self._encode_branch(
                    post_feature_map,
                    kwargs,
                    return_attention=True,
                )
                for key, value in post_attention.items():
                    results[f"post_attention_{key}"] = value
            else:
                post_embedding = self._encode_branch(post_feature_map, kwargs)
            post_projection = F.normalize(
                self.post_projection_head(post_embedding),
                dim=-1,
            )

        classifier_input = patient_embedding
        if self.supervised_uses_post and post_embedding is not None:
            classifier_input = torch.cat([patient_embedding, post_embedding], dim=-1)

        logits = self.fc(classifier_input)
        y_prob = self.prepare_y_prob(logits)
        results.update(
            {
                "logit": logits,
                "y_prob": y_prob,
                "pre_projection": pre_projection,
            }
        )

        if post_embedding is not None and post_projection is not None:
            results["post_embed"] = post_embedding
            results["post_projection"] = post_projection
            contrastive_loss, similarity = self._compute_contrastive_loss(
                pre_projection,
                post_projection,
            )
            results["contrastive_loss"] = contrastive_loss
            results["contrastive_logits"] = similarity
            results["loss"] = self.contrastive_weight * contrastive_loss

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            supervised_loss = self.get_loss_function()(logits, y_true)
            results["supervised_loss"] = supervised_loss
            results["y_true"] = y_true
            if "loss" in results:
                results["loss"] = results["loss"] + supervised_loss
            else:
                results["loss"] = supervised_loss

        if kwargs.get("embed", False):
            results["embed"] = patient_embedding
        return results
