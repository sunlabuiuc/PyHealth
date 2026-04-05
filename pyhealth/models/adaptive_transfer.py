from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class AdaptiveTransferModel(BaseModel):
    """Adaptive transfer model for multi-source time-series classification.

    This model is a practical PyHealth-style implementation inspired by
    "Daily Physical Activity Monitoring: Adaptive Learning from Multi-Source
    Motion Sensor Data".

    This class provides:

    1. a standard forward pass for supervised training/evaluation.
    2. utilities to compute source-target similarity using paired samples.
    3. utilities for similarity-weighted transfer in example scripts.
    4. dependency injection for both the neural encoder and the distance
       function used inside IPD-style similarity computation.

    Expected input format
    ---------------------
    The selected feature_key should correspond to a dense time-series input.
    The processor schema should expose at least a "value" tensor, optionally
    a "mask" tensor.

    Supported shapes for value:
    - [batch, seq_len] -> interpreted as 1 feature per timestep
    - [batch, seq_len, input_dim]

    Supported shapes for mask:
    - [batch, seq_len]
    - [batch, seq_len, input_dim] (collapsed across feature dimension)

    Custom backbone contract
    ------------------------
    If a custom backbone is provided, it should accept a tensor of shape
    [batch, seq_len, input_dim] and return either:
    - [batch, backbone_output_dim], or
    - [batch, seq_len, backbone_output_dim] (we will pool over time).

    Args:
        dataset: PyHealth sample dataset.
        feature_key: The dense time-series feature to use. If None, the first
            feature in dataset.input_schema is used.
        hidden_dim: Hidden size of the default LSTM backbone.
        num_layers: Number of LSTM layers for the default backbone.
        dropout: Dropout used in the encoder/classifier.
        bidirectional: Whether to use a bidirectional LSTM for the default
            backbone.
        backbone: Optional injected encoder module. If None, a default LSTM
            encoder is used.
        backbone_name: Name for the backbone. Supported strings:
            "lstm" (default), "gru", "mlp".
            Ignored if backbone is explicitly provided.
        backbone_output_dim: Output embedding dimension for the injected
            backbone. If None, we try to infer it from common attributes.
        distance_fn: Distance function used for IPD-style similarity.
            Supported strings: "euclidean", "cosine", "manhattan".
            A custom callable can also be passed.
        use_similarity_weighting: Whether transfer learning rates should be
            scaled by source-target similarity.
        use_kde_smoothing: Whether to smooth pairwise distances before forming
            a similarity score. This is a lightweight approximation for the
            smoothing idea in the paper.
        smoothing_std: Standard deviation used in the smoothing approximation.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        backbone: Optional[nn.Module] = None,
        backbone_name: str = "lstm",
        backbone_output_dim: Optional[int] = None,
        distance_fn: Union[str, DistanceFn] = "euclidean",
        use_similarity_weighting: bool = True,
        use_kde_smoothing: bool = True,
        smoothing_std: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(dataset)

        if len(self.label_keys) != 1:
            raise ValueError("AdaptiveTransferModel supports exactly one label key.")

        self.label_key = self.label_keys[0]
        self.feature_key = feature_key or self.feature_keys[0]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.use_similarity_weighting = use_similarity_weighting
        self.use_kde_smoothing = use_kde_smoothing
        self.smoothing_std = smoothing_std
        self.eps = eps
        self.backbone_name = backbone_name.lower()

        # Infer input dimension from dataset statistics, if possible.
        input_dim = self._infer_input_dim(self.feature_key)

        # Build encoder through dependency injection:
        # 1. explicit module injection takes highest priority.
        # 2. otherwise build from a string backbone name.
        self.encoder, inferred_output_dim, self._is_recurrent_backbone = (
            self._build_encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                backbone=backbone,
                backbone_name=self.backbone_name,
                backbone_output_dim=backbone_output_dim,
            )
        )

        classifier_in_dim = inferred_output_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in_dim, self.get_output_size())

        # Distance function injection:
        # allow simple strings for common metrics, but also support any custom
        # callable with signature (source_emb, target_emb) -> [batch] distance.
        self.distance_fn = self._resolve_distance_fn(distance_fn)

    def _build_encoder(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        backbone: Optional[nn.Module],
        backbone_name: str,
        backbone_output_dim: Optional[int],
    ) -> Tuple[nn.Module, int, bool]:
        """Build the encoder and infer the classifier input dimension."""
        if backbone is not None:
            encoder = backbone
            output_dim = self._infer_backbone_output_dim(backbone, backbone_output_dim)
            return encoder, output_dim, isinstance(backbone, (nn.LSTM, nn.GRU))

        # Default recurrent encoder.
        if backbone_name == "lstm":
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )
            direction_factor = 2 if bidirectional else 1
            output_dim = hidden_dim * direction_factor
            return encoder, output_dim, True

        # Alternative recurrent encoder.
        if backbone_name == "gru":
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )
            direction_factor = 2 if bidirectional else 1
            output_dim = hidden_dim * direction_factor
            return encoder, output_dim, True

        # Simple feed-forward baseline:
        # pool over time, then run through an MLP.
        if backbone_name == "mlp":
            encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return encoder, hidden_dim, False

        raise ValueError(
            f"Unsupported backbone_name: {backbone_name}. "
            "Expected one of {'lstm', 'gru', 'mlp'} or pass a custom backbone."
        )

    def _infer_backbone_output_dim(
        self, backbone: nn.Module, backbone_output_dim: Optional[int]
    ) -> int:
        """Infer output embedding size for a custom backbone."""
        if backbone_output_dim is not None:
            return backbone_output_dim

        # Common attribute names used by many custom modules.
        candidate_attrs = [
            "output_dim",
            "hidden_dim",
            "hidden_size",
            "embedding_dim",
        ]
        for attr in candidate_attrs:
            if hasattr(backbone, attr):
                value = getattr(backbone, attr)
                if isinstance(value, int) and value > 0:
                    return value

        raise ValueError(
            "Could not infer backbone output dimension. "
            "Please provide backbone_output_dim when injecting a custom backbone."
        )

    def _resolve_distance_fn(
        self, distance_fn: Union[str, DistanceFn]
    ) -> DistanceFn:
        """Resolve a string or callable distance function into a callable."""
        if callable(distance_fn):
            return distance_fn

        distance_name = distance_fn.lower()

        if distance_name == "euclidean":
            return lambda x, y: torch.norm(x - y, p=2, dim=1)

        if distance_name == "manhattan":
            return lambda x, y: torch.norm(x - y, p=1, dim=1)

        if distance_name == "cosine":
            return lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=1)

        raise ValueError(
            f"Unsupported distance_fn: {distance_fn}. "
            "Expected one of {'euclidean', 'manhattan', 'cosine'} "
            "or a custom callable."
        )

    def _infer_input_dim(self, feature_key: str) -> int:
        """Infer dense feature dimensionality from dataset statistics.

        Falls back to 1 when shape metadata is unavailable.
        """
        if self.dataset is None:
            return 1

        # Prefer explicit dimension statistics if they exist.
        try:
            stats = self.dataset.input_info[feature_key]
            if "len" in stats and isinstance(stats["len"], int):
                # For dense vectors represented at each timestep.
                return max(1, int(stats["len"]))
            if "dim" in stats and isinstance(stats["dim"], int):
                return max(1, int(stats["dim"]))
        except Exception:
            pass

        # Conservative fallback.
        return 1

    def _get_feature_value_and_mask(
        self, feature: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract value and mask tensors from a PyHealth feature tuple."""
        if isinstance(feature, torch.Tensor):
            value = feature
            mask = None
        else:
            schema = self.dataset.input_processors[self.feature_key].schema()
            if "value" not in schema:
                raise ValueError(
                    f"Feature '{self.feature_key}' must contain 'value' in its schema."
                )

            value = feature[schema.index("value")]
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if mask is None and len(feature) == len(schema) + 1:
                mask = feature[-1]

        value = value.to(self.device).float()
        if mask is not None:
            mask = mask.to(self.device).float()

        # Normalize shapes:
        # [B, T] -> [B, T, 1]
        if value.dim() == 2:
            value = value.unsqueeze(-1)
        elif value.dim() != 3:
            raise ValueError(
                f"Unsupported input shape for '{self.feature_key}': {tuple(value.shape)}"
            )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.any(dim=-1).float()
            elif mask.dim() != 2:
                raise ValueError(
                    f"Unsupported mask shape for '{self.feature_key}': {tuple(mask.shape)}"
                )

        return value, mask

    def _masked_mean_pool(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool a sequence tensor [B, T, D] into [B, D] using an optional mask."""
        if mask is None:
            return x.mean(dim=1)

        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_sequence(
        self, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode a dense time series into a fixed-size embedding."""
        # Recurrent backbones return full hidden state structures that need to be
        # handled explicitly.
        if isinstance(self.encoder, nn.LSTM):
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    value, lengths, batch_first=True, enforce_sorted=False
                )
                _, (h_n, _) = self.encoder(packed)
            else:
                _, (h_n, _) = self.encoder(value)

            if self.bidirectional:
                # Last layer forward/backward states.
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h_last = h_n[-1]

            return self.dropout(h_last)

        if isinstance(self.encoder, nn.GRU):
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    value, lengths, batch_first=True, enforce_sorted=False
                )
                _, h_n = self.encoder(packed)
            else:
                _, h_n = self.encoder(value)

            if self.bidirectional:
                # Last layer forward/backward states.
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h_last = h_n[-1]

            return self.dropout(h_last)

        # Non-recurrent encoders are treated more generically.
        # If the encoder returns [B, T, D], we pool over time.
        # If it already returns [B, D], we use it directly.
        encoder_out = self.encoder(value)

        if encoder_out.dim() == 3:
            emb = self._masked_mean_pool(encoder_out, mask)
        elif encoder_out.dim() == 2:
            emb = encoder_out
        else:
            raise ValueError(
                "Custom backbone must return a tensor of shape [B, D] or [B, T, D]."
            )

        return self.dropout(emb)

    def forward(
        self, **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass following the PyHealth BaseModel convention."""
        if self.feature_key not in kwargs:
            raise ValueError(
                f"Expected feature key '{self.feature_key}' in model inputs."
            )

        value, mask = self._get_feature_value_and_mask(kwargs[self.feature_key])
        patient_emb = self._encode_sequence(value, mask)
        logits = self.classifier(patient_emb)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)

            # Cross-entropy expects [B] labels for multiclass.
            if self.mode == "multiclass" and y_true.dim() > 1:
                y_true = y_true.squeeze(-1).long()
            elif self.mode == "binary":
                y_true = y_true.float()
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(-1)

            loss_fn = self.get_loss_function()
            loss = loss_fn(logits, y_true)

            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def forward_from_embedding(
        self, **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Dict[str, torch.Tensor]:
        """Compatibility hook for interpretability.

        Since this model already operates on dense values, we reuse forward().
        """
        return self.forward(**kwargs)

    @torch.no_grad()
    def extract_embedding(
        self, batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
    ) -> torch.Tensor:
        """Return the latent embedding for a batch."""
        value, mask = self._get_feature_value_and_mask(batch[self.feature_key])
        return self._encode_sequence(value, mask)

    @torch.no_grad()
    def compute_pairwise_distances(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        """Compute paired distances between source and target embeddings.

        The two batches are assumed to be paired by row index.
        """
        source_emb = self.extract_embedding(source_batch)
        target_emb = self.extract_embedding(target_batch)

        if source_emb.shape[0] != target_emb.shape[0]:
            raise ValueError(
                "Source and target batches must have the same batch size for paired IPD."
            )

        return self.distance_fn(source_emb, target_emb)

    @torch.no_grad()
    def compute_ipd(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> float:
        """Compute an IPD-style distance between one source batch and target batch.

        This is a lightweight approximation:
        - compute paired embedding distances,
        - optionally smooth them with Gaussian perturbation,
        - average the result.
        """
        distances = self.compute_pairwise_distances(source_batch, target_batch)

        if self.use_kde_smoothing:
            noise = torch.randn_like(distances) * self.smoothing_std
            distances = (distances + noise).clamp_min(0.0)

        return float(distances.mean().item())

    @torch.no_grad()
    def compute_source_similarities(
        self,
        source_batches: Sequence[
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
        ],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> List[float]:
        """Compute source-target similarities for multiple source domains.

        Similarity is defined as inverse distance:
            similarity = 1 / (ipd + eps)
        """
        similarities: List[float] = []
        for source_batch in source_batches:
            ipd = self.compute_ipd(source_batch, target_batch)
            sim = 1.0 / (ipd + self.eps)
            similarities.append(sim)
        return similarities

    @torch.no_grad()
    def rank_source_domains(
        self,
        source_batches: Sequence[
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
        ],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> List[int]:
        """Return source domain indices sorted by descending similarity."""
        similarities = self.compute_source_similarities(source_batches, target_batch)
        ranked = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )
        return ranked

    def get_adaptive_lr(self, base_lr: float, similarity: float) -> float:
        """Scale the base learning rate by similarity."""
        if not self.use_similarity_weighting:
            return base_lr
        return base_lr * max(similarity, self.eps)
