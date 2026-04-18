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

    This model is inspired by
    "Daily Physical Activity Monitoring: Adaptive Learning from Multi-Source
    Motion Sensor Data".

    The model supports:
        1. standard supervised forward passes;
        2. paired source-target similarity computation;
        3. similarity-weighted transfer utilities for example scripts;
        4. dependency injection for both the backbone and distance function.

    Args:
        dataset: PyHealth sample dataset.
        feature_key: Dense time-series feature key. If None, uses the first
            available feature key.
        hidden_dim: Hidden size for built-in backbones.
        num_layers: Number of recurrent layers for built-in backbones.
        dropout: Dropout probability.
        bidirectional: Whether recurrent backbones are bidirectional.
        backbone: Backbone encoder specification. Supported string values are
            {"lstm", "gru", "mlp"}, or a custom ``nn.Module`` can be passed.
        backbone_output_dim: Output dimension of a custom backbone. Required
            when it cannot be inferred from the module itself.
        distance_fn: Distance function for IPD-style similarity. One of
            {"euclidean", "manhattan", "cosine"} or a callable. Euclidean and
            Manhattan use ``torch.nn.functional.pairwise_distance``; cosine
            uses ``1 - cosine_similarity``.
        use_similarity_weighting: Whether to scale learning rates by similarity.
        use_kde_smoothing: Whether to smooth pairwise distances before
            averaging.
        smoothing_std: Standard deviation of Gaussian smoothing noise.
        eps: Small constant for numerical stability.
        input_dim: Optional per-time-step input width; inferred from the
            dataset when omitted.
    Raises:
        ValueError: If the dataset does not expose exactly one label key.

    Example:
        >>> model = AdaptiveTransferModel(dataset=dataset, feature_key="signal")
        >>> output = model(**batch)
        >>> output["logit"].shape
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        backbone: Union[str, nn.Module] = "lstm",
        backbone_output_dim: Optional[int] = None,
        distance_fn: Union[str, DistanceFn] = "euclidean",
        use_similarity_weighting: bool = True,
        use_kde_smoothing: bool = True,
        smoothing_std: float = 0.01,
        eps: float = 1e-8,
        input_dim: Optional[int] = None,
    ) -> None:
        """Initialize the adaptive transfer model.

        Args:
            dataset: PyHealth sample dataset.
            feature_key: Dense input feature key. If None, uses the first
                available feature key from the dataset.
            hidden_dim: Hidden size for built-in backbones.
            num_layers: Number of recurrent layers for built-in backbones.
            dropout: Dropout probability.
            bidirectional: Whether recurrent backbones are bidirectional.
            backbone: Backbone encoder specification. Supported string values
                are {"lstm", "gru", "mlp"}, or a custom ``nn.Module``.
            backbone_output_dim: Output dimension of a custom backbone.
            distance_fn: Distance function identifier or callable used for
                IPD-style similarity.
            use_similarity_weighting: Whether adaptive learning rates should be
                scaled by source-target similarity.
            use_kde_smoothing: Whether to apply smoothing to pairwise
                distances before averaging.
            smoothing_std: Standard deviation of the Gaussian smoothing noise.
            eps: Small constant for numerical stability.
            input_dim: If set, per-time-step input size for built-in backbones
                (e.g. number of DSA channels). When ``None``, the model infers
                from ``dataset.input_info`` when present, otherwise from the
                first training sample.

        Raises:
            ValueError: If the dataset exposes more than one label key.
        """
        super().__init__(dataset)

        if len(self.label_keys) != 1:
            raise ValueError("AdaptiveTransferModel supports exactly one label key.")

        self.label_key = self.label_keys[0]
        self.feature_key = feature_key or self.feature_keys[0]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_similarity_weighting = use_similarity_weighting
        self.use_kde_smoothing = use_kde_smoothing
        self.smoothing_std = smoothing_std
        self.eps = eps

        encoder_input_dim = (
            max(1, int(input_dim))
            if input_dim is not None
            else self._infer_input_dim(self.feature_key)
        )
        self.encoder, encoder_output_dim = self._build_encoder(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            backbone=backbone,
            backbone_output_dim=backbone_output_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_output_dim, self.get_output_size())
        self.distance_fn = self._resolve_distance_fn(distance_fn)

    def _build_encoder(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        backbone: Union[str, nn.Module],
        backbone_output_dim: Optional[int],
    ) -> Tuple[nn.Module, int]:
        """Build the encoder and infer its output size."""
        if isinstance(backbone, nn.Module):
            output_dim = self._infer_backbone_output_dim(
                backbone, backbone_output_dim
            )
            return backbone, output_dim

        backbone_name = backbone.lower()

        if backbone_name == "lstm":
            encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            output_dim = hidden_dim * (2 if bidirectional else 1)
            return encoder, output_dim

        if backbone_name == "gru":
            encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            output_dim = hidden_dim * (2 if bidirectional else 1)
            return encoder, output_dim

        if backbone_name == "mlp":
            encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return encoder, hidden_dim

        raise ValueError(
            f"Unsupported backbone: {backbone}. Expected one of "
            "{'lstm', 'gru', 'mlp'} or a custom backbone module."
        )

    def _infer_backbone_output_dim(
        self,
        backbone: nn.Module,
        backbone_output_dim: Optional[int],
    ) -> int:
        """Infer output size for a custom backbone."""
        if backbone_output_dim is not None:
            return backbone_output_dim

        for attr in ["output_dim", "hidden_dim", "hidden_size", "embedding_dim"]:
            if hasattr(backbone, attr):
                value = getattr(backbone, attr)
                if isinstance(value, int) and value > 0:
                    return value

        raise ValueError(
            "Could not infer backbone output dimension. Please provide "
            "backbone_output_dim for a custom backbone."
        )

    def _resolve_distance_fn(
        self,
        distance_fn: Union[str, DistanceFn],
    ) -> DistanceFn:
        """Resolve a string or callable distance function."""
        if callable(distance_fn):
            return distance_fn

        name = distance_fn.lower()
        if name == "euclidean":
            return lambda x, y: F.pairwise_distance(x, y, p=2)
        if name == "manhattan":
            return lambda x, y: F.pairwise_distance(x, y, p=1)
        if name == "cosine":
            return lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=1)

        raise ValueError(
            f"Unsupported distance_fn: {distance_fn}. Expected one of "
            "{'euclidean', 'manhattan', 'cosine'} or a callable."
        )

    def _infer_input_dim(self, feature_key: str) -> int:
        """Infer per-time-step width from ``input_info`` or the first sample."""
        if self.dataset is None:
            return 1

        try:
            stats = self.dataset.input_info[feature_key]
            if "len" in stats and isinstance(stats["len"], int):
                return max(1, int(stats["len"]))
            if "dim" in stats and isinstance(stats["dim"], int):
                return max(1, int(stats["dim"]))
        except (KeyError, TypeError, AttributeError):
            pass

        try:
            n = len(self.dataset)
            if n == 0 or feature_key not in self.dataset[0]:
                return 1
            feature = self.dataset[0][feature_key]
            if isinstance(feature, torch.Tensor):
                value = feature
            else:
                proc = self.dataset.input_processors[feature_key]
                schema = proc.schema()
                if "value" not in schema:
                    return 1
                value = feature[schema.index("value")]
            if value.dim() == 1:
                return 1
            if value.dim() >= 2:
                return max(1, int(value.shape[-1]))
        except Exception:
            pass

        return 1

    def _get_feature_value_and_mask(
        self,
        feature: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract normalized value and mask tensors from a feature input."""
        if isinstance(feature, torch.Tensor):
            value = feature
            mask = None
        else:
            schema = self.dataset.input_processors[self.feature_key].schema()
            if "value" not in schema:
                raise ValueError(
                    f"Feature '{self.feature_key}' must contain 'value'."
                )

            value = feature[schema.index("value")]
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if mask is None and len(feature) == len(schema) + 1:
                mask = feature[-1]

        value = value.to(self.device).float()
        if mask is not None:
            mask = mask.to(self.device).float()

        # [B, T] -> [B, T, 1]
        if value.dim() == 2:
            value = value.unsqueeze(-1)
        elif value.dim() != 3:
            raise ValueError(
                f"Unsupported input shape for '{self.feature_key}': "
                f"{tuple(value.shape)}"
            )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.any(dim=-1).float()
            elif mask.dim() != 2:
                raise ValueError(
                    f"Unsupported mask shape for '{self.feature_key}': "
                    f"{tuple(mask.shape)}"
                )

        return value, mask

    def _masked_mean_pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply masked mean pooling over the time dimension."""
        if mask is None:
            return x.mean(dim=1)

        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_sequence(
        self,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a dense time series into a fixed-size embedding."""
        if isinstance(self.encoder, nn.LSTM):
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    value,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, (h_n, _) = self.encoder(packed)
            else:
                _, (h_n, _) = self.encoder(value)

            if self.bidirectional:
                emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                emb = h_n[-1]
            return self.dropout(emb)

        if isinstance(self.encoder, nn.GRU):
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    value,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, h_n = self.encoder(packed)
            else:
                _, h_n = self.encoder(value)

            if self.bidirectional:
                emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                emb = h_n[-1]
            return self.dropout(emb)

        # Non-recurrent encoders may return [B, D] or [B, T, D].
        encoder_out = self.encoder(value)

        if encoder_out.dim() == 3:
            emb = self._masked_mean_pool(encoder_out, mask)
        elif encoder_out.dim() == 2:
            emb = encoder_out
        else:
            raise ValueError(
                "Custom backbone must return a tensor of shape [B, D] "
                "or [B, T, D]."
            )

        return self.dropout(emb)

    def forward(
        self,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Dict[str, torch.Tensor]:
        """Run the forward pass and optionally compute loss.

        Args:
            **kwargs: Keyword inputs expected by PyHealth. Must contain the
                configured feature key. May also contain the label key.

        Returns:
            A dictionary containing model outputs and, when labels are
            provided, the loss and ground-truth labels.

        Raises:
            ValueError: If the configured feature key is missing from inputs.
        """
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

            if self.mode == "multiclass" and y_true.dim() > 1:
                y_true = y_true.squeeze(-1).long()
            elif self.mode == "binary":
                y_true = y_true.float()
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(-1)

            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def forward_from_embedding(
        self,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Dict[str, torch.Tensor]:
        """Forward hook kept for compatibility with PyHealth interfaces."""
        return self.forward(**kwargs)

    @torch.no_grad()
    def extract_embedding(
        self,
        batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        """Extract latent embeddings for a batch.

        Args:
            batch: Batch dictionary containing the configured feature key.

        Returns:
            Batch embedding tensor of shape [B, H].
        """
        value, mask = self._get_feature_value_and_mask(batch[self.feature_key])
        return self._encode_sequence(value, mask)

    @torch.no_grad()
    def compute_pairwise_distances(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        """Compute paired distances between source and target embeddings.

        Args:
            source_batch: Source-domain batch dictionary.
            target_batch: Target-domain batch dictionary.

        Returns:
            Distance tensor of shape [B].

        Raises:
            ValueError: If the source and target batch sizes differ.
        """
        source_emb = self.extract_embedding(source_batch)
        target_emb = self.extract_embedding(target_batch)

        if source_emb.shape[0] != target_emb.shape[0]:
            raise ValueError(
                "Source and target batches must have the same batch size "
                "for paired IPD."
            )

        return self.distance_fn(source_emb, target_emb)

    @torch.no_grad()
    def compute_ipd(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> float:
        """Compute an IPD-style distance between one source and target batch.

        Args:
            source_batch: Source-domain batch dictionary.
            target_batch: Target-domain batch dictionary.

        Returns:
            Mean paired distance as a float.
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
        """Compute inverse-distance similarities for multiple source batches.

        Args:
            source_batches: Sequence of source-domain batches.
            target_batch: Target-domain batch dictionary.

        Returns:
            List of similarity scores, one per source batch.
        """
        similarities: List[float] = []
        for source_batch in source_batches:
            ipd = self.compute_ipd(source_batch, target_batch)
            similarities.append(1.0 / (ipd + self.eps))
        return similarities

    @torch.no_grad()
    def rank_source_domains(
        self,
        source_batches: Sequence[
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
        ],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> List[int]:
        """Rank source domains by descending similarity to the target.

        Args:
            source_batches: Sequence of source-domain batches.
            target_batch: Target-domain batch dictionary.

        Returns:
            List of source-domain indices sorted from most similar to least
            similar.
        """
        similarities = self.compute_source_similarities(source_batches, target_batch)
        return sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )

    def get_adaptive_lr(self, base_lr: float, similarity: float) -> float:
        """Scale the base learning rate by similarity if enabled.

        Args:
            base_lr: Base learning rate before adaptation.
            similarity: Source-target similarity score.

        Returns:
            Adapted learning rate.
        """
        if not self.use_similarity_weighting:
            return base_lr
        return base_lr * max(similarity, self.eps)
