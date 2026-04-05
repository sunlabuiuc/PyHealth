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

    Raises:
        ValueError: If the dataset does not expose exactly one label key.

    Example:
        >>> model = AdaptiveTransferModel(
        ...     dataset=dataset,
        ...     feature_key="signal",
        ...     backbone_name="lstm",
        ...     distance_fn="euclidean",
        ... )
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
        backbone: Optional[nn.Module] = None,
        backbone_name: str = "lstm",
        backbone_output_dim: Optional[int] = None,
        distance_fn: Union[str, DistanceFn] = "euclidean",
        use_similarity_weighting: bool = True,
        use_kde_smoothing: bool = True,
        smoothing_std: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the adaptive transfer model.

        Args:
            dataset: PyHealth sample dataset.
            feature_key: Dense input feature key. If None, uses the first
                available feature key from the dataset.
            hidden_dim: Hidden size for the default recurrent encoder.
            num_layers: Number of recurrent layers in the default encoder.
            dropout: Dropout probability applied in encoder/classifier.
            bidirectional: Whether the default recurrent encoder is
                bidirectional.
            backbone: Optional custom encoder module.
            backbone_name: Name of built-in fallback encoder to construct when
                backbone is not provided.
            backbone_output_dim: Output embedding dimension for a custom
                backbone.
            distance_fn: Distance function identifier or callable used for
                IPD-style similarity.
            use_similarity_weighting: Whether adaptive learning rates should be
                scaled by source-target similarity.
            use_kde_smoothing: Whether to apply smoothing to pairwise
                distances before averaging.
            smoothing_std: Standard deviation of the Gaussian smoothing noise.
            eps: Small constant for numerical stability.

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
        """Build the encoder module and infer its output size.

        Args:
            input_dim: Input feature dimension per timestep.
            hidden_dim: Hidden dimension for built-in encoders.
            num_layers: Number of recurrent layers for built-in encoders.
            dropout: Dropout probability for built-in encoders.
            bidirectional: Whether built-in recurrent encoders are
                bidirectional.
            backbone: Explicit custom backbone module, if provided.
            backbone_name: Name of built-in encoder to construct when
                backbone is not provided.
            backbone_output_dim: Explicit output dimension for a custom
                backbone.

        Returns:
            A tuple containing:
                - the constructed encoder module,
                - the encoder output dimension,
                - whether the encoder is recurrent.

        Raises:
            ValueError: If backbone_name is unsupported.
        """
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
        """Infer the embedding size of a custom backbone.

        Args:
            backbone: Custom encoder module.
            backbone_output_dim: Explicit output dimension, if provided.

        Returns:
            The inferred output embedding dimension.

        Raises:
            ValueError: If the output dimension cannot be inferred from either
                the argument or common module attributes.
        """
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
        """Resolve a distance function specification into a callable.

        Args:
            distance_fn: String name of a built-in distance function or a
                callable that maps two embedding tensors of shape [B, D] to a
                distance tensor of shape [B].

        Returns:
            A callable distance function.

        Raises:
            ValueError: If a string distance name is provided but is not
                supported.
        """
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
        """Infer dense input dimensionality from dataset metadata.

        Args:
            feature_key: Feature name whose dimension should be inferred.

        Returns:
            The inferred input dimension. Returns 1 as a conservative fallback
            when metadata is unavailable.
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
        """Extract dense value and optional mask tensors from a feature input.

        Args:
            feature: Either a raw tensor or a PyHealth processor tuple that
                contains a "value" tensor and optionally a "mask" tensor.

        Returns:
            A tuple of:
                - value tensor with normalized shape [B, T, D]
                - optional mask tensor with shape [B, T]

        Raises:
            ValueError: If the feature schema is missing "value", or if the
                value/mask shapes are unsupported.
        """
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
        """Apply masked mean pooling over the time dimension.

        Args:
            x: Sequence tensor of shape [B, T, D].
            mask: Optional binary mask of shape [B, T].

        Returns:
            Pooled tensor of shape [B, D].
        """
        if mask is None:
            return x.mean(dim=1)

        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_sequence(
        self, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode a dense time series into a fixed-size embedding.

        Args:
            value: Input value tensor of shape [B, T, D].
            mask: Optional mask tensor of shape [B, T].

        Returns:
            Embedding tensor of shape [B, H], where H is the backbone output
            dimension.

        Raises:
            ValueError: If a custom non-recurrent backbone returns an
                unsupported tensor rank.
        """
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
        """Run the forward pass and optionally compute loss.

        Args:
            **kwargs: Keyword inputs expected by PyHealth. Must contain the
                configured feature key. May also contain the label key.

        Returns:
            A dictionary containing:
                - "logit": raw logits
                - "y_prob": post-processed prediction probabilities
                - optionally "loss" and "y_true" when labels are provided

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
        """Forward method used for compatibility with PyHealth hooks.

        Args:
            **kwargs: Same keyword arguments accepted by ``forward``.

        Returns:
            The same dictionary returned by ``forward``.
        """
        return self.forward(**kwargs)

    @torch.no_grad()
    def extract_embedding(
        self, batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
    ) -> torch.Tensor:
        """Extract latent embeddings for a batch without gradient tracking.

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

        The two batches are assumed to be aligned by row index.

        Args:
            source_batch: Source-domain batch dictionary.
            target_batch: Target-domain batch dictionary.

        Returns:
            Distance tensor of shape [B], where each entry is the distance
            between paired source and target embeddings.

        Raises:
            ValueError: If the source and target batch sizes differ.
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
        """Compute an IPD-style distance between one source and target batch.

        This implementation is a lightweight approximation of the paper's
        similarity computation:
        1. compute paired embedding distances,
        2. optionally smooth them with Gaussian perturbation,
        3. average the result.

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
        """Compute source-target similarities for multiple source domains.

        Similarity is defined as inverse distance:
            similarity = 1 / (ipd + eps)

        Args:
            source_batches: Sequence of source-domain batches.
            target_batch: Target-domain batch dictionary.

        Returns:
            A list of similarity scores, one per source batch.
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
        """Rank source domains by descending similarity to the target.

        Args:
            source_batches: Sequence of source-domain batches.
            target_batch: Target-domain batch dictionary.

        Returns:
            List of source-domain indices sorted from most similar to least
            similar.
        """
        similarities = self.compute_source_similarities(source_batches, target_batch)
        ranked = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )
        return ranked

    def get_adaptive_lr(self, base_lr: float, similarity: float) -> float:
        """Compute the learning rate used for a source domain.

        Args:
            base_lr: Base learning rate before adaptation.
            similarity: Source-target similarity score.

        Returns:
            Adapted learning rate. If similarity weighting is disabled,
            returns the base learning rate unchanged.
        """
        if not self.use_similarity_weighting:
            return base_lr
        return base_lr * max(similarity, self.eps)
