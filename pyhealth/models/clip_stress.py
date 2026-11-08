"""Multimodal contrastive pretraining for wearable stress detection.

Contributor: Siqi Zhu (UIUC NetID: siqizhu4).

Paper: Yang, Yu, and Sano, CHIL 2025, "Contrastive Pretraining for Stress
Detection with Multimodal Wearable Sensor Data and Surveys".
https://proceedings.mlr.press/v287/yang25a.html

Independent implementation of the paper's method for PyHealth.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class CLIPStress(BaseModel):
    """Align wearable/survey representations with cached text embeddings.

    A Transformer encodes wearable windows, a residual MLP encodes survey
    features, and a linear layer fuses their pooled representations. Two
    projection heads are trained with symmetric cross-entropy on normalized
    similarities. Downstream classification uses the fused representation.
    No language model is downloaded by this class.

    Args:
        dataset: Fitted PyHealth dataset. Feature widths come from processors.
            Use ``tensor`` for fixed windows (time, channels), tabular vectors,
            and cached text vectors; ``timeseries`` also supports wearable data.
            At most one binary or multiclass output is supported. An unlabeled
            dataset with an empty output schema supports pretraining only.
        signal_key: Wearable feature key; None disables the wearable branch.
        tabular_key: Survey feature key; None disables the survey branch.
        text_key: Cached text embedding key, required only for pretraining.
        mask_key: Optional boolean tensor key (batch, time), True for observed
            timesteps. Required for padded variable-length windows; zero sensor
            values are never treated as padding automatically.
        hidden_dim: Encoder and fused representation width.
        projection_dim: Width of the contrastive embedding space.
        num_layers: Number of Transformer and tabular residual blocks.
        num_heads: Number of Transformer attention heads.
        dropout: Dropout probability.
        temperature: Positive contrastive softmax temperature.
        stage: ``pretrain``, ``linear_probe``, or ``finetune``. A randomly
            initialized model in ``finetune`` is the supervised baseline.

    Notes:
        Pretraining never reads labels. Text embeddings must be generated from
        predictor features only, without target labels or target-derived fields.
        The frozen language encoder can be run once outside the training loop.
        ``mode`` is None during pretraining, so Trainer evaluates contrastive
        loss only. Classification returns the usual PyHealth prediction keys.

        This is a paper-method implementation, not bitwise reproduction of the
        authors' repository. It uses hard paired targets and cosine similarity,
        a residual MLP for tables, and sinusoidal positions for sensor windows.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"signal": [[1., 2.], [3., 4.]], "tabular": [0., 1.],
        ...      "text_embedding": [1., 0., 1.], "label": 0},
        ...     {"signal": [[2., 1.], [4., 3.]], "tabular": [1., 0.],
        ...      "text_embedding": [0., 1., 0.], "label": 1},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {k: "tensor" for k in
        ...               ("signal", "tabular", "text_embedding")},
        ...     {"label": "binary"})
        >>> model = CLIPStress(dataset, hidden_dim=8, num_layers=1, num_heads=2)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> model(**batch)["loss"].ndim
        0
        >>> model.set_stage("linear_probe")
        >>> model(**batch)["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        signal_key: str | None = "signal",
        tabular_key: str | None = "tabular",
        text_key: str = "text_embedding",
        mask_key: str | None = None,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
        stage: str = "pretrain",
    ) -> None:
        super().__init__(dataset)
        if any(v <= 0 for v in (hidden_dim, projection_dim, num_layers, num_heads)):
            raise ValueError("Dimensions, num_layers, and num_heads must be positive.")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if signal_key is None and tabular_key is None:
            raise ValueError("At least one of signal_key and tabular_key is required.")
        keys = [k for k in (signal_key, tabular_key, text_key, mask_key) if k]
        if len(keys) != len(set(keys)):
            raise ValueError("Feature keys must be distinct.")
        if mask_key is not None and signal_key is None:
            raise ValueError("mask_key requires a signal branch.")
        if len(self.label_keys) > 1 or (
            self.label_keys and self.mode not in ("binary", "multiclass")
        ):
            raise ValueError("Use no output or one binary/multiclass output.")
        self.label_key = self.label_keys[0] if self.label_keys else None
        self._classification_mode = self.mode
        self.signal_key = signal_key
        self.tabular_key = tabular_key
        self.text_key = text_key
        self.mask_key = mask_key
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.encoders = nn.ModuleDict()
        if signal_key is not None:
            self.encoders["signal_input"] = nn.Linear(
                self._feature_width(signal_key, 2), hidden_dim
            )
            layer = nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoders["signal"] = nn.TransformerEncoder(
                layer, num_layers, enable_nested_tensor=False
            )
        if tabular_key is not None:
            self.encoders["tabular"] = nn.Sequential(
                nn.Linear(self._feature_width(tabular_key, 1), hidden_dim),
                *[_ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)],
                nn.LayerNorm(hidden_dim),
            )
        if mask_key is not None and mask_key not in dataset.input_processors:
            raise ValueError(f"Mask field {mask_key!r} is absent from input_schema.")
        branches = int(signal_key is not None) + int(tabular_key is not None)
        self.fusion = nn.Linear(hidden_dim * branches, hidden_dim)
        self.signal_projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.classifier = (
            nn.Linear(hidden_dim, self.get_output_size())
            if self.label_key is not None
            else None
        )
        # Optional text must not change the supervised classifier's initialization.
        self.text_projection = (
            nn.Linear(self._feature_width(text_key, 1), projection_dim)
            if text_key in dataset.input_processors
            else None
        )
        self.set_stage(stage)

    def _feature_width(self, key: str, rank: int) -> int:
        if key not in self.dataset.input_processors:
            raise ValueError(f"Feature {key!r} is absent from input_schema.")
        processor = self.dataset.input_processors[key]
        schema = processor.schema()
        if "value" not in schema or processor.is_token():
            raise ValueError(f"{key!r} requires a numeric feature processor.")
        if processor.dim()[schema.index("value")] != rank:
            raise ValueError(f"{key!r} must have {rank} dimensions per sample.")
        width = processor.size()
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"{key!r} requires a fitted positive feature width.")
        return width

    def _value(self, batch: dict[str, Any], key: str) -> torch.Tensor:
        if key not in batch:
            raise ValueError(f"Missing required batch feature {key!r}.")
        value = batch[key]
        if isinstance(value, (tuple, list)):
            schema = self.dataset.input_processors[key].schema()
            value = value[schema.index("value")]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key!r} must contain a processed tensor.")
        return value.to(device=self.device, dtype=self.fusion.weight.dtype)

    def set_stage(self, stage: str) -> None:
        """Select an objective and freeze/unfreeze the appropriate parameters.

        Args:
            stage: ``pretrain``, ``linear_probe``, or ``finetune``.

        Raises:
            ValueError: If the stage or dataset schema is incompatible.

        Notes:
            Create a new Trainer/optimizer after switching stages. A linear
            probe keeps encoder dropout disabled even after ``model.train()``.
            For an entirely unlabeled model, use ``load_pretrained_encoder``
            on a new model built on the labeled dataset.
        """
        if stage not in ("pretrain", "linear_probe", "finetune"):
            raise ValueError("stage must be pretrain, linear_probe, or finetune.")
        if stage == "pretrain" and self.text_projection is None:
            raise ValueError(
                "Pretraining requires cached text embeddings in input_schema."
            )
        if stage != "pretrain" and self.classifier is None:
            raise ValueError("Classification requires a labeled dataset schema.")
        self.stage = stage
        self.mode = None if stage == "pretrain" else self._classification_mode
        # Frozen parameters must not retain gradients from the previous stage.
        self.zero_grad(set_to_none=True)
        for module in (self.encoders, self.fusion):
            module.requires_grad_(stage != "linear_probe")
        for module in (self.signal_projection, self.text_projection):
            if module is not None:
                module.requires_grad_(stage == "pretrain")
        if self.classifier is not None:
            self.classifier.requires_grad_(stage != "pretrain")
        self.train(self.training)

    def load_pretrained_encoder(self, pretrained: CLIPStress) -> None:
        """Copy the feature extractor from an unlabeled pretrained model.

        Args:
            pretrained: Model with matching encoder architecture and feature
                ordering. Its text projection and classifier are not copied.

        Raises:
            ValueError: If encoder parameter names, shapes, or attention head
                counts differ.

        Examples:
            >>> # Build each model using its own fitted dataset schema.
            >>> downstream.load_pretrained_encoder(pretrained)  # doctest: +SKIP
            >>> downstream.set_stage("linear_probe")  # doctest: +SKIP
        """
        if "signal" in self.encoders and "signal" in pretrained.encoders:
            destination_layers = self.encoders["signal"].layers
            source_layers = pretrained.encoders["signal"].layers
            if any(
                destination.self_attn.num_heads != source.self_attn.num_heads
                for destination, source in zip(destination_layers, source_layers)
            ):
                raise ValueError("Pretrained encoder attention head counts differ.")
        pairs = ((self.encoders, pretrained.encoders), (self.fusion, pretrained.fusion))
        for destination, source in pairs:
            current, incoming = destination.state_dict(), source.state_dict()
            if current.keys() != incoming.keys() or any(
                current[key].shape != incoming[key].shape for key in current
            ):
                raise ValueError("Pretrained encoder architecture does not match.")
        for destination, source in pairs:
            destination.load_state_dict(source.state_dict())

    def train(self, mode: bool = True) -> CLIPStress:
        """Set training mode while preserving deterministic frozen encoders."""
        super().train(mode)
        if self.stage == "linear_probe":
            self.encoders.eval()
            self.fusion.eval()
        return self

    def encode(self, **kwargs: Any) -> torch.Tensor:
        """Encode predictors without text or labels.

        Args:
            **kwargs: Processed batch fields matching the configured keys.

        Returns:
            Fused features of shape (batch, hidden_dim).
        """
        features = []
        if self.signal_key is not None:
            x = self._value(kwargs, self.signal_key)
            width = self.encoders["signal_input"].in_features
            if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != width:
                raise ValueError(
                    f"signal must have shape (batch, nonempty time, {width})."
                )
            if self.mask_key is None:
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            else:
                mask_value = self._value(kwargs, self.mask_key)
                if mask_value.shape != x.shape[:2] or not torch.all(
                    (mask_value == 0) | (mask_value == 1)
                ):
                    raise ValueError("mask must be boolean with shape (batch, time).")
                mask = mask_value.bool()
            if not mask.any(dim=1).all():
                raise ValueError("Each signal window must contain a valid timestep.")
            # Mask before projecting so arbitrary padded values cannot leak in.
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            if not torch.isfinite(x).all():
                raise ValueError("Observed signal values must be finite; impute first.")
            x = self.encoders["signal_input"](x)
            positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
            frequency = torch.exp(
                torch.arange(0, self.hidden_dim, 2, device=x.device, dtype=x.dtype)
                * (-math.log(10000.0) / self.hidden_dim)
            )
            angles = positions[:, None] * frequency[None, :]
            positional = torch.zeros_like(x[0])
            positional[:, 0::2] = angles.sin()
            positional[:, 1::2] = angles[:, : self.hidden_dim // 2].cos()
            x = self.encoders["signal"](x + positional, src_key_padding_mask=~mask)
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            features.append(x.sum(dim=1) / mask.sum(dim=1, keepdim=True))
        if self.tabular_key is not None:
            x = self._value(kwargs, self.tabular_key)
            width = self.encoders["tabular"][0].in_features
            if x.ndim != 2 or x.shape[1] != width or not torch.isfinite(x).all():
                raise ValueError(f"tabular must be a finite (batch, {width}) tensor.")
            features.append(self.encoders["tabular"](x))
        if features[0].shape[0] == 0 or any(
            f.shape[0] != features[0].shape[0] for f in features
        ):
            raise ValueError("Modalities must have matching nonempty batch sizes.")
        return self.fusion(torch.cat(features, dim=-1))

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute contrastive loss or downstream predictions.

        Args:
            **kwargs: Processed PyHealth batch. ``embed=True`` also returns
                fused features. Text is only needed in the pretraining stage;
                labels are optional for downstream inference.

        Returns:
            During pretraining, loss and paired-retrieval ``logit``, ``y_prob``,
            ``y_true`` (batch indices, not stress labels). During classification,
            logits and probabilities, plus loss and y_true when labels exist.

        Raises:
            ValueError: For malformed inputs or fewer than two pretraining pairs.
        """
        fused = self.encode(**kwargs)
        if self.stage == "pretrain":
            text = self._value(kwargs, self.text_key)
            width = self.text_projection.in_features
            if text.ndim != 2 or text.shape != (fused.shape[0], width):
                raise ValueError(f"Text must have shape ({fused.shape[0]}, {width}).")
            if fused.shape[0] < 2:
                raise ValueError("Contrastive pretraining needs at least two pairs.")
            if not torch.isfinite(text).all():
                raise ValueError("Text embeddings must be finite.")
            sensor_embedding = F.normalize(self.signal_projection(fused), dim=-1)
            text_embedding = F.normalize(self.text_projection(text), dim=-1)
            logit = sensor_embedding @ text_embedding.T / self.temperature
            y_true = torch.arange(logit.shape[0], device=self.device)
            loss = (
                F.cross_entropy(logit, y_true) + F.cross_entropy(logit.T, y_true)
            ) / 2
            result = {
                "loss": loss,
                "logit": logit,
                "y_prob": logit.softmax(dim=-1),
                "y_true": y_true,
            }
        else:
            logit = self.classifier(fused)
            result = {"logit": logit, "y_prob": self.prepare_y_prob(logit)}
            if self.label_key in kwargs:
                y_true = kwargs[self.label_key].to(self.device)
                if self.mode == "binary":
                    y_true = y_true.to(logit.dtype).reshape_as(logit)
                else:
                    y_true = y_true.long().reshape(-1)
                result["y_true"] = y_true
                result["loss"] = self.get_loss_function()(logit, y_true)
        if kwargs.get("embed", False):
            result["embed"] = fused
        return result
