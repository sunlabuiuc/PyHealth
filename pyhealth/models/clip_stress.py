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
from pyhealth.models.clip_stress_layers import (
    StressProjection,
    StressTabularResNet,
    stress_contrastive_loss,
)


class CLIPStress(BaseModel):
    """Align wearable/survey representations with cached text embeddings.

    Matches the authors' channel-attention Transformer, convolutional tabular
    ResNet and residual projection heads. The input API uses (time, channels);
    the model transposes it to (channels, time) before attention. No positional
    encoding or learned sensor input projection is added. Classification uses
    the pretrained projection followed by ReLU and a linear classifier.

    Args:
        dataset: Fitted PyHealth dataset with tensor signal/tabular/text fields
            and zero or one binary/multiclass output. No weights are downloaded.
        signal_key: Wearable field; None disables this branch.
        tabular_key: Survey field; None disables this branch.
        text_key: Cached frozen BERT vector; required only for pretraining.
        mask_key: Optional all-true mask. Partial or variable windows are rejected:
            the reference architecture requires fixed complete windows.
        hidden_dim: Transformer feedforward width and tabular output width.
        projection_dim: Contrastive and downstream projection width (128).
        num_layers: Transformer layers and convolutional residual blocks (3).
        num_heads: Attention heads (4); must divide signal_length.
        dropout: Transformer dropout (0.1).
        temperature: Contrastive temperature (0.07).
        stage: pretrain, linear_probe, or finetune (also supervised from scratch).
        signal_length: Fixed time dimension (24 hourly values).
        fusion_dim: Fused width; defaults to hidden_dim // 2, as in author code.
        residual_dropout: Tabular residual-block dropout (0.5).
        projection_dropout: Projection-head dropout (0.5).
        objective: reference_soft matches the public code; paper_cosine follows
            the printed paired-cosine objective with the minimization sign fixed.

    Notes:
        Pretraining never reads labels; use label-excluded text, an explicit
        configuration in the authors' scripts. Binary classification internally
        uses two logits and cross-entropy, while returning their difference and
        positive-class probability for the PyHealth binary contract.
        Linear probing freezes encoder and projection parameters. As in the
        reference training loop, train() still enables dropout and BatchNorm
        updates; eval() is deterministic. Old MLP checkpoints are incompatible.

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
        >>> model = CLIPStress(dataset, hidden_dim=8, num_layers=1, num_heads=2, signal_length=2)
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
        hidden_dim: int = 64,
        projection_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
        stage: str = "pretrain",
        signal_length: int = 24,
        fusion_dim: int | None = None,
        residual_dropout: float = 0.5,
        projection_dropout: float = 0.5,
        objective: str = "reference_soft",
    ) -> None:
        super().__init__(dataset)
        if any(v <= 0 for v in (hidden_dim, projection_dim, num_layers, num_heads)):
            raise ValueError("Dimensions, num_layers, and num_heads must be positive.")
        fusion_dim = hidden_dim // 2 if fusion_dim is None else fusion_dim
        if signal_length <= 0 or fusion_dim <= 0:
            raise ValueError("signal_length and fusion_dim must be positive.")
        if signal_key is not None and signal_length % num_heads:
            raise ValueError("signal_length must be divisible by num_heads.")
        if objective not in ("reference_soft", "paper_cosine"):
            raise ValueError("objective must be reference_soft or paper_cosine.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if any(not 0 <= p < 1 for p in (dropout, residual_dropout, projection_dropout)):
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
        self.signal_length = signal_length
        self.fusion_dim = fusion_dim
        self.objective = objective
        self.signal_width = self._feature_width(signal_key, 2) if signal_key else None

        self.encoders = nn.ModuleDict()
        if signal_key is not None:
            layer = nn.TransformerEncoderLayer(
                signal_length,
                num_heads,
                hidden_dim,
                dropout,
                activation="relu",
                batch_first=True,
            )
            self.encoders["signal"] = nn.TransformerEncoder(
                layer, num_layers, enable_nested_tensor=False
            )
        if tabular_key is not None:
            self.encoders["tabular"] = StressTabularResNet(
                self._feature_width(tabular_key, 1),
                hidden_dim,
                num_layers,
                residual_dropout,
            )
        if mask_key is not None and mask_key not in dataset.input_processors:
            raise ValueError("mask_key is absent from input_schema.")
        fused_input = (signal_length if signal_key else 0) + (
            hidden_dim if tabular_key else 0
        )
        self.fusion = nn.Linear(fused_input, fusion_dim)
        self.signal_projection = StressProjection(
            fusion_dim, projection_dim, projection_dropout
        )
        self.classifier = (
            nn.Linear(
                projection_dim, 2 if self.mode == "binary" else self.get_output_size()
            )
            if self.label_key is not None
            else None
        )
        # Optional text must not affect initialization of the supervised network.
        self.text_projection = (
            StressProjection(
                self._feature_width(text_key, 1), projection_dim, projection_dropout
            )
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
        for module in (self.encoders, self.fusion, self.signal_projection):
            module.requires_grad_(stage != "linear_probe")
        for module in (self.text_projection,):
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
        pairs = (
            (self.encoders, pretrained.encoders),
            (self.fusion, pretrained.fusion),
            (self.signal_projection, pretrained.signal_projection),
        )
        for destination, source in pairs:
            current, incoming = destination.state_dict(), source.state_dict()
            if current.keys() != incoming.keys() or any(
                current[key].shape != incoming[key].shape for key in current
            ):
                raise ValueError("Pretrained encoder architecture does not match.")
        for destination, source in pairs:
            destination.load_state_dict(source.state_dict())

    def encode(self, **kwargs: Any) -> torch.Tensor:
        """Encode predictors without text or labels.

        Args:
            **kwargs: Processed batch fields matching the configured keys.

        Returns:
            Fused features of shape (batch, fusion_dim).
        """
        features = []
        if self.signal_key is not None:
            x = self._value(kwargs, self.signal_key)
            if x.ndim != 3 or x.shape[1:] != (self.signal_length, self.signal_width):
                raise ValueError(
                    f"signal must have shape (batch, {self.signal_length}, {self.signal_width})."
                )
            if self.mask_key is not None:
                mask = self._value(kwargs, self.mask_key)
                if mask.shape != x.shape[:2] or not torch.all(mask == 1):
                    raise ValueError(
                        "Reference channel attention requires complete fixed windows."
                    )
            if not torch.isfinite(x).all():
                raise ValueError("signal must be finite; impute before training.")
            features.append(self.encoders["signal"](x.transpose(1, 2)).mean(1))
        if self.tabular_key is not None:
            x = self._value(kwargs, self.tabular_key)
            width = self.encoders["tabular"].input_dim
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
            width = self.text_projection.input_dim
            if text.ndim != 2 or text.shape != (fused.shape[0], width):
                raise ValueError(f"Text must have shape ({fused.shape[0]}, {width}).")
            if fused.shape[0] < 2 and self.objective == "paper_cosine":
                raise ValueError("Contrastive pretraining needs at least two pairs.")
            if not torch.isfinite(text).all():
                raise ValueError("Text embeddings must be finite.")
            sensor_embedding = self.signal_projection(fused)
            text_embedding = self.text_projection(text)
            logit, loss = stress_contrastive_loss(
                sensor_embedding, text_embedding, self.temperature, self.objective
            )
            y_true = torch.arange(logit.shape[0], device=self.device)
            result = {
                "loss": loss,
                "logit": logit,
                "y_prob": logit.softmax(dim=-1),
                "y_true": y_true,
            }
        else:
            class_logits = self.classifier(F.relu(self.signal_projection(fused)))
            binary = self._classification_mode == "binary"
            logit = (
                class_logits[:, 1:2] - class_logits[:, :1] if binary else class_logits
            )
            result = {"logit": logit, "y_prob": self.prepare_y_prob(logit)}
            if self.label_key in kwargs:
                target = kwargs[self.label_key].to(self.device).long().reshape(-1)
                result["y_true"] = (
                    target.to(logit.dtype).unsqueeze(-1) if binary else target
                )
                result["loss"] = F.cross_entropy(class_logits, target)
        if kwargs.get("embed", False):
            result["embed"] = fused
        return result
