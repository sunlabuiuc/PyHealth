# Author: Sean Nian
# Description: MedFuse model implementation for PyHealth.

from __future__ import annotations

from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class MedFuseLayer(nn.Module):
    """MedFuse fusion layer.

    Fuses EHR time-series and chest X-ray (CXR) representations using an
    LSTM-based sequential fusion strategy that supports missing CXR modality.

    Paper:
        Hayat, N., Geras, K. J., & Shamout, F. E. (2022).
        MedFuse: Multi-modal fusion with clinical time-series data and chest
        X-ray images. MLHC 2022.

    Args:
        ehr_input_dim: Dimension of EHR features at each timestep.
        ehr_hidden_dim: Hidden dimension of the EHR LSTM encoder.
            Default is 256.
        ehr_num_layers: Number of stacked LSTM layers for EHR.
            Default is 2.
        cxr_backbone: ResNet variant for CXR encoder.
            Default is ``"resnet34"``.
        cxr_pretrained: Whether to use pretrained CXR encoder weights.
            Default is True.
        fusion_hidden_dim: Hidden dimension of fusion LSTM.
            Default is 512.
        projection_dim: Dimension to project CXR features to.
            Must match ``ehr_hidden_dim``.
        dropout: Dropout rate.
        num_labels: Number of output labels.
    """

    SUPPORTED_CXR_BACKBONES = ("resnet18", "resnet34", "resnet50")

    def __init__(
        self,
        ehr_input_dim: int,
        ehr_hidden_dim: int = 256,
        ehr_num_layers: int = 2,
        cxr_backbone: str = "resnet34",
        cxr_pretrained: bool = True,
        fusion_hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.5,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        if projection_dim != ehr_hidden_dim:
            raise ValueError(
                "projection_dim must match ehr_hidden_dim for fusion sequencing."
            )
        if cxr_backbone not in self.SUPPORTED_CXR_BACKBONES:
            raise ValueError(
                f"Unsupported cxr_backbone: {cxr_backbone}. "
                f"Supported values: {self.SUPPORTED_CXR_BACKBONES}."
            )

        self.ehr_encoder = nn.LSTM(
            input_size=ehr_input_dim,
            hidden_size=ehr_hidden_dim,
            num_layers=ehr_num_layers,
            batch_first=True,
            dropout=dropout if ehr_num_layers > 1 else 0.0,
        )

        self.cxr_encoder, cxr_feature_dim = self._build_cxr_encoder(
            cxr_backbone=cxr_backbone,
            cxr_pretrained=cxr_pretrained,
        )
        self.projection = nn.Linear(cxr_feature_dim, projection_dim)

        self.fusion_lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=fusion_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_hidden_dim, num_labels)

    def _build_cxr_encoder(
        self,
        cxr_backbone: str,
        cxr_pretrained: bool,
    ) -> Tuple[nn.Module, int]:
        """Builds the CXR backbone and returns encoder + feature dimension."""
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required to use MedFuse CXR backbones."
            ) from exc

        constructor = getattr(tv_models, cxr_backbone)

        if cxr_pretrained:
            try:
                weights = tv_models.get_model_weights(cxr_backbone).DEFAULT
                backbone = constructor(weights=weights)
            except Exception:
                backbone = constructor(pretrained=True)
        else:
            try:
                backbone = constructor(weights=None)
            except TypeError:
                backbone = constructor(pretrained=False)

        if not hasattr(backbone, "fc") or not isinstance(backbone.fc, nn.Linear):
            raise ValueError(
                f"Backbone {cxr_backbone} must expose a linear `fc` layer."
            )

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    def _encode_cxr(self, cxr_input: torch.Tensor) -> torch.Tensor:
        """Encodes CXR images and projects them to the fusion dimension."""
        if cxr_input.dim() != 4:
            raise ValueError(
                "cxr_input must have shape [batch, channels, height, width]."
            )

        if cxr_input.size(1) == 1:
            cxr_input = cxr_input.repeat(1, 3, 1, 1)
        elif cxr_input.size(1) != 3:
            raise ValueError("cxr_input must have 1 or 3 channels.")

        cxr_features = self.cxr_encoder(cxr_input)
        if cxr_features.dim() > 2:
            cxr_features = torch.flatten(cxr_features, start_dim=1)

        projected = self.projection(cxr_features)
        return projected

    def forward(
        self,
        ehr_input: torch.Tensor,
        cxr_input: Optional[torch.Tensor] = None,
        cxr_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs MedFuse fusion.

        Args:
            ehr_input: EHR tensor of shape ``[batch, seq_len, ehr_input_dim]``.
            cxr_input: Optional CXR tensor of shape
                ``[batch, channels, height, width]``.
            cxr_mask: Optional tensor of shape ``[batch]`` indicating CXR
                availability (1 = present, 0 = missing).

        Returns:
            Logits tensor of shape ``[batch, num_labels]``.
        """
        if ehr_input.dim() != 3:
            raise ValueError(
                "ehr_input must have shape [batch, seq_len, ehr_input_dim]."
            )

        _, (ehr_hidden, _) = self.ehr_encoder(ehr_input)
        ehr_repr = self.dropout_layer(ehr_hidden[-1])

        batch_size = ehr_repr.size(0)
        sequence_lengths = torch.ones(
            batch_size,
            dtype=torch.long,
            device=ehr_input.device,
        )
        fusion_tokens = [ehr_repr.unsqueeze(1)]

        if cxr_input is not None:
            if cxr_input.size(0) != batch_size:
                raise ValueError(
                    "cxr_input batch size must match ehr_input batch size."
                )

            if cxr_mask is None:
                cxr_present_mask = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=ehr_input.device,
                )
            else:
                cxr_present_mask = cxr_mask.to(ehr_input.device).view(-1).bool()
                if cxr_present_mask.numel() != batch_size:
                    raise ValueError("cxr_mask must have shape [batch].")

            cxr_projected = torch.zeros(
                batch_size,
                self.projection.out_features,
                dtype=ehr_repr.dtype,
                device=ehr_input.device,
            )

            if cxr_present_mask.any():
                present_cxr = cxr_input[cxr_present_mask]
                present_projected = self._encode_cxr(present_cxr)
                cxr_projected[cxr_present_mask] = present_projected
                sequence_lengths[cxr_present_mask] = 2

            fusion_tokens.append(cxr_projected.unsqueeze(1))

        fusion_sequence = torch.cat(fusion_tokens, dim=1)
        packed_sequence = rnn_utils.pack_padded_sequence(
            fusion_sequence,
            lengths=sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (fusion_hidden, _) = self.fusion_lstm(packed_sequence)
        fused_repr = self.dropout_layer(fusion_hidden[-1])
        logits = self.classifier(fused_repr)
        return logits


class MedFuse(BaseModel):
    """MedFuse model for multi-modal clinical prediction.

    This model fuses clinical time-series (EHR) and chest X-ray (CXR)
    representations using an LSTM-based fusion module. It handles missing CXR
    via variable sequence lengths, where each sample contributes either:

    - ``[v_ehr]`` (EHR only), or
    - ``[v_ehr, v_cxr]`` (EHR + CXR).

    Paper:
        Hayat, N., Geras, K. J., & Shamout, F. E. (2022).
        MedFuse: Multi-modal fusion with clinical time-series data and chest
        X-ray images. MLHC 2022.

    Args:
        dataset: Sample dataset used for model configuration.
        ehr_feature_key: Input key for EHR tensor. Default is ``"ehr"``.
        cxr_feature_key: Input key for CXR tensor. Default is ``"cxr"``.
        cxr_mask_key: Optional input key for per-sample CXR availability mask.
            Default is ``"cxr_mask"``.
        ehr_hidden_dim: Hidden dimension for EHR encoder.
        ehr_num_layers: Number of EHR LSTM layers.
        cxr_backbone: CXR ResNet backbone. Default is ``"resnet34"``.
        cxr_pretrained: Whether to use pretrained CXR backbone.
        fusion_hidden_dim: Hidden dimension for fusion LSTM.
        projection_dim: CXR projection output dimension.
        dropout: Dropout rate.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import MedFuse
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "ehr": torch.randn(5, 10).tolist(),
        ...         "cxr": torch.randn(3, 32, 32).tolist(),
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "ehr": torch.randn(5, 10).tolist(),
        ...         "cxr": torch.randn(3, 32, 32).tolist(),
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ehr": "tensor", "cxr": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> model = MedFuse(dataset=dataset, cxr_pretrained=False)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> output["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        ehr_feature_key: str = "ehr",
        cxr_feature_key: str = "cxr",
        cxr_mask_key: Optional[str] = "cxr_mask",
        ehr_hidden_dim: int = 256,
        ehr_num_layers: int = 2,
        cxr_backbone: str = "resnet34",
        cxr_pretrained: bool = True,
        fusion_hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(dataset=dataset)

        if len(self.label_keys) != 1:
            raise ValueError("MedFuse supports exactly one label key.")

        if ehr_feature_key not in self.feature_keys:
            raise ValueError(
                f"ehr_feature_key '{ehr_feature_key}' not found in dataset "
                f"features: {self.feature_keys}."
            )

        self.ehr_feature_key = ehr_feature_key
        self.cxr_feature_key = cxr_feature_key
        self.cxr_mask_key = cxr_mask_key
        self.label_key = self.label_keys[0]
        self.mode = self._resolve_mode(self.dataset.output_schema[self.label_key])

        self.has_cxr_feature = cxr_feature_key in self.feature_keys
        self.ehr_input_dim = self._infer_ehr_input_dim(ehr_feature_key)

        output_size = self.get_output_size()
        self.layer = MedFuseLayer(
            ehr_input_dim=self.ehr_input_dim,
            ehr_hidden_dim=ehr_hidden_dim,
            ehr_num_layers=ehr_num_layers,
            cxr_backbone=cxr_backbone,
            cxr_pretrained=cxr_pretrained,
            fusion_hidden_dim=fusion_hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            num_labels=output_size,
        )

    def _extract_feature_value(
        self,
        feature_key: str,
        feature: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Extracts the semantic ``value`` tensor from a feature payload."""
        if isinstance(feature, torch.Tensor):
            return feature

        if feature_key not in self.dataset.input_processors:
            raise ValueError(
                f"Feature '{feature_key}' is not defined in dataset processors."
            )

        schema = self.dataset.input_processors[feature_key].schema()
        if "value" in schema:
            value = feature[schema.index("value")]
            if isinstance(value, torch.Tensor):
                return value

        raise ValueError(
            f"Feature '{feature_key}' must provide a tensor value in its schema."
        )

    def _infer_ehr_input_dim(self, ehr_feature_key: str) -> int:
        """Infers EHR input feature dimension from processed dataset samples."""
        for sample in self.dataset:
            if ehr_feature_key not in sample:
                continue
            feature = sample[ehr_feature_key]
            if isinstance(feature, tuple):
                value = self._extract_feature_value(ehr_feature_key, feature)
            elif isinstance(feature, torch.Tensor):
                value = feature
            else:
                value = torch.as_tensor(feature)

            if value.dim() >= 2:
                return int(value.shape[-1])

        raise ValueError(
            "Unable to infer EHR input dimension. Ensure EHR feature tensors "
            "have shape [seq_len, feature_dim]."
        )

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Model inputs. Expected keys include:
                - ``ehr_feature_key`` (required): EHR tensor.
                - ``cxr_feature_key`` (optional): CXR tensor.
                - ``cxr_mask_key`` (optional): CXR availability mask.
                - ``label_key`` (optional): Label tensor for loss computation.

        Returns:
            A dictionary containing:
                - ``logit``: Raw logits.
                - ``y_prob``: Predicted probabilities.
                - ``loss``: Loss tensor when labels are provided.
                - ``y_true``: Ground-truth labels when provided.
        """
        if self.ehr_feature_key not in kwargs:
            raise ValueError(
                f"Missing required EHR feature key: '{self.ehr_feature_key}'."
            )

        ehr_input = self._extract_feature_value(
            self.ehr_feature_key,
            cast(torch.Tensor | tuple[torch.Tensor, ...], kwargs[self.ehr_feature_key]),
        )
        ehr_input = ehr_input.to(self.device).float()

        cxr_input: Optional[torch.Tensor] = None
        if self.cxr_feature_key in kwargs:
            cxr_input = self._extract_feature_value(
                self.cxr_feature_key,
                cast(
                    torch.Tensor | tuple[torch.Tensor, ...],
                    kwargs[self.cxr_feature_key],
                ),
            )
            cxr_input = cxr_input.to(self.device).float()

        cxr_mask: Optional[torch.Tensor] = None
        if self.cxr_mask_key is not None and self.cxr_mask_key in kwargs:
            raw_mask = kwargs[self.cxr_mask_key]
            if isinstance(raw_mask, tuple):
                raise ValueError("cxr_mask must be provided as a tensor.")
            if isinstance(raw_mask, torch.Tensor):
                cxr_mask = raw_mask.to(self.device)
            else:
                cxr_mask = torch.as_tensor(raw_mask, device=self.device)

        logits = self.layer(
            ehr_input=ehr_input,
            cxr_input=cxr_input,
            cxr_mask=cxr_mask,
        )
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results
