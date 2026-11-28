# Authors: Christopher Perez (cjperez3), Kai Cheung (kaicc2), Sarah Lee (slee537)
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
#   Health Records and Chest X-Ray Scans (Elsharief et al., CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/elsharief25a.html
# Description: DAFT fusion of clinical time series and chest X-rays for PyHealth.

"""DAFT: Dynamic Affine Feature Map Transform for EHR + chest X-ray fusion.

Paper:
    Polsterl, S., Wolf, T. N., & Wachinger, C. (2021). Combining 3D image and
    tabular data via the dynamic affine feature map transform. MICCAI 2021.

As a baseline in:
    Elsharief, S., et al. (2025). MedMod: Multimodal benchmark for medical
    prediction tasks with electronic health records and chest X-ray scans.
    CHIL 2025. https://proceedings.mlr.press/v287/elsharief25a.html
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class DAFTBlock(nn.Module):
    """One dynamic affine feature map transform.

    Predicts a per-channel scale and shift for ``target`` from a conditioning
    vector, and applies them. The auxiliary network sees the conditioning
    vector together with the target's own global average, as in the DAFT
    paper: ``Linear(cond_dim + channels, bottleneck) -> ReLU ->
    Linear(bottleneck, 2 * channels)``, both without bias.

    ``target`` is either a sequence ``[batch, time, channels]`` (averaged over
    time) or a feature map ``[batch, channels, height, width]`` (averaged over
    space); the same scale and shift is applied at every position.

    Args:
        channels: Number of channels of the target.
        cond_dim: Size of the conditioning vector.
        bottleneck_dim: Hidden size of the auxiliary network.
        scale: Whether to predict a scale. If False, the scale is 1.
        shift: Whether to predict a shift. If False, the shift is 0.
        activation: Activation applied to the scale: ``"linear"`` (none),
            ``"sigmoid"`` or ``"tanh"``.

    Examples:
        >>> block = DAFTBlock(channels=8, cond_dim=4, bottleneck_dim=3)
        >>> block(torch.randn(2, 5, 8), torch.randn(2, 4)).shape
        torch.Size([2, 5, 8])
    """

    ACTIVATIONS = ("linear", "sigmoid", "tanh")

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        bottleneck_dim: int,
        scale: bool = True,
        shift: bool = True,
        activation: str = "linear",
    ) -> None:
        super().__init__()
        if not (scale or shift):
            raise ValueError("At least one of scale and shift must be True.")
        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {self.ACTIVATIONS}, got '{activation}'."
            )
        if bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be at least 1.")

        self.channels = channels
        self.use_scale = scale
        self.use_shift = shift
        self.activation = activation
        n_out = channels * (int(scale) + int(shift))
        self.aux = nn.Sequential(
            OrderedDict(
                [
                    (
                        "aux_base",
                        nn.Linear(cond_dim + channels, bottleneck_dim, bias=False),
                    ),
                    ("aux_relu", nn.ReLU()),
                    ("aux_out", nn.Linear(bottleneck_dim, n_out, bias=False)),
                ]
            )
        )

    def film_params(
        self, target: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the scale and shift, each of shape ``[batch, channels]``."""
        if target.dim() == 3:
            pooled = target.mean(dim=1)
        elif target.dim() == 4:
            pooled = target.mean(dim=(2, 3))
        else:
            raise ValueError(
                "target must be [batch, time, channels] or "
                "[batch, channels, height, width]."
            )
        params = self.aux(torch.cat([pooled, cond], dim=1))

        if self.use_scale and self.use_shift:
            v_scale, v_shift = params.split(self.channels, dim=1)
        elif self.use_scale:
            v_scale, v_shift = params, torch.zeros_like(params)
        else:
            v_scale, v_shift = torch.ones_like(params), params

        if self.use_scale and self.activation == "sigmoid":
            v_scale = torch.sigmoid(v_scale)
        elif self.use_scale and self.activation == "tanh":
            v_scale = torch.tanh(v_scale)
        return v_scale, v_shift

    def forward(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Applies the transform.

        Args:
            target: ``[batch, time, channels]`` or
                ``[batch, channels, height, width]``.
            cond: Conditioning vector, ``[batch, cond_dim]``.

        Returns:
            A tensor of the same shape as ``target``.
        """
        v_scale, v_shift = self.film_params(target, cond)
        if target.dim() == 3:
            return v_scale.unsqueeze(1) * target + v_shift.unsqueeze(1)
        return v_scale[:, :, None, None] * target + v_shift[:, :, None, None]


class DAFTLayer(nn.Module):
    """DAFT fusion of an EHR LSTM and a ResNet chest X-ray encoder.

    Two directions are supported:

    - ``"cxr_to_ehr"`` (MedMod's implementation, the default): the ResNet
      feature map after ``layer_after`` conditions the hidden sequence of the
      first EHR LSTM layer; a second LSTM layer reads the transformed sequence
      and its last hidden state is classified. The ResNet only runs up to
      ``layer_after``.
    - ``"ehr_to_cxr"`` (the DAFT paper's direction): the last hidden state of
      the EHR LSTM conditions the ResNet feature map after ``layer_after``;
      the pooled image features are concatenated with the EHR state and
      classified.

    Samples without an image (``cxr_mask == 0``) never go through the ResNet.
    With ``missing_cxr="skip"`` they keep the untransformed EHR sequence
    (``cxr_to_ehr``) or zero image features (``ehr_to_cxr``). With
    ``missing_cxr="learned"`` a learned vector stands in for the image
    summary.

    Args:
        ehr_input_dim: Features per EHR time step.
        num_labels: Output size.
        direction: ``"cxr_to_ehr"`` or ``"ehr_to_cxr"``.
        ehr_hidden_dim: Hidden size of the EHR LSTM.
        cxr_backbone: ``"resnet18"``, ``"resnet34"`` or ``"resnet50"``.
        cxr_pretrained: Whether to load ImageNet weights.
        layer_after: Where DAFT is applied: 0 after the ResNet stem, 1-4
            after ``layer1``-``layer4``.
        bottleneck_dim: Hidden size of the DAFT auxiliary network. None uses
            MedMod's ``(16 + cxr_channels) // 7`` for ``cxr_to_ehr`` and the
            DAFT paper's ``(ehr_hidden_dim + cxr_channels) // 7`` for
            ``ehr_to_cxr``.
        scale: Whether DAFT predicts a scale.
        shift: Whether DAFT predicts a shift.
        activation: Activation of the scale (``"linear"``, ``"sigmoid"``,
            ``"tanh"``).
        missing_cxr: ``"skip"`` or ``"learned"``.
        dropout: Dropout before the classifier.

    Examples:
        >>> layer = DAFTLayer(
        ...     ehr_input_dim=34, cxr_backbone="resnet18", cxr_pretrained=False
        ... )
        >>> ehr = torch.randn(2, 24, 34)
        >>> cxr = torch.randn(2, 3, 64, 64)
        >>> layer(ehr, cxr, cxr_mask=torch.tensor([1.0, 0.0])).shape
        torch.Size([2, 1])
    """

    DIRECTIONS = ("cxr_to_ehr", "ehr_to_cxr")
    BACKBONES = ("resnet18", "resnet34", "resnet50")
    MISSING = ("skip", "learned")

    def __init__(
        self,
        ehr_input_dim: int,
        num_labels: int = 1,
        direction: str = "cxr_to_ehr",
        ehr_hidden_dim: int = 256,
        cxr_backbone: str = "resnet34",
        cxr_pretrained: bool = True,
        layer_after: int = 4,
        bottleneck_dim: int | None = None,
        scale: bool = True,
        shift: bool = True,
        activation: str = "linear",
        missing_cxr: str = "skip",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if direction not in self.DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self.DIRECTIONS}, got '{direction}'."
            )
        if cxr_backbone not in self.BACKBONES:
            raise ValueError(
                f"cxr_backbone must be one of {self.BACKBONES}, got '{cxr_backbone}'."
            )
        if layer_after not in range(5):
            raise ValueError(f"layer_after must be 0-4, got {layer_after}.")
        if missing_cxr not in self.MISSING:
            raise ValueError(
                f"missing_cxr must be one of {self.MISSING}, got '{missing_cxr}'."
            )

        self.direction = direction
        self.layer_after = layer_after
        self.missing_cxr = missing_cxr

        self.cxr_stages, cxr_channels = self._build_cxr_stages(
            cxr_backbone, cxr_pretrained, direction, layer_after
        )
        self.cxr_channels = cxr_channels

        if direction == "cxr_to_ehr":
            self.ehr_lstm_1 = nn.LSTM(ehr_input_dim, ehr_hidden_dim, batch_first=True)
            self.ehr_lstm_2 = nn.LSTM(ehr_hidden_dim, ehr_hidden_dim, batch_first=True)
            if bottleneck_dim is None:
                bottleneck_dim = max(1, (16 + cxr_channels) // 7)
            self.daft = DAFTBlock(
                channels=ehr_hidden_dim,
                cond_dim=cxr_channels,
                bottleneck_dim=bottleneck_dim,
                scale=scale,
                shift=shift,
                activation=activation,
            )
            missing_dim = cxr_channels
            head_dim = ehr_hidden_dim
        else:
            self.ehr_lstm = nn.LSTM(
                ehr_input_dim, ehr_hidden_dim, num_layers=2, batch_first=True
            )
            if bottleneck_dim is None:
                bottleneck_dim = max(1, (ehr_hidden_dim + cxr_channels) // 7)
            self.daft = DAFTBlock(
                channels=cxr_channels,
                cond_dim=ehr_hidden_dim,
                bottleneck_dim=bottleneck_dim,
                scale=scale,
                shift=shift,
                activation=activation,
            )
            self.final_channels = self._stage_channels(self.cxr_stages[-1])
            missing_dim = self.final_channels
            head_dim = ehr_hidden_dim + self.final_channels

        self.missing_embedding: nn.Parameter | None = None
        if missing_cxr == "learned":
            self.missing_embedding = nn.Parameter(torch.zeros(missing_dim))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_dim, num_labels)

    @staticmethod
    def _stage_channels(stage: nn.Sequential) -> int:
        """Output channels of the ResNet stem or of a residual stage."""
        last = stage[-1]
        if isinstance(last, nn.MaxPool2d):  # stem: conv1, bn1, relu, maxpool
            return int(stage[0].out_channels)
        norm = getattr(last, "bn3", None)  # Bottleneck (resnet50)
        if norm is None:
            norm = last.bn2  # BasicBlock (resnet18/34)
        return int(norm.num_features)

    @classmethod
    def _build_cxr_stages(
        cls,
        backbone_name: str,
        pretrained: bool,
        direction: str,
        layer_after: int,
    ) -> tuple[nn.ModuleList, int]:
        """Splits a torchvision ResNet into the stem and its four stages.

        Returns the stages that are needed and the channel count after
        ``layer_after``. ``cxr_to_ehr`` only needs the stages up to
        ``layer_after``; ``ehr_to_cxr`` needs all of them.
        """
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError("torchvision is required for DAFT.") from exc

        weights = None
        if pretrained:
            weights = tv_models.get_model_weights(backbone_name).DEFAULT
        resnet = getattr(tv_models, backbone_name)(weights=weights)
        stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        stages = [stem, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
        channels = cls._stage_channels(stages[layer_after])
        if direction == "cxr_to_ehr":
            stages = stages[: layer_after + 1]
        return nn.ModuleList(stages), channels

    def _check_inputs(
        self,
        ehr_input: torch.Tensor,
        cxr_input: torch.Tensor | None,
        cxr_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Validates shapes and returns a boolean ``[batch]`` presence mask."""
        if ehr_input.dim() != 3:
            raise ValueError("ehr_input must have shape [batch, time, features].")
        batch_size = ehr_input.size(0)
        if cxr_input is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=ehr_input.device)
        if cxr_input.dim() != 4 or cxr_input.size(0) != batch_size:
            raise ValueError(
                "cxr_input must have shape [batch, channels, height, width] with "
                "the same batch size as ehr_input."
            )
        if cxr_input.size(1) not in (1, 3):
            raise ValueError("cxr_input must have 1 or 3 channels.")
        if cxr_mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=ehr_input.device)
        present = cxr_mask.to(ehr_input.device).reshape(-1).bool()
        if present.numel() != batch_size:
            raise ValueError("cxr_mask must have shape [batch].")
        return present

    @contextmanager
    def _single_image_batch_norm(self, num_images: int) -> Iterator[None]:
        """Uses the running BatchNorm statistics for a lone image in training.

        Only the samples with an image go through the ResNet, so a training
        batch can hold a single image. BatchNorm cannot compute batch
        statistics from one value per channel (a 1x1 feature map on small
        images), so the ResNet's BatchNorm layers use their running
        statistics for that batch.
        """
        norms = [
            module
            for module in self.cxr_stages.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
        ]
        if num_images > 1:
            norms = []
        for module in norms:
            module.eval()
        try:
            yield
        finally:
            for module in norms:
                module.train()

    def _run_stages(self, x: torch.Tensor, stages: nn.ModuleList) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        for stage in stages:
            x = stage(x)
        return x

    def forward(
        self,
        ehr_input: torch.Tensor,
        cxr_input: torch.Tensor | None = None,
        cxr_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs DAFT fusion.

        Args:
            ehr_input: ``[batch, time, ehr_input_dim]``.
            cxr_input: Optional ``[batch, 1 or 3, height, width]``.
            cxr_mask: Optional ``[batch]``, 1 where the image is present.
                Without a mask every given image counts as present.

        Returns:
            Logits of shape ``[batch, num_labels]``.
        """
        present = self._check_inputs(ehr_input, cxr_input, cxr_mask)
        if self.direction == "cxr_to_ehr":
            return self._forward_cxr_to_ehr(ehr_input, cxr_input, present)
        return self._forward_ehr_to_cxr(ehr_input, cxr_input, present)

    def _forward_cxr_to_ehr(
        self,
        ehr_input: torch.Tensor,
        cxr_input: torch.Tensor | None,
        present: torch.Tensor,
    ) -> torch.Tensor:
        hidden_seq, _ = self.ehr_lstm_1(ehr_input)
        transformed = hidden_seq

        cond = None
        if self.missing_embedding is not None:
            cond = self.missing_embedding.expand(ehr_input.size(0), -1).clone()
        if present.any():
            with self._single_image_batch_norm(int(present.sum())):
                feature_map = self._run_stages(cxr_input[present], self.cxr_stages)
            pooled = feature_map.mean(dim=(2, 3))
            if cond is None:
                cond = pooled.new_zeros(ehr_input.size(0), pooled.size(1))
            cond[present] = pooled

        if cond is not None:
            filmed = self.daft(hidden_seq, cond)
            if self.missing_embedding is None:
                keep = present.view(-1, 1, 1)
                transformed = torch.where(keep, filmed, hidden_seq)
            else:
                transformed = filmed

        _, (h_n, _) = self.ehr_lstm_2(transformed)
        return self.classifier(self.dropout(h_n[-1]))

    def _forward_ehr_to_cxr(
        self,
        ehr_input: torch.Tensor,
        cxr_input: torch.Tensor | None,
        present: torch.Tensor,
    ) -> torch.Tensor:
        _, (h_n, _) = self.ehr_lstm(ehr_input)
        ehr_state = h_n[-1]

        if self.missing_embedding is not None:
            image_repr = self.missing_embedding.expand(ehr_input.size(0), -1).clone()
        else:
            image_repr = ehr_state.new_zeros(ehr_input.size(0), self.final_channels)

        if present.any():
            x = cxr_input[present]
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            with self._single_image_batch_norm(x.size(0)):
                for index, stage in enumerate(self.cxr_stages):
                    x = stage(x)
                    if index == self.layer_after:
                        x = self.daft(x, ehr_state[present])
            image_repr[present] = x.mean(dim=(2, 3))

        fused = torch.cat([ehr_state, image_repr], dim=1)
        return self.classifier(self.dropout(fused))


class DAFT(BaseModel):
    """DAFT model for EHR time series + chest X-ray prediction.

    Wraps :class:`DAFTLayer` for PyHealth datasets. It reads the ``ehr``,
    ``cxr`` and ``cxr_mask`` inputs produced by
    :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4` and
    :class:`~pyhealth.tasks.EHRCXRPhenotypingMIMIC4`, and works with binary,
    multiclass and multilabel outputs.

    The default configuration follows the DAFT baseline of MedMod
    (Elsharief et al., CHIL 2025): the chest X-ray conditions the EHR LSTM
    after the last ResNet-34 stage, with a linear scale.

    Args:
        dataset: Sample dataset used to configure the model.
        ehr_feature_key: Key of the EHR tensor ``[time, features]``.
        cxr_feature_key: Key of the image tensor ``[channels, height,
            width]``. If the dataset has no such key, the model runs on EHR
            alone.
        cxr_mask_key: Key of the per-sample image availability (1 present,
            0 missing). If the batch has no such key, every image counts as
            present.
        **kwargs: Options of :class:`DAFTLayer` (``direction``,
            ``ehr_hidden_dim``, ``cxr_backbone``, ``cxr_pretrained``,
            ``layer_after``, ``bottleneck_dim``, ``scale``, ``shift``,
            ``activation``, ``missing_cxr``, ``dropout``).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "visit_id": f"v{i}",
        ...         "ehr": torch.randn(6, 4).tolist(),
        ...         "cxr": torch.randn(3, 64, 64).tolist(),
        ...         "cxr_mask": float(i % 2),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(2)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ehr": "tensor", "cxr": "tensor", "cxr_mask": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = DAFT(dataset, cxr_backbone="resnet18", cxr_pretrained=False)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        ehr_feature_key: str = "ehr",
        cxr_feature_key: str = "cxr",
        cxr_mask_key: str | None = "cxr_mask",
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("DAFT supports exactly one label key.")
        if ehr_feature_key not in self.feature_keys:
            raise ValueError(
                f"ehr_feature_key '{ehr_feature_key}' not found in dataset "
                f"features: {self.feature_keys}."
            )

        self.ehr_feature_key = ehr_feature_key
        self.cxr_feature_key = cxr_feature_key
        self.cxr_mask_key = cxr_mask_key
        self.label_key = self.label_keys[0]
        self.ehr_input_dim = self._infer_ehr_input_dim()
        self.layer = DAFTLayer(
            ehr_input_dim=self.ehr_input_dim,
            num_labels=self.label_processor.output_size(),
            **kwargs,
        )

    def _value(self, key: str, feature) -> torch.Tensor:
        """Returns the value tensor of a feature, whatever its processor."""
        if isinstance(feature, torch.Tensor):
            return feature
        if isinstance(feature, tuple):
            schema = self.dataset.input_processors[key].schema()
            if "value" in schema:
                return feature[schema.index("value")]
        return torch.as_tensor(feature)

    def _infer_ehr_input_dim(self) -> int:
        for sample in self.dataset:
            if self.ehr_feature_key in sample:
                value = self._value(self.ehr_feature_key, sample[self.ehr_feature_key])
                if value.dim() >= 2:
                    return int(value.shape[-1])
        raise ValueError(
            "Unable to infer the EHR input size; EHR tensors must be [time, features]."
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`,
                with the EHR key, optionally the image and mask keys, and
                optionally the label key.

        Returns:
            A dict with ``logit`` and ``y_prob``, plus ``loss`` and
            ``y_true`` when the batch has labels.
        """
        if self.ehr_feature_key not in kwargs:
            raise ValueError(f"Missing EHR feature '{self.ehr_feature_key}'.")
        ehr = self._value(self.ehr_feature_key, kwargs[self.ehr_feature_key])
        ehr = ehr.to(self.device).float()

        cxr = None
        if self.cxr_feature_key in kwargs:
            cxr = self._value(self.cxr_feature_key, kwargs[self.cxr_feature_key])
            cxr = cxr.to(self.device).float()

        mask = None
        if self.cxr_mask_key is not None and self.cxr_mask_key in kwargs:
            mask = self._value(self.cxr_mask_key, kwargs[self.cxr_mask_key])
            mask = mask.to(self.device)

        logits = self.layer(ehr, cxr, mask)
        results: dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
        }
        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        return results
