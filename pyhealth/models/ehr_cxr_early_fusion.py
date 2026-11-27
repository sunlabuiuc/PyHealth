# Authors: Yuqing Hu (yhu672@gatech.edu), Hanh Trang Do (hdo61@gatech.edu)
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with
#     Electronic Health Records and Chest X-Ray Scans (Elsharief et al., 2025)
#     https://proceedings.mlr.press/v287/elsharief25a.html
# Description: Early fusion of an EHR LSTM and a chest X-ray ResNet.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

_RESNETS = ("resnet18", "resnet34", "resnet50")


def build_resnet(name: str) -> tuple[nn.Module, int]:
    """Builds an untrained torchvision ResNet without its classifier.

    Args:
        name: One of ``"resnet18"``, ``"resnet34"`` and ``"resnet50"``.

    Returns:
        The backbone, which maps ``[batch, 3, H, W]`` to ``[batch, dim]``, and
        ``dim``.

    Examples:
        >>> backbone, dim = build_resnet("resnet18")
        >>> dim
        512
    """
    if name not in _RESNETS:
        raise ValueError(f"Unsupported backbone '{name}'. Use one of {_RESNETS}.")
    import torchvision.models as tv_models

    backbone = getattr(tv_models, name)(weights=None)
    dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, dim


def value_tensor(
    dataset: SampleDataset, key: str, feature: torch.Tensor | tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Returns the ``value`` tensor of an input feature.

    Args:
        dataset: The dataset that holds the input processors.
        key: The input key.
        feature: A tensor, or a tuple of tensors described by the schema of
            the processor of ``key``.

    Returns:
        The value tensor.

    Examples:
        >>> value_tensor(None, "ehr", torch.zeros(2, 3)).shape
        torch.Size([2, 3])
    """
    if isinstance(feature, torch.Tensor):
        return feature
    schema = dataset.input_processors[key].schema()
    if "value" not in schema:
        raise ValueError(f"Feature '{key}' must provide a 'value' tensor.")
    return feature[schema.index("value")]


class EHRCXREarlyFusion(BaseModel):
    """Early fusion of an EHR LSTM and a chest X-ray (CXR) ResNet.

    The EHR time series goes through an LSTM and the image through a ResNet.
    The image feature is projected to the size of the EHR feature, the two are
    concatenated and a linear layer gives the logit. This is the Early Fusion
    baseline of MedMod (Elsharief et al., 2025), which builds on the original
    MedFuse code (Hayat et al., 2022, https://github.com/nyuad-cai/MedFuse),
    not on PyHealth's :class:`~pyhealth.models.MedFuse`. With
    ``freeze_encoders=True`` the two encoders are not trained: only the
    projection and the classifier are. With ``freeze_encoders=False`` all
    parts are trained.

    Frozen encoders are in eval mode from construction on and stay in it when
    the model is set to train mode, so the ResNet's BatchNorm statistics do
    not change and the LSTM applies no dropout. The original MedFuse code
    instead sets the whole model to train mode, frozen encoders included, so
    results can differ slightly from it.

    The model expects samples that have an image. With
    :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4` use ``paired_only=True``,
    or select the samples with ``cxr_mask == 1``.

    Pretrained weights are passed as state dicts, so that this class stays
    independent of how a checkpoint file is laid out. They are loaded strictly
    into :attr:`ehr_encoder` (an ``nn.LSTM``) and :attr:`cxr_encoder` (a
    ResNet without its ``fc`` layer).

    Args:
        dataset: Sample dataset used for model configuration.
        ehr_feature_key: Input key of the EHR tensor ``[batch, time, dim]``.
            All samples must have the same number of time steps, e.g. 24 for
            :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4` with its default
            2-hour bins over 48 hours, or 48 with 1-hour bins (``bin_hours=1``,
            MedMod's setting). The LSTM reads the last step, so zero padding
            is not skipped.
        cxr_feature_key: Key of the image tensor ``[batch, 3, H, W]`` in the
            batch. It need not be a dataset input: a training loop can add the
            images of each batch under this key. With ``None`` the model is
            EHR-only: with ``freeze_encoders=False``, only the LSTM and the
            classifier are trained, e.g. to pretrain the LSTM on EHR alone.
        ehr_hidden_dim: Hidden size of the LSTM.
        ehr_num_layers: Number of LSTM layers.
        dropout: Dropout between the LSTM layers.
        cxr_backbone: ``"resnet18"``, ``"resnet34"`` or ``"resnet50"`` for an
            untrained torchvision ResNet, or a module that maps images to
            ``[batch, cxr_feature_dim]`` (for example a frozen MedSigLIP).
        cxr_feature_dim: Output size of ``cxr_backbone`` when it is a module.
            Leave it unset for a ResNet name, which sets its own size.
        freeze_encoders: Whether to freeze the LSTM and the image encoder.
            Frozen encoders also stay in eval mode when the model is set to
            train mode, so BatchNorm statistics do not change.
        ehr_state_dict: Optional weights of :attr:`ehr_encoder`.
        cxr_state_dict: Optional weights of :attr:`cxr_encoder`.
        augment_cxr: Whether to augment images in training mode: horizontal
            flip and a random affine transform, as in MedFuse. Only for a
            ResNet backbone, which reads images.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "visit_id": f"v{i}",
        ...         "ehr": torch.randn(4, 6).tolist(),
        ...         "cxr": torch.randn(3, 32, 32).tolist(),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ehr": "tensor", "cxr": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = EHRCXREarlyFusion(
        ...     dataset, ehr_hidden_dim=8, ehr_num_layers=1, cxr_backbone="resnet18"
        ... )
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        ehr_feature_key: str = "ehr",
        cxr_feature_key: str | None = "cxr",
        ehr_hidden_dim: int = 256,
        ehr_num_layers: int = 2,
        dropout: float = 0.3,
        cxr_backbone: str | nn.Module = "resnet34",
        cxr_feature_dim: int | None = None,
        freeze_encoders: bool = True,
        ehr_state_dict: Mapping[str, Any] | None = None,
        cxr_state_dict: Mapping[str, Any] | None = None,
        augment_cxr: bool = False,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("EHRCXREarlyFusion supports exactly one label key.")
        if ehr_feature_key not in self.feature_keys:
            raise ValueError(
                f"ehr_feature_key '{ehr_feature_key}' is not an input of the "
                f"dataset: {self.feature_keys}."
            )
        if cxr_feature_key is None and freeze_encoders:
            raise ValueError(
                "An EHR-only model has nothing else to train: set "
                "freeze_encoders=False."
            )
        self.ehr_feature_key = ehr_feature_key
        self.cxr_feature_key = cxr_feature_key
        self.label_key = self.label_keys[0]
        self.freeze_encoders = freeze_encoders
        self.augment_cxr = augment_cxr

        self.ehr_encoder = nn.LSTM(
            input_size=self._infer_ehr_dim(),
            hidden_size=ehr_hidden_dim,
            num_layers=ehr_num_layers,
            batch_first=True,
            dropout=dropout if ehr_num_layers > 1 else 0.0,
        )
        if ehr_state_dict is not None:
            self.ehr_encoder.load_state_dict(ehr_state_dict)

        self.cxr_encoder: nn.Module | None = None
        self.projection: nn.Linear | None = None
        classifier_in = ehr_hidden_dim
        if cxr_feature_key is not None:
            if isinstance(cxr_backbone, str):
                if cxr_feature_dim is not None:
                    raise ValueError(
                        "cxr_feature_dim is only for a module backbone; a "
                        "ResNet sets its own output size."
                    )
                self.cxr_encoder, cxr_dim = build_resnet(cxr_backbone)
            else:
                if cxr_feature_dim is None:
                    raise ValueError("cxr_feature_dim is required for a module.")
                if augment_cxr:
                    raise ValueError("Augmentation needs images (a ResNet backbone).")
                self.cxr_encoder, cxr_dim = cxr_backbone, cxr_feature_dim
            if cxr_state_dict is not None:
                self.cxr_encoder.load_state_dict(cxr_state_dict)
            self.projection = nn.Linear(cxr_dim, ehr_hidden_dim)
            classifier_in = 2 * ehr_hidden_dim
        elif cxr_state_dict is not None:
            raise ValueError("cxr_state_dict needs a cxr_feature_key.")

        if freeze_encoders:
            for module in (self.ehr_encoder, self.cxr_encoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        self.classifier = nn.Linear(classifier_in, self.label_processor.output_size())
        self.augmentation = self._build_augmentation() if augment_cxr else None
        # A new module is in train mode; this puts frozen encoders in eval mode
        # now, not only on the first train() or eval() call.
        self.train(self.training)

    @staticmethod
    def _build_augmentation() -> nn.Module:
        """Flip and affine transform of the MedFuse training images."""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=45, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=0
                ),
            ]
        )

    def _infer_ehr_dim(self) -> int:
        """Infers the EHR feature size from the first processed sample."""
        for sample in self.dataset:
            if self.ehr_feature_key in sample:
                value = value_tensor(
                    self.dataset, self.ehr_feature_key, sample[self.ehr_feature_key]
                )
                return int(value.shape[-1])
        raise ValueError(f"No sample has the EHR key '{self.ehr_feature_key}'.")

    def train(self, mode: bool = True) -> EHRCXREarlyFusion:
        """Sets the training mode, keeping frozen encoders in eval mode.

        Args:
            mode: Whether to set training mode.

        Returns:
            The model.

        Examples:
            >>> model.train().training  # doctest: +SKIP
            True
        """
        super().train(mode)
        if self.freeze_encoders:
            self.ehr_encoder.eval()
            if self.cxr_encoder is not None:
                self.cxr_encoder.eval()
        return self

    def _augment(self, cxr: torch.Tensor) -> torch.Tensor:
        """Augments each image on its own, as one transform per call."""
        return torch.stack([self.augmentation(image) for image in cxr])

    def forward(
        self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Inputs by key: the EHR tensor, the image tensor (unless
                EHR-only) and, to get a loss, the label.

        Returns:
            A dictionary with ``logit``, ``y_prob`` and, when the label is
            given, ``loss`` and ``y_true``.
        """
        train_encoders = torch.is_grad_enabled() and not self.freeze_encoders

        ehr = value_tensor(
            self.dataset, self.ehr_feature_key, kwargs[self.ehr_feature_key]
        )
        with torch.set_grad_enabled(train_encoders):
            _, (hidden, _) = self.ehr_encoder(ehr.to(self.device).float())
        features = hidden[-1]

        if self.cxr_encoder is not None:
            cxr = (
                value_tensor(
                    self.dataset, self.cxr_feature_key, kwargs[self.cxr_feature_key]
                )
                .to(self.device)
                .float()
            )
            if self.augmentation is not None and self.training:
                cxr = self._augment(cxr)
            with torch.set_grad_enabled(train_encoders):
                cxr_features = self.cxr_encoder(cxr)
            features = torch.cat([features, self.projection(cxr_features)], dim=-1)

        logit = self.classifier(features)
        results = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logit, y_true)
            results["y_true"] = y_true
        return results
