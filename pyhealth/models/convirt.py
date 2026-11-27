# Authors: Yuqing Hu (yhu672@gatech.edu), Hanh Trang Do (hdo61@gatech.edu)
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with
#     Electronic Health Records and Chest X-Ray Scans (Elsharief et al., 2025)
#     https://proceedings.mlr.press/v287/elsharief25a.html
# Description: ConVIRT contrastive pretraining of an EHR LSTM and a chest
#     X-ray ResNet, and a linear probe on the frozen encoders.

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.ehr_cxr_early_fusion import build_resnet, value_tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def info_nce_loss(
    z_ehr: torch.Tensor, z_cxr: torch.Tensor, temperature: float = 0.01
) -> torch.Tensor:
    """Bidirectional InfoNCE loss between paired EHR and image embeddings.

    Row ``i`` of ``z_ehr`` and row ``i`` of ``z_cxr`` are a positive pair, and
    every other row of the batch is a negative. The cosine similarities are
    divided by ``temperature``. As in the MedMod code, the denominator of each
    term holds the negatives only: the positive pair is masked out of it.

    Args:
        z_ehr: EHR embeddings, ``[batch, dim]``.
        z_cxr: Image embeddings, ``[batch, dim]``.
        temperature: Softmax temperature.

    Returns:
        The mean loss, a scalar.

    Examples:
        >>> z = torch.eye(3)
        >>> bool(info_nce_loss(z, z, temperature=0.5) < info_nce_loss(z, z.flip(0)))
        True
    """
    if z_ehr.shape != z_cxr.shape or z_ehr.dim() != 2:
        raise ValueError("z_ehr and z_cxr must both have shape [batch, dim].")
    if z_ehr.size(0) < 2:
        raise ValueError("InfoNCE needs at least 2 samples per batch.")
    similarity = (
        F.normalize(z_cxr, dim=-1) @ F.normalize(z_ehr, dim=-1).T
    ) / temperature
    positive = torch.eye(similarity.size(0), dtype=torch.bool, device=z_ehr.device)
    negatives = similarity.masked_fill(positive, -9e15)
    cxr_to_ehr = similarity[positive] - torch.logsumexp(negatives, dim=1)
    ehr_to_cxr = similarity[positive] - torch.logsumexp(negatives, dim=0)
    return -(cxr_to_ehr + ehr_to_cxr).mean()


class _SimCLRAugmentation(nn.Module):
    """SimCLRv2-style augmentation of one image, as in the MedMod code.

    Random affine, random crop of 60 to 100% of the side, resize back, random
    flip, colour jitter (80% of the time) and random grayscale. It expects
    pixel values in ``[0, 1]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.affine = transforms.RandomAffine(
            degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.7, 1.5), shear=(-25, 25)
        )
        self.flip = transforms.RandomHorizontalFlip()
        self.jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.gray = transforms.RandomGrayscale(p=0.2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        size = image.shape[-1]
        image = self.affine(image)
        crop = int(torch.empty(1).uniform_(0.6 * size, size).item())
        image = transforms.RandomCrop(crop)(image)
        image = transforms.functional.resize(image, [size, size], antialias=True)
        image = self.flip(image)
        if torch.rand(1).item() < 0.8:
            image = self.jitter(image)
        return self.gray(image)


class _ConVIRTEncoders(nn.Module):
    """The EHR LSTM and the image ResNet of ConVIRT.

    Returns the encoder outputs, which are the features that the linear probe
    reads, before the projection heads.
    """

    def __init__(
        self,
        ehr_dim: int,
        ehr_hidden_dim: int,
        ehr_num_layers: int,
        cxr_backbone: str | nn.Module,
        cxr_feature_dim: int | None,
        augment_cxr: bool,
        cxr_mean: Sequence[float] | None,
        cxr_std: Sequence[float] | None,
    ) -> None:
        super().__init__()
        self.ehr_encoder = nn.LSTM(
            ehr_dim, ehr_hidden_dim, ehr_num_layers, batch_first=True
        )
        self._init_lstm(self.ehr_encoder)
        if isinstance(cxr_backbone, str):
            self.cxr_encoder, self.cxr_dim = build_resnet(cxr_backbone)
        else:
            self.cxr_encoder, self.cxr_dim = cxr_backbone, cxr_feature_dim
        self.ehr_dim = ehr_hidden_dim
        self.augmentation = _SimCLRAugmentation() if augment_cxr else None
        self.register_buffer(
            "cxr_mean",
            None if cxr_mean is None else torch.tensor(cxr_mean).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "cxr_std",
            None if cxr_std is None else torch.tensor(cxr_std).view(1, -1, 1, 1),
        )

    @staticmethod
    def _init_lstm(lstm: nn.LSTM) -> None:
        """Orthogonal recurrent weights, as in the MedMod EHR encoder."""
        for name, parameter in lstm.named_parameters():
            if name.startswith("weight_hh"):
                nn.init.orthogonal_(parameter)
            elif name.startswith("weight_ih"):
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(
        self,
        ehr: torch.Tensor,
        cxr: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lengths is not None:
            # Zero-padded stays of different lengths: packing makes the LSTM
            # stop at each stay's last bin, as MedMod's EHR encoder does.
            ehr = nn.utils.rnn.pack_padded_sequence(
                ehr, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        if self.cxr_mean is not None:
            # MedMod feeds ConVIRT pixel values in [0, 1], without ImageNet
            # normalization, so undo the normalization of the input.
            cxr = cxr * self.cxr_std + self.cxr_mean
        if self.augmentation is not None and self.training:
            cxr = torch.stack([self.augmentation(image) for image in cxr])
        _, (hidden, _) = self.ehr_encoder(ehr)
        return hidden[-1], self.cxr_encoder(cxr)


def _projection_head(in_dim: int, dim: int) -> nn.Sequential:
    """Linear-ReLU-Linear-ReLU-Linear (no bias), as the MedMod heads."""
    return nn.Sequential(
        nn.Linear(in_dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim, bias=False),
    )


class ConVIRT(BaseModel):
    """ConVIRT contrastive pretraining of an EHR LSTM and a chest X-ray ResNet.

    The EHR time series and the image of the same ICU stay are two views of
    one sample. Each encoder is followed by a projection head, and
    :func:`info_nce_loss` pulls the two embeddings of a stay together and
    pushes those of different stays apart. Both encoders start from random
    weights. Labels are not used.

    The model scores no label, so :class:`~pyhealth.trainer.Trainer` trains it
    on the loss alone and reports the validation loss. Follow it with
    :class:`ConVIRTLinearProbe`.

    The loss compares each pair with the other pairs of its batch, so a batch
    of one sample has no negatives: its loss is zero and gives no gradient.
    It is not a full no-op: in train mode the ResNet's BatchNorm still updates
    its running statistics (and raises an error for very small images), and
    an optimizer with momentum or weight decay still moves the weights. Use
    ``get_dataloader(..., drop_last=True)`` for pretraining, as MedMod does,
    so every batch has the full set of negatives. In validation, a last batch
    of one sample would also lower the mean loss.

    Paper:
        Zhang et al. (2022). Contrastive Learning of Medical Visual
        Representations from Paired Images and Text. MLHC 2022. Setup as in
        MedMod (Elsharief et al., 2025).

    Args:
        dataset: Sample dataset used for model configuration.
        ehr_feature_key: Input key of the EHR tensor ``[batch, time, dim]``.
            Stays of different lengths may come zero-padded with their
            lengths, when the key's processor schema is
            ``("value", "lengths")``; the LSTM then packs them and stops at
            each stay's last step.
        cxr_feature_key: Key of the image input in the batch: an image tensor
            ``[batch, 3, H, W]`` for a ResNet, or ``[batch, cxr_feature_dim]``
            embeddings with ``cxr_backbone=nn.Identity()``. It need not be a
            dataset input: a training loop can load the images (or look up
            the embeddings) of each batch and add them under this key.
        ehr_hidden_dim: Hidden size of the LSTM.
        ehr_num_layers: Number of LSTM layers.
        projection_dim: Size of the embeddings the loss compares.
        temperature: Temperature of the InfoNCE loss.
        cxr_backbone: ``"resnet18"``, ``"resnet34"`` or ``"resnet50"`` for an
            untrained ResNet, or a module that maps the image input to
            ``[batch, cxr_feature_dim]``. Pass ``nn.Identity()`` to feed
            precomputed image embeddings (e.g. frozen MedSigLIP, 1,152 values)
            under ``cxr_feature_key``. A module is used as given, so freeze
            it beforehand if it should not train.
        cxr_feature_dim: Output size of ``cxr_backbone`` when it is a module.
            Leave it unset for a ResNet name, which sets its own size.
        augment_cxr: Whether to augment images in training mode, with the
            SimCLRv2-style transforms of MedMod. They expect pixels in
            ``[0, 1]`` (the colour jitter clamps to that range), so for
            ImageNet-normalized images also pass ``cxr_mean`` and ``cxr_std``.
        cxr_mean: Mean the images were normalized with, to undo before
            encoding, as MedMod feeds ConVIRT pixel values in ``[0, 1]``.
            Default None for images that are not normalized. For
            ImageNet-normalized images pass :data:`IMAGENET_MEAN`.
        cxr_std: Standard deviation the images were normalized with, e.g.
            :data:`IMAGENET_STD`. Default None.

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
        >>> model = ConVIRT(dataset, ehr_hidden_dim=8, projection_dim=8,
        ...                 cxr_backbone="resnet18")
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> model(**batch)["loss"].dim()
        0
    """

    def __init__(
        self,
        dataset: SampleDataset,
        ehr_feature_key: str = "ehr",
        cxr_feature_key: str = "cxr",
        ehr_hidden_dim: int = 128,
        ehr_num_layers: int = 1,
        projection_dim: int = 128,
        temperature: float = 0.01,
        cxr_backbone: str | nn.Module = "resnet34",
        cxr_feature_dim: int | None = None,
        augment_cxr: bool = False,
        cxr_mean: Sequence[float] | None = None,
        cxr_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__(dataset=dataset)
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if isinstance(cxr_backbone, str) and cxr_feature_dim is not None:
            raise ValueError(
                "cxr_feature_dim is only for a module backbone; a ResNet sets "
                "its own output size."
            )
        if not isinstance(cxr_backbone, str):
            if cxr_feature_dim is None:
                raise ValueError("cxr_feature_dim is required for a module.")
            if augment_cxr or cxr_mean is not None:
                raise ValueError("Augmentation and un-normalization need images.")
        if (cxr_mean is None) != (cxr_std is None):
            raise ValueError("Pass both cxr_mean and cxr_std, or neither.")
        if ehr_feature_key not in self.feature_keys:
            raise ValueError(
                f"ehr_feature_key '{ehr_feature_key}' is not an input of the "
                f"dataset: {self.feature_keys}."
            )
        self.ehr_feature_key = ehr_feature_key
        self.cxr_feature_key = cxr_feature_key
        self.temperature = temperature

        self.encoders = _ConVIRTEncoders(
            _infer_ehr_dim(dataset, ehr_feature_key),
            ehr_hidden_dim,
            ehr_num_layers,
            cxr_backbone,
            cxr_feature_dim,
            augment_cxr,
            cxr_mean,
            cxr_std,
        )
        self.ehr_head = _projection_head(ehr_hidden_dim, projection_dim)
        self.cxr_head = _projection_head(self.encoders.cxr_dim, projection_dim)

    @property
    def label_processors(self) -> dict:
        """No label is scored, so the trainer reports the loss only."""
        return {}

    def forward(
        self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Inputs by key: the EHR tensor and the image tensor.

        Returns:
            A dictionary with ``loss`` and the embeddings ``z_ehr`` and
            ``z_cxr``.
        """
        ehr = value_tensor(
            self.dataset, self.ehr_feature_key, kwargs[self.ehr_feature_key]
        )
        cxr = value_tensor(
            self.dataset, self.cxr_feature_key, kwargs[self.cxr_feature_key]
        )
        lengths = _ehr_lengths(
            self.dataset, self.ehr_feature_key, kwargs[self.ehr_feature_key]
        )
        v_ehr, v_cxr = self.encoders(
            ehr.to(self.device).float(), cxr.to(self.device).float(), lengths
        )
        z_ehr, z_cxr = self.ehr_head(v_ehr), self.cxr_head(v_cxr)
        if z_ehr.size(0) < 2:
            # A single pair has no negatives. A zero loss that keeps the graph
            # lets backward() run, so a last batch of one sample is a no-op.
            loss = (z_ehr.sum() + z_cxr.sum()) * 0.0
        else:
            loss = info_nce_loss(z_ehr, z_cxr, self.temperature)
        return {"loss": loss, "z_ehr": z_ehr, "z_cxr": z_cxr}


def _ehr_lengths(
    dataset: SampleDataset, key: str, feature: torch.Tensor | tuple[torch.Tensor, ...]
) -> torch.Tensor | None:
    """The true lengths of zero-padded EHR, or None for fixed-length EHR."""
    if isinstance(feature, torch.Tensor):
        return None
    schema = dataset.input_processors[key].schema()
    return feature[schema.index("lengths")] if "lengths" in schema else None


def _infer_ehr_dim(dataset: SampleDataset, key: str) -> int:
    """Infers the EHR feature size from the first processed sample."""
    for sample in dataset:
        if key in sample:
            return int(value_tensor(dataset, key, sample[key]).shape[-1])
    raise ValueError(f"No sample has the EHR key '{key}'.")


class ConVIRTLinearProbe(BaseModel):
    """Linear probe on the frozen encoders of a pretrained :class:`ConVIRT`.

    The encoders are copied from ``convirt``, frozen and kept in eval mode
    from construction on, and the projection heads are dropped. A single
    linear layer reads the concatenated encoder outputs
    (``ehr_hidden_dim + cxr_dim`` features) and is the only trained part.
    ``cxr_dim`` is 512 for a ResNet-34, giving ``641`` parameters with the
    default sizes, or ``cxr_feature_dim`` for precomputed embeddings (1,152
    for MedSigLIP, giving 1,281 parameters).

    As the encoders are frozen and images are not augmented, the features can
    be computed once with :meth:`extract_features`, and the layer
    :attr:`linear` trained on them without the images.

    Args:
        dataset: Sample dataset with the label to predict.
        convirt: A pretrained :class:`ConVIRT`.

    Examples:
        >>> probe = ConVIRTLinearProbe(dataset, convirt)  # doctest: +SKIP
        >>> probe.linear.in_features + 1  # doctest: +SKIP
        641
    """

    def __init__(self, dataset: SampleDataset, convirt: ConVIRT) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("ConVIRTLinearProbe supports exactly one label key.")
        self.ehr_feature_key = convirt.ehr_feature_key
        self.cxr_feature_key = convirt.cxr_feature_key
        self.label_key = self.label_keys[0]

        self.encoders = copy.deepcopy(convirt.encoders)
        self.encoders.augmentation = None
        for parameter in self.encoders.parameters():
            parameter.requires_grad = False
        self.linear = nn.Linear(
            self.encoders.ehr_dim + self.encoders.cxr_dim,
            self.label_processor.output_size(),
        )
        # The copied encoders keep ConVIRT's train mode; put them in eval mode
        # now, not only on the first train() or eval() call.
        self.train(self.training)

    def train(self, mode: bool = True) -> ConVIRTLinearProbe:
        """Sets the training mode, keeping the frozen encoders in eval mode.

        Args:
            mode: Whether to set training mode.

        Returns:
            The model.

        Examples:
            >>> probe.train().encoders.training  # doctest: +SKIP
            False
        """
        super().train(mode)
        self.encoders.eval()
        return self

    def _features(
        self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        ehr = value_tensor(
            self.dataset, self.ehr_feature_key, kwargs[self.ehr_feature_key]
        )
        cxr = value_tensor(
            self.dataset, self.cxr_feature_key, kwargs[self.cxr_feature_key]
        )
        lengths = _ehr_lengths(
            self.dataset, self.ehr_feature_key, kwargs[self.ehr_feature_key]
        )
        with torch.no_grad():
            v_ehr, v_cxr = self.encoders(
                ehr.to(self.device).float(), cxr.to(self.device).float(), lengths
            )
        return torch.cat([v_ehr, v_cxr], dim=-1)

    def extract_features(
        self, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the encoder features and the labels of a dataloader.

        Args:
            dataloader: Dataloader of the dataset.

        Returns:
            The features ``[samples, ehr_hidden_dim + cxr_dim]`` and the
            labels.

        Examples:
            >>> features, labels = probe.extract_features(loader)  # doctest: +SKIP
        """
        self.eval()
        features, labels = [], []
        for batch in dataloader:
            features.append(self._features(**batch).cpu())
            labels.append(batch[self.label_key].cpu())
        return torch.cat(features), torch.cat(labels)

    def forward(
        self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Inputs by key: the EHR tensor, the image tensor and, to
                get a loss, the label.

        Returns:
            A dictionary with ``logit``, ``y_prob`` and, when the label is
            given, ``loss`` and ``y_true``.
        """
        logit = self.linear(self._features(**kwargs))
        results = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logit, y_true)
            results["y_true"] = y_true
        return results
