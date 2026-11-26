"""Plain 1D ResNet-18 classifier for multilabel ECG tasks."""

import torch
from torch import nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel
from .protoecgnet import ResNet1dBackbone


class PTBXLResNet18(BaseModel):
    """A plain ResNet-18 ECG classifier with a multilabel prediction head.

    The model uses PyHealth's existing 1D ResNet backbone, global average
    pooling over time, and a linear output layer trained with binary
    cross-entropy.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "signal": torch.zeros(12, 256).tolist(),
        ...      "labels": ["NORM"]},
        ...     {"patient_id": "p1", "signal": torch.ones(12, 256).tolist(),
        ...      "labels": ["MI"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"labels": "multilabel"},
        ... )
        >>> model = PTBXLResNet18(
        ...     dataset, base_width=4, layers=(1, 1, 1, 1)
        ... )
        >>> result = model(
        ...     signal=torch.randn(2, 12, 256),
        ...     labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ... )
        >>> result["logit"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 12,
        base_width: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__(dataset)

        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError(
                "PTBXLResNet18 expects one signal input and one label."
            )
        if self.mode != "multilabel":
            raise ValueError("PTBXLResNet18 requires multilabel output.")

        self.signal_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.in_channels = in_channels

        self.backbone = ResNet1dBackbone(
            in_channels=in_channels,
            base_width=base_width,
            layers=layers,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.label_processor.output_size()),
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Return multilabel logits, probabilities, and loss when labels exist.

        Args:
            **kwargs: Batch containing the ECG tensor under the configured
                signal key, with shape ``(batch, leads, time)``. May also
                contain the multi-hot labels.

        Returns:
            A dictionary containing ``logit`` and ``y_prob``; when labels are
            provided, it also contains ``loss`` and ``y_true``.
        """
        if self.signal_key not in kwargs:
            raise KeyError(f"Missing ECG input: {self.signal_key}")

        signal = kwargs[self.signal_key].to(self.device).float()
        if signal.ndim != 3 or signal.shape[1] != self.in_channels:
            raise ValueError(
                "ECG input must have shape "
                f"(batch, {self.in_channels}, time); got {tuple(signal.shape)}."
            )

        features = self.backbone(signal)
        embedding = self.pool(features).squeeze(-1)
        logits = self.head(embedding)
        result = {"logit": logits, "y_prob": torch.sigmoid(logits)}

        labels = kwargs.get(self.label_key)
        if labels is not None:
            labels = labels.to(self.device).float()
            if labels.shape != logits.shape:
                raise ValueError(
                    f"Labels must have shape {tuple(logits.shape)}; "
                    f"got {tuple(labels.shape)}."
                )
            result["loss"] = F.binary_cross_entropy_with_logits(logits, labels)
            result["y_true"] = labels

        return result
