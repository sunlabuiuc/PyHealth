# Author: Yongda Fan
# NetID: yongdaf2
# Description: CNN model implementation for PyHealth 2.0

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from pyhealth.models.embedding import EmbeddingModel
from pyhealth.processors import (
    ImageProcessor,
    MultiHotProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)


class CNNBlock(nn.Module):
    """Convolutional neural network block.

    This block wraps the PyTorch convolutional neural network layer with batch
    normalization and residual connection. It is used in the CNN layer.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int, spatial_dim: int):
        super(CNNBlock, self).__init__()
        if spatial_dim not in (1, 2, 3):
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")

        conv_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dim]
        bn_cls = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[spatial_dim]

        self.conv1 = nn.Sequential(
            # stride=1 by default
            conv_cls(in_channels, out_channels, kernel_size=3, padding=1),
            bn_cls(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            # stride=1 by default
            conv_cls(out_channels, out_channels, kernel_size=3, padding=1),
            bn_cls(out_channels),
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                # stride=1, padding=0 by default
                conv_cls(in_channels, out_channels, kernel_size=1),
                bn_cls(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input tensor of shape [batch size, in_channels, *].

        Returns:
            output tensor of shape [batch size, out_channels, *].
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CNNLayer(nn.Module):
    """Convolutional neural network layer.

    This layer stacks multiple CNN blocks and applies adaptive average pooling
    at the end. It expects the input tensor to be in channels-first format.

    Args:
        input_size: number of input channels.
        hidden_size: hidden feature size.
        num_layers: number of convolutional layers. Default is 1.
        spatial_dim: spatial dimensionality (1, 2, or 3).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        spatial_dim: int = 1,
    ):
        super(CNNLayer, self).__init__()
        if spatial_dim not in (1, 2, 3):
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.spatial_dim = spatial_dim

        self.cnn = nn.ModuleList()
        in_channels = input_size
        for _ in range(num_layers):
            self.cnn.append(CNNBlock(in_channels, hidden_size, spatial_dim))
            in_channels = hidden_size

        pool_cls = {
            1: nn.AdaptiveAvgPool1d,
            2: nn.AdaptiveAvgPool2d,
            3: nn.AdaptiveAvgPool3d,
        }[spatial_dim]
        self.pooling = pool_cls(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, input channels, *spatial_dims].

        Returns:
            outputs: a tensor of shape [batch size, hidden size, *spatial_dims].
            pooled_outputs: a tensor of shape [batch size, hidden size], containing
                the pooled output features.
        """
        for block in self.cnn:
            x = block(x)

        pooled_outputs = self.pooling(x).reshape(x.size(0), -1)

        outputs = x
        return outputs, pooled_outputs


class CNN(BaseModel):
    """Convolutional neural network model for PyHealth 2.0 datasets.

    Each feature is embedded independently, processed by a dedicated
    :class:`CNNLayer`, and the pooled representations are concatenated for the
    final prediction head. The model works with sequence-style processors such as
    ``SequenceProcessor``, ``TimeseriesProcessor``, ``TensorProcessor``, and
    nested categorical inputs produced by ``StageNetProcessor``.

    Args:
        dataset (SampleDataset): Dataset with fitted input and output processors.
        embedding_dim (int): Size of the intermediate embedding space.
        hidden_dim (int): Number of channels produced by each CNN block.
        num_layers (int): Number of convolutional blocks per feature.
        **kwargs: Additional keyword arguments forwarded to :class:`CNNLayer`.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A05B", "A05C", "A06A"],
        ...         "labs": [[1.0, 2.5], [3.0, 4.0]],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v1",
        ...         "conditions": ["A05B"],
        ...         "labs": [[0.5, 1.0]],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "labs": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="toy",
        ... )
        >>> model = CNN(dataset)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        **kwargs,
    ):
        super(CNN, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.feature_conv_dims = {}
        self.cnn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            processor = self.dataset.input_processors.get(feature_key)
            spatial_dim = self._determine_spatial_dim(processor)
            self.feature_conv_dims[feature_key] = spatial_dim
            input_channels = self._determine_input_channels(feature_key, spatial_dim)
            self.cnn[feature_key] = CNNLayer(
                input_size=input_channels,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                spatial_dim=spatial_dim,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    @staticmethod
    def _extract_feature_tensor(feature):
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature[1]
        return feature

    @staticmethod
    def _ensure_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    def _determine_spatial_dim(self, processor) -> int:
        if isinstance(
            processor,
            (
                SequenceProcessor,
                StageNetProcessor,
                StageNetTensorProcessor,
                TensorProcessor,
                TimeseriesProcessor,
                MultiHotProcessor,
            ),
        ):
            return 1
        if isinstance(processor, ImageProcessor):
            return 2
        raise ValueError(
            f"Unsupported processor type for feature convolution: {type(processor).__name__}"
        )

    def _determine_input_channels(self, feature_key: str, spatial_dim: int) -> int:
        if spatial_dim == 1:
            return self.embedding_dim

        min_dim = spatial_dim + 1
        for sample in self.dataset:
            if feature_key not in sample:
                continue
            feature = self._extract_feature_tensor(sample[feature_key])
            if feature is None:
                continue
            tensor = self._ensure_tensor(feature)
            if tensor.dim() < min_dim:
                continue
            return tensor.shape[0]

        raise ValueError(
            f"Unable to infer input channels for feature '{feature_key}' with spatial dimension {spatial_dim}."
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        patient_emb = []

        embed_inputs = {
            feature_key: self._extract_feature_tensor(kwargs[feature_key])
            for feature_key in self.feature_keys
        }
        embedded = self.embedding_model(embed_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            else:
                x = x.to(self.device)

            spatial_dim = self.feature_conv_dims[feature_key]
            expected_dims = {1: 3, 2: 4, 3: 5}[spatial_dim]
            if x.dim() != expected_dims:
                raise ValueError(
                    f"Expected {expected_dims}-D tensor for feature {feature_key}, got shape {x.shape}"
                )
            if spatial_dim == 1:
                x = x.permute(0, 2, 1)

            x = x.float()
            _, pooled = self.cnn[feature_key](x)
            patient_emb.append(pooled)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset
    from pyhealth.datasets import get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["A05B", "A05C", "A06A"],
            "labs": [[1.0, 2.5], [3.0, 4.0]],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "conditions": ["A05B"],
            "labs": [[0.5, 1.0]],
            "label": 0,
        },
    ]

    input_schema = {"conditions": "sequence", "labs": "tensor"}
    output_schema = {"label": "binary"}
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,  # type: ignore[arg-type]
        output_schema=output_schema,  # type: ignore[arg-type]
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = CNN(dataset=dataset)

    data_batch = next(iter(train_loader))

    ret = model(**data_batch)
    print(ret)

    ret["loss"].backward()
