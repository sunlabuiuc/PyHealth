# Author: Yongda Fan
# NetID: yongdaf2
# Description: CNN model implementation for PyHealth 2.0

from typing import Dict, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from pyhealth.models.embedding import EmbeddingModel


class CNNBlock(nn.Module):
    """Convolutional neural network block.

    This block wraps the PyTorch convolutional neural network layer with batch
    normalization and residual connection. It is used in the CNN layer.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Sequential(
            # stride=1 by default
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            # stride=1 by default
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                # stride=1, padding=0 by default
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
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
    at the end. It is used in the CNN model. But it can also be used as a
    standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        num_layers: number of convolutional layers. Default is 1.

    Examples:
        >>> from pyhealth.models import CNNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = CNNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super(CNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cnn = nn.ModuleDict()
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            self.cnn[f"CNN-{i}"] = CNNBlock(in_channels, out_channels)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            pooled_outputs: a tensor of shape [batch size, hidden size], containing
                the pooled output features.
        """
        # [batch size, input size, sequence len]
        x = x.permute(0, 2, 1)
        for idx in range(len(self.cnn)):
            x = self.cnn[f"CNN-{idx}"](x)
        outputs = x.permute(0, 2, 1)
        # pooling
        pooled_outputs = self.pooling(x).squeeze(-1)
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
        >>> from pyhealth.datasets import SampleDataset
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
        >>> dataset = SampleDataset(
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

        self.cnn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.cnn[feature_key] = CNNLayer(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    @staticmethod
    def _extract_feature_tensor(feature):
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature[1]
        return feature

    def _reshape_for_convolution(
        self, tensor: torch.Tensor, feature_key: str
    ) -> torch.Tensor:
        if tensor.dim() == 4:
            tensor = tensor.sum(dim=2)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(1).unsqueeze(-1)
        if tensor.dim() != 3:
            raise ValueError(
                f"Unsupported tensor shape for feature {feature_key}: {tensor.shape}"
            )
        return tensor

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

            x = self._reshape_for_convolution(x, feature_key).float()
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
    from pyhealth.datasets import SampleDataset
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
    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = CNN(dataset=dataset)

    data_batch = next(iter(train_loader))

    ret = model(**data_batch)
    print(ret)

    ret["loss"].backward()
