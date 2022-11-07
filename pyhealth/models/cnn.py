from typing import List

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

VALID_OPERATION_LEVEL = ["visit", "event"]


class CNNBlock(nn.Module):
    """Convolutional neural network block.

    This block wraps the PyTorch convolutional neural network layer with batch
    normalization and residual connection. It is used in the CNN layer.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Sequential(
            # stride=1 by default
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            # stride=1 by default
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                # stride=1, padding=0 by default
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: input tensor of shape [batch size, in_channels, sequence len].

        Returns:
            output tensor of shape [batch size, out_channels, sequence len].
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
        num_layers: number of cnn blocks. Default is 1.

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

    def forward(self, x: torch.tensor):
        """
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
    """Convolutional neural network model.

    This model applies a separate CNN layer for each feature, and then concatenates
    the final hidden states of each CNN layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        This model can operate on both visit and event level, as designated by
            the operation_level parameter.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        operation_level: one of "visit", "event".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the CNN layer.
    """

    def __init__(
            self,
            dataset: BaseDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            operation_level: str,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            **kwargs,
    ):
        super(CNN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        assert operation_level in VALID_OPERATION_LEVEL, \
            f"operation_level must be one of {VALID_OPERATION_LEVEL}"
        self.operation_level = operation_level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)
        self.cnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.cnn[feature_key] = CNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def visit_level_forward(self, **kwargs):
        """Visit-level CNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == list
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            # (patient, visit, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, hidden_dim)
            _, x = self.cnn[feature_key](x)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def event_level_forward(self, **kwargs):
        """Event-level CNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == str
            x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
            # (patient, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, hidden_dim)
            _, x = self.cnn[feature_key](x)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def forward(self, **kwargs):
        if self.operation_level == "visit":
            return self.visit_level_forward(**kwargs)
        elif self.operation_level == "event":
            return self.event_level_forward(**kwargs)
        else:
            raise NotImplementedError
