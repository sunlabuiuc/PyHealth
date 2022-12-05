from typing import List, Tuple, Dict

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

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
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
    """Convolutional neural network model.

    This model applies a separate CNN layer for each feature, and then concatenates
    the final hidden states of each CNN layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        We use separate CNN layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the CNN model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply CNN on the code level
            - case 2. [1.5, 2.0, 8, 1.2, 4.5, 2.1]
                - we use a two-layer MLP
            - case 3. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                CNN one the braket level
            - case 4. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run CNN directly
                on the inner bracket level
            - case 5. (developing) high-dimensional tensor
                - we will flatten the tensor into case 3 or case 4 and run CNN

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
        assert (
            operation_level in VALID_OPERATION_LEVEL
        ), f"operation_level must be one of {VALID_OPERATION_LEVEL}"
        self.operation_level = operation_level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # validate kwargs for CNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
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

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        If `operation_level` is "visit", then the input is a list of visits
        for each patient. Each visit is a list of codes. For example,
        `kwargs["conditions"]` is a list of visits for each patient. Each
        visit is a list of condition codes.

        If `operation_level` is "event", then the input is a list of events
        for each patient. Each event is a code. For example, `kwargs["conditions"]`
        is a list of condition codes for each patient.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        if self.operation_level == "visit":
            return self.visit_level_forward(**kwargs)
        elif self.operation_level == "event":
            return self.event_level_forward(**kwargs)
        else:
            raise NotImplementedError
