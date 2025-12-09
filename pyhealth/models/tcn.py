from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


# From TCN original paper https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNLayer(nn.Module):
    """Temporal Convolutional Networks layer.

    Shaojie Bai et al. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.

    This layer wraps the PyTorch TCN layer with masking and dropout support. It is
    used in the TCN model. But it can also be used as a standalone layer.

    Args:
        input_dim: input feature size.
        num_channels: int or list of ints. If int, the depth will be automatically decided by the max_seq_length. If list, number of channels in each layer.
        max_seq_length: max sequence length. Used to compute the depth of the TCN.
        kernel_size: kernel size of the TCN.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            TCN blocks. Default is 0.5.

    Examples:
        >>> from pyhealth.models import TCNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = TCNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: Union[int, List[int]] = 128,
        max_seq_length: int = 20,
        kernel_size: int = 2,
        dropout: float = 0.5,
    ):
        super(TCNLayer, self).__init__()

        layers = []

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(num_channels, int):
            num_channels = [num_channels] * int(
                np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size))
            )

        # Store the actual output dimension (last layer's output size)
        self.num_channels = num_channels[-1]

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
        """
        batch_size = x.size(0)
        # TCN expects (batch, channels, seq_len) so we permute
        outputs = self.network(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Extract last valid output using mask (similar to RNN)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64, device=x.device
            )
        else:
            # Ensure mask is on the same device as x to avoid device mismatch
            mask = mask.to(x.device)
            lengths = torch.sum(mask.int(), dim=-1)

        # Clamp lengths to at least 1 to handle empty sequences
        lengths = torch.clamp(lengths, min=1)
        last_outputs = outputs[torch.arange(batch_size, device=x.device), (lengths - 1), :]
        return outputs, last_outputs


class TCN(BaseModel):
    """Temporal Convolutional Networks model.

    This model applies a separate TCN layer for each feature, and then concatenates
    the final hidden states of each TCN layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        We use separate TCN layers for different feature_keys.
        Currently, we support two types of input formats:
            - Sequence of codes (e.g., diagnosis codes, procedure codes)
                - Input format: (batch_size, sequence_length)
                - Each code is embedded into a vector and TCN is applied on the sequence
            - Timeseries values (e.g., lab tests, vital signs)
                - Input format: (batch_size, sequence_length, num_features)
                - Each timestep contains a fixed number of measurements
                - TCN is applied directly on the timeseries data

    Args:
        dataset (SampleDataset): the dataset to train the model. It is used to query certain
            information such as the set of all tokens. The dataset's input_schema and
            output_schema define the feature_keys, label_key, and mode.
        embedding_dim (int): the embedding dimension. Default is 128.
        num_channels (Union[int, List[int]]): the number of channels in the TCN layer.
            If int, depth is auto-computed from max_seq_length. If list, specifies
            channels for each layer. Default is 128.
        **kwargs: other parameters for the TCN layer (e.g., max_seq_length, kernel_size, dropout).
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_channels: Union[int, List[int]] = 128,
        **kwargs
    ):
        super(TCN, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        # validate kwargs for TCN layer
        if "input_dim" in kwargs:
            raise ValueError("input_dim is determined by embedding_dim")
        assert len(self.label_keys) == 1, "Only one label key is supported if TCN is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.tcn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.tcn[feature_key] = TCNLayer(
                input_dim=embedding_dim, num_channels=num_channels, **kwargs
            )

        # Get the actual output dimension from TCNLayer instances
        # All TCNLayers have the same output dimension
        self.num_channels = next(iter(self.tcn.values())).num_channels

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.num_channels, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient embeddings if requested.
        """
        patient_emb = []
        embedded = self.embedding_model(kwargs)
        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            mask = (x.sum(dim=-1) != 0).int()
            _, x = self.tcn[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
