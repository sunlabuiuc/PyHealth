from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

VALID_OPERATION_LEVEL = ["visit", "event"]


class RNNLayer(nn.Module):
    """Recurrent neural network layer.

    This layer wraps the PyTorch RNN layer with masking and dropout support. It is
    used in the RNN model. But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        rnn_type: type of rnn, one of "RNN", "LSTM", "GRU". Default is "GRU".
        num_layers: number of recurrent layers. Default is 1.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            RNN layer. Default is 0.5.
        bidirectional: whether to use bidirectional recurrent layers. If True,
            a fully-connected layer is applied to the concatenation of the forward
            and backward hidden states to reduce the dimension to hidden_size.
            Default is False.

    Examples:
        >>> from pyhealth.models import RNNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = RNNLayer(5, 64)
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
            rnn_type: str = "GRU",
            num_layers: int = 1,
            dropout: float = 0.5,
            bidirectional: bool = False,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
            self,
            x: torch.tensor,
            mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
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
        # pytorch's rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            return outputs, last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return outputs, last_outputs


class RNN(BaseModel):
    """Recurrent neural network model.

    This model applies a separate RNN layer for each feature, and then concatenates
    the final hidden states of each RNN layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        This model can operate on both visit and event level, as designated by
            the `operation_level` parameter.

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
        **kwargs: other parameters for the RNN layer.
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
            **kwargs
    ):
        super(RNN, self).__init__(
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

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        self.rnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def visit_level_forward(self, **kwargs):
        """Visit-level RNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            # [patient, visit, code]
            assert type(kwargs[feature_key][0][0]) == list
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            # (patient, visit, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.sum(x, dim=2) != 0
            # (patient, hidden_dim)
            _, x = self.rnn[feature_key](x, mask)
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
        """Event-level RNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            # [patient, code]
            assert type(kwargs[feature_key][0][0]) == str
            x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
            # (patient, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, code)
            mask = torch.sum(x, dim=2) != 0
            # (patient, hidden_dim)
            _, x = self.rnn[feature_key](x, mask)
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
