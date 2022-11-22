from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

# VALID_OPERATION_LEVEL = ["visit", "event"]


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
            dropout=dropout,
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
        We use separate RNN layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the RNN model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply RNN on the code level
            - case 2. [1.5, 2.0, 8, 1.2, 4.5, 2.1]
                - we use a two-layer MLP
            - case 3. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                RNN one the braket level
            - case 4. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run RNN directly
                on the inner bracket level
            - case 5. (developing) high-dimensional tensor
                - we will flatten the tensor into case 3 or case 4 and run RNN

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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        # pick the first sample to initialize the linear transformation float/int features
        sample = self.dataset.samples[0]
        self.linear = nn.ModuleDict()
        for feature_key in feature_keys:
            if feature_key not in self.feat_tokenizers:
                input_dim = (
                    len(sample[feature_key])
                    if type(sample[feature_key][0]) != list
                    else len(sample[feature_key][0])
                )

                self.linear[feature_key] = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )

        self.rnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

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
        patient_emb = []
        for feature_key in self.feature_keys:
            if (type(kwargs[feature_key][0][0]) == list) and (
                type(kwargs[feature_key][0][0][0]) != list
            ):

                # for case 3: [[code1, code2], [code3, ...], ...]
                if type(kwargs[feature_key][0][0][0]) == str:
                    x = self.feat_tokenizers[feature_key].batch_encode_3d(
                        kwargs[feature_key]
                    )
                    # (patient, visit, code)
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    # (patient, visit, code, embedding_dim)
                    x = self.embeddings[feature_key](x)
                    # (patient, visit, embedding_dim)
                    x = torch.sum(x, dim=2)
                    # (patient, visit)
                    mask = torch.sum(x, dim=2) != 0

                # for case 4: [[1.5, 2.0, 0.0], ...]
                else:
                    x, mask = self.padding3d(kwargs[feature_key])
                    # (patient, visit, values)
                    x = torch.tensor(x, dtype=torch.float, device=self.device)
                    # (patient, visit, embedding_dim)
                    x = self.linear[feature_key](x)
                    # (patient, visit)
                    mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

                # (patient, embedding_dim)
                _, x = self.rnn[feature_key](x, mask)
                patient_emb.append(x)

            elif (type(kwargs[feature_key][0]) == list) and (
                type(kwargs[feature_key][0][0]) != list
            ):

                # for case 1: [code1, code2, code3, ...]
                if type(kwargs[feature_key][0][0]) == str:
                    x = self.feat_tokenizers[feature_key].batch_encode_2d(
                        kwargs[feature_key]
                    )
                    # (patient, code)
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    # (patient, code, embedding_dim)
                    x = self.embeddings[feature_key](x)
                    # (patient, code)
                    mask = torch.sum(x, dim=2) != 0
                    # (patient, embedding_dim)
                    _, x = self.rnn[feature_key](x, mask)

                # for case 2: [1.5, 2.0, 0.0, ...]
                else:
                    # (patient, values)
                    x = torch.tensor(
                        kwargs[feature_key], dtype=torch.float, device=self.device
                    )
                    # (patient, embedding_dim)
                    x = self.linear[feature_key](x)
                patient_emb.append(x)

            else:
                raise NotImplementedError

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


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": [["cond-33", "cond-86", "cond-80"]],
            "procedures": [1.0, 2.0, 3.5, 4],
            "label": 0,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": [["cond-33", "cond-86", "cond-80"]],
            "procedures": [5.0, 2.0, 3.5, 4],
            "label": 1,
        },
    ]

    # dataset
    dataset = SampleDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = RNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="label",
        mode="binary",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
