from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
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
        We use separate rnn layers for different feature_keys.
        Currently, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the rnn model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply rnn on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                rnn one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run rnn directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run rnn directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the RNN layer.

    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import RNN
        >>> model = RNN(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_codes",
        ...             "list_vectors",
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         mode="binary",
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.8056, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.5906],
                            [0.6620]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                            [0.]]),
            'logit': tensor([[0.3666],
                            [0.6721]], grad_fn=<AddmmBackward0>)
        }
        >>>


    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        pretrained_emb: str = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(RNN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature RNN layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "RNN only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "RNN only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "RNN only supports 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

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
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x !=0, dim=2)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x !=0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError

            # transform x to (patient, event, embedding_dim)
            if self.pretrained_emb != None:
                x = self.linear_layers[feature_key](x)

            _, x = self.rnn[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            # "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            # "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
        },
    ]

    # dataset
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = RNN(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
            "list_list_vectors",
        ],
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
