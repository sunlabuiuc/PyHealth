from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(
            start=1, end=number_of_logits + 1, dtype=torch.float32
        ).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class Recalibration(nn.Module):
    def __init__(
        self, channel, reduction=9, use_h=True, use_c=True, activation="sigmoid"
    ):
        super(Recalibration, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.use_h = use_h
        self.use_c = use_c
        scale_dim = 0
        self.activation = activation

        self.nn_c = nn.Linear(channel, channel // reduction)
        scale_dim += channel // reduction

        self.nn_rescale = nn.Linear(scale_dim, channel)
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, t = x.size()

        y_origin = x.permute(0, 2, 1).reshape(b * t, c).contiguous()
        se_c = self.nn_c(y_origin)
        se_c = torch.relu(se_c)
        y = se_c

        y = self.nn_rescale(y).view(b, t, c).permute(0, 2, 1).contiguous()
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation == "sparsemax":
            y = self.sparsemax(y)
        else:
            y = self.softmax(y)
        return x * y.expand_as(x), y.permute(0, 2, 1)


class AdaCareLayer(nn.Module):
    """AdaCare layer.

    Paper: Liantao Ma et al. Adacare: Explainable clinical health status representation learning
    via scale-adaptive feature extraction and recalibration. AAAI 2020.

    This layer is used in the AdaCare model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: the input feature size.
        hidden_dim: the hidden dimension of the GRU layer. Default is 128.
        kernel_size: the kernel size of the causal convolution layer. Default is 2.
        kernel_num: the kernel number of the causal convolution layer. Default is 64.
        r_v: the number of the reduction rate for the original feature calibration. Default is 4.
        r_c: the number of the reduction rate for the convolutional feature recalibration. Default is 4.
        activation: the activation function for the recalibration layer (sigmoid, sparsemax, softmax). Default is "sigmoid".
        dropout: dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import AdaCareLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = AdaCareLayer(64)
        >>> c, _, inputatt, convatt = layer(input)
        >>> c.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 2,
        kernel_num: int = 64,
        r_v: int = 4,
        r_c: int = 4,
        activation: str = "sigmoid",
        rnn_type: str = "gru",
        dropout: float = 0.5,
    ):
        super(AdaCareLayer, self).__init__()

        if activation not in ["sigmoid", "softmax", "sparsemax"]:
            raise ValueError(
                "Only sigmoid, softmax and sparsemax are supported for activation."
            )
        if rnn_type not in ["gru", "lstm"]:
            raise ValueError("Only gru and lstm are supported for rnn_type.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.r_v = r_v
        self.r_c = r_c
        self.dropout = dropout

        self.nn_conv1 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 1)
        self.nn_conv3 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 3)
        self.nn_conv5 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 5)
        torch.nn.init.xavier_uniform_(self.nn_conv1.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv3.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv5.weight)

        self.nn_convse = Recalibration(
            3 * kernel_num, r_c, use_h=False, use_c=True, activation="sigmoid"
        )
        self.nn_inputse = Recalibration(
            input_dim, r_v, use_h=False, use_c=True, activation=activation
        )

        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim + 3 * kernel_num, hidden_dim)
        else:
            self.rnn = nn.LSTM(input_dim + 3 * kernel_num, hidden_dim)
        # self.nn_output = nn.Linear(hidden_dim, output_dim)
        self.nn_dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            last_output: a tensor of shape [batch size, input_dim] representing the
                patient embedding.
            output: a tensor of shape [batch size, sequence_len, input_dim] representing the patient embedding at each time step.
            inputatt: a tensor of shape [batch size, sequence_len, input_dim] representing the feature importance of the input.
            convatt: a tensor of shape [batch size, sequence_len, 3 * kernel_num] representing the feature importance of the convolutional features.
        """
        # batch_size = x.size(0)
        # time_step = x.size(1)
        # feature_dim = x.size(2)

        conv_input = x.permute(0, 2, 1)
        conv_res1 = self.nn_conv1(conv_input)
        conv_res3 = self.nn_conv3(conv_input)
        conv_res5 = self.nn_conv5(conv_input)

        conv_res = torch.cat((conv_res1, conv_res3, conv_res5), dim=1)
        conv_res = self.relu(conv_res)

        convse_res, convatt = self.nn_convse(conv_res)
        inputse_res, inputatt = self.nn_inputse(x.permute(0, 2, 1))
        concat_input = torch.cat((convse_res, inputse_res), dim=1).permute(0, 2, 1)
        output, _ = self.rnn(concat_input)
        last_output = get_last_visit(output, mask)
        if self.dropout > 0.0:
            last_output = self.nn_dropout(last_output)
        return last_output, output, inputatt, convatt


class AdaCare(BaseModel):
    """AdaCare model.

    Paper: Liantao Ma et al. Adacare: Explainable clinical health status representation learning
    via scale-adaptive feature extraction and recalibration. AAAI 2020.

    Note:
        We use separate AdaCare layers for different feature_keys.
        Currently, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        Since the AdaCare model calibrate the original features to provide interpretability, we do not recommend use embeddings for the input features.
        We follow the current convention for the AdaCare model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply AdaCare on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                AdaCare one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run AdaCare directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run AdaCare directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        use_embedding: list of bools indicating whether to use embedding for each feature type,
            e.g. [True, False].
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the AdaCare layer.


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
        >>> from pyhealth.models import AdaCare
        >>> model = AdaCare(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_codes",
        ...             "list_vectors",
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         use_embedding=[True, False, True, False],
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
            'loss': tensor(0.7167, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.5009], [0.4779]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[0.], [1.]]),
            'logit': tensor([[ 0.0036], [-0.0886]], grad_fn=<AddmmBackward0>)
        }
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        use_embedding: List[bool],
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(AdaCare, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.use_embedding = use_embedding
        self.hidden_dim = hidden_dim

        if "input_dim" in kwargs:
            raise ValueError("input_dim is automatically determined")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()
        self.adacare = nn.ModuleDict()

        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "AdaCare only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "AdaCare only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] == str) and (use_embedding[idx] == False):
                raise ValueError(
                    "AdaCare only supports embedding for str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "AdaCare only supports 2-dim or 3-dim float and int as input types"
                )

            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            if use_embedding[idx]:
                self.add_feature_transform_layer(feature_key, input_info)
                self.adacare[feature_key] = AdaCareLayer(
                    input_dim=embedding_dim, hidden_dim=self.hidden_dim, **kwargs
                )
            else:
                self.adacare[feature_key] = AdaCareLayer(
                    input_dim=input_info["len"], hidden_dim=self.hidden_dim, **kwargs
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
                feature_importance: a list of tensors with shape (feature_type, batch_size, time_step, features)
                                    representing the feature importance.
                conv_feature_importance: a list of tensors with shape (feature_type, batch_size, time_step, 3*kernal_size)
                                        representing the convolutional feature importance.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        feature_importance = []
        conv_feature_importance = []
        for idx, feature_key in enumerate(self.feature_keys):
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
                if self.use_embedding[idx]:
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

                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError

            x, _, inputatt, convatt = self.adacare[feature_key](x, mask)
            feature_importance.append(inputatt)
            conv_feature_importance.append(convatt)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
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
    model = AdaCare(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
            "list_list_vectors",
        ],
        label_key="label",
        mode="binary",
        use_embedding=[True, False, True, False],
        hidden_dim=64,
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
