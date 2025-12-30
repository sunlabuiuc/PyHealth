from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import (
    SequenceProcessor,
    NestedSequenceProcessor,
    NestedFloatsProcessor,
    DeepNestedFloatsProcessor,
)
from .base_model import BaseModel
from .embedding import EmbeddingModel
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
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the AdaCare layer and BaseModel (e.g., feature_keys, label_keys).


    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "vector": [[0.1], [0.2], [0.3]],
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
        ...             "list_vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-1",
        ...             "vector": [[0.7], [0.8], [0.9]],
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_vectors": [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = create_sample_dataset(samples=samples,
        ...                         input_schema={
        ...                             'vector': 'nested_sequence_floats',
        ...                             'list_codes': 'sequence',
        ...                             'list_list_codes': 'nested_sequence',
        ...                             'list_vectors': 'nested_sequence_floats',
        ...                             'list_list_vectors': 'deep_nested_sequence_floats'
        ...                             },
        ...                         output_schema={'label': 'binary'},
        ...                         dataset_name='test'
        ...                         )
        >>>
        >>> from pyhealth.models import AdaCare
        >>> model = AdaCare(
        ...         dataset=dataset,
        ...         hidden_dim=64,
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
            'logit': tensor([[ 0.0036], [-0.0886]], grad_fn=<AddmmBackward0>),
            'feature_importance': [...],
            'conv_feature_importance': [...]
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            **kwargs,
        )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        if "input_dim" in kwargs:
            raise ValueError("input_dim is automatically determined")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        
        # Use EmbeddingModel for unified embedding handling
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        # AdaCare layers for each feature
        self.adacare = nn.ModuleDict()

        for idx, feature_key in enumerate(self.feature_keys):
            input_proc = self.dataset.input_processors[feature_key]
            # sanity check
            if not isinstance(
                input_proc,
                (
                    SequenceProcessor,
                    NestedSequenceProcessor,
                    NestedFloatsProcessor,
                    DeepNestedFloatsProcessor,
                ),
            ):
                raise ValueError(
                    """AdaCare only supports SequenceProcessor, NestedSequenceProcessor,
                    NestedFloatsProcessor, DeepNestedFloatsProcessor."""
                )

            self.adacare[feature_key] = AdaCareLayer(
                input_dim=embedding_dim, hidden_dim=self.hidden_dim, **kwargs
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_keys[0]]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                feature_importance: a list of tensors with shape (batch_size, time_step, input_dim)
                                    representing the feature importance for each feature type.
                conv_feature_importance: a list of tensors with shape (batch_size, time_step, 3*kernel_num)
                                        representing the convolutional feature importance for each feature type.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
                logit: a tensor representing the logits.
                embed: (optional) a tensor representing the patient embeddings if 'embed' is True in kwargs.
        """
        patient_emb = []
        embedded, masks = self.embedding_model(kwargs, output_mask=True)
        feature_importance = []
        conv_feature_importance = []
        
        for _, feature_key in enumerate(self.feature_keys):
            embeds = embedded[feature_key]
            mask = masks[feature_key]
            processor = self.dataset.input_processors[feature_key]
            
            if embeds.dim() == 3:
                if isinstance(processor, NestedFloatsProcessor):
                    mask = torch.any(mask, dim=2)
                elif isinstance(processor, SequenceProcessor):
                    pass
                else:
                    raise ValueError(f"Expected NestedFloatsProcessor or SequenceProcessor for 3D input, got {type(processor)}")
            elif embeds.dim() == 4:
                if isinstance(processor, NestedSequenceProcessor):
                    embeds = torch.sum(embeds, dim=2)
                    mask = torch.any(mask, dim=2)
                elif isinstance(processor, DeepNestedFloatsProcessor):
                    embeds = torch.sum(embeds, dim=2)
                    mask = torch.any(mask, dim=(2, 3))
                else:
                    raise ValueError(f"Expected NestedSequenceProcessor or DeepNestedFloatsProcessor for 4D input, got {type(processor)}")
            else:
                raise NotImplementedError(f"Unsupported input dimension {feature_key}: {embeds.dim()} for AdaCare")
            
            embeds, _, inputatt, convatt = self.adacare[feature_key](embeds, mask)
            feature_importance.append(inputatt)
            conv_feature_importance.append(convatt)
            patient_emb.append(embeds)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_keys[0]].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
            "feature_importance": feature_importance,
            "conv_feature_importance": conv_feature_importance,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results