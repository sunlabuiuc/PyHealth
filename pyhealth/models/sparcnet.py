#Author: Joshua Chen
#Paper: Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation
#Paper Link: https://pubmed.ncbi.nlm.nih.gov/36878708/ 
#Description: SparcNet implementation for Pyhealth 2.0
import math
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class DenseLayer(nn.Sequential):
    """Densely connected layer
    Args:
        input_channels: number of input channels
        growth_rate: rate of growth of channels in this layer
        bn_size: multiplicative factor for the bottleneck layer (does not affect the output size)
        drop_rate: dropout rate
        conv_bias: whether to use bias in convolutional layers
        batch_norm: whether to use batch normalization

    Example:
        >>> x = torch.randn(128, 5, 1000)
        >>> batch, channels, length = x.shape
        >>> model = DenseLayer(channels, 5, 2)
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 10, 1000])
    """

    def __init__(
        self,
        input_channels,
        growth_rate,
        bn_size,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseLayer, self).__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(input_channels)),
        self.add_module("elu1", nn.ELU()),
        self.add_module(
            "conv1",
            nn.Conv1d(
                input_channels,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        ),
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module("elu2", nn.ELU()),
        self.add_module(
            "conv2",
            nn.Conv1d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    """Densely connected block
    Args:
        num_layers: number of layers in this block
        input_channls: number of input channels
        growth_rate: rate of growth of channels in this layer
        bn_size: multiplicative factor for the bottleneck layer (does not affect the output size)
        drop_rate: dropout rate
        conv_bias: whether to use bias in convolutional layers
        batch_norm: whether to use batch normalization

    Example:
        >>> x = torch.randn(128, 5, 1000)
        >>> batch, channels, length = x.shape
        >>> model = DenseBlock(3, channels, 5, 2)
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 20, 1000])
    """

    def __init__(
        self,
        num_layers,
        input_channels,
        growth_rate,
        bn_size,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseBlock, self).__init__()
        for idx_layer in range(num_layers):
            layer = DenseLayer(
                input_channels + idx_layer * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                conv_bias,
                batch_norm,
            )
            self.add_module("denselayer%d" % (idx_layer + 1), layer)


class TransitionLayer(nn.Sequential):
    """pooling transition layer

    Args:
        input_channls: number of input channels
        output_channels: number of output channels
        conv_bias: whether to use bias in convolutional layers
        batch_norm: whether to use batch normalization

    Example:
        >>> x = torch.randn(128, 5, 1000)
        >>> model = TransitionLayer(5, 18)
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 18, 500])

    """

    def __init__(
        self, input_channels, output_channels, conv_bias=True, batch_norm=True
    ):
        super(TransitionLayer, self).__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(input_channels))
        self.add_module("elu", nn.ELU())
        self.add_module(
            "conv",
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class SparcNet(BaseModel):
    """The SparcNet model for sleep staging.

    Paper: Jin Jing, et al. Development of Expert-level Classification of Seizures and Rhythmic and
    Periodic Patterns During EEG Interpretation. Neurology 2023.

    Note:
        We use one encoder to handle multiple channel together.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: (not used now) the embedding dimension. Default is 128.
        hidden_dim: (not used now) the hidden dimension. Default is 128.
        block_layer: the number of layers in each dense block. Default is 4.
        growth_rate: the growth rate of each dense layer. Default is 16.
        bn_size: the bottleneck size of each dense layer. Default is 16.
        conv_bias: whether to use bias in convolutional layers. Default is True.
        batch_norm: whether to use batch normalization. Default is True.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...         {
        ...             "patient_id": "p0",
        ...             "visit_id": "v0",
        ...             "signal": np.random.randn(2, 256).astype(np.float32),
        ...             "label": 0,
        ...         },
        ...         {
        ...             "patient_id": "p1",
        ...             "visit_id": "v0",
        ...             "signal": np.random.randn(2, 256).astype(np.float32),
        ...             "label": 1,
        ...         }
        ...     ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="test",
        ... )
        >>>
        >>> from pyhealth.models import SparcNet
        >>> model = SparcNet(dataset=dataset)
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6530, device='cuda:0', grad_fn=<NllLossBackward0>),
            'y_prob': tensor([[0.4459, 0.5541],
                            [0.5111, 0.4889]], device='cuda:0', grad_fn=<SoftmaxBackward0>),
            'y_true': tensor([1, 1], device='cuda:0'),
            'logit': tensor([[-0.2750, -0.0577],
                            [-0.1319, -0.1763]], device='cuda:0', grad_fn=<AddmmBackward0>)
        }

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(SparcNet, self).__init__(dataset=dataset)

        """ common """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if SparcNet is initialized"
        )
        assert len(self.feature_keys) == 1, (
            "Only one feature key is supported if SparcNet is initialized"
        )

        """ input statistics """
        print(f"\n=== Input data statistics ===")
        in_channels, length = self._determine_input_channels_length()
        # input signal size (batch, n_channels, length)
        print(f"n_channels: {in_channels}")
        print(f"length: {length}")

        """ define sparcnet """
        # add initial convolutional layer
        out_channels = 2 ** (math.floor(np.log2(in_channels)) + 1)
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        first_conv["norm0"] = nn.BatchNorm1d(out_channels)
        first_conv["elu0"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.encoder = nn.Sequential(first_conv)

        n_channels = out_channels

        # add dense blocks
        for n_layer in range(int(math.floor(np.log2(length // 4)))):
            block = DenseBlock(
                num_layers=block_layers,
                input_channels=n_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("denseblock%d" % (n_layer + 1), block)
            # update number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            trans = TransitionLayer(
                input_channels=n_channels,
                output_channels=n_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("transition%d" % (n_layer + 1), trans)
            # update number of channels after each transition layer
            n_channels = n_channels // 2

        """ prediction layer """
        output_size = self.get_output_size()
        self.fc = nn.Linear(n_channels, output_size)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _determine_input_channels_length(self):
        for sample in self.dataset:
            if self.feature_keys[0] not in sample:
                continue

            if len(sample[self.feature_keys[0]].shape) == 1:
                return 1, sample[self.feature_keys[0]].shape[0]
            if len(sample[self.feature_keys[0]].shape) == 2:
                return sample[self.feature_keys[0]].shape[0], sample[
                    self.feature_keys[0]
                ].shape[1]

            raise ValueError(
                f"Invalid shape for feature key {self.feature_keys[0]}: {sample[self.feature_keys[0]].shape}"
            )

        raise ValueError(
            f"Unable to infer input channels and length from dataset for feature key {self.feature_keys[0]}"
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]].to(self.device)

        # final layer embedding (batch, embedding)
        emb = self.encoder(x).view(x.shape[0], -1)
        # (patient, label_size)
        logits = self.fc(emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_keys[0]].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results
