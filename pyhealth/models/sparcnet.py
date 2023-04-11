import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import BaseSignalDataset
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
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: (not used now) the embedding dimension. Default is 128.
        hidden_dim: (not used now) the hidden dimension. Default is 128.
        block_layer: the number of layers in each dense block. Default is 4.
        growth_rate: the growth rate of each dense layer. Default is 16.
        bn_size: the bottleneck size of each dense layer. Default is 16.
        conv_bias: whether to use bias in convolutional layers. Default is True.
        batch_norm: whether to use batch normalization. Default is True.
        **kwargs: other parameters for the Deepr layer.

    Examples:
        >>> from pyhealth.datasets import SampleSignalDataset
        >>> samples = [
        ...         {
        ...             "record_id": "SC4001-0",
        ...             "patient_id": "SC4001",
        ...             "epoch_path": "/home/chaoqiy2/.cache/pyhealth/datasets/2f06a9232e54254cbcb4b62624294d71/SC4001-0.pkl",
        ...             "label": "W",
        ...         },
        ...         {
        ...             "record_id": "SC4001-1",
        ...             "patient_id": "SC4001",
        ...             "epoch_path": "/home/chaoqiy2/.cache/pyhealth/datasets/2f06a9232e54254cbcb4b62624294d71/SC4001-1.pkl",
        ...             "label": "R",
        ...         }
        ...     ]
        >>> dataset = SampleSignalDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import SparcNet
        >>> model = SparcNet(
        ...         dataset=dataset,
        ...         feature_keys=["signal"], # dataloader will load the signal from "epoch_path" and put it in "signal"
        ...         label_key="label",
        ...         mode="multiclass",
        ...     )
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
        dataset: BaseSignalDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
        **kwargs,
    ):

        super(SparcNet, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )

        """ common """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # TODO: Use more tokens for <gap> for different lengths once the input has such information
        self.label_tokenizer = self.get_label_tokenizer()

        """ input statistics """
        print(f"\n=== Input data statistics ===")
        # obtain input signal size
        signal_info = self.dataset.input_info["signal"]
        in_channels, length = signal_info["n_channels"], signal_info["length"]
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
        for n_layer in np.arange(math.floor(np.log2(length // 4))):
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
        output_size = self.get_output_size(self.label_tokenizer)
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

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = torch.tensor(
            np.array(kwargs[self.feature_keys[0]]), device=self.device
        ).float()

        # final layer embedding (batch, embedding)
        emb = self.encoder(x).view(x.shape[0], -1)
        # (patient, label_size)
        logits = self.fc(emb)
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
            results["embed"] = emb
        return results


if __name__ == "__main__":
    """
    For dense layer
    """
    # x = torch.randn(128, 5, 1000)
    # batch, channels, length = x.shape
    # model = DenseLayer(channels, 5, 2)
    # y = model(x)
    # print(y.shape)

    """
    For dense block
    """
    # x = torch.randn(128, 5, 1000)
    # batch, channels, length = x.shape
    # model = DenseBlock(3, channels, 5, 2)
    # y = model(x)
    # print(y.shape)

    """
    For transition layer
    """
    # x = torch.randn(128, 5, 1000)
    # batch, channels, length = x.shape
    # model = TransitionLayer(channels, 18)
    # y = model(x)
    # print(y.shape)

    """
    For sparcenet
    """
    from pyhealth.datasets import SampleSignalDataset, get_dataloader

    samples = [
        {
            "record_id": "SC4001-0",
            "patient_id": "SC4001",
            "epoch_path": "/home/chaoqiy2/.cache/pyhealth/datasets/2f06a9232e54254cbcb4b62624294d71/SC4001-0.pkl",
            "label": "W",
        },
        {
            "record_id": "SC4001-0",
            "patient_id": "SC4001",
            "epoch_path": "/home/chaoqiy2/.cache/pyhealth/datasets/2f06a9232e54254cbcb4b62624294d71/SC4001-1.pkl",
            "label": "R",
        },
    ]

    # dataset
    dataset = SampleSignalDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = SparcNet(
        dataset=dataset,
        feature_keys=["signal"],
        label_key="label",
        mode="multiclass",
    ).to("cuda:0")

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
