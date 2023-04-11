import functools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import BaseSignalDataset
from pyhealth.models import BaseModel


class ResBlock2D(nn.Module):
    """Convolutional Residual Block 2D

    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the convolutional layers.
        downsample: whether to use a downsampling residual connection.
        pooling: whether to use max pooling.

    Example:
        >>> import torch
        >>> from pyhealth.models import ResBlock2D
        >>>
        >>> model = ResBlock2D(6, 16, 1, True, True)
        >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
        >>> output = model(input_)
        >>> output.shape
        torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        downsample: bool = True,
        pooling: bool = True,
    ):
        super(ResBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward propagation.

        Args:
            x: input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            out: output tensor of shape (batch_size, out_channels, *, *).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
            out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class ContraWR(BaseModel):
    """The encoder model of ContraWR (a supervised model, STFT + 2D CNN layers)

    Paper: Yang, Chaoqi, Danica Xiao, M. Brandon Westover, and Jimeng Sun.
    "Self-supervised eeg representation learning for automatic sleep staging."
    arXiv preprint arXiv:2110.15278 (2021).

    Note:
        We use one encoder to handle multiple channel together.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
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
        >>> from pyhealth.models import ContraWR
        >>> model = ContraWR(
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
            'loss': tensor(2.8425, device='cuda:0', grad_fn=<NllLossBackward0>),
            'y_prob': tensor([[0.9345, 0.0655],
                            [0.9482, 0.0518]], device='cuda:0', grad_fn=<SoftmaxBackward0>),
            'y_true': tensor([1, 1], device='cuda:0'),
            'logit': tensor([[ 0.1472, -2.5104],
                            [2.1584, -0.7481]], device='cuda:0', grad_fn=<AddmmBackward0>)
        }
        >>>
    """

    def __init__(
        self,
        dataset: BaseSignalDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_fft: int = 128,
        **kwargs,
    ):
        super(ContraWR, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft

        # TODO: Use more tokens for <gap> for different lengths once the input has such information
        self.label_tokenizer = self.get_label_tokenizer()

        # the ContraWR encoder
        channels, emb_size = self.cal_encoder_stat()
        self.encoder = nn.Sequential(
            *[
                ResBlock2D(channels[i], channels[i + 1], 2, True, True)
                for i in range(len(channels) - 1)
            ]
        )

        output_size = self.get_output_size(self.label_tokenizer)
        # the fully connected layer
        self.fc = nn.Linear(emb_size, output_size)

    def cal_encoder_stat(self):
        """obtain the convolution encoder initialization statistics

        Note:
            We show an example to illustrate the encoder statistics.
            input x:
                - torch.Size([5, 7, 3000])
            after stft transform
                - torch.Size([5, 7, 65, 90])
            we design the first CNN (out_channels = 8)
                - torch.Size([5, 8, 16, 22])
                - here: 8 * 16 * 22 > 256, we continute the convolution
            we design the second CNN (out_channels = 16)
                - torch.Size([5, 16, 4, 5])
                - here: 16 * 4 * 5 > 256, we continute the convolution
            we design the second CNN (out_channels = 32)
                - torch.Size([5, 32, 1, 1])
                - here: 32 * 1 * 1, we stop the convolution
            output:
                - channels = [7, 8, 16, 32]
                - emb_size = 32 * 1 * 1 = 32
        """

        print(f"\n=== Input data statistics ===")
        # obtain input signal size
        signal_info = self.dataset.input_info["signal"]
        in_channels, length = signal_info["n_channels"], signal_info["length"]
        # input signal size (batch, n_channels, length)
        print(f"n_channels: {in_channels}")
        print(f"length: {length}")

        # after stft transform (batch, n_channels, freq, time_steps)
        freq = self.n_fft // 2 + 1
        time_steps = (length - self.n_fft) // (self.n_fft // 4) + 1
        print(f"=== Spectrogram statistics ===")
        print(f"n_channels: {in_channels}")
        print(f"freq_dim: {freq}")
        print(f"time_steps: {time_steps}")

        if freq < 4 or time_steps < 4:
            raise ValueError("The input signal is too short or n_fft is too small.")

        # obtain stats at each cnn layer
        channels = [in_channels]
        cur_freq_dim = freq
        cur_time_dim = time_steps

        print(f"=== Convolution Statistics ===")
        while (cur_freq_dim >= 4 and cur_time_dim >= 4) and (
            len(channels) == 1 or cur_freq_dim * cur_time_dim * channels[-1] > 256
        ):
            channels.append(2 ** (math.floor(np.log2(channels[-1])) + 1))
            cur_freq_dim = (cur_freq_dim + 1) // 4
            cur_time_dim = (cur_time_dim + 1) // 4

            print(
                f"in_channels: {channels[-2]}, out_channels: {channels[-1]}, freq_dim: {cur_freq_dim}, time_steps: {cur_time_dim}"
            )
        print()

        emb_size = cur_freq_dim * cur_time_dim * channels[-1]
        return channels, emb_size

    def torch_stft(self, X):
        """torch short time fourier transform (STFT)

        Args:
            X: (batch, n_channels, length)

        Returns:
            signal: (batch, n_channels, freq, time_steps)
        """
        signal = []
        for s in range(X.shape[1]):
            spectral = torch.stft(
                X[:, s, :],
                n_fft=self.n_fft,
                hop_length=self.n_fft // 4,
                center=False,
                onesided=True,
                return_complex=False,
            )
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)
        signal = (signal1**2 + signal2**2) ** 0.5
        return signal

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = torch.tensor(
            np.array(kwargs[self.feature_keys[0]]), device=self.device
        ).float()
        # obtain the stft spectrogram (batch, channel, freq, time step)
        x_spectrogram = self.torch_stft(x)
        # final layer embedding (batch, embedding)
        emb = self.encoder(x_spectrogram).view(x.shape[0], -1)

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
    test the ResBlock2D
    """
    # import torch

    # input_ = torch.randn((16, 6, 3, 75))  # (batch, channel, height, width)
    # model = ResBlock2D(6, 16, 1, True, True)
    # output = model(input_)
    # print("input shape: ", input_.shape)
    # print("output.shape:", output.shape)

    # """
    # test ContraWR
    # """
    # from pyhealth.datasets import split_by_patient, get_dataloader
    # from pyhealth.trainer import Trainer
    # from pyhealth.datasets import SleepEDFDataset
    # from pyhealth.tasks import sleep_staging_sleepedf_fn

    # # step 1: load signal data
    # dataset = SleepEDFDataset(
    #     root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
    #     dev=True,
    #     refresh_cache=False,
    # )

    # # step 2: set task
    # sleep_staging_ds = dataset.set_task(sleep_staging_sleepedf_fn)
    # sleep_staging_ds.stat()
    # print(sleep_staging_ds.input_info)

    # # split dataset
    # train_dataset, val_dataset, test_dataset = split_by_patient(
    #     sleep_staging_ds, [0.6, 0.2, 0.2]
    # )
    # train_dataloader = get_dataloader(train_dataset, batch_size=5, shuffle=True)
    # val_dataloader = get_dataloader(val_dataset, batch_size=5, shuffle=False)
    # test_dataloader = get_dataloader(test_dataset, batch_size=5, shuffle=False)
    # print(
    #     "loader size: train/val/test",
    #     len(train_dataset),
    #     len(val_dataset),
    #     len(test_dataset),
    # )

    # batch = next(iter(train_dataloader))

    # # step 3: define model
    # model = ContraWR(
    #     sleep_staging_ds,
    #     feature_keys=["signal"],
    #     label_key="label",
    #     mode="multiclass",
    #     n_fft=128,
    # )

    # result = model(**batch)
    # print(result)

    """
    test ContraWR 2
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
    model = ContraWR(
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
