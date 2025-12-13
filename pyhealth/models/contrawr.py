import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
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
        self.downsampler = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
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
        if self.downsample:
            residual = self.downsampler(x)
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
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        n_fft: the number of FFT points for STFT. Default is 128.

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
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_fft: int = 128,
    ):
        super(ContraWR, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft

        assert len(self.label_keys) == 1, (
            "Only one label key is supported if ContraWR is initialized"
        )
        assert len(self.feature_keys) == 1, (
            "Only one feature key is supported if ContraWR is initialized"
        )

        # the ContraWR encoder
        channels, emb_size = self.determine_encoder_params()
        self.encoder = nn.Sequential(
            *[
                ResBlock2D(channels[i], channels[i + 1], 2, True, True)
                for i in range(len(channels) - 1)
            ]
        )

        output_size = self.get_output_size()
        # the fully connected layer
        self.fc = nn.Linear(emb_size, output_size)

    def _determine_input_channels_length(self) -> int:
        for sample in self.dataset:
            if self.feature_keys[0] not in sample:
                continue

            if len(sample[self.feature_keys[0]].shape) == 1:
                return 1, sample[self.feature_keys[0]].shape[0]
            elif len(sample[self.feature_keys[0]].shape) == 2:
                return sample[self.feature_keys[0]].shape[0], sample[
                    self.feature_keys[0]
                ].shape[1]
            else:
                raise ValueError(
                    f"Invalid shape for feature key {self.feature_keys[0]}: {sample[self.feature_keys[0]].shape}"
                )

        raise ValueError(
            f"Unable to infer input channels and length from dataset for feature key {self.feature_keys[0]}"
        )

    def determine_encoder_params(self):
        """obtain the convolution encoder parameters based on input signal size

        Note:
            We show an example to illustrate the design process here.
            assume the input signal size is (batch = 5, n_channels = 7, length = 3000)
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

        print("\n=== Input data dimensions ===")
        in_channels, length = self._determine_input_channels_length()
        print(f"n_channels: {in_channels}")
        print(f"length: {length}")

        freq = self.n_fft // 2 + 1
        time_steps = (length - self.n_fft) // (self.n_fft // 4) + 1
        print("=== Spectrogram parameters ===")
        print(f"n_channels: {in_channels}")
        print(f"freq_dim: {freq}")
        print(f"time_steps: {time_steps}")

        if freq < 4 or time_steps < 4:
            raise ValueError("The input signal is too short or n_fft is too small.")

        # obtain stats at each cnn layer
        channels = [in_channels]
        cur_freq_dim = freq
        cur_time_dim = time_steps

        print("=== Convolution Parameters ===")
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
                return_complex=True,
                window=torch.hann_window(self.n_fft).to(X.device),
            )
            signal.append(torch.view_as_real(spectral))

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)
        signal = (signal1**2 + signal2**2) ** 0.5
        return signal

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]]
        # obtain the stft spectrogram (batch, channel, freq, time step)
        x_spectrogram = self.torch_stft(x)
        # final layer embedding (batch, embedding)
        emb = self.encoder(x_spectrogram).view(x.shape[0], -1)

        # (patient, label_size)
        logits = self.fc(emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_keys[0]]
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
