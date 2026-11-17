import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import LabelProcessor, MultiClassLabelProcessor

from .base_model import BaseModel


def _same_padding(length: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """TensorFlow's "SAME" padding: the output has ``ceil(length / stride)`` steps."""
    out = math.ceil(length / stride)
    total = max((out - 1) * stride + kernel_size - length, 0)
    return total // 2, total - total // 2


class _ConvBlock(nn.Module):
    """Conv1d, then BatchNorm and ReLU.

    With ``same=True`` the padding is TensorFlow's "SAME", otherwise none
    ("VALID").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        same: bool,
        bias: bool,
        momentum: float,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.same = same
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=momentum)

    def output_length(self, length: int) -> int:
        if self.same:
            return math.ceil(length / self.stride)
        return (length - self.kernel_size) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same:
            x = F.pad(x, _same_padding(x.shape[-1], self.kernel_size, self.stride))
        return F.relu(self.bn(self.conv(x)))


class _MaxPool(nn.Module):
    """MaxPool1d with "SAME" (padded steps never win the max) or no padding."""

    def __init__(self, kernel_size: int, same: bool):
        super().__init__()
        self.kernel_size = kernel_size
        self.same = same

    def output_length(self, length: int) -> int:
        if self.same:
            return math.ceil(length / self.kernel_size)
        return length // self.kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same:
            pad = _same_padding(x.shape[-1], self.kernel_size, self.kernel_size)
            x = F.pad(x, pad, value=float("-inf"))
        return F.max_pool1d(x, self.kernel_size)


class _Branch(nn.Module):
    """One CNN branch of DeepSleepNet's representation learning part.

    A wide strided convolution with 64 filters, max-pooling, then
    ``n_conv - 1`` convolutions with 128 filters and a second max-pooling.
    In the ``"sleepkd"`` layout the first convolution and the poolings have
    no padding, and the convolutions have biases.
    """

    def __init__(
        self,
        in_channels: int,
        n_conv: int,
        kernel_size: int,
        stride: int,
        pool1: int,
        kernel_size2: int,
        pool2: int,
        layout: str,
    ):
        super().__init__()
        same = layout == "supratak"
        # The reference's moving averages decay by 0.999 per step, Keras's
        # (the SleepKD code) by 0.99.
        block = {"bias": not same, "momentum": 0.001 if same else 0.01}
        layers: list[nn.Module] = [
            _ConvBlock(in_channels, 64, kernel_size, stride, same=same, **block),
            _MaxPool(pool1, same=same),
        ]
        channels = 64
        for _ in range(n_conv - 1):
            layers.append(
                _ConvBlock(channels, 128, kernel_size2, 1, same=True, **block)
            )
            channels = 128
        layers.append(_MaxPool(pool2, same=same))
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def output_dim(self, length: int) -> int:
        """Size of the flattened output for an input of ``length`` steps."""
        for layer in self.layers:
            length = layer.output_length(length)
        return self.out_channels * length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).flatten(1)


class DeepSleepNet(BaseModel):
    """DeepSleepNet: sequence-to-sequence sleep staging from raw single-channel EEG.

    Paper: Akara Supratak, Hao Dong, Chao Wu, and Yike Guo. "DeepSleepNet: a
    model for automatic sleep stage scoring based on raw single-channel EEG."
    IEEE Transactions on Neural Systems and Rehabilitation Engineering 25.11
    (2017): 1998-2008.

    A PyTorch port of the reference TensorFlow implementation
    (https://github.com/akaraspt/deepsleepnet, Apache-2.0). Each 30 s epoch
    goes through two CNN branches, a small-filter one (kernel ``sfreq / 2``,
    stride ``sfreq / 16``) for temporal detail and a large-filter one (kernel
    ``4 * sfreq``, stride ``sfreq / 2``) for frequency content. Their
    concatenated outputs (the *epoch features*) feed a bidirectional LSTM over
    the sequence of epochs, whose output is added to a residual projection of
    the epoch features (the *sequence features*) and classified per epoch.
    As in the reference, every convolution is followed by BatchNorm and ReLU,
    and padding is TensorFlow's "SAME".

    ``n_conv`` and ``hidden_dim`` scale the model: the paper's model is
    ``n_conv=4, hidden_dim=512``. SleepKD (Liang et al., IJCAI 2023) distills
    it into a teacher assistant (2, 256) and a student (1, 128).

    ``layout="sleepkd"`` is the smaller variant that SleepKD actually uses,
    rebuilt from the layout of its authors' Keras code
    (https://github.com/AI4Brain/SleepKD); with ``n_rnn_layers=1`` its
    teacher (4, 512) and student (1, 128) have the 5,502,474 and 1,552,906
    parameters of the SleepKD paper's Table 5, counted as Keras does. In it:

    - The first convolutions and the poolings have no padding, and the
      convolutions have biases.
    - The CNN output goes through a 1024-unit linear layer (the epoch
      features) and a linear layer to ``n_classes`` scores. The authors
      apply dropout between the two, whose output reaches BatchNorm through
      the residual projection; here there is none (see below).
    - The LSTM and the residual projection read those ``n_classes`` scores
      instead of the CNN output, and the sequence features are the LSTM
      output, before the residual is added.
    - The authors' LSTM uses ReLU activations; this one keeps tanh. The
      authors also pre-train the CNN on single epochs before training the
      sequence model, which the paper does not describe; here the model is
      trained end to end.

    Differences from the reference:

    - No BatchNorm layer gets dropped-out inputs. The reference drops out
      the input of the convolutions after the first max-pooling and of the
      residual projection, all followed by BatchNorm. Dropout changes the
      variance of what BatchNorm sees in training, so its running statistics
      are wrong in evaluation: on Sleep-EDF-20, validation accuracy fell
      from 0.71 with batch statistics to 0.35 with the running ones, for the
      same weights. Here the CNN branches have no dropout and the residual
      projection takes the epoch features before dropout; dropout remains on
      the LSTM input and before the classifier.
    - The model is trained end to end (the reference first pre-trains the
      CNN on class-balanced epochs, then fine-tunes the whole model).
    - The LSTM has no peephole connections and applies dropout between its
      layers only, and the L2 weight decay of the reference's first
      convolutions is left to the optimizer.

    The dataset needs one input, a ``(seq_len, n_channels, epoch_len)`` float
    tensor per sample, and one label, a ``(seq_len,)`` long tensor of class
    indices, as produced by
    :class:`~pyhealth.tasks.SleepStagingSequenceSleepEDF`. The per-epoch
    predictions are flattened, so ``y_prob`` is ``(batch * seq_len,
    n_classes)`` and ``y_true`` is ``(batch * seq_len,)``, and
    :class:`~pyhealth.trainer.Trainer` scores every epoch as a multiclass
    label.

    Args:
        dataset: The dataset to train the model.
        n_classes: Number of sleep stages. Default is 5 (W, N1, N2, N3, REM).
        sfreq: Sampling rate of the signal in Hz, which sets the kernel sizes
            and strides of the first convolutions. Default is 100.
        n_conv: Convolutions per CNN branch, the first one included.
            Default is 4.
        hidden_dim: Hidden size of each direction of the LSTM. The sequence
            features have ``2 * hidden_dim`` entries. Default is 512.
        n_rnn_layers: Number of LSTM layers. Default is 2 (the SleepKD
            layout uses 1).
        dropout: Dropout rate. Default is 0.5.
        layout: ``"supratak"``, the DeepSleepNet paper's model, or
            ``"sleepkd"``, the variant of the SleepKD paper (see above).
            Default is ``"supratak"``.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import DeepSleepNet
        >>> samples = [
        ...     {
        ...         "patient_id": str(i),
        ...         "signal": np.random.randn(4, 1, 3000).astype("float32"),
        ...         "label": np.array([0, 1, 2, 4]),
        ...     }
        ...     for i in range(2)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": ("tensor", {"dtype": torch.long})},
        ... )
        >>> model = DeepSleepNet(dataset=dataset, n_conv=1, hidden_dim=16)
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> output = model(**batch)
        >>> output["y_prob"].shape
        torch.Size([8, 5])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_classes: int = 5,
        sfreq: int = 100,
        n_conv: int = 4,
        hidden_dim: int = 512,
        n_rnn_layers: int = 2,
        dropout: float = 0.5,
        layout: Literal["supratak", "sleepkd"] = "supratak",
    ):
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError(
                "DeepSleepNet needs one input and one label field, got inputs "
                f"{self.feature_keys} and labels {self.label_keys}."
            )
        if n_conv < 1:
            raise ValueError(f"n_conv must be at least 1, got {n_conv}.")
        if layout not in ("supratak", "sleepkd"):
            raise ValueError(f"layout must be 'supratak' or 'sleepkd', got {layout!r}.")
        self.layout = layout
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self._label_processor = MultiClassLabelProcessor(labels=n_classes)

        shape = tuple(dataset[0][self.feature_key].shape)
        if len(shape) != 3:
            raise ValueError(
                f"'{self.feature_key}' must be a (seq_len, n_channels, epoch_len) "
                f"tensor per sample, got shape {shape}."
            )
        _, in_channels, epoch_len = shape

        self.small = _Branch(
            in_channels,
            n_conv,
            kernel_size=sfreq // 2,
            stride=sfreq // 16,
            pool1=8,
            kernel_size2=8,
            pool2=4,
            layout=layout,
        )
        self.large = _Branch(
            in_channels,
            n_conv,
            kernel_size=sfreq * 4,
            stride=sfreq // 2,
            pool1=4,
            kernel_size2=6,
            pool2=2,
            layout=layout,
        )
        cnn_dim = self.small.output_dim(epoch_len) + self.large.output_dim(epoch_len)
        self.sequence_dim = 2 * hidden_dim
        self.epoch_dropout = nn.Dropout(dropout)
        if layout == "sleepkd":
            self.epoch_dim = 1024
            self.epoch_fc = nn.Linear(cnn_dim, self.epoch_dim)
            self.epoch_head = nn.Linear(self.epoch_dim, n_classes)
            lstm_input = n_classes
            residual_bias, momentum = True, 0.01
        else:
            self.epoch_dim = cnn_dim
            lstm_input = cnn_dim
            residual_bias, momentum = False, 0.001
        self.residual = nn.Sequential(
            nn.Linear(lstm_input, self.sequence_dim, bias=residual_bias),
            nn.BatchNorm1d(self.sequence_dim, momentum=momentum),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            lstm_input,
            hidden_dim,
            num_layers=n_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_rnn_layers > 1 else 0.0,
        )
        self.sequence_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.sequence_dim, n_classes)

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The per-epoch label kind used by :class:`~pyhealth.trainer.Trainer`.

        The label is a ``"tensor"`` of one class per epoch, which is not a
        label processor. DeepSleepNet flattens its per-epoch outputs and
        supplies a multiclass label processor for them, so the Trainer scores
        every epoch (accuracy, macro-F1, ...).
        """
        return {self.label_key: self._label_processor}

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Runs the network on a batch of epoch sequences.

        Args:
            x: A ``(batch, seq_len, n_channels, epoch_len)`` signal.

        Returns:
            A dict with ``epoch_feature`` ``(batch, seq_len, epoch_dim)``, the
            CNN output of each epoch (in the ``"sleepkd"`` layout, the
            1024-unit layer after it); ``sequence_feature`` ``(batch, seq_len,
            2 * hidden_dim)``, the LSTM output plus the residual (in the
            ``"sleepkd"`` layout, the LSTM output alone); and ``logit``
            ``(batch, seq_len, n_classes)``. The features are taken before
            dropout.
        """
        batch, seq_len = x.shape[:2]
        x = x.flatten(0, 1)
        epoch_feature = torch.cat([self.small(x), self.large(x)], dim=1)
        if self.layout == "sleepkd":
            epoch_feature = self.epoch_fc(epoch_feature)
            h = self.epoch_head(epoch_feature)
        else:
            h = epoch_feature
        # Dropout only where no BatchNorm follows (see the class docstring).
        residual = self.residual(h)
        h = self.epoch_dropout(h)
        lstm_out, _ = self.lstm(h.view(batch, seq_len, -1))
        residual = residual.view(batch, seq_len, -1)
        if self.layout == "sleepkd":
            sequence_feature = lstm_out
            hidden = lstm_out + residual
        else:
            sequence_feature = hidden = lstm_out + residual
        logit = self.fc(self.sequence_dropout(hidden))
        return {
            "epoch_feature": epoch_feature.view(batch, seq_len, -1),
            "sequence_feature": sequence_feature,
            "logit": logit,
        }

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Returns:
            A dict with ``loss``, ``y_prob`` and ``logit`` of shape
            ``(batch * seq_len, n_classes)``, and ``y_true`` of shape
            ``(batch * seq_len,)``. With ``embed=True``, also ``embed``, the
            ``(batch * seq_len, 2 * hidden_dim)`` sequence features.
        """
        x = kwargs[self.feature_key].to(self.device)
        out = self.encode(x)
        logits = out["logit"].flatten(0, 1)
        y_true = kwargs[self.label_key].to(self.device).flatten()
        results = {
            "loss": self._label_processor.loss(logits, y_true),
            "y_prob": self._label_processor.activation(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = out["sequence_feature"].flatten(0, 1)
        return results
