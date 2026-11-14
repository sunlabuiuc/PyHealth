"""EEGNet: a compact convolutional network for EEG classification.

Author: Axel Coiffard-Donaty
Paper: V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung,
    B. J. Lance. "EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces". Journal of Neural Engineering,
    15(5):056013, 2018.
Paper link: https://doi.org/10.1088/1741-2552/aace8c (preprint:
    https://arxiv.org/abs/1611.08024)
Description: PyTorch implementation of EEGNet-8,2 as described in the
    published version of the paper (arXiv v4, section 2.2.1 and table 2),
    written from the paper. It reads a ``(channels, time)`` signal and
    classifies it with any PyHealth label processor (binary, multiclass,
    multilabel).
"""

from typing import Any

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

# Sampling rate the paper's kernel lengths and pooling sizes were chosen for.
PAPER_SAMPLING_RATE = 128.0
# Keras BatchNormalization defaults used by the reference implementation:
# Keras momentum 0.99 is PyTorch momentum 0.01.
BN_MOMENTUM = 0.01
BN_EPS = 1e-3


def same_time_padding(kernel_length: int) -> nn.ZeroPad2d:
    """Zero-pad the time axis so a stride-1 convolution keeps its length.

    Matches Keras ``padding="same"``, used by the authors' code: an even
    kernel gets one more zero on the right than on the left. Explicit padding
    avoids the extra input copy that ``padding="same"`` costs in PyTorch for
    even kernels.

    Args:
        kernel_length: Kernel length along time.

    Returns:
        A padding layer for ``(batch, channels, height, time)`` inputs.

    Examples:
        >>> same_time_padding(64).padding
        (31, 32, 0, 0)
    """
    left = (kernel_length - 1) // 2
    return nn.ZeroPad2d((left, kernel_length - 1 - left, 0, 0))


class Conv2dWithMaxNorm(nn.Conv2d):
    """2D convolution whose filters are renormalized to a maximum L2 norm.

    The paper constrains the depthwise (spatial) filters with a max-norm of 1.
    As in Keras, where a kernel constraint is applied after each update, the
    weights are renormalized in place before each training forward pass.

    Args:
        *args: Positional arguments of :class:`torch.nn.Conv2d`.
        max_norm: Maximum L2 norm of each output filter.
        **kwargs: Keyword arguments of :class:`torch.nn.Conv2d`.

    Examples:
        >>> conv = Conv2dWithMaxNorm(1, 4, (3, 1), max_norm=1.0, bias=False)
        >>> _ = conv(torch.randn(2, 1, 3, 10))
        >>> bool(conv.weight.flatten(1).norm(dim=1).max() <= 1.0 + 1e-6)
        True
    """

    def __init__(self, *args: Any, max_norm: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renormalize the filters (in training mode), then convolve.

        Args:
            x: Input tensor of shape ``(batch, in_channels, height, width)``.

        Returns:
            The convolution output.
        """
        if self.training:
            with torch.no_grad():
                self.weight.copy_(
                    torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
                )
        return super().forward(x)


class LinearWithMaxNorm(nn.Linear):
    """Linear layer whose output units have a maximum L2 weight norm.

    The paper constrains the classification layer with a max-norm of 0.25.

    Args:
        *args: Positional arguments of :class:`torch.nn.Linear`.
        max_norm: Maximum L2 norm of the weights of each output unit.
        **kwargs: Keyword arguments of :class:`torch.nn.Linear`.

    Examples:
        >>> fc = LinearWithMaxNorm(8, 3, max_norm=0.25)
        >>> _ = fc(torch.randn(2, 8))
        >>> bool(fc.weight.norm(dim=1).max() <= 0.25 + 1e-6)
        True
    """

    def __init__(self, *args: Any, max_norm: float = 0.25, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renormalize the weights (in training mode), then apply the layer.

        Args:
            x: Input tensor of shape ``(batch, in_features)``.

        Returns:
            The layer output of shape ``(batch, out_features)``.
        """
        if self.training:
            with torch.no_grad():
                self.weight.copy_(
                    torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
                )
        return super().forward(x)


class EEGNet(BaseModel):
    """EEGNet model for EEG signal classification (Lawhern et al., 2018).

    The input is one signal feature of shape ``(channels, time)`` per sample,
    for example the ``signal`` field of
    :class:`~pyhealth.tasks.EEGMotorImageryEEGBCI` or
    :class:`~pyhealth.tasks.SeizureDetectionCHBMIT`. The network has two
    blocks:

    1. a temporal convolution (``F1`` filters, ``kernel_length`` samples),
       then a depthwise spatial convolution over all channels (``D`` filters
       per temporal filter, max-norm 1), batch norm, ELU, average pooling
       by 4 and dropout;
    2. a separable convolution (depthwise ``separable_kernel_length`` then
       pointwise to ``F2`` filters), batch norm, ELU, average pooling by 8
       and dropout;

    followed by a linear classifier with max-norm 0.25. Convolutions have no
    bias. The defaults are EEGNet-8,2 (``F1=8, D=2, F2=16``).

    Sampling rate: the paper uses 128 Hz, a 64-sample temporal kernel (half
    a second) and a 16-sample separable kernel (half a second after pooling
    by 4). By default this class keeps those durations at ``sampling_rate``:
    ``kernel_length = sampling_rate / 2`` and ``separable_kernel_length =
    sampling_rate / 8``, while the pooling sizes stay at 4 and 8. The paper
    only evaluates 128 Hz; other rates follow the authors' code comments and
    are not validated by the paper. Resampling the data to 128 Hz reproduces
    the published configuration exactly.

    Dropout: the paper uses 0.5 for within-subject and 0.25 for cross-subject
    classification. The default, 0.25, matches subject-wise splits.

    Weights use PyTorch's default initialization; the authors' Keras code
    uses Glorot uniform.

    Args:
        dataset: The :class:`~pyhealth.datasets.SampleDataset` to train on.
            It must have exactly one label key and one signal feature key
            (an ``"stft"`` key, if present, is ignored).
        sampling_rate: Sampling rate of the signal in Hz. Used only to derive
            the default kernel lengths. Default is 128.
        n_temporal_filters: ``F1`` in the paper, the number of temporal
            filters. Default is 8.
        depth_multiplier: ``D`` in the paper, the number of spatial filters
            per temporal filter. Default is 2.
        n_pointwise_filters: ``F2`` in the paper, the number of pointwise
            filters. Default is 16 (``D * F1``).
        kernel_length: Temporal kernel length in samples. Default is
            ``round(sampling_rate / 2)``.
        separable_kernel_length: Separable kernel length in samples, after
            the first pooling. Default is ``round(sampling_rate / 8)``.
        dropout: Dropout probability. Default is 0.25.

    Raises:
        ValueError: If the dataset does not have exactly one label key and
            one signal feature key, if the signal is not 1D or 2D, or if it is
            shorter than the total pooling factor (32 samples).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import EEGNet
        >>> rng = np.random.default_rng(0)
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "signal": rng.standard_normal((4, 256)).astype(np.float32),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="eegnet_demo",
        ... )
        >>> model = EEGNet(dataset=dataset, sampling_rate=128)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> out = model(**batch)
        >>> sorted(out)
        ['logit', 'loss', 'y_prob', 'y_true']
        >>> tuple(out["y_prob"].shape)
        (4, 1)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        sampling_rate: float = PAPER_SAMPLING_RATE,
        n_temporal_filters: int = 8,
        depth_multiplier: int = 2,
        n_pointwise_filters: int = 16,
        kernel_length: int | None = None,
        separable_kernel_length: int | None = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("EEGNet supports exactly one label key.")
        # Tasks such as EEGEventsTUEV also emit a precomputed "stft"; EEGNet
        # works on the raw signal only.
        self.feature_keys = [k for k in self.feature_keys if k != "stft"]
        if len(self.feature_keys) != 1:
            raise ValueError(
                "EEGNet supports exactly one signal feature key, got "
                f"{self.feature_keys}."
            )
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")

        self.sampling_rate = float(sampling_rate)
        self.n_temporal_filters = n_temporal_filters
        self.depth_multiplier = depth_multiplier
        self.n_pointwise_filters = n_pointwise_filters
        f1, f2 = n_temporal_filters, n_pointwise_filters
        n_spatial = f1 * depth_multiplier
        self.kernel_length = (
            kernel_length
            if kernel_length is not None
            else max(1, round(self.sampling_rate / 2))
        )
        self.separable_kernel_length = (
            separable_kernel_length
            if separable_kernel_length is not None
            else max(1, round(self.sampling_rate / 8))
        )
        self.dropout = dropout
        self.pool1, self.pool2 = 4, 8

        self.n_channels, self.n_times = self._infer_input_shape()
        n_pooled = self.n_times // (self.pool1 * self.pool2)
        if n_pooled < 1:
            raise ValueError(
                f"The signal has {self.n_times} samples; EEGNet pools time by "
                f"{self.pool1 * self.pool2}, so it needs at least that many."
            )

        self.block1 = nn.Sequential(
            same_time_padding(self.kernel_length),
            nn.Conv2d(1, f1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(f1, momentum=BN_MOMENTUM, eps=BN_EPS),
            Conv2dWithMaxNorm(
                f1,
                n_spatial,
                (self.n_channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(n_spatial, momentum=BN_MOMENTUM, eps=BN_EPS),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool1)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            same_time_padding(self.separable_kernel_length),
            nn.Conv2d(
                n_spatial,
                n_spatial,
                (1, self.separable_kernel_length),
                groups=n_spatial,
                bias=False,
            ),
            nn.Conv2d(n_spatial, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2, momentum=BN_MOMENTUM, eps=BN_EPS),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool2)),
            nn.Dropout(dropout),
        )
        self.embedding_dim = f2 * n_pooled
        self.fc = LinearWithMaxNorm(
            self.embedding_dim, self.label_processor.output_size(), max_norm=0.25
        )

    def _infer_input_shape(self) -> tuple[int, int]:
        """Read ``(channels, time)`` from the first sample of the dataset.

        Returns:
            The number of channels and of time samples.

        Raises:
            ValueError: If no sample has the feature, or its shape is not 1D
                or 2D.
        """
        key = self.feature_keys[0]
        for sample in self.dataset:
            if key not in sample:
                continue
            shape = tuple(sample[key].shape)
            if len(shape) == 1:
                return 1, shape[0]
            if len(shape) == 2:
                return shape[0], shape[1]
            raise ValueError(
                f"EEGNet expects a (channels, time) signal for {key!r}, "
                f"got shape {shape}."
            )
        raise ValueError(f"No sample in the dataset has the feature {key!r}.")

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run EEGNet on a batch.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`:
                the signal feature of shape ``(batch, channels, time)`` (or
                ``(batch, time)`` for one channel) and the label. Pass
                ``embed=True`` to also return the flattened features.

        Returns:
            A dict with ``loss`` (scalar), ``y_prob`` and ``logit`` of shape
            ``(batch, output_size)``, ``y_true``, and ``embed`` of shape
            ``(batch, embedding_dim)`` when requested.
        """
        x = kwargs[self.feature_keys[0]].to(self.device).float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # (batch, channels, time) -> (batch, 1, channels, time)
        x = x.unsqueeze(1)
        emb = self.block2(self.block1(x)).flatten(start_dim=1)
        logits = self.fc(emb)

        y_true = kwargs[self.label_keys[0]].to(self.device)
        results = {
            "loss": self.label_processor.loss(logits, y_true),
            "y_prob": self.label_processor.activation(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results
