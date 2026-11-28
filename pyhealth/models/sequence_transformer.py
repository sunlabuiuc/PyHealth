"""Sequence Transformer Networks for clinical time series.

Contributors:
    Matthew Nguyen (mnguyen358, mnguyen358@gatech.edu)
    Alireza Ghiam (aghiam3, aghiam3@gatech.edu)

Paper:
    Jeeheh Oh, Jiaxuan Wang and Jenna Wiens. "Learning to Exploit Invariances
    in Clinical Time-Series Data using Sequence Transformer Networks."
    Machine Learning for Healthcare (MLHC) 2018, PMLR 85.
    https://proceedings.mlr.press/v85/oh18a.html

A Sequence Transformer learns a linear warp of the time axis
(t = theta1 * t' + theta0) and a linear map of the values
(x'' = phi1 * x' + phi0) and passes the transformed series to a small 1D CNN
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel


def _leaky_clamp(x: torch.Tensor, bound: float, slope: float = 0.01) -> torch.Tensor:
    """Identity on [-bound, bound], with a small slope outside.

    Subtracts the boundary before scaling so there is no jump. The released
    code uses bound * sign(x) + slope * x instead.
    """
    upper = bound + slope * (x - bound)
    lower = -bound + slope * (x + bound)
    return torch.where(x > bound, upper, torch.where(x < -bound, lower, x))


def temporal_sample(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Resamples series at linearly warped times

    Time runs over [-1, 1], so theta = (1, 0) is the identity. Output step t'
    reads the input at t = theta1 * t' + theta0 by linear interpolation
    between the two nearest steps. The positions beyond either end take the
    edge value (the paper pads with the last known value).

    Args:
        x: Series of shape [batch, channels, time]
        theta: [batch, 2] holding (theta1, theta0) per sample

    Returns:
        The resampled series, shaped like x.

    Examples:
        >>> import torch
        >>> x = torch.arange(5.0).view(1, 1, 5)
        >>> temporal_sample(x, torch.tensor([[1.0, 0.5]]))
        tensor([[[1., 2., 3., 4., 4.]]])
    """
    batch, channels, steps = x.shape
    if steps == 1:
        return x
    target = torch.linspace(-1.0, 1.0, steps, device=x.device, dtype=x.dtype)
    source = theta[:, :1] * target + theta[:, 1:2]
    position = ((source + 1.0) / 2.0 * (steps - 1)).clamp(0, steps - 1)
    left = position.floor().long().clamp(max=steps - 2)
    weight = (position - left.to(x.dtype)).unsqueeze(1)
    left = left.unsqueeze(1).expand(batch, channels, steps)
    x_left = torch.gather(x, 2, left)
    x_right = torch.gather(x, 2, left + 1)
    return x_left * (1.0 - weight) + x_right * weight


class SequenceTransformerLayer(nn.Module):
    """Learns and applies per-sample temporal and magnitude transformations.

    A small CNN reads the whole series and predicts theta = (theta1, theta0)
    and phi = (phi1, phi0). The series is then warped in time with theta and
    mapped as phi1 * x + phi0. The same parameters apply to every channel of
    a sample. The layer starts at the identity transformation, and parameters
    beyond [-bound, bound] change with slope 0.01.

    Args:
        in_channels: Number of input channels
        seq_len: Number of time steps; at least 18 for the network's
            unpadded convolutions and pooling
        temporal: Whether to learn the temporal transformation
        magnitude: Whether to learn the magnitude transformation
        bound: Range [-bound, bound] in which the parameters change freely
               Must exceed 1, so that the identity (1, 0) is inside.

    Examples:
        >>> import torch
        >>> layer = SequenceTransformerLayer(in_channels=3, seq_len=24)
        >>> out, theta, phi = layer(torch.randn(2, 3, 24))
        >>> out.shape, theta.shape, phi.shape
        (torch.Size([2, 3, 24]), torch.Size([2, 2]), torch.Size([2, 2]))
        >>> theta[0].tolist(), phi[0].tolist()
        ([1.0, 0.0], [1.0, 0.0])
    """

    MIN_SEQ_LEN = 18

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        temporal: bool = True,
        magnitude: bool = True,
        bound: float = 2.0,
    ) -> None:
        super().__init__()
        if seq_len < self.MIN_SEQ_LEN:
            raise ValueError(
                f"seq_len must be at least {self.MIN_SEQ_LEN}, got {seq_len}."
            )
        if bound <= 1.0:
            raise ValueError(
                f"bound must exceed 1 to contain the identity (1, 0), got {bound}."
            )
        self.temporal = temporal
        self.magnitude = magnitude
        self.bound = bound
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, seq_len)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, 32), nn.ReLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))
        self.register_buffer("identity", torch.tensor([1.0, 0.0]), persistent=False)

    def parameters_for(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the transformation parameters of each sample

        Args:
            x: Series of shape [batch, channels, time]

        Returns:
            (theta, phi), each [batch, 2]
        """
        params = _leaky_clamp(self.head(self.features(x)), self.bound)
        identity = self.identity.expand(x.shape[0], 2)
        theta = params[:, :2] if self.temporal else identity
        phi = params[:, 2:] if self.magnitude else identity
        return theta, phi

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms the series

        Args:
            x: Series of shape [batch, channels, time]

        Returns:
            The transformed series (same shape), theta and phi
        """
        theta, phi = self.parameters_for(x)
        if self.temporal:
            x = temporal_sample(x, theta)
        if self.magnitude:
            x = phi[:, :1, None] * x + phi[:, 1:, None]
        return x, theta, phi


class _ConvClassifier(nn.Module):
    """CNN with Conv1d/ReLU/max-pool blocks, FC, and output"""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        channels: Sequence[int],
        kernel_size: int,
        hidden_dim: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width, steps = in_channels, seq_len
        for out_channels in channels:
            if steps < kernel_size + 1:
                raise ValueError(
                    f"seq_len={seq_len} is too short for {len(channels)} "
                    f"convolution blocks with kernel_size={kernel_size}."
                )
            layers += [nn.Conv1d(width, out_channels, kernel_size), nn.ReLU()]
            layers.append(nn.MaxPool1d(2, 2))
            steps = (steps - kernel_size + 1) // 2
            width = out_channels
        self.features = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, seq_len)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SequenceTransformerNetwork(BaseModel):
    """Sequence Transformer Network (Oh et al., MLHC 2018).

    A SequenceTransformerLayer warps each sample's series in time and value,
    and a 1D CNN classifies the result.
    With temporal and magnitude both False, the model is the CNN baseline.

    Inputs: every input feature is turned into channels of one
    [batch, channels, seq_len] series

    - Series: [batch, time, features] tensor or a single-level
      jagged/sequence_v2 series field
    - Static vectors: [batch, features] tensor or a single jagged field,
      copied across all time steps

    Series are cut to their first seq_len steps, or extended by repeating
    their last step.
    Non-finite values are set to 0.

    Implementation choices:

    - Time runs over [-1, 1], and the layer starts at the identity.
    - The localizer uses 1D convolutions from the paper's diagram. The
      released code uses 2D convolutions over time and features.
    - The clamp has slope 0.01 outside [-bound, bound], without the small
      jump in the released code.
    - Dropout goes before the output layer. The released code puts it after
      the second convolution, with fixed 0.5 dropout after the hidden FC.

    Released code: https://github.com/MLD3/MLHC2018_SequenceTransformerNetworks

    Args:
        dataset: The ~pyhealth.datasets.SampleDataset to train on (it must
            have exactly one label)
        seq_len: Number of time steps the model sees (48 for hourly paper data)
        temporal: Learn the temporal transformation (theta)
        magnitude: Learn the magnitude transformation (phi)
        conv_channels: Output channels of each CNN block (Paper uses (10, 20))
        kernel_size: CNN kernel size
        hidden_dim: Size of the CNN's fully connected hidden layer
        dropout: Dropout before the output layer. The default is 0.2;
            placement differs from the released code as described above.
        bound: Range [-bound, bound] (must exceed 1)

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> rng = np.random.default_rng(0)
        >>> samples = [
        ...     {
        ...         'patient_id': f'p{i}',
        ...         'vitals': rng.normal(size=(24, 3)).astype('float32'),
        ...         'mortality': i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={'vitals': 'tensor'},
        ...     output_schema={'mortality': 'binary'},
        ... )
        >>> model = SequenceTransformerNetwork(dataset, seq_len=24)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> out = model(**batch)
        >>> sorted(out), out['y_prob'].shape
        (['logit', 'loss', 'y_prob', 'y_true'], torch.Size([4, 1]))
    """

    def __init__(
        self,
        dataset: SampleDataset,
        seq_len: int = 48,
        temporal: bool = True,
        magnitude: bool = True,
        conv_channels: Sequence[int] = (10, 20),
        kernel_size: int = 5,
        hidden_dim: int = 50,
        dropout: float = 0.2,
        bound: float = 2.0,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("SequenceTransformerNetwork needs exactly one label")
        if not self.feature_keys:
            raise ValueError("SequenceTransformerNetwork needs at least one input")
        self.label_key = self.label_keys[0]
        self.seq_len = int(seq_len)
        self.temporal = temporal
        self.magnitude = magnitude
        self.in_channels = self._count_channels(dataset)
        self.transformer = (
            SequenceTransformerLayer(
                self.in_channels, self.seq_len, temporal, magnitude, bound
            )
            if temporal or magnitude
            else None
        )
        self.cnn = _ConvClassifier(
            self.in_channels,
            self.seq_len,
            list(conv_channels),
            kernel_size,
            hidden_dim,
            dropout,
            self.label_processor.output_size(),
        )

    def _count_channels(self, dataset: SampleDataset) -> int:
        """Number of input channels, from the layout of the first sample"""
        sample = dataset[0]
        batch = {}
        for key in self.feature_keys:
            value = sample[key]
            processor = dataset.input_processors[key]
            if isinstance(value, tuple):
                batch[key] = processor.collate([value])
            else:
                batch[key] = torch.as_tensor(value).unsqueeze(0)
        return self.series(**batch).shape[1]

    def _jagged(self, key: str, value: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Turns a single-level jagged field into ``[batch, channels, time]``."""
        processor = self.dataset.input_processors[key]
        # The batch holds the fields of ``schema()``, then the batch offsets.
        fields = dict(zip(processor.schema(), value[:-1]))
        roles = {role for _, role in fields}
        if "tokens" in roles or len(processor.ordering()) != 1:
            raise ValueError(
                f"{key!r}: only single-level jagged fields of values are "
                "supported (not tokens or nested records)."
            )
        offsets = value[-1].long()
        parts = [fields[(0, "values")].flatten(1).float()]
        if (0, "mask") in fields:
            parts.append(fields[(0, "mask")].flatten(1).float())
        rows = torch.cat(parts, dim=1)
        if processor.ordering()[0] == "single":
            return rows.unsqueeze(-1)
        lengths = (offsets[1:] - offsets[:-1]).tolist()
        series = nn.utils.rnn.pad_sequence(
            list(torch.split(rows, lengths)), batch_first=True
        )
        return self._fit_length(series, torch.as_tensor(lengths))

    def _fit_length(
        self, series: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[batch, time, features] to [batch, features, seq_len].

        Keeps the first seq_len steps
        Shorter series repeat their last valid step and empty ones are zero.
        """
        batch, steps, _ = series.shape
        if steps == 0:
            return series.new_zeros(batch, series.shape[2], self.seq_len)
        if lengths is None:
            lengths = torch.full((batch,), steps)
        lengths = lengths.to(series.device)
        index = torch.arange(self.seq_len, device=series.device)
        index = torch.minimum(index.unsqueeze(0), (lengths - 1).clamp(min=0)[:, None])
        out = torch.gather(
            series, 1, index.unsqueeze(-1).expand(-1, -1, series.shape[2])
        )
        out = out * (lengths > 0).to(out.dtype)[:, None, None]
        return out.transpose(1, 2)

    def series(self, **kwargs: Any) -> torch.Tensor:
        """Builds the model input from a batch.

        Args:
            **kwargs: A batch from pyhealth.datasets.get_dataloader (only the
                input features are used)
        Returns:
            [batch, channels, seq_len]
        """
        channels = []
        for key in self.feature_keys:
            value = kwargs[key]
            if isinstance(value, (tuple, list)):
                value = tuple(torch.as_tensor(v).to(self.device) for v in value)
                part = self._jagged(key, value)
            else:
                value = torch.as_tensor(value).to(self.device).float()
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
                if value.dim() == 2:
                    part = value.unsqueeze(-1)
                elif value.dim() == 3:
                    part = self._fit_length(value)
                else:
                    raise ValueError(
                        f"{key!r}: expected a [batch, time, features] or "
                        "[batch, features] tensor, "
                        f"got shape {tuple(value.shape)}."
                    )
            if part.shape[-1] == 1:
                part = part.expand(-1, -1, self.seq_len)
            channels.append(part)
        x = torch.cat(channels, dim=1)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Returns the input, its transformation and the learned parameters

        Args:
            **kwargs: A batch from pyhealth.datasets.get_dataloader

        Returns:
            A dict with ``x`` and ``x_transformed`` (``[batch, channels,
            seq_len]``), and ``theta`` and ``phi`` (``[batch, 2]``; the
            identity ``(1, 0)`` for a disabled transformation).
        """
        x = self.series(**kwargs)
        if self.transformer is None:
            identity = x.new_tensor([1.0, 0.0]).expand(x.shape[0], 2)
            return {"x": x, "x_transformed": x, "theta": identity, "phi": identity}
        x_transformed, theta, phi = self.transformer(x)
        return {"x": x, "x_transformed": x_transformed, "theta": theta, "phi": phi}

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass

        Args:
            **kwargs: A batch from pyhealth.datasets.get_dataloader

        Returns:
            A dict with logit and y_prob. If labels are supplied, it also
            includes loss and y_true.
        """
        x = self.series(**kwargs)
        if self.transformer is not None:
            x = self.transformer(x)[0]
        logit = self.cnn(x)
        out = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            y_true = torch.as_tensor(kwargs[self.label_key]).to(self.device)
            out["loss"] = self.label_processor.loss(logit, y_true)
            out["y_true"] = y_true
        return out
