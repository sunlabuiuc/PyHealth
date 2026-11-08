"""APN: adaptive patching for irregular multivariate time-series forecasting.

Paper:
    Xvyuan Liu, Xiangfei Qiu, Xingjian Wu, Zhengyu Li, Chenjuan Guo, Jilin
    Hu and Bin Yang. "Rethinking Irregular Time Series Forecasting: A Simple
    yet Effective Baseline." AAAI 2026. https://arxiv.org/abs/2505.11250

This is a reimplementation from the paper; the authors' code
(https://github.com/decisionintelligence/APN) was used only to confirm
details such as the default sizes.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from ..processors import LabelProcessor, RegressionLabelProcessor
from .base_model import BaseModel


class LearnableTimeEncoding(nn.Module):
    """A linear term and ``dim - 1`` learned sinusoids of a time.

    Args:
        dim: Output size (at least 2).

    Examples:
        >>> import torch
        >>> LearnableTimeEncoding(8)(torch.rand(2, 5, 1)).shape
        torch.Size([2, 5, 8])
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("The time encoding needs at least 2 dimensions.")
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encodes times ``[..., 1]`` as ``[..., dim]``."""
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


class TimeAwarePatchAggregation(nn.Module):
    """Time-Aware Patch Aggregation (TAPA), the patching module of APN.

    Each variable gets ``num_patches`` patches with learnable boundaries:
    the left edge is an offset from evenly spaced initial positions and the
    width is learned in log space. An observation's weight in a patch is a
    soft indicator of lying inside it, ``sigmoid((right - t) / tau) *
    sigmoid((t - left) / tau)``, with a learned per-variable temperature
    ``tau``. Each patch is the weight-normalized average of the features of
    the observed points, projected and refined by a residual feed-forward
    layer.

    Args:
        num_variables: Number of variables (channels).
        num_patches: Patches per variable.
        feature_dim: Size of each observation's feature (value and time
            encoding).
        hidden_dim: Size of a patch representation.
        history: Length of the time span the patches initially cover.
        dropout: Dropout in the feed-forward layer.

    Examples:
        >>> import torch
        >>> tapa = TimeAwarePatchAggregation(3, 4, feature_dim=5, hidden_dim=8)
        >>> t = torch.rand(2, 3, 10)
        >>> h = tapa(t, torch.randn(2, 3, 10, 5), torch.ones(2, 3, 10))
        >>> h.shape
        torch.Size([2, 3, 4, 8])
    """

    def __init__(
        self,
        num_variables: int,
        num_patches: int,
        feature_dim: int,
        hidden_dim: int,
        history: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.history = history
        width = max(history / num_patches, 1e-6)
        centers = torch.linspace(width / 2, history - width / 2, num_patches)
        self.register_buffer("initial_left", centers - width / 2)
        self.left_offset = nn.Parameter(torch.zeros(num_variables, num_patches))
        self.log_width = nn.Parameter(
            torch.full((num_variables, num_patches), math.log(width))
        )
        self.raw_tau = nn.Parameter(torch.zeros(num_variables))
        self.project = nn.Linear(feature_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def boundaries(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(left, right)`` patch edges, each ``[num_variables, num_patches]``."""
        left = self.initial_left + self.left_offset
        right = left + torch.exp(self.log_width) + 1e-6
        return left, right

    def weights(self, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Soft patch memberships of observations, ``[B, N, P, L]``.

        Args:
            t: Observation times, ``[B, N, L]``.
            mask: 1 for observed points, ``[B, N, L]``.
        """
        left, right = self.boundaries()
        tau = (F.softplus(self.raw_tau) + 1e-6)[None, :, None, None]
        t = t.unsqueeze(2)  # [B, N, 1, L]
        inside = torch.sigmoid((right[None, :, :, None] - t) / tau) * torch.sigmoid(
            (t - left[None, :, :, None]) / tau
        )
        return inside * mask.unsqueeze(2)

    def forward(
        self, t: torch.Tensor, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Aggregates observations into patches.

        Args:
            t: Observation times, ``[B, N, L]``.
            features: Observation features, ``[B, N, L, feature_dim]``.
            mask: 1 for observed points, ``[B, N, L]``.

        Returns:
            Patch representations, ``[B, N, P, hidden_dim]``.
        """
        w = self.weights(t, mask)
        pooled = torch.einsum("bnpl,bnlf->bnpf", w, features) / (
            w.sum(dim=-1, keepdim=True) + 1e-9
        )
        h = self.project(pooled)
        return self.norm(h + self.ffn(h))


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * scale)
    table[:, 1::2] = torch.cos(position * scale)[:, : dim // 2]
    return table


class APN(BaseModel):
    """APN forecaster for irregular multivariate time series (AAAI 2026).

    Every variable is handled on its own (channel-independent). Its
    observations, each a value plus a learnable time encoding, are pooled
    into adaptive patches by :class:`TimeAwarePatchAggregation`; a learned
    per-variable query attends over the patches (with sinusoidal patch
    positions) to give one summary vector, and a shallow MLP maps the summary
    and the encoding of each query time to a forecast.

    The model reads three inputs and one label from the dataset:

    - ``x``: values ``[B, L, N]``, NaN where not observed;
    - ``x_time``: their times ``[B, L]`` (shared by all variables);
    - ``y_time``: the times to forecast ``[B, K]``;
    - ``y``: the targets ``[B, K, N]``, NaN where not observed.

    :class:`~pyhealth.tasks.PhysioNet2012Forecasting` produces these fields;
    other names can be given through ``value_key`` etc. The loss is the mean
    squared error over the observed targets. ``y_true`` and ``y_prob`` in the
    output are the observed targets and their forecasts, flattened, so the
    :class:`~pyhealth.trainer.Trainer` scores the model in ``"regression"``
    mode with the MSE and MAE over all observed targets (use
    ``metrics=["mse", "mae"]``). ``forecast`` holds the full ``[B, K, N]``
    forecasts.

    Args:
        dataset: The samples; used for the field names and the number of
            variables.
        hidden_dim: Size of the patch and summary representations
            (``d_model``; 24 for PhysioNet 2012 in the paper).
        num_patches: Patches per variable (20 for PhysioNet 2012).
        time_dim: Size of the time encoding (8).
        dropout: Dropout rate (0.1).
        history: Time span the patches initially cover, in the units of the
            times (1.0 when times are scaled to [0, 1]).
        value_key: Input field with the observed values.
        time_key: Input field with their times.
        query_key: Input field with the times to forecast.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.datasets import get_dataloader
        >>> rng = np.random.default_rng(0)
        >>> samples = [
        ...     {"patient_id": f"p{i}", "x": rng.normal(size=(6, 2)).astype("float32"),
        ...      "x_time": np.arange(6, dtype="float32") / 8,
        ...      "y_time": np.array([6, 7], dtype="float32") / 8,
        ...      "y": rng.normal(size=(2, 2)).astype("float32")}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     {"x": "tensor", "x_time": "tensor", "y_time": "tensor"},
        ...     {"y": "tensor"},
        ... )
        >>> model = APN(dataset, hidden_dim=8, num_patches=3, time_dim=4)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> out = model(**batch)
        >>> out["forecast"].shape
        torch.Size([4, 2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 24,
        num_patches: int = 20,
        time_dim: int = 8,
        dropout: float = 0.1,
        history: float = 1.0,
        value_key: str = "x",
        time_key: str = "x_time",
        query_key: str = "y_time",
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("APN expects exactly one label field (the targets).")
        self.label_key = self.label_keys[0]
        self.value_key, self.time_key, self.query_key = value_key, time_key, query_key
        self.num_variables = self._num_variables(dataset)
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.time_encoding = LearnableTimeEncoding(time_dim)
        self.patching = TimeAwarePatchAggregation(
            self.num_variables, num_patches, 1 + time_dim, hidden_dim, history, dropout
        )
        self.register_buffer(
            "patch_positions", _sinusoidal_positions(num_patches, hidden_dim)
        )
        self.queries = nn.Parameter(torch.randn(self.num_variables, hidden_dim))
        self.summary_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    def _num_variables(self, dataset: SampleDataset) -> int:
        processor = dataset.input_processors[self.value_key]
        size = processor.size() if hasattr(processor, "size") else None
        if size:
            return int(size)
        return int(torch.as_tensor(dataset[0][self.value_key]).shape[-1])

    def encode(
        self, x: torch.Tensor, x_time: torch.Tensor
    ) -> torch.Tensor:
        """Summarizes each variable's history, ``[B, N, hidden_dim]``.

        Args:
            x: Values ``[B, L, N]``, NaN where not observed.
            x_time: Times ``[B, L]``.
        """
        batch, length, n_vars = x.shape
        mask = torch.isfinite(x).transpose(1, 2).float()  # [B, N, L]
        values = torch.nan_to_num(x, nan=0.0).transpose(1, 2)  # [B, N, L]
        t = x_time.unsqueeze(1).expand(batch, n_vars, length)
        features = torch.cat(
            [values.unsqueeze(-1), self.time_encoding(t.unsqueeze(-1))], dim=-1
        )
        patches = self.patching(t, features, mask) + self.patch_positions
        scores = torch.einsum("nh,bnph->bnp", self.queries, patches)
        attention = torch.softmax(scores / math.sqrt(self.hidden_dim), dim=-1)
        summary = torch.einsum("bnp,bnph->bnh", attention, patches)
        return self.summary_norm(summary)

    def decode(self, summary: torch.Tensor, y_time: torch.Tensor) -> torch.Tensor:
        """Forecasts ``[B, K, N]`` at the query times ``[B, K]``."""
        batch, n_vars, _ = summary.shape
        steps = y_time.shape[1]
        te = self.time_encoding(y_time.unsqueeze(-1))  # [B, K, T]
        inputs = torch.cat(
            [
                summary.unsqueeze(2).expand(batch, n_vars, steps, -1),
                te.unsqueeze(1).expand(batch, n_vars, steps, -1),
            ],
            dim=-1,
        )
        return self.decoder(inputs).squeeze(-1).transpose(1, 2)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forecasts the query times and, with targets, the masked MSE.

        Returns:
            A dict with ``forecast`` (``[B, K, N]``), ``logit`` (the same),
            and, when the targets are given, ``loss``, ``y_true`` and
            ``y_prob`` (the observed targets and their forecasts, flattened)
            and ``mask`` (``[B, K, N]``, 1 where a target was observed).
        """
        x = kwargs[self.value_key].to(self.device).float()
        x_time = kwargs[self.time_key].to(self.device).float()
        y_time = kwargs[self.query_key].to(self.device).float()
        forecast = self.decode(self.encode(x, x_time), y_time)
        out = {"forecast": forecast, "logit": forecast}
        if self.label_key in kwargs:
            y = kwargs[self.label_key].to(self.device).float()
            mask = torch.isfinite(y)
            out["loss"] = masked_mse(forecast, y)
            out["y_true"] = y[mask]
            out["y_prob"] = forecast[mask]
            out["mask"] = mask.float()
        else:
            out["y_prob"] = forecast
        return out

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The forecasts, scored as a regression with :func:`masked_mse`.

        The label is a ``"tensor"`` field, which is not a label processor, so
        APN supplies its own: the Trainer scores the observed targets as a
        regression.

        Examples:
            >>> model.label_processor.loss(pred, target)  # doctest: +SKIP
        """
        return {self.label_key: _ForecastLabel()}


class _ForecastLabel(RegressionLabelProcessor):
    """A regression over the observed (finite) targets of a forecast."""

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return masked_mse(logits, y_true)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the finite entries of ``target``.

    Args:
        prediction: Forecasts, any shape.
        target: Targets of the same shape, NaN where not observed.

    Returns:
        A scalar; 0 (with gradient) when nothing is observed.

    Examples:
        >>> import torch
        >>> masked_mse(torch.tensor([1.0, 2.0]), torch.tensor([0.0, float("nan")]))
        tensor(1.)
    """
    mask = torch.isfinite(target)
    if not mask.any():
        return prediction.sum() * 0.0
    return ((prediction[mask] - target[mask]) ** 2).mean()
