# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: CNN-LSTM over irregular measurement tuples, trained either with
#   temporal-difference (TD) learning against a soft-updated target network or with
#   ordinary supervised targets. Written from scratch for PyHealth.
"""CNN-LSTM over measurement tuples with temporal-difference training."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..datasets import SampleDataset
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Column layout of a measurement tuple, as produced by
# :class:`~pyhealth.tasks.RealTimeICUMortalityMIMIC4`.
FEATURE, TIME, VALUE, DELTA_TIME, DELTA_VALUE = range(5)


class _TokenBatchNorm(nn.Module):
    """Batch norm over the valid (unmasked) vectors of a batch.

    Falls back to the running statistics when a training batch holds a single
    vector, where batch statistics are undefined.

    Args:
        num_features: Size of each vector.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalises ``x`` of shape ``(..., C)`` over its valid vectors.

        Args:
            x: Input of shape ``(..., C)``.
            mask: Optional boolean mask of shape ``x.shape[:-1]``; masked
                vectors are left as zeros.

        Returns:
            The normalised tensor, same shape as ``x``.
        """
        norm = self.norm
        flat = x.reshape(-1, x.shape[-1])
        weights = (
            mask.reshape(-1, 1).to(flat.dtype)
            if mask is not None
            else torch.ones_like(flat[:, :1])
        )
        # Masked sums instead of boolean indexing: no host-device sync.
        count = weights.sum()
        mean = (flat * weights).sum(0) / count.clamp(min=1.0)
        var = (((flat - mean) ** 2) * weights).sum(0) / count.clamp(min=1.0)
        if self.training:
            # A single vector has no batch statistics: use the running ones.
            use_batch = count >= 2
            with torch.no_grad():
                unbiased = var * count / (count - 1).clamp(min=1.0)
                momentum = norm.momentum
                new_mean = (1 - momentum) * norm.running_mean + momentum * mean
                new_var = (1 - momentum) * norm.running_var + momentum * unbiased
                norm.running_mean.copy_(
                    torch.where(use_batch, new_mean, norm.running_mean)
                )
                norm.running_var.copy_(
                    torch.where(use_batch, new_var, norm.running_var)
                )
                norm.num_batches_tracked.add_(1)
            mean = torch.where(use_batch, mean, norm.running_mean)
            var = torch.where(use_batch, var, norm.running_var)
        else:
            mean, var = norm.running_mean, norm.running_var
        out = (flat - mean) * torch.rsqrt(var + norm.eps) * norm.weight + norm.bias
        return (out * weights).reshape(x.shape)


class RealTimeTDCNNLSTMEncoder(nn.Module):
    """The CNN-LSTM value network of Frost et al. (2024).

    Each measurement tuple ``(t, v, f, dt, dv)`` is embedded by four small
    one-to-many MLPs (time, value, delta time, delta value) and a feature
    embedding; the five embeddings are summed (as in STraTS) and batch
    normalised. Convolution + max-pool blocks shorten the sequence, a
    multi-layer LSTM reads it oldest to newest, and an MLP head maps the
    LSTM's final hidden state to one logit. Padding is masked throughout, so
    a sequence's output does not depend on how much padding its batch has.

    Args:
        num_features: Number of feature ids (ids ``1..num_features``; 0 is
            padding and ``num_features + 1`` is used for unknown ids).
        embedding_dim: Size of the tuple embedding. Defaults to 32.
        lstm_hidden_dim: LSTM hidden size. Defaults to ``8 * embedding_dim``.
        num_cnn_layers: Number of convolution + max-pool blocks. Defaults to 2.
        num_lstm_layers: Number of LSTM layers. Defaults to 2.
        dropout: Dropout between LSTM layers. Defaults to 0.5.

    Examples:
        >>> import torch
        >>> from pyhealth.models import RealTimeTDCNNLSTMEncoder
        >>> encoder = RealTimeTDCNNLSTMEncoder(num_features=10, embedding_dim=8)
        >>> ids = torch.randint(1, 11, (2, 6))
        >>> x = torch.randn(2, 6)
        >>> mask = torch.ones(2, 6, dtype=torch.bool)
        >>> encoder(ids, x, x, x, x, mask).shape
        torch.Size([2])
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 32,
        lstm_hidden_dim: int | None = None,
        num_cnn_layers: int = 2,
        num_lstm_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim or 8 * embedding_dim
        self.num_cnn_layers = num_cnn_layers
        reduced = max(int(math.sqrt(embedding_dim)), 1)

        def scalar_mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(1, reduced), nn.ReLU(), nn.Linear(reduced, embedding_dim)
            )

        self.time_embedding = scalar_mlp()
        self.value_embedding = scalar_mlp()
        self.delta_time_embedding = scalar_mlp()
        self.delta_value_embedding = scalar_mlp()
        self.feature_embedding = nn.Embedding(
            num_features + 2, embedding_dim, padding_idx=0
        )
        self.embedding_norm = _TokenBatchNorm(embedding_dim)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2, padding=1)
            for _ in range(num_cnn_layers)
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_norm1 = _TokenBatchNorm(self.lstm_hidden_dim)
        self.head_fc1 = nn.Linear(self.lstm_hidden_dim, embedding_dim)
        self.head_norm2 = _TokenBatchNorm(embedding_dim)
        self.head_fc2 = nn.Linear(embedding_dim, 1)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-normal initialisation of all weight matrices (as the authors)."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.feature_embedding.weight[0].zero_()

    @staticmethod
    def _zero_beyond(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Zeros positions ``>= lengths`` of ``h`` with shape ``(B, C, L)``."""
        positions = torch.arange(h.shape[-1], device=h.device)
        keep = positions[None, :] < lengths[:, None]
        return h * keep[:, None, :].to(h.dtype)

    def forward(
        self,
        feature_ids: torch.Tensor,
        times: torch.Tensor,
        values: torch.Tensor,
        delta_times: torch.Tensor,
        delta_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes one logit per sequence.

        Args:
            feature_ids: Long tensor ``(B, L)``; 0 marks padding.
            times: Standardised times ``(B, L)``.
            values: Standardised values ``(B, L)``.
            delta_times: Standardised delta times ``(B, L)``, NaN if missing.
            delta_values: Standardised delta values ``(B, L)``, NaN if missing.
            mask: Boolean ``(B, L)``, True for real measurements. Real
                measurements must come first in each row.

        Returns:
            Logits of shape ``(B,)``.
        """
        delta_missing = torch.isnan(delta_times) | torch.isnan(delta_values)
        delta_times = torch.nan_to_num(delta_times, nan=0.0)
        delta_values = torch.nan_to_num(delta_values, nan=0.0)
        keep_delta = (~delta_missing).unsqueeze(-1).to(values.dtype)
        embedded = (
            self.time_embedding(times.unsqueeze(-1))
            + self.value_embedding(values.unsqueeze(-1))
            + self.feature_embedding(feature_ids)
            + keep_delta * self.delta_time_embedding(delta_times.unsqueeze(-1))
            + keep_delta * self.delta_value_embedding(delta_values.unsqueeze(-1))
        )
        embedded = self.embedding_norm(embedded, mask)  # (B, L, D), pads = 0

        lengths = mask.sum(dim=1)
        h = embedded.transpose(1, 2)  # (B, D, L)
        for conv in self.convs:
            h = self.relu(conv(h))  # length L + 1
            lengths = lengths + 1
            h = self._zero_beyond(h, lengths)
            if h.shape[-1] >= 2:
                h = F.max_pool1d(h, kernel_size=2, stride=2)
                lengths = torch.clamp(lengths // 2, min=1)
            h = self._zero_beyond(h, lengths)

        packed = pack_padded_sequence(
            h.transpose(1, 2),
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        z = self.head_norm1(hidden[-1])
        z = self.relu(self.head_fc1(z))
        z = self.head_norm2(z)
        return self.head_fc2(z).squeeze(-1)


class RealTimeTDCNNLSTM(BaseModel):
    """CNN-LSTM mortality model trained with temporal-difference learning.

    Implements the method of Frost, Li & Harris, "Robust Real-Time Mortality
    Prediction in the Intensive Care Unit using Temporal Difference
    Learning" (ML4H 2024). A value network :math:`V(O_t)` reads the
    irregular measurement tuples of an observation and outputs a mortality
    risk. With ``training_target="td"`` it is trained on the semi-Markov TD
    target

    .. math::

        V(O_t) \\leftarrow R \\text{ if } O_t \\text{ is terminal, else }
        V'(O_{t+k}),

    where :math:`V'` is a frozen *target network* soft-updated after every
    optimiser step, :math:`\\theta' \\leftarrow (1-\\tau)\\theta' +
    \\tau\\theta`, and :math:`R` is the terminal reward (28-day mortality).
    With ``training_target="supervised"`` the same network is trained on the
    dataset's binary label (the paper's baselines), optionally with
    class-balanced cross-entropy.

    The soft update happens at the start of every training ``forward`` call
    (for the previous optimiser step), so the model trains with the stock
    :class:`~pyhealth.trainer.Trainer`, which calls ``forward`` exactly once
    per optimiser step and offers no post-step hook.

    :class:`~pyhealth.models.TDCNNLSTM` is an independent implementation of
    the same paper's network. It expects separate event channels and a
    custom training loop that calls ``update_target()``. This class instead
    reads the tuple samples of the ``RealTimeICUMortality*`` tasks, trains
    with the stock ``Trainer``, and also covers the paper's class-balanced
    supervised baselines and λ-mixed targets.

    Per-feature standardisation (values, delta values and delta times) and
    global time standardisation use statistics computed from ``dataset``;
    values are first clipped to per-feature quantiles to tame outliers. Pass
    the *training* split as ``dataset`` so no statistics leak from
    validation or test data.

    The model expects the fields produced by
    :class:`~pyhealth.tasks.RealTimeICUMortalityMIMIC4`: ``measurements``
    and ``next_measurements`` (``(L, 5)`` tuples ``[feature_id, minutes
    before, value, delta minutes, delta value]``), ``terminal``,
    ``terminal_reward`` and one binary label.

    Args:
        dataset: The (training) :class:`~pyhealth.datasets.SampleDataset`.
        embedding_dim: Tuple embedding size. Defaults to 32 (the paper).
        lstm_hidden_dim: LSTM hidden size. Defaults to ``8 * embedding_dim``.
        num_cnn_layers: Number of convolution + max-pool blocks. Defaults
            to 2.
        num_lstm_layers: Number of LSTM layers. Defaults to 2.
        dropout: Dropout between LSTM layers. Defaults to 0.5.
        training_target: ``"td"`` or ``"supervised"``. Defaults to ``"td"``.
        target_update_rate: Soft-update rate :math:`\\tau` (the paper's
            :math:`1-\\alpha`). Defaults to 0.01.
        td_lambda: Mixes the supervised label into the TD target,
            ``y = td_lambda * label + (1 - td_lambda) * td_target``; 0 is pure
            TD and 1 is pure supervision. Defaults to 0.0.
        class_balanced: Weight the supervised loss by normalised inverse
            class frequency. Defaults to False.
        target_network_train_mode: Keep the target network in training mode
            (dropout and batch statistics active), as in the authors' code.
            Defaults to True.
        num_features: Number of feature ids. Defaults to the largest id seen
            in the statistics sample.
        measurements_key: Field holding the current observation.
        next_measurements_key: Field holding the next-state observation.
        terminal_key: Field holding the terminal flag.
        terminal_reward_key: Field holding the terminal reward.
        label_horizon_index: If set, the label used for training, ``y_true``
            and model selection is column ``label_horizon_index`` of
            ``horizons_key`` instead of the binary label field, so one sample
            cache serves supervised baselines at every horizon. Defaults to
            ``None``.
        horizons_key: Field holding the multi-horizon labels. Defaults to
            ``"mortality_horizons"``.
        normalization_samples: Number of samples used to compute the
            statistics, or ``None`` for all samples. Defaults to 20000.
            Feature ids (or deltas) never seen in these samples fall back to
            the bounded transform ``sign(x) * log1p(|x|)``, and a warning is
            logged.
        clip_quantiles: Per-feature value quantiles used for clipping.
            Delta values are recomputed from the clipped current and previous
            values, so outliers do not leak into them. Defaults to
            ``(0.001, 0.999)``.
        seed: Seed for choosing the statistics sample. Defaults to 0.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import RealTimeTDCNNLSTM
        >>> obs = [[1, 60.0, 4.2, float("nan"), float("nan")], [2, 0.0, 1.1, 5.0, 0.2]]
        >>> samples = [
        ...     {"patient_id": f"p{i}", "measurements": obs,
        ...      "next_measurements": obs, "terminal": float(i % 2),
        ...      "terminal_reward": float(i % 2), "mortality": i % 2}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={"measurements": "tensor",
        ...                   "next_measurements": "tensor"},
        ...     output_schema={"mortality": "binary", "terminal": "tensor",
        ...                    "terminal_reward": "tensor"},
        ... )
        >>> model = RealTimeTDCNNLSTM(dataset, embedding_dim=4)
        >>> model.training_target
        'td'
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 32,
        lstm_hidden_dim: int | None = None,
        num_cnn_layers: int = 2,
        num_lstm_layers: int = 2,
        dropout: float = 0.5,
        training_target: str = "td",
        target_update_rate: float = 0.01,
        td_lambda: float = 0.0,
        class_balanced: bool = False,
        target_network_train_mode: bool = True,
        num_features: int | None = None,
        measurements_key: str = "measurements",
        next_measurements_key: str = "next_measurements",
        terminal_key: str = "terminal",
        terminal_reward_key: str = "terminal_reward",
        label_horizon_index: int | None = None,
        horizons_key: str = "mortality_horizons",
        normalization_samples: int | None = 20000,
        clip_quantiles: tuple[float, float] = (0.001, 0.999),
        seed: int = 0,
    ) -> None:
        super().__init__(dataset=dataset)
        if training_target not in ("td", "supervised"):
            raise ValueError("training_target must be 'td' or 'supervised'")
        if not 0.0 < target_update_rate <= 1.0:
            raise ValueError("target_update_rate must be in (0, 1]")
        if not 0.0 <= td_lambda <= 1.0:
            raise ValueError("td_lambda must be in [0, 1]")
        if self.mode not in (None, "binary"):
            raise ValueError("RealTimeTDCNNLSTM needs exactly one binary label")
        self.training_target = training_target
        self.target_update_rate = float(target_update_rate)
        self.td_lambda = float(td_lambda)
        self.class_balanced = class_balanced
        self.target_network_train_mode = target_network_train_mode
        self.measurements_key = measurements_key
        self.next_measurements_key = next_measurements_key
        self.terminal_key = terminal_key
        self.terminal_reward_key = terminal_reward_key
        self.label_horizon_index = label_horizon_index
        self.horizons_key = horizons_key
        self.label_key = next(iter(self.label_processors), None)
        if self.label_key is None:
            raise ValueError("RealTimeTDCNNLSTM needs a binary label in output_schema")

        stats, labels, max_id = self._collect_statistics(
            dataset, normalization_samples, clip_quantiles, seed
        )
        self.num_features = int(num_features or max_id)
        self._register_statistics(stats)
        self.register_buffer("class_weights", self._class_weights(labels))

        self.encoder = RealTimeTDCNNLSTMEncoder(
            num_features=self.num_features,
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_cnn_layers=num_cnn_layers,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )
        self.target_encoder: RealTimeTDCNNLSTMEncoder | None = None
        if training_target == "td":
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad_(False)
        self._pending_target_update = False
        self.num_target_updates = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _collect_statistics(
        self,
        dataset: SampleDataset,
        max_samples: int | None,
        clip_quantiles: tuple[float, float],
        seed: int,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
        """Computes standardisation statistics from a sample of the dataset.

        Args:
            dataset: Dataset whose samples hold ``measurements_key``.
            max_samples: Maximum number of samples to read (``None``: all).
            clip_quantiles: Lower and upper per-feature value quantiles.
            seed: Seed for choosing the samples.

        Returns:
            ``(stats, labels, max_feature_id)`` where ``stats`` maps buffer
            names to arrays indexed by feature id and ``labels`` holds the
            binary labels of the sampled rows.
        """
        n = len(dataset) if dataset is not None else 0
        rng = np.random.default_rng(seed)
        size_ = n if max_samples is None else min(n, max_samples)
        picked = np.sort(rng.choice(n, size=size_, replace=False)) if n else []
        rows, labels = [], []
        # Iterating a subset streams the disk cache (tens of times faster than
        # random access by index); fall back to indexing for other datasets.
        if len(picked) and hasattr(dataset, "subset"):
            samples = iter(dataset.subset([int(i) for i in picked]))
        else:
            samples = (dataset[int(i)] for i in picked)
        for sample in samples:
            x = torch.as_tensor(sample[self.measurements_key]).reshape(-1, 5)
            rows.append(x[x[:, FEATURE] > 0].numpy())
            if self.label_horizon_index is not None and self.horizons_key in sample:
                horizons = torch.as_tensor(sample[self.horizons_key]).reshape(-1)
                labels.append(float(horizons[self.label_horizon_index]))
            elif self.label_key in sample:
                labels.append(
                    float(torch.as_tensor(sample[self.label_key]).reshape(-1)[0])
                )
        data = np.concatenate(rows, axis=0) if rows else np.zeros((0, 5))
        max_id = int(data[:, FEATURE].max()) if len(data) else 1
        size = max_id + 2
        stats = {
            "value_low": np.full(size, -np.inf),
            "value_high": np.full(size, np.inf),
            "value_mean": np.zeros(size),
            "value_std": np.ones(size),
            "delta_time_mean": np.zeros(size),
            "delta_time_std": np.ones(size),
            "delta_value_mean": np.zeros(size),
            "delta_value_std": np.ones(size),
            "time_mean": np.zeros(1),
            "time_std": np.ones(1),
            "has_value_stats": np.zeros(size),
            "has_delta_stats": np.zeros(size),
        }
        if len(data):
            stats["time_mean"][0] = data[:, TIME].mean()
            stats["time_std"][0] = data[:, TIME].std() or 1.0
            ids = data[:, FEATURE].astype(np.int64)
            for f in np.unique(ids):
                rows_f = data[ids == f]
                low, high = np.quantile(rows_f[:, VALUE], clip_quantiles)
                clipped = np.clip(rows_f[:, VALUE], low, high)
                stats["value_low"][f], stats["value_high"][f] = low, high
                stats["value_mean"][f] = clipped.mean()
                stats["value_std"][f] = clipped.std() or 1.0
                stats["has_value_stats"][f] = 1.0
                finite = np.isfinite(rows_f[:, DELTA_TIME]) & np.isfinite(
                    rows_f[:, DELTA_VALUE]
                )
                if finite.any():
                    rows_d = rows_f[finite]
                    # Deltas between clipped values: "clean, then difference".
                    previous = np.clip(
                        rows_d[:, VALUE] - rows_d[:, DELTA_VALUE], low, high
                    )
                    delta_v = np.clip(rows_d[:, VALUE], low, high) - previous
                    delta_t = rows_d[:, DELTA_TIME]
                    stats["delta_value_mean"][f] = delta_v.mean()
                    stats["delta_value_std"][f] = delta_v.std() or 1.0
                    stats["delta_time_mean"][f] = delta_t.mean()
                    stats["delta_time_std"][f] = delta_t.std() or 1.0
                    stats["has_delta_stats"][f] = 1.0
        return stats, np.asarray(labels), max_id

    def _register_statistics(self, stats: dict[str, np.ndarray]) -> None:
        """Registers the statistics as buffers sized ``num_features + 2``.

        Feature ids without statistics (e.g. unseen in the sample) are marked
        in ``has_value_stats`` / ``has_delta_stats`` and use the bounded
        fallback transform in :meth:`standardize`.
        """
        size = self.num_features + 2
        defaults = {"value_low": -np.inf, "value_high": np.inf}
        for name, array in stats.items():
            if not name.startswith("time_"):
                default = defaults.get(name, 1.0 if name.endswith("std") else 0.0)
                if name.startswith("has_"):
                    default = 0.0
                full = np.full(size, default, dtype=np.float64)
                n = min(size, array.shape[0])
                full[:n] = array[:n]
                array = full
            self.register_buffer(name, torch.tensor(array, dtype=torch.float32))
        missing = int((self.has_value_stats[1 : self.num_features + 1] == 0).sum())
        if missing:
            logger.warning(
                "RealTimeTDCNNLSTM: %d of %d feature ids have no normalisation statistics "
                "(not in the statistics sample); they use sign(x)*log1p(|x|).",
                missing,
                self.num_features,
            )

    @staticmethod
    def _class_weights(labels: np.ndarray) -> torch.Tensor:
        """Normalised inverse class frequencies ``[w_negative, w_positive]``."""
        counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=np.float64)
        if counts.min() == 0:
            return torch.ones(2) / 2
        inverse = 1.0 / counts
        return torch.tensor(inverse / inverse.sum(), dtype=torch.float32)

    def standardize(self, measurements: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Splits and standardises a batch of measurement tuples.

        Args:
            measurements: Float tensor ``(B, L, 5)`` of ``[feature_id,
                minutes_before, value, delta_minutes, delta_value]`` rows,
                zero-padded.

        Returns:
            ``(feature_ids, times, values, delta_times, delta_values, mask)``
            ready for :class:`RealTimeTDCNNLSTMEncoder`.
        """
        x = measurements.to(self.device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        ids = torch.nan_to_num(x[..., FEATURE], nan=0.0).round().long()
        unknown = self.num_features + 1
        ids = torch.where(ids > self.num_features, torch.full_like(ids, unknown), ids)
        ids = ids.clamp(min=0)
        mask = ids > 0
        low, high = self.value_low[ids], self.value_high[ids]
        raw_values, raw_deltas = x[..., VALUE], x[..., DELTA_VALUE]
        clipped = torch.maximum(torch.minimum(raw_values, high), low)
        # Recompute the delta from clipped values so outliers cannot leak into it.
        previous = torch.maximum(torch.minimum(raw_values - raw_deltas, high), low)
        has_value = self.has_value_stats[ids] > 0
        has_delta = self.has_delta_stats[ids] > 0
        values = torch.where(
            has_value,
            (clipped - self.value_mean[ids]) / self.value_std[ids],
            self._fallback(raw_values),
        )
        times = (x[..., TIME] - self.time_mean) / self.time_std
        delta_times = torch.where(
            has_delta,
            (x[..., DELTA_TIME] - self.delta_time_mean[ids]) / self.delta_time_std[ids],
            self._fallback(x[..., DELTA_TIME]),
        )
        delta_values = torch.where(
            has_delta,
            (clipped - previous - self.delta_value_mean[ids])
            / self.delta_value_std[ids],
            self._fallback(raw_deltas),
        )
        values = torch.where(
            mask, torch.nan_to_num(values, nan=0.0), torch.zeros_like(values)
        )
        times = torch.where(mask, times, torch.zeros_like(times))
        return ids, times, values, delta_times, delta_values, mask

    @staticmethod
    def _fallback(x: torch.Tensor) -> torch.Tensor:
        """Bounded transform for inputs without statistics (NaN preserved)."""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> RealTimeTDCNNLSTM:
        """Sets the mode; the target network keeps its configured mode.

        With ``target_network_train_mode`` the target network always runs in
        training mode (dropout and batch statistics), as in the authors'
        code; otherwise it always runs in evaluation mode.
        """
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.train(self.target_network_train_mode)
        return self

    @torch.no_grad()
    def update_target_network(self) -> None:
        """Soft update ``theta' <- (1 - tau) * theta' + tau * theta``."""
        if self.target_encoder is None:
            return
        tau = self.target_update_rate
        online = dict(self.encoder.named_parameters())
        online.update(dict(self.encoder.named_buffers()))
        for name, tensor in list(self.target_encoder.named_parameters()) + list(
            self.target_encoder.named_buffers()
        ):
            if tensor.is_floating_point():
                tensor.mul_(1.0 - tau).add_(online[name], alpha=tau)
        self.num_target_updates += 1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Computes risks and the TD or supervised loss for a batch.

        Args:
            **kwargs: A collated batch with the measurement fields, the
                binary label and (for TD) ``terminal`` and
                ``terminal_reward``. Extra fields such as
                ``mortality_horizons`` are passed through to the output.

        Returns:
            A dict with ``loss``, ``y_prob``, ``y_true`` and ``logit`` (all
            ``(B, 1)`` except the scalar loss), plus pass-through fields. In
            evaluation mode a TD model reports the binary cross-entropy against
            the label as ``loss`` (the target network is only used for
            training).
        """
        if (
            self.training
            and self.target_encoder is not None
            and torch.is_grad_enabled()
        ):
            # The Trainer calls forward once per optimiser step, so this applies
            # the update owed for the previous step before computing targets.
            if self._pending_target_update:
                self.update_target_network()
            self._pending_target_update = True

        logit = self.encoder(*self.standardize(kwargs[self.measurements_key]))
        logit = logit.unsqueeze(-1)
        if self.label_horizon_index is not None:
            horizons = kwargs[self.horizons_key].to(self.device).float()
            y_true = horizons[:, self.label_horizon_index].reshape(-1, 1)
        else:
            y_true = kwargs[self.label_key].to(self.device).float().reshape(-1, 1)

        if self.training_target == "td" and self.training:
            target = self.td_target(kwargs).reshape(-1, 1)
            if self.td_lambda > 0:
                target = self.td_lambda * y_true + (1.0 - self.td_lambda) * target
            loss = F.binary_cross_entropy_with_logits(logit, target)
        elif self.training_target == "td":
            # Evaluation: the TD target network is not run (it would cost a second
            # pass and touch its batch-norm buffers); report BCE on the label.
            loss = F.binary_cross_entropy_with_logits(logit, y_true)
        else:
            weights = None
            if self.class_balanced:
                weights = self.class_weights[y_true.long()]
            loss = F.binary_cross_entropy_with_logits(logit, y_true, weight=weights)

        output = {
            "loss": loss,
            "y_prob": torch.sigmoid(logit),
            "y_true": y_true,
            "logit": logit,
        }
        for key, value in kwargs.items():
            if (
                key not in output
                and key not in self.feature_keys
                and torch.is_tensor(value)
            ):
                output[key] = value
        return output

    @torch.no_grad()
    def td_target(self, batch: dict[str, Any]) -> torch.Tensor:
        """Computes the TD target ``R`` (terminal) or ``V'(O_{t+k})``.

        Args:
            batch: A collated batch with the next-state measurements,
                ``terminal`` and ``terminal_reward``.

        Returns:
            Targets in ``[0, 1]`` of shape ``(B,)``.
        """
        terminal = batch[self.terminal_key].to(self.device).float().reshape(-1)
        reward = batch[self.terminal_reward_key].to(self.device).float().reshape(-1)
        encoder = (
            self.target_encoder if self.target_encoder is not None else self.encoder
        )
        next_value = torch.sigmoid(
            encoder(*self.standardize(batch[self.next_measurements_key]))
        )
        return torch.where(terminal > 0.5, reward, next_value)

    # ------------------------------------------------------------------
    # Optimiser settings from the paper
    # ------------------------------------------------------------------
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters of the value network."""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def paper_optimizer_params(self, batches_per_epoch: int) -> dict[str, float]:
        """AdamW settings of the paper (Appendix D / authors' code).

        Args:
            batches_per_epoch: Number of training batches per epoch.

        Returns:
            ``{"lr": 1 / sqrt(n_params), "weight_decay": 1 / (lr *
            batches_per_epoch)}``. Pass ``lr`` in ``optimizer_params`` and
            ``weight_decay`` as the Trainer's ``weight_decay`` argument.

        Note:
            With decoupled weight decay this shrinks the weights by about a
            factor of e per epoch, whatever the data size, so the decay grows as
            the training set shrinks. On PhysioNet 2012 it gives 0.17 for the
            full training set and 0.33-1.66 for 50-10% subsets, where every TD
            and 28-day supervised seed failed to train. Labs-only models (0.93)
            trained, but TD and the 28-day model lost 0.01-0.04 AUROC. When
            comparing training-set sizes or feature sets, compute it once on the
            largest set and hold it fixed.
        """
        lr = 1.0 / math.sqrt(self.num_trainable_parameters())
        return {"lr": lr, "weight_decay": 1.0 / (lr * max(batches_per_epoch, 1))}
