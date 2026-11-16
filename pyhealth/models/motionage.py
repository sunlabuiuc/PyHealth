"""MotionAge: mortality risk and biological age from wearable activity.

Paper: Song, Yin, et al., "MotionAge: A Deep Learning Framework for Biological
Age Prediction from Wearable Activity." https://openreview.net/pdf?id=xGw1o9acWa

:class:`MotionAge` encodes a wear-aware activity sequence with a GRU, pools it
over the worn epochs only, optionally fuses routine covariates, and predicts
the risk of death within the follow-up horizon. :class:`MotionAgeMapping`
turns those risks into MotionAge, the age at which a typical participant of
the same sex has that risk. The architecture and mapping follow the authors'
code (https://github.com/jackie-jiaqi-yin/MotionAge, MIT License).
"""

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel

_SEQUENCE_KEYS = ("intensity", "hour", "day", "wear_mask")
_FUSION_MODES = ("late", "residual")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Module:
    """Builds Linear-ReLU-Dropout-Linear, as in the authors' ``build_mlp``."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MotionAge(BaseModel):
    """GRU over wear-aware activity with masked pooling and covariate fusion.

    Each epoch's input is its normalized intensity concatenated with learned
    hour-of-day and day-of-week embeddings. A stacked GRU encodes the window,
    and its hidden states are averaged over the worn epochs only
    (``wear_mask == 1``), so non-wear time cannot dilute the summary. Without
    covariates, a linear layer maps the summary to a logit (MotionAge-F).
    With covariates (MotionAge-FA / -FRC), numeric covariates with their
    missing-value flags and embedded categorical codes go through an MLP, and
    then either:

    * ``fusion="late"``: an MLP on [activity summary, covariate summary]
      predicts the logit; or
    * ``fusion="residual"``: a linear activity-only logit plus an MLP
      correction from both summaries, whose last layer starts at zero so
      training begins from the activity-only prediction.

    Designed for :class:`~pyhealth.tasks.NHANESMortalityPrediction` samples:
    ``intensity`` ``[T, 1]`` (or ``[T]``), ``hour`` and ``day`` ``[T]``,
    ``wear_mask`` ``[T]``, and optionally ``covariates`` and
    ``covariate_codes``; the label must be binary. Defaults are the paper's
    hyperparameters.

    Args:
        dataset: The training :class:`~pyhealth.datasets.SampleDataset`.
        hidden_dim: GRU hidden size.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (ignored for one layer).
        hour_embedding_dim: Size of the hour-of-day embedding.
        day_embedding_dim: Size of the day-of-week embedding.
        category_sizes: Number of codes (including 0 for missing) of each
            entry of ``covariate_codes``, e.g. the task's ``category_sizes``.
            Required when the dataset has ``covariate_codes``.
        category_embedding_dim: Embedding size per categorical covariate.
        covariate_hidden_dim: Hidden size of the covariate MLP.
        covariate_output_dim: Output size of the covariate MLP.
        fusion_hidden_dim: Hidden size of the fusion MLP.
        fusion_dropout: Dropout in the covariate and fusion MLPs.
        fusion: ``"late"`` or ``"residual"``; used only with covariates.

    Raises:
        ValueError: If a required input or a binary label is missing, a size
            is not positive, ``fusion`` is unknown, or ``category_sizes`` does
            not match ``covariate_codes``.

    Examples:
        >>> import tempfile
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.datasets.nhanes import NHANESDataset
        >>> from pyhealth.datasets.nhanes_synthetic import generate_synthetic_nhanes
        >>> from pyhealth.models.motionage import MotionAge
        >>> from pyhealth.tasks.nhanes_mortality_prediction import (
        ...     NHANESMortalityPrediction,
        ... )
        >>> root = tempfile.mkdtemp()
        >>> _ = generate_synthetic_nhanes(  # doctest: +SKIP
        ...     root, n_participants=10, n_days=7
        ... )
        >>> task = NHANESMortalityPrediction()
        >>> samples = NHANESDataset(root=root).set_task(task)  # doctest: +SKIP
        >>> model = MotionAge(  # doctest: +SKIP
        ...     samples, category_sizes=task.category_sizes
        ... )
        >>> batch = next(iter(get_dataloader(samples, batch_size=4)))  # doctest: +SKIP
        >>> model(**batch)["y_prob"].shape  # doctest: +SKIP
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.117,
        hour_embedding_dim: int = 8,
        day_embedding_dim: int = 2,
        category_sizes: list[int] | None = None,
        category_embedding_dim: int = 4,
        covariate_hidden_dim: int = 64,
        covariate_output_dim: int = 32,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.1,
        fusion: str = "late",
    ) -> None:
        super().__init__(dataset)
        missing = [key for key in _SEQUENCE_KEYS if key not in self.feature_keys]
        if missing:
            raise ValueError(f"MotionAge needs inputs {missing} in the dataset")
        if len(self.label_keys) != 1 or self.mode != "binary":
            raise ValueError("MotionAge needs exactly one binary label")
        sizes = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "hour_embedding_dim": hour_embedding_dim,
            "day_embedding_dim": day_embedding_dim,
            "category_embedding_dim": category_embedding_dim,
            "covariate_hidden_dim": covariate_hidden_dim,
            "covariate_output_dim": covariate_output_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
        }
        for name, value in sizes.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if fusion not in _FUSION_MODES:
            raise ValueError(f"fusion must be one of {_FUSION_MODES}, got {fusion!r}")

        self.label_key = self.label_keys[0]
        self.fusion = fusion
        self.hour_embedding = nn.Embedding(24, hour_embedding_dim)
        self.day_embedding = nn.Embedding(7, day_embedding_dim)
        self.gru = nn.GRU(
            input_size=1 + hour_embedding_dim + day_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.has_covariates = "covariates" in self.feature_keys
        self.has_codes = "covariate_codes" in self.feature_keys
        self.category_sizes = list(category_sizes or [])
        if self.has_codes:
            n_codes = dataset.input_processors["covariate_codes"].size()
            if len(self.category_sizes) != n_codes or min(self.category_sizes) < 1:
                raise ValueError(
                    f"category_sizes must give a positive size for each of the "
                    f"{n_codes} covariate codes (e.g. task.category_sizes), "
                    f"got {category_sizes}"
                )
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(size, category_embedding_dim)
            for size in (self.category_sizes if self.has_codes else [])
        )
        static_dim = len(self.category_embeddings) * category_embedding_dim
        if self.has_covariates:
            static_dim += dataset.input_processors["covariates"].size()

        self.uses_static = static_dim > 0
        self.classifier = None
        if not self.uses_static or fusion == "residual":
            self.classifier = nn.Linear(hidden_dim, 1)
        if self.uses_static:
            self.covariate_encoder = _mlp(
                static_dim, covariate_hidden_dim, covariate_output_dim, fusion_dropout
            )
            self.fusion_mlp = _mlp(
                hidden_dim + covariate_output_dim, fusion_hidden_dim, 1, fusion_dropout
            )
            if fusion == "residual":
                nn.init.zeros_(self.fusion_mlp[-1].weight)
                nn.init.zeros_(self.fusion_mlp[-1].bias)

    def encode_activity(
        self,
        intensity: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        wear_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Averages the GRU hidden states over the worn epochs.

        Args:
            intensity: Normalized intensity, ``[B, T, 1]`` or ``[B, T]``.
            hour: Hour of day, ``[B, T]``, 0-23.
            day: Day of week, ``[B, T]``, 0-6.
            wear_mask: 1 for worn epochs, ``[B, T]``.

        Returns:
            The pooled representation, ``[B, hidden_dim]``; all zeros for a
            window with no worn epochs.
        """
        intensity = intensity.to(self.device, torch.float32)
        if intensity.dim() == 2:
            intensity = intensity.unsqueeze(-1)
        inputs = torch.cat(
            [
                intensity,
                self.hour_embedding(hour.to(self.device, torch.long)),
                self.day_embedding(day.to(self.device, torch.long)),
            ],
            dim=-1,
        )
        hidden, _ = self.gru(inputs)
        mask = wear_mask.to(self.device, hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Predicts mortality risk for a batch of windows.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`,
                with the inputs above and the label. ``embed=True`` also
                returns the pooled activity representation.

        Returns:
            ``loss``, ``y_prob`` and ``y_true`` (each ``[B, 1]`` apart from the
            scalar loss), ``logit``, and ``embed`` if requested.
        """
        pooled = self.encode_activity(
            kwargs["intensity"], kwargs["hour"], kwargs["day"], kwargs["wear_mask"]
        )
        if not self.uses_static:
            logit = self.classifier(pooled)
        else:
            parts = []
            if self.has_covariates:
                parts.append(kwargs["covariates"].to(self.device, torch.float32))
            if self.has_codes:
                codes = kwargs["covariate_codes"].to(self.device, torch.long)
                parts += [
                    embedding(codes[:, i])
                    for i, embedding in enumerate(self.category_embeddings)
                ]
            static = self.covariate_encoder(torch.cat(parts, dim=-1))
            logit = self.fusion_mlp(torch.cat([pooled, static], dim=-1))
            if self.fusion == "residual":
                logit = self.classifier(pooled) + logit

        y_true = kwargs[self.label_key].to(self.device, torch.float32)
        y_true = y_true.reshape(logit.shape)
        results = {
            "loss": self.get_loss_function()(logit, y_true),
            "y_prob": self.prepare_y_prob(logit),
            "y_true": y_true,
            "logit": logit,
        }
        if kwargs.get("embed", False):
            results["embed"] = pooled
        return results


class MotionAgeMapping:
    """Maps predicted mortality risk to MotionAge, a risk-equivalent age.

    For each sex, :meth:`fit` takes the median predicted risk of participants
    at each (rounded) age, converts it to log-odds and fits a straight line,
    ``logit(risk) = alpha + beta * age``, weighting each age by its number of
    participants. :meth:`transform` inverts the line:
    ``MotionAge = (logit(risk) - alpha) / beta``. Fit on training
    participants only; risks should first be averaged over each
    participant's windows. MotionAge acceleration is MotionAge minus age.

    Args:
        clip_eps: Risks are clipped to ``[clip_eps, 1 - clip_eps]`` before
            taking log-odds.
        weighted: Weight each age by its number of participants.
        min_participants: Minimum participants per sex to fit a line.

    Attributes:
        params: Maps each sex code to its fitted ``(alpha, beta)``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.models.motionage import MotionAgeMapping
        >>> age = np.repeat(np.arange(20, 80), 4).astype(float)
        >>> sex = np.tile([1, 2], age.size // 2)
        >>> risk = 1 / (1 + np.exp(-(-9 + 0.1 * age)))
        >>> mapping = MotionAgeMapping(min_participants=10).fit(risk, age, sex)
        >>> np.round(mapping.transform(risk[:3], sex[:3]), 1)
        array([20., 20., 20.])
    """

    def __init__(
        self,
        clip_eps: float = 1e-4,
        weighted: bool = True,
        min_participants: int = 100,
    ) -> None:
        if not 0.0 < clip_eps < 0.5:
            raise ValueError(f"clip_eps must be in (0, 0.5), got {clip_eps}")
        self.clip_eps = clip_eps
        self.weighted = weighted
        self.min_participants = min_participants
        self.params: dict[int, tuple[float, float]] = {}

    def _logit(self, risk: np.ndarray) -> np.ndarray:
        risk = np.clip(
            np.asarray(risk, dtype=np.float64), self.clip_eps, 1 - self.clip_eps
        )
        return np.log(risk / (1.0 - risk))

    def fit(
        self, risk: np.ndarray, age: np.ndarray, sex: np.ndarray
    ) -> "MotionAgeMapping":
        """Fits one line per sex from training participants.

        Args:
            risk: Predicted risk per participant.
            age: Chronological age per participant.
            sex: Sex code per participant.

        Returns:
            This mapping, fitted.

        Raises:
            ValueError: If a sex has too few participants or ages, or its
                fitted slope is not positive (risk must rise with age).
        """
        risk, age, sex = (np.asarray(x, dtype=np.float64) for x in (risk, age, sex))
        keep = ~(np.isnan(risk) | np.isnan(age) | np.isnan(sex))
        risk, age, sex = risk[keep], age[keep], sex[keep]
        bins = np.rint(age).astype(np.int64)
        self.params = {}
        for code in np.unique(sex):
            in_sex = sex == code
            if in_sex.sum() < self.min_participants:
                raise ValueError(
                    f"sex {code:g} has {int(in_sex.sum())} participants, fewer "
                    f"than min_participants={self.min_participants}"
                )
            ages = np.unique(bins[in_sex])
            if ages.size < 2:
                raise ValueError(f"sex {code:g} needs at least two distinct ages")
            median = np.array([np.median(risk[in_sex & (bins == a)]) for a in ages])
            counts = np.array([np.sum(in_sex & (bins == a)) for a in ages], float)
            weights = counts if self.weighted else np.ones_like(counts)
            alpha, beta = _weighted_line(
                ages.astype(float), self._logit(median), weights
            )
            if not beta > 0:
                raise ValueError(
                    f"sex {code:g}: fitted slope {beta:.4g} is not positive"
                )
            self.params[int(code)] = (alpha, beta)
        return self

    def transform(self, risk: np.ndarray, sex: np.ndarray) -> np.ndarray:
        """Converts risks to MotionAge.

        Args:
            risk: Predicted risk per participant.
            sex: Sex code per participant.

        Returns:
            MotionAge per participant; NaN for a sex the mapping was not
            fitted on.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.params:
            raise RuntimeError("MotionAgeMapping.transform called before fit")
        logit = self._logit(risk)
        sex = np.asarray(sex, dtype=np.float64)
        out = np.full(logit.shape, math.nan)
        for code, (alpha, beta) in self.params.items():
            rows = sex == code
            out[rows] = (logit[rows] - alpha) / beta
        return out


def _weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares intercept and slope of ``y = a + b * x``."""
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    slope = np.sum(w * (x - x_mean) * (y - y_mean)) / np.sum(w * (x - x_mean) ** 2)
    return float(y_mean - slope * x_mean), float(slope)
