"""GRAFT survival ranking with separate, training-only Cox calibration.

Method: Ashhad, Hoehndorf and Henao, arXiv:2602.07884v2 (2026).
This implementation uses training-only preprocessing, average target ranks,
explicit L2, step-KM finite-tail sampling and validation checkpoint selection.
These choices differ from the supplied author's executable experiment.
Optional torchsort, lifelines and scikit-survival imports occur at fitting time.

Architecture adapted from https://github.com/anonymous-785-u/GRAFT,
revision 49d9d747869fd9410bf329baa57289a6497f9d60. Original notice:

MIT License
Copyright (c) 2026 anonymous-785-u

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributor: Taejun Oh (NetID: taejuno2).
Paper: GRAFT: Decoupling Ranking and Calibration for Survival Analysis.
Paper link: https://arxiv.org/abs/2602.07884v2.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import torch
from torch import nn

from .base_model import BaseModel


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _outcomes(times: Any, events: Any) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(_numpy(times), dtype=float).reshape(-1)
    events = np.asarray(_numpy(events), dtype=float).reshape(-1)
    if not len(times) or times.shape != events.shape:
        raise ValueError("Duration and event must be nonempty aligned vectors.")
    if not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("Durations must be finite and strictly positive.")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("Event must be 0=censored or 1=observed event.")
    return times, events.astype(bool)


def _fingerprint(x: np.ndarray, t: np.ndarray, e: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (x, t, e):
        values = np.asarray(values, dtype="<f8")
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


class LocalKMImputer:
    """Cache conditional event-time distributions from training neighbors only.

    Self is excluded. Ties in Euclidean distance use ``row_ids``. Positive
    event masses strictly after censoring are renormalized over represented
    finite events. This does not identify the residual tail. ``tail_policy``
    is either ``"fallback"`` (1.1 times censoring time) or ``"raise"``.
    Insufficient observed events always raise, independently of tail policy.

    Args:
        min_neighbors: Minimum neighborhood size, bounded by available subjects.
        min_events: Required observed events in each censored neighborhood.
        max_neighbors: Maximum nonself neighbors.
        tail_policy: Explicit action when no represented conditional tail exists.

    Examples:
        >>> from pyhealth.models.graft import LocalKMImputer
        >>> imputer = LocalKMImputer(min_neighbors=2, min_events=1)
        >>> imputer.tail_policy
        'fallback'
    """

    def __init__(
        self,
        min_neighbors: int = 30,
        min_events: int = 10,
        max_neighbors: int = 199,
        tail_policy: str = "fallback",
    ) -> None:
        if not 1 <= min_events <= min_neighbors <= max_neighbors:
            raise ValueError(
                "Require 1 <= min_events <= min_neighbors <= max_neighbors."
            )
        if tail_policy not in {"fallback", "raise"}:
            raise ValueError("tail_policy must be 'fallback' or 'raise'.")
        self.min_neighbors = min_neighbors
        self.min_events = min_events
        self.max_neighbors = max_neighbors
        self.tail_policy = tail_policy

    def fit(
        self,
        x: np.ndarray,
        times: Any,
        events: Any,
        row_ids: Sequence[int] | None = None,
    ) -> "LocalKMImputer":
        """Fit on standardized ungated training features and original outcomes.

        Args:
            x: Finite training matrix with shape (subjects, features). The
                caller fits its scaling on this training partition only.
            times: Original strictly positive follow-up durations.
            events: Aligned indicators, 1=observed event and 0=censored.
            row_ids: Unique source-row IDs for deterministic distance ties.
                Defaults to the supplied row positions.

        Returns:
            This fitted imputer. No held-out features or outcomes are needed.

        Raises:
            ValueError: For invalid inputs, insufficient local events, or an
                empty conditional tail when tail_policy is ``"raise"``.
        """
        from scipy.spatial.distance import cdist

        self.times_, self.events_ = _outcomes(times, events)
        x = np.asarray(x, dtype=float)
        n = len(self.times_)
        if x.ndim != 2 or len(x) != n or not np.isfinite(x).all():
            raise ValueError("Local KM requires a finite aligned feature matrix.")
        ids = np.arange(n) if row_ids is None else np.asarray(row_ids)
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("row_ids must be unique and aligned with training rows.")
        self.distributions_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        residuals, sizes, fallback_rows = [], [], []
        censored = np.flatnonzero(~self.events_)
        # Bound temporary distance memory; include all rows before resolving ties.
        for start in range(0, len(censored), 64):
            rows = censored[start : start + 64]
            distances = cdist(x[rows], x, metric="euclidean")
            for local, i in enumerate(rows):
                order = np.lexsort((ids, distances[local]))
                order = order[order != i][: self.max_neighbors]
                counts = np.cumsum(self.events_[order])
                candidates = np.flatnonzero(
                    (counts >= self.min_events)
                    & (np.arange(len(order)) + 1 >= min(self.min_neighbors, n - 1))
                )
                if not len(candidates):
                    raise ValueError(
                        f"Insufficient local events for training row {ids[i]}: "
                        f"need {self.min_events} within {len(order)} nonself neighbors."
                    )
                neighbors = order[: candidates[0] + 1]
                sizes.append(len(neighbors))
                nt, ne = self.times_[neighbors], self.events_[neighbors]
                knots = np.unique(nt)
                risk = len(nt)
                survival, masses, values = 1.0, [], []
                survival_at_censor = 1.0
                for knot in knots:
                    at_time = nt == knot
                    deaths = int(ne[at_time].sum())
                    mass = survival * deaths / risk
                    survival -= mass
                    if knot <= self.times_[i]:
                        survival_at_censor = survival
                    elif mass > 0:
                        values.append(float(knot))
                        masses.append(float(mass))
                    risk -= int(at_time.sum())
                residuals.append(
                    float(survival / survival_at_censor)
                    if survival_at_censor > 0
                    else None
                )
                if survival_at_censor <= 0 or not masses:
                    if self.tail_policy == "raise":
                        raise ValueError(
                            f"Empty conditional KM tail at training row {ids[i]}."
                        )
                    fallback = self.times_[i] * 1.1
                    if not np.isfinite(fallback) or fallback <= self.times_[i]:
                        raise ValueError(
                            "Cannot represent a finite time above censoring."
                        )
                    values, masses = [fallback], [1.0]
                    fallback_rows.append(int(i))
                cdf = np.cumsum(masses) / np.sum(masses)
                cdf[-1] = 1.0
                self.distributions_[int(i)] = (np.asarray(values), cdf)
        self.diagnostics_ = {
            "training_subjects": n,
            "censored_subjects": len(censored),
            "self_excluded": True,
            "tail_policy": self.tail_policy,
            "finite_tail_renormalized": True,
            "fallback_subjects": len(fallback_rows),
            "fallback_fraction_of_censored": len(fallback_rows) / max(1, len(censored)),
            "fallback_row_positions": fallback_rows,
            "conditional_residual_mass": residuals,
            "neighborhood_sizes": sizes,
        }
        return self

    def sample(self, uniforms: np.ndarray) -> np.ndarray:
        """Return log-times; observed subjects retain their observed log-time.

        Args:
            uniforms: Explicit uniform draws in [0, 1), shaped
                (training_subjects, draws). Controlling these draws controls
                imputation randomness independently of model gates/dropout.

        Returns:
            An array of the same shape containing imputed log-event times.

        Raises:
            RuntimeError: If the training distributions have not been fitted.
            ValueError: If uniform values or their shape are invalid.
        """
        if not hasattr(self, "distributions_"):
            raise RuntimeError("Fit LocalKMImputer before sampling.")
        uniforms = np.asarray(uniforms, dtype=float)
        if uniforms.ndim != 2 or uniforms.shape[0] != len(self.times_):
            raise ValueError("Uniforms must have shape (training_subjects, draws).")
        if not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms >= 1)):
            raise ValueError("Imputation uniforms must lie in [0, 1).")
        targets = np.repeat(self.times_[:, None], uniforms.shape[1], axis=1)
        for row, (values, cdf) in self.distributions_.items():
            targets[row] = values[np.searchsorted(cdf, uniforms[row], side="right")]
        return np.log(targets)


class GRAFT(BaseModel):
    """Gated residual survival ranking followed by one-dimensional Cox fitting.

    ``features`` are dense raw numeric vectors. ``duration`` and ``event``
    are separate numerical labels, not classification labels. ``fit`` owns
    preprocessing, training-only KM caches, Adam and validation selection.
    ``forward`` supplies loss only for explicitly supplied imputed log-targets;
    passing observed held-out labels never triggers imputation or fitting.
    Larger raw scores indicate longer survival; ``risk_score`` negates them.

    Args:
        dataset: PyHealth SampleDataset supplying the input feature width.
            For inference reload, omit it and supply ``n_features`` instead.
        n_features: Dense input width, inferred from dataset when omitted.
        feature_key: Numeric input field.
        duration_key: Positive follow-up duration field.
        event_key: Event indicator field, 0=censored, 1=observed.
        hidden_dim: Residual hidden width.
        sigma: Gaussian gate noise standard deviation.
        dropout: Residual hidden dropout probability.
        gate_l0: Expected-active-gate penalty coefficient.
        l2: Explicit L2 coefficient; optimizer weight decay is zero.
        rank_tau: torchsort L2 soft-rank regularization strength.
        mc_draws: Conditional target draws per subject per epoch.
        use_gates: Include stochastic feature gates.
        use_residual: Include residual network; false with no gates is linear-only.
        min_neighbors: Minimum local neighborhood size.
        min_events: Minimum local observed-event count.
        max_neighbors: Maximum nonself neighbors.
        tail_policy: ``"fallback"`` or strict ``"raise"`` for empty KM tails.

    Examples:
        >>> from pyhealth.models import GRAFT
        >>> model = GRAFT(n_features=14)
        >>> model.fit_outside_trainer, model.mode
        (True, None)
        >>> # model.fit(train_samples, validation_samples)
        >>> # model.fit_calibration(train_samples)
        >>> # curves = model.predict_survival(raw_features, [10., 30.])
    """

    fit_outside_trainer = True

    @property
    def mode(self) -> None:
        """Return no classification mode for the numerical event/time outputs.

        Returns:
            None, independently of the dataset's label processors. Ranking
            fitting and survival evaluation use this model's explicit methods.
        """
        return None

    def __init__(
        self,
        dataset: Any = None,
        n_features: int | None = None,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        hidden_dim: int = 32,
        sigma: float = 0.5,
        dropout: float = 0.2,
        gate_l0: float = 0.01,
        l2: float = 1e-4,
        rank_tau: float = 0.1,
        mc_draws: int = 5,
        use_gates: bool = True,
        use_residual: bool = True,
        min_neighbors: int = 30,
        min_events: int = 10,
        max_neighbors: int = 199,
        tail_policy: str = "fallback",
    ) -> None:
        super().__init__(dataset)
        if n_features is None:
            if dataset is None or not len(dataset):
                raise ValueError("Supply a nonempty dataset or n_features.")
            n_features = int(_numpy(dataset[0][feature_key]).size)
        if not isinstance(n_features, int) or n_features < 1 or hidden_dim < 1:
            raise ValueError("Feature and hidden dimensions must be positive integers.")
        numbers = [sigma, dropout, gate_l0, l2, rank_tau]
        if not np.isfinite(numbers).all() or sigma <= 0 or rank_tau <= 0:
            raise ValueError(
                "Finite hyperparameters and positive sigma/rank_tau required."
            )
        if not 0 <= dropout < 1 or min(gate_l0, l2) < 0 or mc_draws < 1:
            raise ValueError("Invalid dropout, penalty or Monte Carlo draw count.")
        LocalKMImputer(min_neighbors, min_events, max_neighbors, tail_policy)
        self.config = {
            "n_features": n_features,
            "feature_key": feature_key,
            "duration_key": duration_key,
            "event_key": event_key,
            "hidden_dim": hidden_dim,
            "sigma": sigma,
            "dropout": dropout,
            "gate_l0": gate_l0,
            "l2": l2,
            "rank_tau": rank_tau,
            "mc_draws": mc_draws,
            "use_gates": use_gates,
            "use_residual": use_residual,
            "min_neighbors": min_neighbors,
            "min_events": min_events,
            "max_neighbors": max_neighbors,
            "tail_policy": tail_policy,
        }
        self.feature_keys, self.label_keys = [feature_key], [duration_key, event_key]
        self.n_features = n_features
        self.register_parameter(
            "eta", nn.Parameter(torch.full((n_features,), 0.5)) if use_gates else None
        )
        self.residual = (
            nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_features),
            )
            if use_residual
            else None
        )
        self.beta = nn.Parameter(torch.randn(n_features) * 0.1)
        self.intercept = nn.Parameter(torch.zeros(1))
        self.register_buffer("medians", torch.zeros(n_features, dtype=torch.float64))
        self.register_buffer(
            "feature_mean", torch.zeros(n_features, dtype=torch.float64)
        )
        self.register_buffer(
            "feature_scale", torch.ones(n_features, dtype=torch.float64)
        )
        self.preprocessing_fitted_ = False
        self.ranking_fitted_ = False
        self.calibration_: dict[str, Any] | None = None
        self.training_info_: dict[str, Any] = {}
        self._training_fingerprint: str | None = None
        self._gate_generator: torch.Generator | None = None

    def _cohort(self, data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        f, d, e = (self.config[k] for k in ("feature_key", "duration_key", "event_key"))
        if isinstance(data, Mapping):
            x = _numpy(data[f]).astype(float)
            times, events = _outcomes(data[d], data[e])
            ids = data.get("patient_id")
        else:
            samples = list(data)
            if not samples:
                raise ValueError("Empty survival cohort.")
            x = np.stack([_numpy(s[f]).reshape(-1) for s in samples]).astype(float)
            times, events = _outcomes(
                [float(_numpy(s[d]).item()) for s in samples],
                [float(_numpy(s[e]).item()) for s in samples],
            )
            ids = (
                [s["patient_id"] for s in samples]
                if all("patient_id" in s for s in samples)
                else None
            )
        if x.shape != (len(times), self.n_features) or np.isinf(x).any():
            raise ValueError(
                "Features must be an aligned numeric matrix without infinity."
            )
        if ids is not None:
            ids = [str(v) for v in ids]
            if len(ids) != len(x) or len(set(ids)) != len(ids):
                raise ValueError(
                    "One unique patient_id is required per survival subject."
                )
        return x, times, events, ids

    def _fit_preprocessing(self, x: np.ndarray) -> None:
        if np.any(np.isnan(x).all(axis=0)):
            raise ValueError("Training features include an entirely missing column.")
        median = np.nanmedian(x, axis=0)
        filled = np.where(np.isnan(x), median, x)
        scale = filled.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        for buffer, values in (
            (self.medians, median),
            (self.feature_mean, filled.mean(axis=0)),
            (self.feature_scale, scale),
        ):
            buffer.copy_(torch.as_tensor(values, device=self.device))
        self.preprocessing_fitted_ = True

    def _transform(self, x: Any) -> torch.Tensor:
        if not self.preprocessing_fitted_:
            raise RuntimeError(
                "Fit GRAFT preprocessing on the training partition first."
            )
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        if x.ndim != 2 or x.shape[1] != self.n_features or torch.isinf(x).any():
            raise ValueError("Expected a two-dimensional raw numeric feature matrix.")
        x = torch.where(torch.isnan(x), self.medians, x)
        transformed = (x - self.feature_mean) / self.feature_scale
        if not torch.isfinite(transformed).all():
            raise ValueError("Nonfinite standardized features.")
        return transformed.to(dtype=self.beta.dtype)

    def deterministic_gates(self) -> torch.Tensor:
        """Return clipped gate means, or ones for the no-gate configuration.

        Returns:
            A feature-length tensor on the model's device. These clipped means
            are not mathematical expectations of clipped Gaussian variables.
        """
        return (
            self.eta.clamp(0, 1) if self.eta is not None else torch.ones_like(self.beta)
        )

    def _scores(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.deterministic_gates()
        if self.training and self.eta is not None:
            noise = (
                torch.randn(
                    self.n_features,
                    device=self.device,
                    generator=self._gate_generator,
                    dtype=self.eta.dtype,
                )
                * self.config["sigma"]
            )
            gates = (self.eta + noise).clamp(0, 1)
        gated = x * gates  # One global mask shared across all subjects in this batch.
        phi = gated if self.residual is None else gated + self.residual(gated)
        return phi @ self.beta + self.intercept

    def _loss(self, scores: torch.Tensor, targets: Any) -> dict[str, torch.Tensor]:
        try:
            import torchsort
        except ImportError as exc:
            raise ImportError("GRAFT fitting requires optional torchsort.") from exc
        targets = torch.as_tensor(targets, device=self.device, dtype=scores.dtype)
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.ndim != 2 or targets.shape[0] != len(scores) or not targets.shape[1]:
            raise ValueError("Imputed log-targets need shape (batch, draws).")
        if targets.requires_grad or not torch.isfinite(targets).all():
            raise ValueError("Imputed targets must be finite, detached log-times.")
        soft = torchsort.soft_rank(
            scores[None, :],
            regularization="l2",
            regularization_strength=self.config["rank_tau"],
        )[0]
        soft = soft - soft.mean()
        # Average midranks handle tied observed and imputed times.
        values = targets.T
        ranks = 1.0 + (values[:, None, :] < values[:, :, None]).sum(-1)
        ranks = ranks + 0.5 * ((values[:, None, :] == values[:, :, None]).sum(-1) - 1)
        centered = ranks.to(scores.dtype) - ranks.to(scores.dtype).mean(
            -1, keepdim=True
        )
        correlations = (centered * soft).sum(-1) / (
            centered.norm(dim=-1) * soft.norm() + 1e-8
        )
        rank_loss = -correlations.mean()
        gate_penalty = scores.sum() * 0
        if self.eta is not None:
            gate_penalty = (
                self.config["gate_l0"]
                * (
                    0.5
                    * (1 + torch.erf(self.eta / (self.config["sigma"] * math.sqrt(2))))
                ).sum()
            )
        l2_penalty = self.config["l2"] * sum(
            p.square().sum() for p in self.parameters()
        )
        return {
            "loss": rank_loss + gate_penalty + l2_penalty,
            "ranking_loss": rank_loss,
            "gate_penalty": gate_penalty,
            "l2_penalty": l2_penalty,
            "constant_target_draws": (centered.norm(dim=-1) == 0).sum(),
        }

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return scores; compute loss only for explicit imputed training targets.

        Args:
            **kwargs: The configured feature_key supplies raw numeric inputs
                of shape (batch, features). Optional ``imputed_log_times``
                supplies detached training targets of shape (batch, draws),
                or a one-draw vector. Optional duration/event fields are
                returned as tensors; they never trigger target imputation.

        Returns:
            ``score`` and its negation ``risk_score``, each shaped (batch,).
            With imputed targets, also returns scalar loss, ranking_loss,
            gate_penalty, l2_penalty and a constant_target_draws count.

        Raises:
            RuntimeError: Before training-only preprocessing is fitted.
            ValueError: For invalid features or imputed targets.
        """
        scores = self._scores(self._transform(kwargs[self.config["feature_key"]]))
        output = {"score": scores, "risk_score": -scores}
        if "imputed_log_times" in kwargs:
            output.update(self._loss(scores, kwargs["imputed_log_times"]))
        for key in self.label_keys:
            if key in kwargs:
                output[key] = torch.as_tensor(kwargs[key], device=self.device)
        return output

    def predict_score(self, features: Any, batch_size: int = 1024) -> np.ndarray:
        """Predict deterministic survival-oriented scores without outcome labels.

        Args:
            features: Raw numeric array/tensor of shape (subjects, features).
                Missing values use fitted training medians.
            batch_size: Positive prediction batch size.

        Returns:
            A finite NumPy vector, increasing with predicted survival time.
            Prediction temporarily disables dropout and stochastic gates,
            then restores the caller's training/evaluation mode.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        x = self._transform(features)
        if not len(x):
            raise ValueError("Cannot predict an empty cohort.")
        previous = self.training
        self.eval()
        try:
            with torch.no_grad():
                result = (
                    torch.cat([self._scores(part) for part in x.split(batch_size)])
                    .cpu()
                    .numpy()
                )
        finally:
            self.train(previous)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite GRAFT score.")
        return result

    def fit(
        self,
        train_data: Any,
        validation_data: Any,
        *,
        epochs: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        patience: int = 10,
        min_delta: float = 0.0001,
        seed: int = 42,
        fold: int = 0,
        row_ids: Sequence[int] | None = None,
    ) -> "GRAFT":
        """Fit training state and restore the earliest accepted validation best.

        Cohorts are SampleDataset/Subset, sequences of sample dictionaries, or
        dictionaries of aligned numeric arrays. Only train_data fits medians,
        scales, KM neighborhoods and parameters. Validation labels select a
        checkpoint by deterministic Harrell C-index. No calibration is fitted.
        Stable source-row IDs govern distance ties; supplied IDs must increase.
        RNG streams follow SeedSequence([5982026, seed, fold, stream_id]).

        Args:
            train_data: Training SampleDataset, subset, sequence of sample
                dictionaries, or mapping of aligned arrays. Contains the
                configured feature, duration and event fields.
            validation_data: Separate labeled cohort used only for checkpoint
                selection. Patient IDs cannot overlap when supplied.
            epochs: Positive maximum number of optimization epochs.
            batch_size: Minibatch size, at least two. Final singletons merge
                into the preceding minibatch.
            learning_rate: Positive Adam learning rate.
            patience: Consecutive epochs without an accepted improvement.
            min_delta: Required validation C-index improvement.
            seed: Nonnegative training seed for the four independent streams.
            fold: Nonnegative fold ID used to derive those streams.
            row_ids: Increasing integer source-row indices aligned with the
                training cohort, used to resolve neighbor-distance ties.

        Returns:
            This model, in evaluation mode with the selected ranking state.
            Calibration is cleared and must be fitted separately.

        Raises:
            ValueError: For invalid/overlapping cohorts, unsupported device,
                or insufficient local observed events.
            FloatingPointError: If loss, gradients or predictions are nonfinite.
        """
        from lifelines.utils import concordance_index

        if self.device.type != "cpu":
            raise ValueError("This GRAFT fitting route is validated for CPU only.")
        if min(epochs, patience) < 1 or batch_size < 2:
            raise ValueError("Positive epochs/patience and batch_size >= 2 required.")
        if not np.isfinite([learning_rate, min_delta]).all():
            raise ValueError("Learning rate and min_delta must be finite.")
        if learning_rate <= 0 or min_delta < 0:
            raise ValueError("Require learning_rate > 0 and min_delta >= 0.")
        x, t, e, ids = self._cohort(train_data)
        vx, vt, ve, vids = self._cohort(validation_data)
        if len(x) < 2 or len(vx) < 2 or not e.any():
            raise ValueError(
                "Need at least two train/validation subjects and training events."
            )
        if ids is not None and vids is not None and set(ids) & set(vids):
            raise ValueError("Training and validation patient IDs overlap.")
        rows = np.arange(len(x)) if row_ids is None else np.asarray(row_ids)
        if rows.shape != (len(x),) or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("row_ids must be aligned integer source-row indices.")
        if np.any(np.diff(rows) <= 0):
            raise ValueError(
                "Training rows must be ordered by increasing source-row ID."
            )
        self.calibration_ = None
        self.ranking_fitted_ = False
        self._fit_preprocessing(x)
        transformed = self._transform(x)
        start = time.perf_counter()
        imputer = LocalKMImputer(
            **{
                key: self.config[key]
                for key in (
                    "min_neighbors",
                    "min_events",
                    "max_neighbors",
                    "tail_policy",
                )
            }
        ).fit(transformed.detach().cpu().numpy(), t, e, row_ids=rows)
        km_seconds = time.perf_counter() - start
        seeds = [
            int(
                np.random.SeedSequence([5982026, seed, fold, stream]).generate_state(1)[
                    0
                ]
            )
            for stream in range(1, 5)
        ]
        permutation_rng = np.random.Generator(np.random.PCG64(seeds[2]))
        imputation_rng = np.random.Generator(np.random.PCG64(seeds[3]))
        self._gate_generator = torch.Generator(device=self.device).manual_seed(seeds[1])
        history, best_state = [], None
        best_score, stale, best_epoch, constant_draws = -math.inf, 0, 0, 0
        gradient_norms, parameter_delta = [], 0.0
        start = time.perf_counter()
        try:
            # Keep initialization/dropout reproducible without changing caller RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seeds[0])
                if self.residual is not None:
                    for module in self.residual.modules():
                        if isinstance(module, nn.Linear):
                            module.reset_parameters()
                with torch.no_grad():
                    self.beta.normal_(0, 0.1)
                    self.intercept.zero_()
                    if self.eta is not None:
                        self.eta.fill_(0.5)
                initial_beta = self.beta.detach().clone()
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0,
                )
                for epoch in range(1, epochs + 1):
                    epoch_start = time.perf_counter()
                    targets = imputer.sample(
                        imputation_rng.random((len(x), self.config["mc_draws"]))
                    )
                    permutation = permutation_rng.permutation(len(x))
                    batches = [
                        permutation[i : i + batch_size]
                        for i in range(0, len(x), batch_size)
                    ]
                    if len(batches) > 1 and len(batches[-1]) == 1:
                        singleton = batches.pop()
                        batches[-1] = np.concatenate([batches[-1], singleton])
                    self.train()
                    losses = []
                    for batch in batches:
                        optimizer.zero_grad(set_to_none=True)
                        output = self._loss(
                            self._scores(transformed[batch]), targets[batch]
                        )
                        if not torch.isfinite(output["loss"]):
                            raise FloatingPointError(
                                f"Nonfinite loss at epoch {epoch}."
                            )
                        output["loss"].backward()
                        grads = [
                            p.grad for p in self.parameters() if p.grad is not None
                        ]
                        if not all(torch.isfinite(g).all() for g in grads):
                            raise FloatingPointError(
                                f"Nonfinite gradient at epoch {epoch}."
                            )
                        if not gradient_norms:
                            gradient_norms = [
                                float(g.norm()) for g in grads if g.numel()
                            ]
                        optimizer.step()
                        losses.append(float(output["loss"].detach()))
                        constant_draws += int(output["constant_target_draws"])
                    score = float(concordance_index(vt, self.predict_score(vx), ve))
                    if not math.isfinite(score):
                        raise FloatingPointError("Nonfinite validation C-index.")
                    history.append(
                        {
                            "epoch": epoch,
                            "mean_batch_loss": float(np.mean(losses)),
                            "validation_c_index": score,
                            "seconds": time.perf_counter() - epoch_start,
                        }
                    )
                    if score > best_score + min_delta:
                        best_score, best_epoch, stale = score, epoch, 0
                        best_state = deepcopy(self.state_dict())
                    else:
                        stale += 1
                    if stale >= patience:
                        break
                if best_state is None:
                    raise RuntimeError("No valid validation checkpoint.")
                self.load_state_dict(best_state)
                parameter_delta = float((self.beta.detach() - initial_beta).norm())
        finally:
            self._gate_generator = None
            self.eval()
        self.ranking_fitted_ = True
        self._training_fingerprint = _fingerprint(x, t, e)
        self.training_info_ = {
            "status": "fitted",
            "seed": seed,
            "fold": fold,
            "stream_seeds": seeds,
            "training_subjects": len(x),
            "validation_subjects": len(vx),
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_c_index": best_score,
            "best_state_restored": True,
            "stop_reason": "patience" if stale >= patience else "maximum_epochs",
            "km_seconds": km_seconds,
            "fit_seconds": time.perf_counter() - start,
            "first_gradient_norms": gradient_norms,
            "score_head_parameter_delta": parameter_delta,
            "constant_target_draws": constant_draws,
            "optimizer_weight_decay": 0,
            "explicit_l2": self.config["l2"],
            "imputation": imputer.diagnostics_,
            "fallback_draws": imputer.diagnostics_["fallback_subjects"]
            * self.config["mc_draws"]
            * len(history),
        }
        return self

    def fit_calibration(self, train_data: Any) -> "GRAFT":
        """Fit centered Cox/Breslow and censoring KM on the identical training rows.

        Clears any prior calibrator before attempting a new fit. Cox failure
        raises; no penalty tuning or alternative calibrator is tried.

        Args:
            train_data: The identical ordered training cohort used by fit,
                including original duration/event outcomes.

        Returns:
            This model with fitted centered Cox/Breslow and censor-survival
            state. A nonnegative Cox coefficient emits RuntimeWarning;
            discrimination remains measured using unmodified raw scores.

        Raises:
            RuntimeError: Before ranking fitting and checkpoint restoration.
            ValueError: If the supplied cohort differs from the training data.
        """
        from lifelines import CoxPHFitter
        import pandas as pd
        from sksurv.nonparametric import CensoringDistributionEstimator
        from sksurv.util import Surv

        self.calibration_ = None
        if not self.ranking_fitted_:
            raise RuntimeError("Fit and restore GRAFT ranking before calibration.")
        x, t, e, _ = self._cohort(train_data)
        if _fingerprint(x, t, e) != self._training_fingerprint:
            raise ValueError(
                "Calibrate on the identical training cohort in the same order."
            )
        score = self.predict_score(x).astype(float)
        frame = pd.DataFrame({"score": score, "duration": t, "event": e})
        cox = CoxPHFitter(penalizer=0.0).fit(frame, "duration", "event")
        coefficient = float(cox.params_["score"])
        center = float(cox._norm_mean["score"])
        baseline = cox.baseline_cumulative_hazard_
        bt = baseline.index.to_numpy(dtype=float)
        bh = baseline.iloc[:, 0].to_numpy(dtype=float)
        if not np.isfinite([coefficient, center]).all() or not np.isfinite(bh).all():
            raise FloatingPointError("Nonfinite Cox calibration state.")
        if np.any(bh < 0) or np.any(np.diff(bh) < 0):
            raise ValueError("Invalid Breslow cumulative hazard.")
        if coefficient >= 0:
            warnings.warn(
                "Cox slope is nonnegative; calibration reverses or loses orientation. "
                "Raw-score C-index remains the discrimination metric.",
                RuntimeWarning,
                stacklevel=2,
            )
        censor = CensoringDistributionEstimator().fit(Surv.from_arrays(e, t))
        relative_hazard = np.exp(coefficient * (score - center))
        np.testing.assert_allclose(
            relative_hazard,
            cox.predict_partial_hazard(frame).to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        self.calibration_ = {
            "coefficient": coefficient,
            "score_mean": center,
            "baseline_times": bt.tolist(),
            "baseline_cumulative_hazard": bh.tolist(),
            "training_max_time": float(t.max()),
            "penalizer": 0.0,
            "nonnegative_slope": coefficient >= 0,
            "subjects": len(t),
            "censor_times": censor.unique_time_.tolist(),
            "censor_probabilities": censor.prob_.tolist(),
            "lifelines_relative_hazard_verified": True,
        }
        return self

    def predict_survival(self, features: Any, times: Any) -> np.ndarray:
        """Return survival curves, using no prediction-cohort outcomes.

        Args:
            features: Raw numeric array/tensor of shape (subjects, features).
            times: Nonnegative, strictly increasing query times up to the
                maximum fitted training follow-up. Cumulative hazard is
                linearly interpolated, including the origin (0, 0).

        Returns:
            A NumPy array shaped (subjects, times), bounded in [0, 1] and
            nonincreasing over time. Cox calibration assumes proportional
            hazards; fitting it does not guarantee probability accuracy.

        Raises:
            RuntimeError: If calibration has not been fitted.
            ValueError: For invalid or unsupported query times.
            FloatingPointError: For nonfinite predictions.
        """
        if self.calibration_ is None:
            raise RuntimeError(
                "Call fit_calibration before predicting survival curves."
            )
        times = np.asarray(times, dtype=float)
        state = self.calibration_
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError("Query times must be a nonempty finite vector.")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("Query times must be nonnegative and strictly increasing.")
        if times[-1] > state["training_max_time"]:
            raise ValueError(
                "Survival query exceeds fitted training follow-up support."
            )
        bt = np.r_[0.0, state["baseline_times"]]
        bh = np.r_[0.0, state["baseline_cumulative_hazard"]]
        hazard = np.interp(times, bt, bh, left=0)
        log_relative = state["coefficient"] * (
            self.predict_score(features).astype(float) - state["score_mean"]
        )
        with np.errstate(over="raise", invalid="raise"):
            relative = np.exp(log_relative)
            curves = np.exp(-relative[:, None] * hazard[None, :])
        if not np.isfinite(curves).all():
            raise FloatingPointError("Nonfinite survival prediction.")
        return curves

    def evaluate_survival(
        self,
        data: Any,
        times: Any,
        horizon: float,
        probability_floor: float = 0.05,
    ) -> dict[str, Any]:
        """Compute raw-score C-index and training-censor-KM IPCW Brier/IBS.

        Supply a prespecified training-supported grid ending strictly below
        horizon. Administratively censor evaluation outcomes above horizon;
        keep all subjects. C-index uses original outcomes. The reverse-KM
        convention is the fitted scikit-survival convention, right-continuous
        at ties. This is a prediction-error measure, not pure calibration.

        Args:
            data: Labeled evaluation cohort, in the same formats as fit.
            times: A prespecified training-supported, increasing time grid
                with at least two points. Nonuniform grids are supported.
            horizon: Training-supported administrative censoring horizon,
                strictly larger than the final grid time.
            probability_floor: Minimum permitted censor-survival probability.
                Unsupported weights raise instead of being clipped.

        Returns:
            C-index on original outcomes, integrated Brier score, pointwise
            Brier values, the actual grid, horizon and subject/censor counts.

        Raises:
            ValueError: For invalid support or non-comparable outcomes. If
                IBS fails, callers can retain C-index by separately evaluating
                predict_score against the original outcomes.
        """
        from lifelines.utils import concordance_index

        x, t, e, _ = self._cohort(data)
        c_index = float(concordance_index(t, self.predict_score(x), e))
        grid = np.asarray(times, dtype=float)
        curves = self.predict_survival(x, grid)
        if len(grid) < 2 or not np.isfinite(horizon) or grid[-1] >= horizon:
            raise ValueError("IBS requires two grid points strictly below horizon.")
        if not 0 < probability_floor <= 1:
            raise ValueError("Censor probability floor must be in (0, 1].")
        if horizon > self.calibration_["training_max_time"]:
            raise ValueError("Administrative horizon exceeds training support.")
        clipped_t = np.minimum(t, horizon)
        clipped_e = e & (t <= horizon)
        ct = np.asarray(self.calibration_["censor_times"])
        cp = np.asarray(self.calibration_["censor_probabilities"])

        def censor_probability(query: np.ndarray) -> np.ndarray:
            indices = np.searchsorted(ct, query, side="right") - 1
            return np.where(indices < 0, 1.0, cp[np.maximum(indices, 0)])

        g_grid, g_subjects = censor_probability(grid), censor_probability(clipped_t)
        if min(g_grid.min(), g_subjects.min()) < probability_floor:
            raise ValueError(
                "Unsupported IPCW grid/horizon: censor survival below floor."
            )
        cases = (clipped_t[:, None] <= grid) & clipped_e[:, None]
        controls = clipped_t[:, None] > grid
        brier = (
            cases * curves**2 / g_subjects[:, None]
            + controls * (1 - curves) ** 2 / g_grid
        ).mean(axis=0)
        return {
            "c_index": c_index,
            "ibs": float(np.trapezoid(brier, grid) / (grid[-1] - grid[0])),
            "brier": brier.tolist(),
            "grid": grid.tolist(),
            "horizon": horizon,
            "subjects": len(t),
            "administratively_censored": int((t > horizon).sum()),
        }

    def get_extra_state(self) -> dict[str, Any]:
        """Include inference metadata in ordinary PyTorch state_dict checkpoints."""
        return deepcopy(
            {
                "version": 1,
                "config": self.config,
                "preprocessing_fitted": self.preprocessing_fitted_,
                "ranking_fitted": self.ranking_fitted_,
                "calibration": self.calibration_,
                "training_info": self.training_info_,
                "training_fingerprint": self._training_fingerprint,
            }
        )

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore metadata without needing the dataset or imputation caches."""
        if state["version"] != 1 or state["config"] != self.config:
            raise ValueError("Checkpoint GRAFT version/configuration does not match.")
        self.preprocessing_fitted_ = state["preprocessing_fitted"]
        self.ranking_fitted_ = state["ranking_fitted"]
        self.calibration_ = deepcopy(state["calibration"])
        self.training_info_ = deepcopy(state["training_info"])
        self._training_fingerprint = state["training_fingerprint"]

    def save(self, path: str | Path) -> None:
        """Save constructor configuration and fitted state, excluding the dataset.

        Args:
            path: Checkpoint destination in an existing directory.
        """
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "GRAFT":
        """Load inference state with weights_only=True and no training records.

        Args:
            path: Checkpoint previously written with save.
            map_location: Torch destination device; CPU is verified.

        Returns:
            A model in evaluation mode without an attached dataset. Fitted
            preprocessing/calibration are restored when present in the file.
        """
        saved = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**saved["config"]).to(map_location)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return model
