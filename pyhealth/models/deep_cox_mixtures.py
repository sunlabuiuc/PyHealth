# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Deep Cox Mixtures (Nagpal et al., MLHC 2021) for right-censored survival.

The network, the survival curves and the loss reported by ``forward`` are
native PyHealth code (PyTorch, NumPy, SciPy). Training uses the authors'
expectation-maximization routine from the optional ``auton-survival`` package
through :meth:`DeepCoxMixtures.fit_survival`; install it with the ``dcm``
extra described in the class docstring. The model runs on CPU only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline

from ..datasets import SampleDataset
from ..processors import LabelProcessor
from ._dcm_reference import (
    DCMReference,
    DeepCoxMixturesNetwork,
    baselines_from_state,
    baselines_to_state,
)
from .base_model import BaseModel

__all__ = ["DeepCoxMixtures"]

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]
_STATE_VERSION = 1


def _value(value: Any) -> Any:
    """Return the primary value of a processor output (tuple or tensor)."""
    return value[0] if isinstance(value, tuple) else value


def _infer_input_dim(dataset: SampleDataset, feature_key: str) -> int:
    """Read the dense feature width from the first processed sample.

    Args:
        dataset: Processed sample dataset.
        feature_key: Input field holding a 1-D dense vector.

    Returns:
        The vector length.

    Raises:
        ValueError: The field is missing or is not a 1-D vector.
    """
    sample = dataset[0]
    if feature_key not in sample:
        raise ValueError(f"feature_key {feature_key!r} is not in the dataset")
    tensor = torch.as_tensor(_value(sample[feature_key]))
    if tensor.ndim != 1:
        raise ValueError(
            f"DeepCoxMixtures expects {feature_key!r} to be a 1-D dense vector "
            f"per sample; got shape {tuple(tensor.shape)}"
        )
    return int(tensor.shape[0])


def _check_targets(time: np.ndarray, event: np.ndarray, name: str) -> None:
    """Validate censored survival targets.

    Args:
        time: Observed event or censoring times.
        event: Event indicators.
        name: Label used in the error message.

    Raises:
        ValueError: Times are not finite and positive or events are not 0/1.
    """
    if (
        not np.isfinite(time).all()
        or np.any(time <= 0)
        or not np.isin(event, [0, 1]).all()
    ):
        raise ValueError(
            f"{name}: use finite positive follow-up times and events in {{0, 1}}"
        )


def _check_horizons(horizons: ArrayLike) -> np.ndarray:
    """Return horizons as a float array after checking they are usable.

    Args:
        horizons: Candidate evaluation times.

    Returns:
        One-dimensional float64 array.

    Raises:
        ValueError: Horizons are empty, nonfinite, negative or not increasing.
    """
    horizons = np.asarray(horizons, dtype=float)
    if (
        horizons.ndim != 1
        or not len(horizons)
        or not np.isfinite(horizons).all()
        or np.any(np.diff(horizons) <= 0)
        or np.any(horizons < 0)
    ):
        raise ValueError("Horizons must be nonnegative and strictly increasing")
    return horizons


def _structured(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Pack event and time into the structured array the EM routine reads."""
    y = np.empty(len(time), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = event.astype(bool), time.astype(float)
    return y


class DeepCoxMixtures(BaseModel):
    """Deep Cox Mixtures: a mixture of K Cox models with a shared encoder.

    Each patient belongs to one of ``k`` latent groups; a gating head gives
    the group probabilities and an expert head gives one log hazard ratio per
    group. Each group has its own nonparametric baseline survival function,
    estimated with a smoothed Breslow spline. The survival curve is the
    mixture ``S(t | x) = sum_k pi_k(x) * S_k(t) ** exp(r_k(x))``.

    **Training.** The paper fits the model with an EM algorithm that
    alternates gradient steps on the network with refits of the baseline
    splines. Fit with :meth:`fit_survival` (NumPy arrays) or :meth:`fit`
    (PyHealth dataloaders). ``Trainer.train`` raises ``TypeError`` because a
    plain gradient loop cannot refit the splines; ``Trainer.evaluate``,
    ``Trainer.inference`` and ``save_ckpt``/``load_ckpt`` work after fitting.
    Fitting needs scikit-survival and lifelines (``pip install
    'pyhealth[dcm]'``) and the authors' ``auton-survival`` package from
    https://github.com/autonlab/auton-survival at commit
    ``5dde465f7223601717abddc1d075e837707c403b`` (the tested version),
    installed with ``pip install --no-deps`` because its metadata pins
    ``torch<2``.
    Construction, :meth:`forward` and :meth:`predict_survival` need only the
    core PyHealth dependencies.

    Args:
        dataset: Optional processed ``SampleDataset`` with one dense feature
            vector field and ``time``/``event`` output fields. Used to check
            keys and infer ``input_dim``.
        feature_key: Input field with the dense, preprocessed covariates.
        time_key: Output field with positive follow-up times.
        event_key: Output field with 1 for death and 0 for censoring.
        input_dim: Number of covariates; inferred from ``dataset`` if omitted.
        k: Number of latent Cox components.
        width: Width of the single ReLU6 hidden layer.
        seed: Seed for weight initialization and for the EM fit. The global
            random state is not changed.
        horizons: Optional increasing times at which ``forward`` reports
            survival probabilities (``y_prob``). ``fit_survival`` can set them.

    Attributes:
        network: Gating/expert network (``DeepCoxMixturesNetwork``).
        baselines: Per-component baseline survival splines, ``None`` before
            fitting.
        horizons: Evaluation times used by ``forward``, or ``None``.
        fit_audit: Epoch count, validation Q-losses, time and warnings of the
            last fit, or ``None``.

    Raises:
        ValueError: Neither ``dataset`` nor ``input_dim`` is given, or a key is
            missing from the dataset schema.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "features": [0.1, 0.4], "time": [30.0],
        ...      "event": [1.0]},
        ...     {"patient_id": "p1", "features": [0.5, 0.2], "time": [90.0],
        ...      "event": [0.0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"time": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixtures(dataset, k=3, width=8, horizons=[30.0, 60.0])
        >>> model.input_dim, model.mode
        (2, None)
    """

    #: Always ``True``: ``Trainer.train`` refuses this model; use ``fit``.
    fit_outside_trainer = True

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """No label processors: time and event are scored by the model."""
        return {}

    def __init__(
        self,
        dataset: Optional[SampleDataset] = None,
        feature_key: str = "features",
        time_key: str = "time",
        event_key: str = "event",
        input_dim: Optional[int] = None,
        k: int = 3,
        width: int = 50,
        seed: int = 0,
        horizons: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(dataset)
        self.feature_key, self.time_key, self.event_key = (
            feature_key,
            time_key,
            event_key,
        )
        if dataset is not None:
            if feature_key not in dataset.input_schema:
                raise ValueError(f"{feature_key!r} is not in dataset.input_schema")
            for key in (time_key, event_key):
                if key not in dataset.output_schema:
                    raise ValueError(f"{key!r} is not in dataset.output_schema")
            if input_dim is None:
                input_dim = _infer_input_dim(dataset, feature_key)
        if input_dim is None:
            raise ValueError("Pass a dataset or input_dim")
        self.input_dim = int(input_dim)
        self.k, self.width, self.seed = k, width, seed
        self.learner = DCMReference(k=k, width=width, seed=seed)
        # Seed a forked generator so construction leaves global state alone.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = DeepCoxMixturesNetwork(
                self.input_dim, k=k, layers=[width], gamma=10
            )
        self.baselines: Optional[dict[int, UnivariateSpline]] = None
        self.horizons = None if horizons is None else _check_horizons(horizons)
        self.fit_audit: Optional[dict[str, Any]] = None

    def fit_survival(
        self,
        features: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        *,
        validation: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        horizons: Optional[ArrayLike] = None,
        epochs: int = 100,
        refit: bool = False,
        capture_output: bool = False,
    ) -> DeepCoxMixtures:
        """Fit the network and baseline splines with the authors' EM routine.

        Args:
            features: Finite training matrix ``[patients, input_dim]``; fit any
                preprocessing on the training rows only.
            time: Positive observed event or censoring times.
            event: 1 for an observed death, 0 for censoring.
            validation: Optional ``(features, time, event)`` used for early
                stopping on the authors' validation Q-loss (patience 3).
            horizons: Optional increasing times stored for ``forward``.
            epochs: Maximum number of EM epochs.
            refit: Run exactly ``epochs`` epochs without early stopping.
            capture_output: Store the authors' progress output in
                ``fit_audit`` instead of printing it.

        Returns:
            This model, fitted.

        Raises:
            ValueError: Shapes do not match ``input_dim`` or targets/features
                are invalid.
            ImportError: The optional ``auton_survival`` package is missing.
            FloatingPointError: The EM routine produced no usable baselines.
        """
        x = np.asarray(features, dtype=np.float32)
        time, event = np.asarray(time, dtype=float), np.asarray(event)
        if (
            x.ndim != 2
            or x.shape[1] != self.input_dim
            or len(x) != len(time)
            or len(x) != len(event)
        ):
            raise ValueError("Feature/target dimensions do not match input_dim")
        if not np.isfinite(x).all():
            raise ValueError("Features must be finite; impute before fitting")
        _check_targets(time, event, "training targets")
        val = None
        if validation is not None:
            vx, vt, ve = validation
            vx = np.asarray(vx, dtype=np.float32)
            vt, ve = np.asarray(vt, dtype=float), np.asarray(ve)
            if (
                vx.ndim != 2
                or vx.shape[1] != self.input_dim
                or len(vx) != len(vt)
                or len(vx) != len(ve)
            ):
                raise ValueError(
                    "Validation feature/target dimensions do not match input_dim"
                )
            if not np.isfinite(vx).all():
                raise ValueError("Validation features must be finite")
            _check_targets(vt, ve, "validation targets")
            val = (vx, _structured(vt, ve))
        checked = None if horizons is None else _check_horizons(horizons)
        self.learner.fit(
            x,
            _structured(time, event),
            validation=val,
            epochs=epochs,
            refit=refit,
            capture_output=capture_output,
        )
        self.network = self.learner.net
        self.baselines = self.learner.baselines
        self.fit_audit = self.learner.audit
        if checked is not None:
            self.horizons = checked
        return self

    def fit(
        self,
        train_dataloader: Iterable[dict[str, Any]],
        val_dataloader: Optional[Iterable[dict[str, Any]]] = None,
        horizons: Optional[ArrayLike] = None,
        epochs: int = 100,
        capture_output: bool = False,
    ) -> DeepCoxMixtures:
        """Fit from PyHealth dataloaders by collecting them into arrays.

        Args:
            train_dataloader: Batches with ``feature_key``, ``time_key`` and
                ``event_key`` fields, e.g. from ``get_dataloader``.
            val_dataloader: Optional batches for early stopping.
            horizons: Optional increasing times stored for ``forward``.
            epochs: Maximum number of EM epochs.
            capture_output: Store the authors' progress output in
                ``fit_audit`` instead of printing it.

        Returns:
            This model, fitted.

        Raises:
            ValueError: A batch is missing a field or targets are invalid.
            ImportError: The optional ``auton_survival`` package is missing.
        """
        train = self._collect(train_dataloader)
        val = None if val_dataloader is None else self._collect(val_dataloader)
        return self.fit_survival(
            *train,
            validation=val,
            horizons=horizons,
            epochs=epochs,
            capture_output=capture_output,
        )

    def _collect(
        self, dataloader: Iterable[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate features, times and events from a dataloader."""
        parts: dict[str, list[np.ndarray]] = {
            self.feature_key: [],
            self.time_key: [],
            self.event_key: [],
        }
        for batch in dataloader:
            for key, values in parts.items():
                if key not in batch:
                    raise ValueError(f"Batch is missing field {key!r}")
                values.append(torch.as_tensor(_value(batch[key])).cpu().numpy())
        x = np.concatenate(parts[self.feature_key]).astype(np.float32)
        time = np.concatenate(parts[self.time_key]).reshape(-1)
        event = np.concatenate(parts[self.event_key]).reshape(-1)
        return x, time, event

    def _features(self, features: ArrayLike) -> torch.Tensor:
        """Convert covariates to a checked float32 CPU tensor."""
        if isinstance(features, torch.Tensor) and features.device.type != "cpu":
            raise ValueError("DeepCoxMixtures runs on CPU tensors only")
        x = torch.as_tensor(features, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not len(x):
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}]; "
                f"got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("features must be finite")
        return x

    def _posteriors(
        self,
        log_weights: torch.Tensor,
        log_hazard_ratios: torch.Tensor,
        time: np.ndarray,
        event: np.ndarray,
    ) -> torch.Tensor:
        """E-step: log posterior group probabilities given the outcome.

        Mirrors the authors' ``e_step``: the likelihood of each outcome under
        each component (survival if censored, density if dead), times the gate
        probability, with log values floored at -10 and normalized per row.
        """
        assert self.baselines is not None
        lrisks = log_hazard_ratios.detach().numpy().astype(float)
        risks = np.exp(lrisks)
        log_likelihood = np.empty_like(lrisks)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(lrisks.shape[1]):
                spline = self.baselines[i]
                s0 = spline(time)
                survival = s0 ** risks[:, i]
                density = -risks[:, i] * s0 ** (risks[:, i] - 1) * (
                    spline.derivative()(time)
                )
                log_likelihood[:, i] = np.log(np.where(event == 1, density, survival))
        probs = log_weights.detach().double() + torch.from_numpy(log_likelihood)
        probs = torch.nan_to_num(probs, nan=-10.0).clamp(min=-10.0)
        return probs - torch.logsumexp(probs, dim=1, keepdim=True)

    def _q_loss(
        self,
        log_weights: torch.Tensor,
        log_hazard_ratios: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        posteriors: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-mean negative M-step objective with hard group assignments."""
        z = torch.argmax(posteriors, dim=1)
        total = -(posteriors.exp().float() * log_weights).sum()
        for i in range(self.k):
            members = z == i
            if not members.any():
                continue
            order = torch.argsort(time[members], descending=True, stable=True)
            risk = log_hazard_ratios[members][:, i][order]
            dead = event[members][order] == 1
            partial = risk - torch.logcumsumexp(risk, dim=0)
            total = total - partial[dead].sum()
        return total / len(time)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Score one batch with the fitted model.

        Args:
            **kwargs: Batch fields. ``feature_key`` is required. When
                ``time_key`` and ``event_key`` are present, ``loss`` and
                ``y_true`` are returned too.

        Returns:
            Dictionary with:

            - ``logit``: ``[batch, 2k]``, the log mixture weights followed by
              the log hazard ratios (differentiable).
            - ``log_weights`` and ``log_hazard_ratios``: the two halves,
              each ``[batch, k]``.
            - ``y_prob``: ``[batch, len(horizons)]`` mixture survival
              probabilities at ``self.horizons`` (computed from the splines,
              no gradient).
            - ``loss`` (with labels): the batch-mean negative M-step objective
              of the paper's EM algorithm. Group posteriors are computed from
              the fitted splines and held fixed; each patient is assigned to
              its most probable group (the authors' ``typ='hard'`` variant);
              the objective is the sum of the per-group negative Cox partial
              log-likelihoods plus the negative posterior-weighted log gate
              probabilities. Ties are broken by input order. It is a training
              diagnostic, not the full mixture likelihood, and its gradient
              is what one M-step follows.
            - ``y_true`` (with labels): ``[batch, 2]`` of ``(time, event)``.

        Raises:
            RuntimeError: The model has not been fitted.
            ValueError: Missing feature field, bad shapes, non-CPU tensors,
                invalid targets, or no horizons for ``y_prob``.
        """
        if self.baselines is None:
            raise RuntimeError("Call fit_survival (or fit) before forward")
        if self.horizons is None:
            raise ValueError("Set horizons (constructor or fit_survival) first")
        if self.feature_key not in kwargs:
            raise ValueError(f"Batch is missing field {self.feature_key!r}")
        x = self._features(_value(kwargs[self.feature_key]))
        log_weights, log_hazard_ratios = self.network(x)
        output = {
            "logit": torch.cat([log_weights, log_hazard_ratios], dim=-1),
            "log_weights": log_weights,
            "log_hazard_ratios": log_hazard_ratios,
            "y_prob": torch.from_numpy(self.predict_survival(x.numpy())),
        }
        if self.time_key in kwargs and self.event_key in kwargs:
            time = torch.as_tensor(_value(kwargs[self.time_key])).float().reshape(-1)
            event = torch.as_tensor(_value(kwargs[self.event_key])).float()
            event = event.reshape(-1)
            if len(time) != len(x) or len(event) != len(x):
                raise ValueError("time/event length does not match the batch")
            _check_targets(time.numpy(), event.numpy(), "batch targets")
            posteriors = self._posteriors(
                log_weights, log_hazard_ratios, time.numpy(), event.numpy()
            )
            output["loss"] = self._q_loss(
                log_weights, log_hazard_ratios, time, event, posteriors
            )
            output["y_true"] = torch.stack([time, event], dim=-1)
        return output

    def predict_survival(
        self, features: ArrayLike, horizons: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Predict survival probabilities at increasing horizons.

        Tiny spline overshoots (at most 1e-3) are projected into ``[0, 1]``
        and made nonincreasing; anything larger raises.

        Args:
            features: Dense covariates ``[patients, input_dim]`` transformed
                with the training-fold preprocessing.
            horizons: Increasing nonnegative times; defaults to
                ``self.horizons``.

        Returns:
            ``[patients, len(horizons)]`` NumPy array of survival
            probabilities.

        Raises:
            RuntimeError: The model has not been fitted.
            ValueError: Bad feature shape or no valid horizons.
            FloatingPointError: The fitted splines give an invalid curve.
        """
        if self.baselines is None:
            raise RuntimeError("Call fit_survival (or fit) before prediction")
        x = self._features(features)
        horizons = self.horizons if horizons is None else horizons
        if horizons is None:
            raise ValueError("Provide evaluation horizons")
        self.learner.net, self.learner.baselines = self.network, self.baselines
        return self.learner.predict_survival(x.numpy(), _check_horizons(horizons))

    def get_extra_state(self) -> dict[str, Any]:
        """Return the fitted non-parameter state for ``state_dict``.

        The baseline splines are stored as knot/coefficient tensors and the
        audit as JSON text, so ``torch.load(weights_only=True)`` (used by
        ``Trainer.load_ckpt``) can read the checkpoint.

        Returns:
            Dictionary with ``version``, ``baselines``, ``horizons`` and
            ``fit_audit`` entries.
        """
        return {
            "version": _STATE_VERSION,
            "baselines": baselines_to_state(self.baselines),
            "horizons": None
            if self.horizons is None
            else torch.from_numpy(np.array(self.horizons, dtype=np.float64)),
            "fit_audit": json.dumps(self.fit_audit),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore the state written by :meth:`get_extra_state`.

        Args:
            state: Dictionary from :meth:`get_extra_state`.

        Raises:
            ValueError: The state was written by an incompatible version.
        """
        if state.get("version") != _STATE_VERSION:
            raise ValueError(
                f"DeepCoxMixtures state version {state.get('version')!r}, "
                f"expected {_STATE_VERSION}"
            )
        self.baselines = baselines_from_state(state["baselines"])
        horizons = state["horizons"]
        self.horizons = None if horizons is None else horizons.numpy()
        self.fit_audit = json.loads(state["fit_audit"])
