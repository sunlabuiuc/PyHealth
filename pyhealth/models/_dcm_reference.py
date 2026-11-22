# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Deep Cox Mixtures network, survival curves and EM training wrapper.

The network and the curve computation are written natively in PyHealth and
need only PyTorch, NumPy and SciPy. Their structure follows the
``DeepCoxMixturesTorch`` module and ``predict_survival`` helper of the authors'
auton-survival package (Copyright (c) 2020 Carnegie Mellon University Auton
Lab, MIT License); ``tests/core/test_deep_cox_mixtures.py`` checks that both
give identical outputs for identical weights. Training calls the authors'
unmodified ``train_dcm`` EM routine, which is imported lazily from the optional
``auton_survival`` package; no author code is copied into this file.
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from torch import nn

__all__ = [
    "DCMReference",
    "DeepCoxMixturesNetwork",
    "baselines_from_state",
    "baselines_to_state",
    "predict_mixture_survival",
]

AUTON_SURVIVAL_COMMIT = "5dde465f7223601717abddc1d075e837707c403b"
INSTALL_HINT = (
    "Deep Cox Mixtures training needs the optional auton-survival and "
    "scikit-survival packages: pip install 'pyhealth[dcm]' and then "
    "pip install --no-deps 'auton-survival @ git+https://github.com/autonlab/"
    f"auton-survival@{AUTON_SURVIVAL_COMMIT}'. (--no-deps: its metadata pins "
    "torch<2 and numpy<2, which it does not need for this model.)"
)

Horizons = Union[np.ndarray, Sequence[float]]


def _import_train_dcm() -> Any:
    """Return the authors' ``train_dcm`` function or raise a clear ImportError.

    Returns:
        The ``auton_survival.models.dcm.dcm_utilities.train_dcm`` function.

    Raises:
        ImportError: ``auton_survival`` or one of its dependencies is missing.
    """
    try:
        from auton_survival.models.dcm.dcm_utilities import train_dcm
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return train_dcm


class DeepCoxMixturesNetwork(nn.Module):
    """Gating and expert heads of Deep Cox Mixtures over a ReLU6 MLP.

    The module creates its layers in the same order as the authors'
    ``DeepCoxMixturesTorch`` (gate, expert, then representation), so the same
    torch seed gives the same initial weights, and it exposes the ``k`` and
    ``optimizer`` attributes that the authors' ``train_dcm`` reads.

    Args:
        input_dim: Number of dense input covariates.
        k: Number of latent Cox components.
        layers: Hidden widths of the bias-free ReLU6 representation. An empty
            list uses the covariates directly.
        gamma: Bound on the absolute log hazard ratio of each component.

    Attributes:
        k: Number of latent components.
        gamma: Log hazard ratio bound.
        optimizer: Optimizer name read by the authors' training routine.
        gate: Linear layer producing component logits.
        expert: Linear layer producing component log hazard ratios.
        embedding: Representation network.

    Raises:
        ValueError: ``k`` is not a positive integer.

    Examples:
        >>> net = DeepCoxMixturesNetwork(input_dim=5, k=3, layers=[8])
        >>> log_weights, log_hazard_ratios = net(torch.zeros(2, 5))
        >>> tuple(log_weights.shape), tuple(log_hazard_ratios.shape)
        ((2, 3), (2, 3))
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 3,
        layers: Optional[Sequence[int]] = None,
        gamma: float = 10,
    ) -> None:
        super().__init__()
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        layers = list(layers or [])
        last = layers[-1] if layers else input_dim
        self.k = k
        self.gamma = gamma
        self.optimizer = "Adam"
        self.gate = nn.Linear(last, k, bias=False)
        self.expert = nn.Linear(last, k, bias=False)
        modules: list[nn.Module] = []
        previous = input_dim
        for hidden in layers:
            modules += [nn.Linear(previous, hidden, bias=False), nn.ReLU6()]
            previous = hidden
        self.embedding = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log mixture weights and bounded log hazard ratios.

        Args:
            x: Dense covariates shaped ``[batch, input_dim]``.

        Returns:
            ``(log_weights, log_hazard_ratios)``, each shaped ``[batch, k]``.
        """
        hidden = self.embedding(x)
        log_hazard_ratios = torch.clamp(
            self.expert(hidden), min=-self.gamma, max=self.gamma
        )
        return torch.log_softmax(self.gate(hidden), dim=1), log_hazard_ratios


def predict_mixture_survival(
    network: nn.Module,
    baselines: Mapping[int, UnivariateSpline],
    x: torch.Tensor,
    horizons: Horizons,
) -> np.ndarray:
    """Evaluate the mixture survival function at the given horizons.

    Computes ``sum_k pi_k(x) * S_k(t) ** exp(r_k(x))`` with the same NumPy
    operations, in the same order, as the authors' ``predict_survival``.

    Args:
        network: Module returning ``(log_weights, log_hazard_ratios)``.
        baselines: One baseline survival spline per component.
        x: Dense covariates shaped ``[patients, input_dim]``.
        horizons: Evaluation times.

    Returns:
        Raw survival probabilities shaped ``[patients, len(horizons)]``.

    Examples:
        >>> spline = UnivariateSpline([1.0, 2.0], [1.0, 0.5], k=1, s=0, ext=3)
        >>> net = DeepCoxMixturesNetwork(input_dim=2, k=1)
        >>> predict_mixture_survival(net, {0: spline}, torch.zeros(1, 2), [1.0])
        array([[1.]])
    """
    with torch.no_grad():
        log_weights, log_hazard_ratios = network(x)
    lrisks = log_hazard_ratios.detach().numpy()
    gate_probs = torch.exp(log_weights).detach().numpy()
    predictions = []
    for t in horizons:
        curves = [
            baselines[i](t) ** np.exp(lrisks[:, i]) for i in range(lrisks.shape[1])
        ]
        predictions.append((gate_probs * np.array(curves).T).sum(axis=1))
    return np.array(predictions).T


def baselines_to_state(
    baselines: Optional[Mapping[int, UnivariateSpline]],
) -> Optional[list[dict[str, Any]]]:
    """Encode fitted baseline splines as tensors for ``weights_only`` loading.

    Args:
        baselines: Component index to fitted spline, or ``None``.

    Returns:
        A list of ``{"knots", "coefficients", "degree", "ext"}`` dictionaries,
        or ``None`` when nothing is fitted.

    Examples:
        >>> spline = UnivariateSpline([1.0, 2.0], [1.0, 0.5], k=1, s=0, ext=3)
        >>> baselines_to_state({0: spline})[0]["degree"]
        1
    """
    if baselines is None:
        return None
    state = []
    for i in range(len(baselines)):
        knots, coefficients, degree = baselines[i]._eval_args
        state.append(
            {
                "knots": torch.from_numpy(np.array(knots, dtype=np.float64)),
                "coefficients": torch.from_numpy(
                    np.array(coefficients, dtype=np.float64)
                ),
                "degree": int(degree),
                "ext": int(baselines[i].ext),
            }
        )
    return state


def baselines_from_state(
    state: Optional[list[dict[str, Any]]],
) -> Optional[dict[int, UnivariateSpline]]:
    """Rebuild baseline splines saved by :func:`baselines_to_state`.

    Args:
        state: Encoded splines or ``None``.

    Returns:
        Component index to spline, evaluating identically to the saved splines,
        or ``None``.

    Examples:
        >>> spline = UnivariateSpline([1.0, 2.0], [1.0, 0.5], k=1, s=0, ext=3)
        >>> restored = baselines_from_state(baselines_to_state({0: spline}))
        >>> float(restored[0](1.5))
        0.75
    """
    if state is None:
        return None
    return {
        i: UnivariateSpline._from_tck(
            (
                item["knots"].numpy(),
                item["coefficients"].numpy(),
                int(item["degree"]),
            ),
            ext=int(item["ext"]),
        )
        for i, item in enumerate(state)
    }


@contextlib.contextmanager
def _isolated_rng(seed: int, num_threads: Optional[int]) -> Iterator[None]:
    """Seed NumPy and torch for one fit, then restore the caller's state.

    The authors' routine draws from the global NumPy and torch generators, so
    they are seeded inside the block and restored afterwards. The torch thread
    count is changed only inside the block.

    Args:
        seed: Seed for the global NumPy and torch generators.
        num_threads: Torch intra-op threads inside the block, or ``None`` to
            leave the setting alone.

    Yields:
        Nothing.
    """
    numpy_state = np.random.get_state()
    previous_threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        try:
            yield
        finally:
            torch.set_num_threads(previous_threads)
            np.random.set_state(numpy_state)


class DCMReference:
    """Fit Deep Cox Mixtures with the authors' EM routine and predict curves.

    This is the training engine behind
    :meth:`pyhealth.models.DeepCoxMixtures.fit_survival`. The five-fold study
    driver in ``examples/support2_dcm`` also uses it directly, because that
    study works on NumPy arrays rather than on a PyHealth dataset.

    Args:
        k: Number of latent components.
        width: Width of the single ReLU6 hidden layer.
        seed: Seed for initialization, minibatch shuffling and sampling.

    Attributes:
        net: Fitted :class:`DeepCoxMixturesNetwork` (after :meth:`fit`).
        baselines: Fitted per-component baseline survival splines.
        audit: Fit record with epochs, validation losses, time and warnings.
        last_prediction_audit: Projection record of the latest prediction.

    Examples:
        >>> learner = DCMReference(k=3, width=50, seed=0)
        >>> learner.k, learner.baselines is None
        (3, True)
    """

    def __init__(self, k: int = 3, width: int = 50, seed: int = 0) -> None:
        self.k, self.width, self.seed = k, width, seed
        self.net: Optional[nn.Module] = None
        self.baselines: Optional[dict[int, UnivariateSpline]] = None
        self.audit: Optional[dict[str, Any]] = None
        self.last_prediction_audit: Optional[dict[str, Any]] = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation: Optional[tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        refit: bool = False,
        capture_output: bool = False,
        num_threads: Optional[int] = 1,
    ) -> DCMReference:
        """Run the authors' EM routine with the study's fixed settings.

        Settings: batch size 128, Adam learning rate 1e-3, posterior-based
        spline updates, spline smoothing 1e-4, patience 3 on validation
        Q-loss. The global NumPy/torch random state and the torch thread count
        are restored when the call returns.

        Args:
            x: Training covariate matrix ``[patients, features]``.
            y: Structured array with ``"event"`` (bool) and ``"time"`` fields.
            validation: Optional ``(x, y)`` pair for early stopping. Without
                it, the training data is used, as in the authors' code.
            epochs: Maximum number of EM epochs.
            refit: Disable early stopping to reuse a selected epoch count.
            capture_output: Capture the authors' progress output into
                ``audit["upstream_output"]`` instead of printing it.
            num_threads: Torch intra-op threads during the fit; ``1`` matches
                the retained study. ``None`` leaves the setting unchanged.

        Returns:
            This learner with fitted network, baselines and audit record.

        Raises:
            ImportError: The optional ``auton_survival`` package is missing.
            FloatingPointError: No baselines were fitted or a loss is not
                finite.
        """
        train_dcm = _import_train_dcm()
        x = np.asarray(x, dtype=np.float32)

        def tensors(a: np.ndarray, b: np.ndarray) -> tuple[torch.Tensor, ...]:
            """Convert covariates and structured labels to float32 tensors."""
            return (
                torch.from_numpy(np.asarray(a, dtype=np.float32)),
                torch.from_numpy(b["time"].astype(np.float32)),
                torch.from_numpy(b["event"].astype(np.float32)),
            )

        train = tensors(x, y)
        val = train if validation is None else tensors(*validation)
        output = io.StringIO()
        streams = contextlib.ExitStack()
        if capture_output:
            streams.enter_context(contextlib.redirect_stdout(output))
            streams.enter_context(contextlib.redirect_stderr(output))
        started = time.monotonic()
        with (
            _isolated_rng(self.seed, num_threads),
            warnings.catch_warnings(record=True) as caught,
            streams,
        ):
            warnings.simplefilter("always")
            net = DeepCoxMixturesNetwork(
                x.shape[1], k=self.k, layers=[self.width], gamma=10
            )
            fitted, losses = train_dcm(
                net,
                train,
                val,
                epochs=epochs,
                patience=epochs + 1 if refit else 3,
                bs=128,
                lr=0.001,
                use_posteriors=True,
                random_seed=self.seed,
                return_losses=True,
                smoothing_factor=1e-4,
            )
        if fitted[1] is None or not all(np.isfinite(losses)):
            raise FloatingPointError(
                "DCM has no fitted baselines or nonfinite training losses"
            )
        self.net, self.baselines = fitted[0].eval(), fitted[1]
        self.audit = {
            "epochs": len(losses),
            "validation_q_losses": [float(v) for v in losses],
            "seconds": time.monotonic() - started,
            "warnings": sorted({str(w.message) for w in caught}),
            "upstream_output": output.getvalue(),
        }
        return self

    def predict_survival(self, x: np.ndarray, horizons: Horizons) -> np.ndarray:
        """Predict curves, projecting only tiny spline overshoots.

        Args:
            x: Dense preprocessed covariates.
            horizons: Strictly increasing, nonnegative evaluation times.

        Returns:
            Finite, nonincreasing survival probabilities in ``[0, 1]`` shaped
            ``[patients, horizons]``.

        Raises:
            RuntimeError: The learner has not been fitted.
            ValueError: Horizons are not a sorted one-dimensional sequence.
            FloatingPointError: Curves are nonfinite or materially invalid.
        """
        if self.net is None or self.baselines is None:
            raise RuntimeError("Call fit before predict_survival")
        horizons = np.asarray(horizons, dtype=float)
        if (
            horizons.ndim != 1
            or not len(horizons)
            or not np.isfinite(horizons).all()
            or np.any(np.diff(horizons) <= 0)
            or np.any(horizons < 0)
        ):
            raise ValueError("Horizons must be nonnegative and strictly increasing")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = predict_mixture_survival(
                self.net,
                self.baselines,
                torch.as_tensor(x, dtype=torch.float32),
                horizons.tolist(),
            )
        # A nonfinite curve is a failed model, not an imputed metric.
        if not np.isfinite(raw).all():
            raise FloatingPointError("Nonfinite spline prediction; fit is invalid")
        fixed = np.minimum.accumulate(np.clip(raw, 0, 1), axis=1)
        self.last_prediction_audit = {
            "out_of_bounds_cells": int(((raw < 0) | (raw > 1)).sum()),
            "increasing_cells": int((np.diff(raw, axis=1) > 1e-12).sum()),
            "max_projection": float(np.max(np.abs(fixed - raw))),
            "warnings": sorted({str(w.message) for w in caught}),
        }
        # Project tiny spline overshoots; material corruption fails visibly.
        if self.last_prediction_audit["max_projection"] > 1e-3:
            raise FloatingPointError(
                f"Materially invalid survival curve: {self.last_prediction_audit}"
            )
        return fixed
