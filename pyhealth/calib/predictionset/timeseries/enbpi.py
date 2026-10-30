"""
Ensemble Batch Prediction Intervals (EnbPI)

Online conformal prediction intervals for time series regression. A bootstrap
ensemble gives every training point an out-of-bag (OOB) prediction from the
models that never saw it, and the absolute residuals of those predictions fill
a sliding window. The interval half-width is a quantile of that window, which
slides forward as test labels are revealed, without refitting any model.

Paper:
    Xu, Chen, and Yao Xie. "Conformal prediction interval for dynamic
    time-series." ICML 2021.
"""

import bisect
import logging
from collections import deque
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.calib.base_classes import IntervalPredictor
from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer

from .utils import as_column, as_numpy, conformal_quantile

logger = logging.getLogger(__name__)

__all__ = ["EnbPI"]

SUPPORTED_AGGREGATIONS = ("mean", "median")
SUPPORTED_CENTERS = ("aggregate", "quantile")

_AGGREGATE_FNS = {"mean": np.nanmean, "median": np.nanmedian}

# Cap on the elements of the 3-D array the median aggregation builds per chunk,
# about 32 MB of float64.
_MAX_CHUNK_ELEMENTS = 2**22


def _oob_aggregate(
    oob: np.ndarray,
    preds: np.ndarray,
    aggregation: str,
    max_chunk_elements: int = _MAX_CHUNK_ELEMENTS,
) -> np.ndarray:
    """Aggregates each OOB ensemble's predictions at a batch of inputs.

    Args:
        oob: Boolean mask of shape (B, T'). Column ``i`` marks the models that
            left training point ``i`` out, and has at least one True.
        preds: Predictions of every model, shape (B, n).
        aggregation: ``"mean"`` or ``"median"``.
        max_chunk_elements: Cap on the size of the 3-D array built for the
            median.

    Returns:
        Array of shape (T', n), the prediction ``f_{-i}(x)`` of each OOB
        ensemble ``i`` at each input ``x``.
    """
    if aggregation == "mean":
        # A masked mean is a matrix product, with no (T', B, n) array. einsum
        # avoids the spurious floating-point warnings some BLAS builds raise
        # from matmul.
        sums = np.einsum("bt,bn->tn", oob, preds.astype(float))
        return sums / oob.sum(axis=0)[:, None]
    # The median has no such shortcut, so bound memory by chunking over T'.
    n_models, n_points = oob.shape
    chunk = max(1, max_chunk_elements // (n_models * preds.shape[1]))
    out = np.empty((n_points, preds.shape[1]))
    for start in range(0, n_points, chunk):
        mask = oob[:, start : start + chunk].T[:, :, None]
        out[start : start + chunk] = np.nanmedian(
            np.where(mask, preds[None, :, :], np.nan), axis=1
        )
    return out


def _oob_mask(bootstrap_indices: np.ndarray) -> np.ndarray:
    """Mask of shape (B, T), True where model ``b`` never saw point ``i``."""
    in_bag = np.zeros(bootstrap_indices.shape, dtype=bool)
    np.put_along_axis(in_bag, bootstrap_indices, True, axis=1)
    return ~in_bag


def _check_regression(model: BaseModel) -> None:
    if model.mode != "regression":
        raise NotImplementedError(
            f"EnbPI only supports regression models, got mode {model.mode!r}."
        )


class EnbPI(IntervalPredictor):
    """Ensemble Batch Prediction Intervals for time series regression.

    EnbPI trains ``n_bootstrap`` models on bootstrap resamples of the
    training data. Each training point ``i`` gets an out-of-bag prediction
    ``f_{-i}(x_i)``, the aggregate of the models whose resample left ``i``
    out (the paper's leave-one-out predictor), and its absolute residual
    enters a sliding window of length ``T``, the number of training points.
    At test time the interval is

        ``f_{-t}(x_t) +/- (1 - alpha) quantile of the window``

    where ``f_{-t}(x_t)`` aggregates the OOB ensembles' predictions at
    ``x_t``. Each time ``feedback_size`` test labels are revealed, their
    residuals enter the window and the oldest leave it. No model is refit.

    Coverage is approximate rather than finite-sample:
    ``P(Y_t in C_t) >= 1 - alpha - o(1)`` when the errors are stationary and
    strongly mixing and the OOB predictors approximate the true regression
    function. Exchangeability is not required.

    ``state_dict`` saves the whole predictor: the bootstrap models' weights
    and the calibration state, including the current residual window. To
    restore it, create an ``EnbPI`` with the same ``model_fn`` and
    ``n_bootstrap`` and call ``load_state_dict``, or use
    ``Trainer.save_ckpt`` and ``Trainer.load_ckpt``.

    Paper:
        Xu, Chen, and Yao Xie. "Conformal prediction interval for dynamic
        time-series." ICML 2021.

    Args:
        model_fn: Returns a fresh, untrained regression ``BaseModel``. Called
            once per bootstrap model, so each starts from its own random
            initialization.
        alpha: Target miscoverage level in (0, 1).
        n_bootstrap: Number of bootstrap models ``B``. The paper finds 20 to
            30 enough. Default 20.
        aggregation: How predictions are aggregated, ``"mean"`` (default, as
            in the paper's experiments) or ``"median"``.
        center: ``"aggregate"`` (default) uses ``aggregation`` over the OOB
            ensembles' predictions at ``x_t``. ``"quantile"`` uses their
            ``(1 - alpha)`` quantile, as written in Algorithm 1, line 12.
        batch_size: Batch size for training the bootstrap models and for
            running them over a dataset. Default 32.
        train_kwargs: Keyword arguments for ``Trainer.train``, such as
            ``epochs``. Default is None, which uses the Trainer defaults.
        device: Device for training. Default is None, which uses the Trainer
            default.
        feedback_size: Points predicted before their labels are revealed in
            ``predict_sequential``, the ``s`` of the paper. Default 1.
        random_state: Seed for drawing the bootstrap resamples. Training also
            uses torch's global RNG, so seed it with ``torch.manual_seed`` for
            fully reproducible models.

    Examples:
        >>> from pyhealth.calib.predictionset import EnbPI
        >>> from pyhealth.models import MLP
        >>> # train_ds and test_ds hold one series in time order, with a
        >>> # "regression" label "y" and lagged values as the feature "x".
        >>> enbpi = EnbPI(
        ...     model_fn=lambda: MLP(dataset=train_ds, feature_keys=["x"],
        ...                          label_key="y", mode="regression"),
        ...     alpha=0.1, n_bootstrap=20, train_kwargs={"epochs": 10},
        ...     random_state=0,
        ... )
        >>> enbpi.calibrate(train_ds)  # trains 20 models
        >>> out = enbpi.predict_sequential(test_ds)
        >>> from pyhealth.metrics import regression_metrics_fn
        >>> # A dict with the MAE of the centers, the share of test points
        >>> # outside their interval (near alpha) and the mean width.
        >>> scores = regression_metrics_fn(
        ...     out["y_true"], out["y_pred"],
        ...     metrics=["mae", "miscoverage_pi", "interval_width"],
        ...     y_lower=out["y_lower"], y_upper=out["y_upper"],
        ... )
    """

    def __init__(
        self,
        model_fn: Callable[[], BaseModel],
        alpha: float,
        n_bootstrap: int = 20,
        aggregation: str = "mean",
        center: str = "aggregate",
        batch_size: int = 32,
        train_kwargs: dict | None = None,
        device: str | None = None,
        feedback_size: int = 1,
        random_state: int | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}.")
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation: {aggregation!r}. Supported: "
                f"{SUPPORTED_AGGREGATIONS}."
            )
        if center not in SUPPORTED_CENTERS:
            raise ValueError(
                f"Unknown center: {center!r}. Supported: {SUPPORTED_CENTERS}."
            )
        if feedback_size < 1:
            raise ValueError(f"feedback_size must be at least 1, got {feedback_size}.")
        super().__init__()
        self.mode = "regression"
        self.model_fn = model_fn
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.aggregation = aggregation
        self.aggregate = _AGGREGATE_FNS[aggregation]
        self.center = center
        self.batch_size = batch_size
        self.train_kwargs = train_kwargs or {}
        self.train_device = device
        self.feedback_size = feedback_size
        self.rng = np.random.default_rng(random_state)

        self.models: torch.nn.ModuleList | None = None
        self.bootstrap_indices: np.ndarray | None = None
        # oob[b, i] is True when model b never saw training point i.
        self.oob: np.ndarray | None = None
        self.residuals: deque | None = None
        # The same residuals in ascending order, so the width is a lookup.
        self.sorted_residuals: list | None = None
        # The window as calibrate left it, restored by reset.
        self.initial_residuals: list | None = None
        self.label_key: str | None = None
        self.calibrated = False

    def _train_models(self, train_dataset) -> None:
        """Algorithm 1, lines 1-4: fit one model per bootstrap resample."""
        n_samples = len(train_dataset)
        self.bootstrap_indices = self.rng.integers(
            0, n_samples, size=(self.n_bootstrap, n_samples)
        )
        models = []
        for b, indices in enumerate(self.bootstrap_indices):
            logger.info(f"EnbPI: training bootstrap model {b + 1}/{self.n_bootstrap}")
            model = self.model_fn()
            # Fail before training the first model, not after all of them.
            _check_regression(model)
            # The loader shuffles, so sorting keeps the same bag while letting
            # disk-backed datasets read contiguous runs.
            loader = get_dataloader(
                train_dataset.subset(np.sort(indices).tolist()),
                self.batch_size,
                shuffle=True,
            )
            Trainer(model=model, enable_logging=False, device=self.train_device).train(
                train_dataloader=loader, **self.train_kwargs
            )
            models.append(model)
        self.models = torch.nn.ModuleList(models)

    def _model_predictions(self, **batch) -> np.ndarray:
        """Predictions of every bootstrap model, shape (n_bootstrap, n)."""
        preds = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                y_prob = model(**batch)["y_prob"]
                preds.append(y_prob.detach().cpu().numpy().reshape(-1))
        return np.stack(preds)

    def _push_residuals(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Appends absolute residuals, skipping points with no label or OOB
        prediction (NaN), so they neither enter nor evict anything."""
        for residual in np.abs(np.asarray(y_true) - np.asarray(y_pred)):
            if np.isnan(residual):
                continue
            if len(self.residuals) == self.residuals.maxlen:
                oldest = self.residuals[0]
                del self.sorted_residuals[
                    bisect.bisect_left(self.sorted_residuals, oldest)
                ]
            self.residuals.append(float(residual))
            bisect.insort(self.sorted_residuals, float(residual))

    def calibrate(self, cal_dataset) -> None:
        """Trains the ensemble and fills the window with OOB residuals.

        Every call trains a new ensemble on ``cal_dataset``, drawing new
        bootstrap samples, so a second call gives different models than the
        first. To rerun an experiment from the same starting window without
        retraining, use :meth:`reset`.

        Args:
            cal_dataset: Training samples. EnbPI uses no separate calibration
                split, so this is the data its models are trained on. Must
                be in time order.
        """
        self.calibrated = False
        self._train_models(cal_dataset)
        self.label_key = self.models[0].label_keys[0]
        n_samples = len(cal_dataset)
        self.oob = _oob_mask(self.bootstrap_indices)

        preds, y_true = [], []
        loader = get_dataloader(cal_dataset, self.batch_size, shuffle=False)
        for batch in loader:
            preds.append(self._model_predictions(**batch))
            y_true.append(as_numpy(batch[self.label_key]))
        preds = np.concatenate(preds, axis=1)
        y_true = np.concatenate(y_true)

        # Algorithm 1, lines 6-10: OOB prediction from the models that left
        # each point out.
        has_oob = self.oob.any(axis=0)
        if not has_oob.any():
            # Every center would be an empty aggregate, so every interval
            # would be NaN, and updates could never fill the window.
            raise ValueError(
                "EnbPI: every training point is in every bootstrap sample, so "
                "no out-of-bag predictions exist. Increase n_bootstrap or use "
                "more training data."
            )
        if not has_oob.all():
            logger.warning(
                f"EnbPI: {int((~has_oob).sum())} training points are in every "
                "bootstrap sample and get no residual. Increase n_bootstrap."
            )
        y_oob = np.full(n_samples, np.nan)
        y_oob[has_oob] = self.aggregate(
            np.where(self.oob, preds, np.nan)[:, has_oob], axis=0
        )
        self.residuals = deque(maxlen=n_samples)
        self.sorted_residuals = []
        self._push_residuals(y_oob, y_true)
        self.initial_residuals = list(self.residuals)
        self.calibrated = True

    def reset(self) -> None:
        """Restores the window to the OOB residuals from :meth:`calibrate`.

        Undoes every :meth:`update` without retraining or needing the
        training data, so an evaluation can be rerun from the same start, for
        example with another ``feedback_size`` or ``alpha``.
        """
        self._check_calibrated()
        self.residuals = deque(self.initial_residuals, maxlen=self.residuals.maxlen)
        self.sorted_residuals = sorted(self.initial_residuals)

    def _check_calibrated(self) -> None:
        if not self.calibrated:
            raise RuntimeError(
                "Model must be calibrated before inference. Call calibrate() first."
            )

    def _center(self, **batch) -> np.ndarray:
        """Algorithm 1, line 12: center from the OOB ensembles at ``x_t``."""
        preds = self._model_predictions(**batch)
        # OOB prediction of every training point's ensemble at each test
        # input, shape (training points with an OOB model, batch size).
        oob = self.oob[:, self.oob.any(axis=0)]
        oob_preds = _oob_aggregate(oob, preds, self.aggregation)
        if self.center == "quantile":
            return np.quantile(oob_preds, 1 - self.alpha, axis=0)
        return self.aggregate(oob_preds, axis=0)

    def _width(self) -> float:
        """Algorithm 1, line 13: the ``(1 - alpha)`` quantile of the window."""
        return conformal_quantile(self.sorted_residuals, self.alpha, presorted=True)

    def _predict_dataset(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        """Centers and labels for a dataset, in dataset order."""
        loader = get_dataloader(dataset, self.batch_size, shuffle=False)
        y_pred, y_true = [], []
        for batch in loader:
            y_pred.append(self._center(**batch))
            y_true.append(as_numpy(batch[self.label_key]))
        return np.concatenate(y_pred), np.concatenate(y_true)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predicts intervals for one batch without changing the window.

        Works inside ``Trainer.inference``. Labels in the batch are not used
        to update the window; call :meth:`update` once they are revealed.
        Running ``Trainer.inference`` over a whole test set therefore gives
        every point the same width, which is not the paper's online method.
        Use :meth:`predict_sequential` to evaluate that.

        Returns:
            A dictionary with ``y_prob`` (the center), ``y_lower`` and
            ``y_upper``, each of shape [batch size, 1]. When the batch carries
            labels it also has ``y_true`` and the squared error ``loss`` of
            ``y_prob``, as ``Trainer.inference`` expects.
        """
        self._check_calibrated()
        y_pred = self._center(**kwargs)
        width = self._width()
        out = {
            "y_prob": as_column(y_pred),
            "y_lower": as_column(y_pred - width),
            "y_upper": as_column(y_pred + width),
        }
        if self.label_key in kwargs:
            out["y_true"] = as_column(as_numpy(kwargs[self.label_key]))
            out["loss"] = F.mse_loss(out["y_prob"], out["y_true"])
        return out

    def update(self, revealed_dataset) -> None:
        """Algorithm 1, lines 15-19: slides the window by new residuals.

        Args:
            revealed_dataset: Points whose labels were just revealed, in time
                order. Their residuals enter the window and as many of the
                oldest leave it.
        """
        self._check_calibrated()
        self._push_residuals(*self._predict_dataset(revealed_dataset))

    def predict_sequential(self, test_dataset) -> dict[str, np.ndarray]:
        """Evaluates EnbPI on a labeled test stream, as in the paper.

        A convenience method for evaluation, not part of the
        :class:`~pyhealth.calib.base_classes.IntervalPredictor` interface. It
        runs Algorithm 1, lines 11-19: predict ``feedback_size`` points from
        the current window, then reveal their labels and slide the window,
        until the stream ends. This gives the same intervals as calling
        :meth:`forward` on each chunk of ``feedback_size`` points followed by
        :meth:`update` on that chunk, but runs the ensemble over the stream
        only once, since the centers do not depend on the window.

        All labels must be known in advance, so this is for measuring
        coverage and width on held-out data. In deployment, where labels
        arrive later, call :meth:`forward` and :meth:`update` directly.

        The window is changed in place: afterwards it holds the latest test
        residuals, so a second call continues from there. Call :meth:`reset`
        first to start over from the calibration window.

        Args:
            test_dataset: Test samples following the training data, in time
                order.

        Returns:
            A dictionary of arrays of shape (n,) in time order: ``y_true``,
            ``y_pred`` (the center), ``y_lower`` and ``y_upper``.
        """
        self._check_calibrated()
        y_pred, y_true = self._predict_dataset(test_dataset)
        width = np.empty(len(y_true))
        for start in range(0, len(y_true), self.feedback_size):
            step = slice(start, start + self.feedback_size)
            width[step] = self._width()
            self._push_residuals(y_pred[step], y_true[step])
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_lower": y_pred - width,
            "y_upper": y_pred + width,
        }

    # ------------------------------------------------------------------
    # Checkpointing, not part of Algorithm 1. PyTorch's extra state hooks
    # let state_dict and load_state_dict carry the calibration state.
    # ------------------------------------------------------------------
    def get_extra_state(self) -> dict:
        """Calibration state that ``state_dict`` saves next to the weights.

        Holds the bootstrap indices, the residual window, the initial window
        and the label key, as tensors and plain values, so checkpoints load
        with ``torch.load(..., weights_only=True)`` as ``Trainer.load_ckpt``
        does.
        """
        if not self.calibrated:
            return {"calibrated": False}
        return {
            "calibrated": True,
            "label_key": self.label_key,
            "bootstrap_indices": torch.as_tensor(self.bootstrap_indices),
            "residuals": torch.tensor(list(self.residuals), dtype=torch.float64),
            "initial_residuals": torch.tensor(
                self.initial_residuals, dtype=torch.float64
            ),
        }

    def set_extra_state(self, state: dict) -> None:
        """Restores the state saved by :meth:`get_extra_state`.

        ``load_state_dict`` calls this before loading the bootstrap models'
        weights, so it first builds ``n_bootstrap`` models with ``model_fn``
        for those weights to load into.
        """
        if not state["calibrated"]:
            self.calibrated = False
            return
        bootstrap_indices = state["bootstrap_indices"].numpy()
        if len(bootstrap_indices) != self.n_bootstrap:
            raise ValueError(
                f"The checkpoint has {len(bootstrap_indices)} bootstrap models, "
                f"but this EnbPI has n_bootstrap={self.n_bootstrap}."
            )
        if self.models is None or len(self.models) != self.n_bootstrap:
            self.models = torch.nn.ModuleList(
                self.model_fn() for _ in range(self.n_bootstrap)
            )
        self.bootstrap_indices = bootstrap_indices
        self.oob = _oob_mask(bootstrap_indices)
        self.label_key = state["label_key"]
        residuals = state["residuals"].tolist()
        self.residuals = deque(residuals, maxlen=bootstrap_indices.shape[1])
        self.sorted_residuals = sorted(residuals)
        self.initial_residuals = state["initial_residuals"].tolist()
        self.calibrated = True
