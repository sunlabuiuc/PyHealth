"""
CalibrationBenchmark: Compare a new CP algorithm against all built-in PyHealth baselines.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd

from pyhealth.calib.calibration import (
    DirichletCalibration,
    HistogramBinning,
    TemperatureScaling,
)
from pyhealth.calib.predictionset import (
    LABEL,
    SCRIB,
    BaseConformal,
    ClusterLabel,
    FavMac,
    NeighborhoodLabel,
)
from pyhealth.calib.utils import extract_embeddings
from pyhealth.datasets import get_dataloader
from pyhealth.trainer import Trainer, get_metrics_fn

logger = logging.getLogger(__name__)

__all__ = ["CalibrationBenchmark"]


# Default metric lists by task type, one set for set predictors and one for
# post-hoc probability calibrators.
_DEFAULT_SET_METRICS: Dict[str, List[str]] = {
    "multiclass": ["accuracy", "set_size", "miscoverage_overall_ps"],
    "binary": ["accuracy", "set_size", "miscoverage_overall_ps"],
    "multilabel": ["set_size", "tp", "fp"],
}

_DEFAULT_CALIB_METRICS: Dict[str, List[str]] = {
    "multiclass": ["accuracy", "ECE", "cwECEt_adapt"],
    "binary": ["accuracy", "ECE"],
    "multilabel": ["accuracy", "cwECE_adapt"],
}


@dataclass
class _BaselineSpec:
    """Internal descriptor for a single baseline method."""

    cls: Type
    is_set_predictor: bool = True
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    calibrate_kwargs: Dict[str, Any] = field(default_factory=dict)
    # SCRIB uses `risk=` as its coverage parameter instead of `alpha=`
    uses_risk: bool = False
    # FavMac exposes no alpha / risk parameter
    no_alpha: bool = False
    # Whether the method needs embeddings extracted from the model
    needs_cal_embeddings: bool = False
    needs_train_embeddings: bool = False
    # None means the method supports all task types
    supported_modes: Optional[Set[str]] = None


# Registry of every built-in baseline included by default.
# KCal and CovariateLabel are intentionally excluded:
#   - KCal requires a separate training loop (too heavyweight for quick benchmarking)
#   - CovariateLabel requires test embeddings at calibration time (non-standard contract)
# Both can be added via the `custom_baselines` argument if needed.
_DEFAULT_BASELINES: Dict[str, _BaselineSpec] = {
    "LABEL": _BaselineSpec(
        cls=LABEL,
        supported_modes={"multiclass"},
    ),
    "BaseConformal_APS": _BaselineSpec(
        cls=BaseConformal,
        init_kwargs={"score_type": "aps"},
        supported_modes={"multiclass"},
    ),
    "BaseConformal_threshold": _BaselineSpec(
        cls=BaseConformal,
        init_kwargs={"score_type": "threshold"},
        supported_modes={"multiclass"},
    ),
    "SCRIB": _BaselineSpec(
        cls=SCRIB,
        uses_risk=True,
        supported_modes={"multiclass"},
    ),
    "NeighborhoodLabel": _BaselineSpec(
        cls=NeighborhoodLabel,
        needs_cal_embeddings=True,
        supported_modes={"multiclass"},
    ),
    "ClusterLabel": _BaselineSpec(
        cls=ClusterLabel,
        needs_cal_embeddings=True,
        needs_train_embeddings=True,
        supported_modes={"multiclass"},
    ),
    "FavMac": _BaselineSpec(
        cls=FavMac,
        no_alpha=True,
        supported_modes={"multilabel"},
    ),
    "TemperatureScaling": _BaselineSpec(
        cls=TemperatureScaling,
        is_set_predictor=False,
    ),
    "HistogramBinning": _BaselineSpec(
        cls=HistogramBinning,
        is_set_predictor=False,
    ),
    "DirichletCalibration": _BaselineSpec(
        cls=DirichletCalibration,
        is_set_predictor=False,
    ),
}


class CalibrationBenchmark:
    """Compare a new conformal prediction algorithm against all built-in baselines.

    Given a pre-trained model plus calibration and test splits, this class
    automatically runs every built-in calibrator and set-predictor in
    ``pyhealth.calib`` as a baseline and returns a comparative metrics table.
    Methods that are incompatible with the specified ``task_type``, or that
    require embeddings the model cannot produce, are silently skipped with a
    warning rather than crashing the whole run.

    Args:
        model: Pre-trained base model.
        cal_dataset: Calibration split used to fit each baseline.
        test_dataset: Test split used to evaluate all baselines.
        alpha: Miscoverage level passed to set predictors. Default ``0.1``.
        task_type: One of ``"multiclass"``, ``"binary"``, ``"multilabel"``.
        set_metrics: Metrics to report for set predictors. Defaults to a
            sensible list based on ``task_type`` (coverage, set size, accuracy).
        calib_metrics: Metrics to report for post-hoc probability calibrators.
            Defaults to calibration-focused metrics for the given ``task_type``.
        device: Device for embedding extraction and inference. Defaults to
            ``model.device`` when available, otherwise ``"cpu"``.
        run_all_defaults: Include all built-in baselines. Default ``True``.
            Set to ``False`` to evaluate only ``custom_baselines`` and the
            new method passed to :meth:`run`.
        custom_baselines: Extra baselines alongside the built-in ones,
            provided as ``{name: (CalibratorClass, init_kwargs_dict)}``.
            ``model`` and ``alpha`` are always injected automatically; only
            pass additional constructor kwargs here.
        train_dataset: Optional training split. Required by ``ClusterLabel``,
            which fits K-means on training-set embeddings.
        batch_size: Batch size used for all internal dataloaders. Default 32.

    Examples:
        >>> from pyhealth.calib import CalibrationBenchmark
        >>> from myproject import MyNewCP
        >>>
        >>> bm = CalibrationBenchmark(
        ...     model=trained_model,
        ...     cal_dataset=cal_data,
        ...     test_dataset=test_data,
        ...     alpha=0.1,
        ...     task_type="multiclass",
        ... )
        >>> results = bm.run(
        ...     new_calibrator=MyNewCP,
        ...     new_calibrator_name="MyNewCP",
        ... )
        >>> bm.summary()
    """

    def __init__(
        self,
        model,
        cal_dataset,
        test_dataset,
        alpha: float = 0.1,
        task_type: str = "multiclass",
        set_metrics: Optional[List[str]] = None,
        calib_metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        run_all_defaults: bool = True,
        custom_baselines: Optional[Dict[str, Tuple[Type, Dict]]] = None,
        train_dataset=None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.cal_dataset = cal_dataset
        self.test_dataset = test_dataset
        self.alpha = alpha
        self.task_type = task_type
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.results: Optional[pd.DataFrame] = None

        self.device: str = device or getattr(model, "device", "cpu")

        # Build baseline registry
        self._baselines: Dict[str, _BaselineSpec] = {}
        if run_all_defaults:
            self._baselines.update(_DEFAULT_BASELINES)
        if custom_baselines:
            for name, (cls, init_kw) in custom_baselines.items():
                self._baselines[name] = _BaselineSpec(cls=cls, init_kwargs=init_kw or {})

        self._set_metrics: List[str] = set_metrics or _DEFAULT_SET_METRICS.get(task_type, [])
        self._calib_metrics: List[str] = calib_metrics or _DEFAULT_CALIB_METRICS.get(task_type, [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        new_calibrator: Optional[Type] = None,
        new_calibrator_name: str = "new_method",
        new_calibrator_kwargs: Optional[Dict] = None,
        calibrate_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """Run all baselines and optionally a new method, returning a metrics table.

        Args:
            new_calibrator: Class (not instance) of the new CP method to
                evaluate alongside the baselines. ``model`` and ``alpha`` are
                injected automatically; pass any extra constructor arguments
                via ``new_calibrator_kwargs``.
            new_calibrator_name: Label for the new method in the results table.
            new_calibrator_kwargs: Extra ``__init__`` keyword arguments for
                the new method.
            calibrate_kwargs: Per-method extra keyword arguments for
                ``calibrate()``. Format: ``{method_name: {kwarg: value}}``.

        Returns:
            :class:`pandas.DataFrame` with methods as rows and metrics as
            columns. Cells are ``NaN`` for metrics that do not apply to a
            method, or when a method fails (a warning is emitted instead of
            raising).
        """
        calibrate_kwargs = calibrate_kwargs or {}

        # Pre-extract embeddings once, shared across all embedding-dependent baselines
        cal_embeddings = self._try_extract_embeddings(self.cal_dataset, "calibration")
        train_embeddings = (
            self._try_extract_embeddings(self.train_dataset, "train")
            if self.train_dataset is not None
            else None
        )

        # Merge built-in baselines with the new method (if supplied)
        all_specs = dict(self._baselines)
        if new_calibrator is not None:
            all_specs[new_calibrator_name] = _BaselineSpec(
                cls=new_calibrator,
                init_kwargs=new_calibrator_kwargs or {},
            )

        test_loader = get_dataloader(self.test_dataset, self.batch_size, shuffle=False)

        rows: Dict[str, Dict[str, Any]] = {}
        for name, spec in all_specs.items():
            rows[name] = self._run_one(
                name=name,
                spec=spec,
                test_loader=test_loader,
                cal_embeddings=cal_embeddings,
                train_embeddings=train_embeddings,
                extra_calibrate_kwargs=calibrate_kwargs.get(name, {}),
            )

        self.results = pd.DataFrame(rows).T
        return self.results

    def summary(self) -> None:
        """Pretty-print the benchmark results table to stdout."""
        if self.results is None:
            print("No results yet. Call run() first.")
            return
        print("\n" + "=" * 60)
        print("CalibrationBenchmark Results")
        print(f"  task_type={self.task_type}  alpha={self.alpha}")
        print("=" * 60)
        print(self.results.to_string(float_format="{:.4f}".format))
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_extract_embeddings(
        self, dataset, label: str
    ) -> Optional[np.ndarray]:
        """Extract embeddings for ``dataset``, returning ``None`` on failure."""
        needs_cal = label == "calibration" and any(
            s.needs_cal_embeddings for s in self._baselines.values()
        )
        needs_train = label == "train" and any(
            s.needs_train_embeddings for s in self._baselines.values()
        )
        if not (needs_cal or needs_train):
            return None
        try:
            logger.info("Extracting %s embeddings...", label)
            return extract_embeddings(
                self.model,
                dataset,
                batch_size=self.batch_size,
                device=self.device,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not extract {label} embeddings ({exc}). "
                "Embedding-dependent baselines will be skipped."
            )
            return None

    def _run_one(
        self,
        name: str,
        spec: _BaselineSpec,
        test_loader,
        cal_embeddings: Optional[np.ndarray],
        train_embeddings: Optional[np.ndarray],
        extra_calibrate_kwargs: Dict,
    ) -> Dict[str, Any]:
        """Instantiate, calibrate, and evaluate one baseline.

        Returns a metric dict, or an empty dict (rendered as NaN in the final
        DataFrame) when the method is skipped or raises an exception.
        """
        try:
            # Mode compatibility check
            if (
                spec.supported_modes is not None
                and self.task_type not in spec.supported_modes
            ):
                logger.info(
                    "Skipping '%s': not supported for task_type='%s'.",
                    name,
                    self.task_type,
                )
                return {}

            # Embedding availability checks
            if spec.needs_cal_embeddings and cal_embeddings is None:
                warnings.warn(
                    f"Skipping '{name}': calibration embeddings unavailable "
                    "(model does not support embed=True)."
                )
                return {}
            if spec.needs_train_embeddings and train_embeddings is None:
                warnings.warn(
                    f"Skipping '{name}': train embeddings unavailable. "
                    "Pass train_dataset= to CalibrationBenchmark to enable ClusterLabel."
                )
                return {}

            # Instantiate calibrator
            init_kw = dict(spec.init_kwargs)
            if spec.is_set_predictor and not spec.no_alpha:
                coverage_kwarg = "risk" if spec.uses_risk else "alpha"
                init_kw[coverage_kwarg] = self.alpha
            calibrator = spec.cls(self.model, **init_kw)

            # Calibrate
            cal_kw = dict(spec.calibrate_kwargs)
            if spec.needs_cal_embeddings:
                cal_kw["cal_embeddings"] = cal_embeddings
            if spec.needs_train_embeddings:
                cal_kw["train_embeddings"] = train_embeddings
            cal_kw.update(extra_calibrate_kwargs)
            calibrator.calibrate(self.cal_dataset, **cal_kw)

            # Inference
            additional_keys = ["y_predset"] if spec.is_set_predictor else None
            inference_out = Trainer(
                model=calibrator, enable_logging=False
            ).inference(test_loader, additional_outputs=additional_keys)

            y_true: np.ndarray = inference_out[0]
            y_prob: np.ndarray = inference_out[1]
            y_predset: Optional[np.ndarray] = None
            if spec.is_set_predictor and len(inference_out) > 3:
                y_predset = inference_out[3].get("y_predset")

            # Compute metrics
            metric_list = self._set_metrics if spec.is_set_predictor else self._calib_metrics
            metrics_fn = get_metrics_fn(self.task_type)
            return metrics_fn(y_true, y_prob, metrics=metric_list, y_predset=y_predset)

        except Exception as exc:
            warnings.warn(f"Baseline '{name}' failed with error: {exc}")
            return {}
