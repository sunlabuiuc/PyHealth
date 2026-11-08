"""XGBoost gradient-boosted trees for tabular and bag-of-codes EHR features."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..datasets import SampleDataset
from .gradient_boosted_trees import GradientBoostedTreeModel


def _import_xgboost():
    try:
        return importlib.import_module("xgboost")
    except ImportError as e:
        raise ImportError(
            "XGBoostModel needs the optional xgboost package: "
            "pip install 'pyhealth[xgboost]' (or pip install xgboost). On macOS, "
            "see the XGBoostModel docs for the libomp note."
        ) from e


class XGBoostModel(GradientBoostedTreeModel):
    """Gradient-boosted trees (XGBoost) with PyHealth's dataset, Trainer and metrics.

    Fit once with :meth:`fit` (not ``Trainer.train``, which raises a
    ``TypeError`` for this model), then use ``Trainer.evaluate`` /
    ``Trainer.inference`` and ``Trainer.save_ckpt`` / ``load_ckpt`` as for any
    other model. See :class:`~pyhealth.models.GradientBoostedTreeModel` for the
    supported input fields and how the feature matrix is built.

    Label modes, read from the output processor:

    - ``binary``: ``XGBClassifier`` with ``objective="binary:logistic"``.
      Predictions are identical to a plain ``XGBClassifier`` with the same
      parameters on the same matrix (see :meth:`build_feature_matrix`).
    - ``multiclass``: ``XGBClassifier`` with ``objective="multi:softprob"``.
    - ``multilabel``: one booster per label (``binary:logistic``), so each
      label gets its own ``scale_pos_weight`` and early-stopping round.
    - ``regression``: ``XGBRegressor`` (``reg:squarederror`` unless
      ``objective`` is given).

    ``logit`` is the booster margin (``output_margin=True``) and ``y_prob`` the
    booster's own probabilities, so no probability is clipped or inverted.

    **Missing values.** ``NaN`` is passed to XGBoost as missing
    (``missing=np.nan``), and each split learns which way missing values go.
    Imputing first (e.g. median imputation in a processor) gives different
    trees from this native handling. XGBoost stores features as float32; a
    processor that imputes or scales should compute its statistics in float64
    and cast to float32 once, which matches scikit-learn pipelines exactly.

    **Class imbalance.** ``scale_pos_weight="balanced"`` (or ``"auto"``) uses
    negatives / positives in the training labels, per label for multilabel.
    It inflates predicted probabilities; if calibration matters, recalibrate
    on a patient-grouped calibration split (e.g. Platt scaling) rather than
    reading the raw probabilities as risks.

    **Early stopping.** Off by default, so results match a plain
    ``XGBClassifier``. With ``early_stopping_rounds`` and a ``val_data`` in
    :meth:`fit`, training stops when ``eval_metric`` (e.g. ``"aucpr"`` for rare
    outcomes) stops improving, and predictions use the best iteration.

    **Interpretation.** :meth:`explain` gives exact TreeSHAP values from
    XGBoost (``pred_contribs=True``); :meth:`mean_abs_shap` ranks columns; and
    :class:`pyhealth.interpret.methods.TreeSHAP` maps them onto the input
    fields like the other interpreters.

    **macOS.** XGBoost and PyTorch each load an OpenMP runtime (Homebrew's
    ``libomp`` and the copy bundled in the torch wheel). Two copies in one
    process can crash multithreaded fits (segfault, or ``OMP: Error #179``).
    Make both use one copy, e.g. replace ``torch/lib/libomp.dylib`` with a
    symlink to Homebrew's ``libomp.dylib``, or use ``n_jobs=1``.

    Args:
        dataset: The dataset defining inputs and the label.
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
            tree_method, random_state, n_jobs, device: Passed to XGBoost; None
            keeps XGBoost's default. ``device=None`` uses ``"cuda"`` when the
            model is on a CUDA device at fit time.
        missing: Value treated as missing. Default ``np.nan``.
        early_stopping_rounds: Rounds without improvement on ``val_data``
            before stopping. Default None (no early stopping).
        eval_metric: Metric for ``val_data`` (e.g. ``"aucpr"``, ``"logloss"``).
        scale_pos_weight: A number, one per label (multilabel), ``"balanced"``
            / ``"auto"``, or None.
        bag_of_codes: Encode code sequences as per-sample code counts.
        feature_names: Optional ``{field: [name, ...]}`` for tensor fields.
        **xgb_params: Any other XGBoost parameter (``nthread``, ``reg_lambda``,
            ``min_child_weight``, ``objective`` for regression, ...).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import XGBoostModel
        >>> samples = [
        ...     {"patient_id": f"p{i}", "labs": [float(i % 7), float(i % 3)],
        ...      "label": int(i % 7 > 3)}
        ...     for i in range(40)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"labs": "tensor"}, {"label": "binary"}
        ... )
        >>> model = XGBoostModel(dataset, n_estimators=20, max_depth=2)
        >>> loader = get_dataloader(dataset, batch_size=16)
        >>> model = model.fit(loader)
        >>> out = model(**next(iter(loader)))
        >>> out["y_prob"].shape
        torch.Size([16, 1])
        >>> model.feature_names
        ['labs[0]', 'labs[1]']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        tree_method: str | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        missing: float = np.nan,
        device: str | None = None,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        scale_pos_weight: float | Sequence[float] | str | None = None,
        bag_of_codes: bool = False,
        feature_names: dict[str, Sequence[str]] | None = None,
        **xgb_params: Any,
    ):
        super().__init__(
            dataset,
            bag_of_codes=bag_of_codes,
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
        )
        named = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": tree_method,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "device": device,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_metric": eval_metric,
        }
        for key in ("scale_pos_weight", "num_class"):
            if key in xgb_params:
                raise ValueError(f"{key} is set by XGBoostModel; do not pass it in xgb_params.")
        if "objective" in xgb_params and self.mode != "regression":
            raise ValueError(
                "objective is set from the label mode; only regression accepts "
                "a custom objective."
            )
        self.xgb_params = {k: v for k, v in named.items() if v is not None}
        self.xgb_params["missing"] = missing
        self.xgb_params.update(xgb_params)
        _import_xgboost()

    def _make_estimator(self, scale_pos_weight: float | None):
        xgb = _import_xgboost()
        params = dict(self.xgb_params)
        if "device" not in params and self.device.type == "cuda":
            params["device"] = "cuda"
        if self.mode == "regression":
            return xgb.XGBRegressor(**params)
        if self.mode == "multiclass":
            return xgb.XGBClassifier(objective="multi:softprob", **params)
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        if self.mode == "multilabel":
            # A regressor with a logistic objective: an XGBClassifier would
            # reject labels that are all 0 in the training set.
            return xgb.XGBRegressor(objective="binary:logistic", **params)
        return xgb.XGBClassifier(objective="binary:logistic", **params)

    def _fit_estimator(self, estimator, X, y, eval_set) -> None:
        if eval_set is None:
            if estimator.get_params().get("early_stopping_rounds") is not None:
                raise ValueError("early_stopping_rounds needs val_data in fit().")
            estimator.fit(X, y)
        else:
            estimator.fit(X, y, eval_set=[eval_set], verbose=False)

    def _predict_margin(self, estimator, X):
        return estimator.predict(X, output_margin=True)

    def _predict_value(self, estimator, X):
        if self.mode in ("binary", "multiclass"):
            return estimator.predict_proba(X)
        return estimator.predict(X)

    def _contributions(self, estimator, X):
        xgb = _import_xgboost()
        booster = estimator.get_booster()
        dmatrix = xgb.DMatrix(X, missing=self.xgb_params["missing"])
        best = getattr(estimator, "best_iteration", None)
        iteration_range = (0, best + 1) if best is not None else (0, 0)
        return booster.predict(dmatrix, pred_contribs=True, iteration_range=iteration_range)

    def _to_bytes(self, estimator) -> bytes:
        return bytes(estimator.get_booster().save_raw("ubj"))

    def _from_bytes(self, raw: bytes):
        estimator = self._make_estimator(None)
        estimator.load_model(bytearray(raw))
        return estimator
