"""Modeling utilities for the EOL mistrust study pipeline.

This module implements the model-facing pieces of the EOL mistrust workflow:

1. three admission-level mistrust metrics
2. feature-weight summaries for the two proxy logistic models
3. race-gap, treatment-disparity, and acuity-control analyses
4. downstream repeated-split prediction experiments

The sentiment metric uses a transformers+torch backend that is already available
in the project environment. That is an intentional practical substitute for the
original Pattern-based notebook implementation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import warnings
from collections import OrderedDict
from datetime import datetime
from itertools import combinations
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd
from pyhealth.tasks.eol_mistrust import get_eol_mistrust_task_map

try:
    from scipy.stats import mannwhitneyu, pearsonr  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mannwhitneyu = None
    pearsonr = None

try:
    # pylint: disable=import-error
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # pragma: no cover
    class LogisticRegression:  # type: ignore[no-redef]
        """Fallback estimator preserving the sklearn constructor surface."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, features, labels):
            del features, labels
            raise ModuleNotFoundError(
                "scikit-learn is required for EOL mistrust model fitting."
            )

        def predict_proba(self, features):
            del features
            raise ModuleNotFoundError(
                "scikit-learn is required for EOL mistrust model inference."
            )

    def train_test_split(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise ModuleNotFoundError(
            "scikit-learn is required for downstream evaluation splits."
        )

    class LogisticRegressionCV:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ModuleNotFoundError(
                "scikit-learn is required for downstream LogisticRegressionCV tuning."
            )

    class GroupShuffleSplit:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ModuleNotFoundError(
                "scikit-learn is required for group-aware downstream evaluation splits."
            )

    def roc_auc_score(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise ModuleNotFoundError(
            "scikit-learn is required for downstream AUC evaluation."
        )


RACE_WHITE = "WHITE"
RACE_BLACK = "BLACK"
DEFAULT_LOGISTIC_C = 0.1
DownstreamEstimatorFactoryResolver = Callable[[str, str], Callable[[], object] | None]

MISTRUST_SCORE_COLUMNS = [
    "noncompliance_score_z",
    "autopsy_score_z",
    "negative_sentiment_score_z",
]

PROXY_LABEL_COLUMNS = OrderedDict(
    [
        ("noncompliance", "noncompliance_label"),
        ("autopsy", "autopsy_label"),
    ]
)

BASELINE_FEATURE_COLUMNS = [
    "age",
    "los_days",
    "gender_f",
    "gender_m",
    "insurance_private",
    "insurance_public",
    "insurance_self_pay",
]

RACE_FEATURE_COLUMNS = [
    "race_white",
    "race_black",
    "race_asian",
    "race_hispanic",
    "race_native_american",
    "race_other",
]

DOWNSTREAM_TASK_MAP = get_eol_mistrust_task_map()

DOWNSTREAM_FEATURE_CONFIGS = OrderedDict(
    [
        ("Baseline", list(BASELINE_FEATURE_COLUMNS)),
        ("Baseline + Race", list(BASELINE_FEATURE_COLUMNS + RACE_FEATURE_COLUMNS)),
        (
            "Baseline + Noncompliant",
            list(BASELINE_FEATURE_COLUMNS + ["noncompliance_score_z"]),
        ),
        ("Baseline + Autopsy", list(BASELINE_FEATURE_COLUMNS + ["autopsy_score_z"])),
        (
            "Baseline + Neg-Sentiment",
            list(BASELINE_FEATURE_COLUMNS + ["negative_sentiment_score_z"]),
        ),
        (
            "Baseline + ALL",
            list(
                BASELINE_FEATURE_COLUMNS
                + RACE_FEATURE_COLUMNS
                + MISTRUST_SCORE_COLUMNS
            ),
        ),
    ]
)


DEFAULT_TRANSFORMERS_SENTIMENT_BATCH_SIZE = 64

_SENTIMENT_BATCH_BACKEND: (
    Callable[[Sequence[str]], list[tuple[float, float]]] | None
) = None


def _parse_transformers_sentiment_output(
    result: Mapping[str, object],
) -> tuple[float, float]:
    """Convert a transformers pipeline output row into the repo sentiment tuple."""

    label = str(result.get("label", "")).upper()
    score = float(result.get("score", 0.0))
    polarity = score if "POS" in label else -score
    return (polarity, 0.0)


def _load_transformers_sentiment_batch(
    batch_size: int = DEFAULT_TRANSFORMERS_SENTIMENT_BATCH_SIZE,
) -> Callable[[Sequence[str]], list[tuple[float, float]]]:
    """Load the project-standard transformers sentiment pipeline with batching.

    GPU is used first when CUDA is available; otherwise the backend falls back
    to CPU without changing the public scorer interface.
    """

    transformers_module = importlib.import_module("transformers")
    torch_module = importlib.import_module("torch")

    pipeline_factory = getattr(transformers_module, "pipeline", None)
    if not callable(pipeline_factory):
        raise ModuleNotFoundError(
            "transformers.pipeline is unavailable in the current environment."
        )

    try:  # pragma: no cover - logging surface depends on transformers version
        transformers_logging = importlib.import_module("transformers.utils.logging")
        set_verbosity_error = getattr(transformers_logging, "set_verbosity_error", None)
        if callable(set_verbosity_error):
            set_verbosity_error()
    except Exception:
        pass

    use_cuda = bool(
        getattr(torch_module, "cuda", None) and torch_module.cuda.is_available()
    )
    device = 0 if use_cuda else -1
    classifier = pipeline_factory(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

    def _transformers_sentiment_batch(
        texts: Sequence[str],
    ) -> list[tuple[float, float]]:
        cleaned_texts = [_prepare_note_text_for_sentiment(text) for text in texts]
        outputs = [(0.0, 0.0) for _ in cleaned_texts]

        non_empty_indices = [index for index, text in enumerate(cleaned_texts) if text]
        if not non_empty_indices:
            return outputs

        non_empty_texts = [cleaned_texts[index][:2048] for index in non_empty_indices]
        batch_results = classifier(
            non_empty_texts,
            truncation=True,
            batch_size=batch_size,
        )

        for index, result in zip(non_empty_indices, batch_results):
            outputs[index] = _parse_transformers_sentiment_output(result)
        return outputs

    return _transformers_sentiment_batch


def _load_transformers_sentiment() -> Callable[[str], tuple[float, float]]:
    """Load the single-text transformers sentiment adapter."""

    def _transformers_sentiment(text: str) -> tuple[float, float]:
        return _default_sentiment_batch_backend([text])[0]

    return _transformers_sentiment


def _default_sentiment_batch_backend(texts: Sequence[str]) -> list[tuple[float, float]]:
    """Resolve and cache the default batched transformers sentiment backend lazily."""

    global _SENTIMENT_BATCH_BACKEND
    if _SENTIMENT_BATCH_BACKEND is None:
        _SENTIMENT_BATCH_BACKEND = _load_transformers_sentiment_batch()
    return _SENTIMENT_BATCH_BACKEND(texts)


def _require_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{df_name} is missing required columns: {missing_str}")


def _prepare_note_text_for_sentiment(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).split())


def _note_present_hadm_ids(note_corpus: pd.DataFrame) -> list[int]:
    """Return sorted admission ids with at least one non-empty aggregated note."""

    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")
    present = note_corpus.copy()
    note_text = present["note_text"].fillna("").astype(str).str.strip()
    hadm_ids = pd.to_numeric(present.loc[note_text != "", "hadm_id"], errors="coerce")
    return sorted(hadm_ids.dropna().astype(int).unique().tolist())


def _default_estimator_factory() -> object:
    return LogisticRegression(
        penalty="l1",
        C=DEFAULT_LOGISTIC_C,
        solver="liblinear",
        max_iter=100,
        tol=0.01,
    )


def build_logistic_estimator_factory(
    *,
    C: float = DEFAULT_LOGISTIC_C,
    class_weight: str | Mapping[int, float] | None = None,
    penalty: str = "l1",
    solver: str = "liblinear",
    max_iter: int = 100,
    tol: float = 0.01,
) -> Callable[[], object]:
    """Return a reusable sklearn logistic-regression factory."""

    def _factory() -> object:
        return LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            tol=tol,
        )

    return _factory


class _AdaptiveLogisticRegressionCV:
    """Binary logistic CV wrapper with fold count chosen from the train labels."""

    def __init__(
        self,
        *,
        Cs: Sequence[float],
        class_weight: str | Mapping[int, float] | None = None,
        penalty: str = "l1",
        solver: str = "liblinear",
        max_iter: int = 1000,
        tol: float = 0.001,
        scoring: str = "roc_auc",
        max_cv_folds: int = 5,
    ) -> None:
        self.Cs = [float(value) for value in Cs]
        self.class_weight = class_weight
        self.penalty = penalty
        self.solver = solver
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scoring = scoring
        self.max_cv_folds = int(max_cv_folds)
        self.estimator_ = None

    def fit(self, X, y):
        y_series = pd.Series(y).reset_index(drop=True).astype(int)
        class_counts = y_series.value_counts(dropna=True)
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0

        if min_class_count < 2:
            fallback_c = self.Cs[0] if self.Cs else DEFAULT_LOGISTIC_C
            estimator = LogisticRegression(
                penalty=self.penalty,
                C=fallback_c,
                solver=self.solver,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            cv_folds = max(2, min(self.max_cv_folds, min_class_count))
            estimator = LogisticRegressionCV(
                Cs=self.Cs,
                penalty=self.penalty,
                solver=self.solver,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                scoring=self.scoring,
                cv=cv_folds,
                refit=True,
            )

        self.estimator_ = estimator.fit(X, y_series)
        self.coef_ = getattr(self.estimator_, "coef_", None)
        self.C_ = getattr(self.estimator_, "C_", None)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        return self

    def predict_proba(self, X):
        if self.estimator_ is None:
            raise AttributeError("Estimator has not been fitted yet.")
        return self.estimator_.predict_proba(X)

    def __getattr__(self, name):
        if name == "estimator_":
            raise AttributeError(name)
        if self.estimator_ is None:
            raise AttributeError(name)
        return getattr(self.estimator_, name)


def build_logistic_cv_estimator_factory(
    *,
    Cs: Sequence[float],
    class_weight: str | Mapping[int, float] | None = None,
    penalty: str = "l1",
    solver: str = "liblinear",
    max_iter: int = 1000,
    tol: float = 0.001,
    scoring: str = "roc_auc",
    max_cv_folds: int = 5,
) -> Callable[[], object]:
    """Return an adaptive LogisticRegressionCV factory for downstream use."""

    candidate_cs = [float(value) for value in Cs]

    def _factory() -> object:
        return _AdaptiveLogisticRegressionCV(
            Cs=candidate_cs,
            class_weight=class_weight,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            scoring=scoring,
            max_cv_folds=max_cv_folds,
        )

    return _factory


def _resolve_downstream_estimator_factory(
    task_name: str,
    config_name: str,
    estimator_factory: Callable[[], object] | None,
    downstream_estimator_factory_resolver: DownstreamEstimatorFactoryResolver | None,
) -> Callable[[], object]:
    if downstream_estimator_factory_resolver is not None:
        resolved = downstream_estimator_factory_resolver(task_name, config_name)
        if resolved is not None:
            return resolved
    return (
        _default_estimator_factory if estimator_factory is None else estimator_factory
    )


def _extract_positive_class_probabilities(probabilities) -> np.ndarray:
    """Validate predict_proba output and return the positive-class column."""

    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 2 or probability_array.shape[1] < 2:
        raise IndexError(
            "Estimator `predict_proba` output must have shape "
            "(n_samples, n_classes>=2)."
        )
    return probability_array[:, 1]


def _score_column_name(label_column: str) -> str:
    if label_column.endswith("_label"):
        return f"{label_column[:-6]}_score"
    return f"{label_column}_score"


class _ConstantProbabilityEstimator:
    """Proxy estimator that predicts a constant positive-class probability."""

    def __init__(self, positive_probability: float):
        self.positive_probability = float(positive_probability)
        self.fit_X = None
        self.fit_y = None
        self.classes_ = np.array([0, 1], dtype=int)
        self.coef_ = np.zeros((1, 0), dtype=float)
        self.intercept_ = np.array([0.0], dtype=float)

    def fit(self, X, y):
        self.fit_X = X.copy() if hasattr(X, "copy") else X
        self.fit_y = y.copy() if hasattr(y, "copy") else y
        n_features = int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) >= 2 else 0
        self.coef_ = np.zeros((1, n_features), dtype=float)
        probability = self.positive_probability
        if 0.0 < probability < 1.0:
            self.intercept_ = np.array(
                [float(np.log(probability / (1.0 - probability)))],
                dtype=float,
            )
        return self

    def predict_proba(self, X):
        n_rows = len(X)
        probability = self.positive_probability
        return np.column_stack(
            [
                np.full(n_rows, 1.0 - probability, dtype=float),
                np.full(n_rows, probability, dtype=float),
            ]
        )


def _warn_degenerate_proxy_training(
    label_column: str,
    class_values: Sequence[int],
    n_rows: int,
) -> None:
    if not class_values:
        detail = "no joined training rows"
    else:
        detail = f"a single observed class ({class_values[0]})"
    warnings.warn(
        f"Proxy training for '{label_column}' has {detail} across {n_rows} rows; "
        "returning constant probabilities and zero feature weights.",
        UserWarning,
        stacklevel=3,
    )


def _iter_downstream_jobs(
    final_model_table: pd.DataFrame,
    feature_configurations: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, str] | None = None,
):
    """Yield prepared downstream task/config jobs in stable order."""

    _require_columns(final_model_table, ["hadm_id"], "final_model_table")
    if feature_configurations is None:
        configs = get_downstream_feature_configurations()
    else:
        configs = OrderedDict(
            (name, list(columns)) for name, columns in feature_configurations.items()
        )
    tasks = get_downstream_task_map() if task_map is None else OrderedDict(task_map)

    for task_name, target_column in tasks.items():
        _require_columns(final_model_table, [target_column], "final_model_table")
        for config_name, feature_columns in configs.items():
            _require_columns(final_model_table, feature_columns, "final_model_table")
            selected_columns = ["hadm_id", target_column, *feature_columns]
            if "subject_id" in final_model_table.columns:
                selected_columns.insert(1, "subject_id")
            usable = final_model_table[selected_columns].dropna().copy()
            usable = usable.sort_values("hadm_id").reset_index(drop=True)
            y = pd.to_numeric(usable[target_column], errors="coerce")
            n_pos = int((y == 1).sum())
            if n_pos < 10:
                warnings.warn(
                    f"Downstream task '{task_name}' / config '{config_name}' has only "
                    f"{n_pos} positive examples in the cohort "
                    "(minimum 10 recommended). "
                    "AUC results for this combination will be NaN.",
                    UserWarning,
                    stacklevel=2,
                )
            X = usable[feature_columns]
            yield task_name, target_column, config_name, feature_columns, usable, X, y


def _iter_downstream_jobs_with_estimators(
    final_model_table: pd.DataFrame,
    feature_configurations: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, str] | None = None,
    estimator_factory: Callable[[], object] | None = None,
    downstream_estimator_factory_resolver: (
        DownstreamEstimatorFactoryResolver | None
    ) = None,
):
    """Yield downstream jobs together with the resolved estimator factory."""

    jobs = _iter_downstream_jobs(
        final_model_table,
        feature_configurations=feature_configurations,
        task_map=task_map,
    )
    for task_name, target_column, config_name, feature_columns, usable, X, y in jobs:
        yield (
            task_name,
            target_column,
            config_name,
            feature_columns,
            usable,
            X,
            y,
            _resolve_downstream_estimator_factory(
                task_name=task_name,
                config_name=config_name,
                estimator_factory=estimator_factory,
                downstream_estimator_factory_resolver=(
                    downstream_estimator_factory_resolver
                ),
            ),
        )


def _downstream_split_with_optional_grouping(
    X: pd.DataFrame,
    y: pd.Series,
    usable: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    split_fn: Callable[..., tuple] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split downstream rows, grouping by subject_id when available."""

    if split_fn is not None:
        return split_fn(X, y, test_size=test_size, random_state=random_state)

    if "subject_id" not in usable.columns:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    groups = pd.to_numeric(usable["subject_id"], errors="coerce")
    if groups.isna().any():
        raise ValueError(
            "Downstream final_model_table contains null subject_id values."
        )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
    )


def _iter_downstream_repetition_splits(
    X: pd.DataFrame,
    y: pd.Series,
    usable: pd.DataFrame,
    *,
    repetitions: int,
    test_size: float,
    split_fn: Callable[..., tuple] | None = None,
):
    """Yield downstream train/test splits, using ``None`` for invalid repeats."""

    for random_state in range(repetitions):
        if usable.empty or y.nunique(dropna=True) < 2:
            yield None
            continue

        X_train, X_test, y_train, y_test = _downstream_split_with_optional_grouping(
            X,
            y,
            usable,
            test_size=test_size,
            random_state=random_state,
            split_fn=split_fn,
        )
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
        if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
            yield None
            continue

        yield X_train, X_test, y_train, y_test


def _standardize_downstream_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize downstream train/test features while preserving DataFrame columns."""

    scaler = StandardScaler()
    train_index = getattr(X_train, "index", None)
    test_index = getattr(X_test, "index", None)
    return (
        pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_columns,
            index=train_index,
        ),
        pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_columns,
            index=test_index,
        ),
    )


def _prepare_proxy_training_frame(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    _require_columns(feature_matrix, ["hadm_id"], "feature_matrix")
    _require_columns(note_labels, ["hadm_id", label_column], "note_labels")

    feature_columns = [
        column for column in feature_matrix.columns if column != "hadm_id"
    ]
    merged = feature_matrix.merge(
        note_labels[["hadm_id", label_column]],
        on="hadm_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("hadm_id")
    return merged.reset_index(drop=True), feature_columns


def _make_metric_result(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float, float, float, float, int, int]:
    left = pd.to_numeric(left, errors="coerce").dropna().astype(float)
    right = pd.to_numeric(right, errors="coerce").dropna().astype(float)
    if left.empty or right.empty:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            len(left),
            len(right),
        )

    left_median = float(left.median())
    right_median = float(right.median())

    if mannwhitneyu is None:  # pragma: no cover
        statistic = float("nan")
        pvalue = float("nan")
    else:
        result = mannwhitneyu(left, right, alternative="two-sided")
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)

    return statistic, pvalue, left_median, right_median, len(left), len(right)


def _pearson_with_pvalue(left: pd.Series, right: pd.Series) -> tuple[float, float, int]:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2:
        return float("nan"), float("nan"), len(frame)

    if pearsonr is not None:  # pragma: no branch
        corr, pvalue = pearsonr(frame["left"], frame["right"])
        return float(corr), float(pvalue), len(frame)

    corr = float(frame["left"].corr(frame["right"], method="pearson"))
    return corr, float("nan"), len(frame)


def _assign_severity_bins(
    frame: pd.DataFrame,
    acuity_column: str = "oasis",
) -> pd.DataFrame:
    """Assign stable low/medium/high terciles from an acuity column."""

    _require_columns(frame, [acuity_column], "acuity_frame")
    labeled = frame.copy()
    acuity_values = pd.to_numeric(labeled[acuity_column], errors="coerce")
    labeled["severity_bin"] = pd.Series(pd.NA, index=labeled.index, dtype="object")

    valid = acuity_values.notna()
    if valid.sum() == 0:
        return labeled

    ordered = acuity_values.loc[valid].rank(method="first")
    if len(ordered) >= 3:
        bins = pd.qcut(ordered, 3, labels=["low", "medium", "high"])
        labeled.loc[valid, "severity_bin"] = bins.astype(str)
        return labeled

    fallback_labels = ["low", "medium", "high"][: len(ordered)]
    fallback = pd.Series(fallback_labels, index=ordered.sort_values().index)
    labeled.loc[fallback.index, "severity_bin"] = fallback.astype(str)
    return labeled


def build_empirical_cdf_curve(values: Iterable[float]) -> pd.DataFrame:
    """Build a plot-ready empirical CDF curve from numeric values."""

    series = (
        pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(float)
    )
    series = series.sort_values().reset_index(drop=True)
    if series.empty:
        return pd.DataFrame(columns=["x", "cdf"])
    cdf = (np.arange(1, len(series) + 1) / len(series)).astype(float)
    return pd.DataFrame({"x": series, "cdf": cdf})


def get_downstream_feature_configurations() -> OrderedDict[str, list[str]]:
    """Return the six required downstream feature configurations."""

    return OrderedDict(
        (name, list(columns)) for name, columns in DOWNSTREAM_FEATURE_CONFIGS.items()
    )


def get_downstream_task_map() -> OrderedDict[str, str]:
    """Return the required downstream prediction targets."""

    return OrderedDict(DOWNSTREAM_TASK_MAP)


def fit_proxy_mistrust_model(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
):
    """Fit the L1 logistic proxy model on the labeled subset only.

    Rows where ``label_column`` is NaN are excluded from training.
    """

    merged, feature_columns = _prepare_proxy_training_frame(
        feature_matrix, note_labels, label_column
    )
    labeled_mask = merged[label_column].notna()
    train = merged.loc[labeled_mask].copy()
    train_labels = train[label_column].astype(int)
    observed_classes = sorted(pd.unique(train_labels).tolist())

    if train.empty or len(observed_classes) < 2:
        _warn_degenerate_proxy_training(label_column, observed_classes, len(train))
        probability = float(observed_classes[0]) if observed_classes else 0.0
        return _ConstantProbabilityEstimator(probability).fit(
            train[feature_columns], train_labels
        )

    estimator = (
        _default_estimator_factory() if estimator_factory is None
        else estimator_factory()
    )
    estimator.fit(train[feature_columns], train_labels)
    return estimator


def build_proxy_probability_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Fit a proxy logistic model and return positive-class probabilities.

    Training uses only rows where ``label_column`` is not NaN (the labeled
    cohort).  Scoring uses the full ``feature_matrix`` so every admission
    receives a score.  This matches the reference notebook behavior where
    autopsy proxy training uses only consent/decline admissions but scores
    are produced for all patients.
    """

    merged, feature_columns = _prepare_proxy_training_frame(
        feature_matrix, note_labels, label_column
    )
    score_column = _score_column_name(label_column)

    labeled_mask = merged[label_column].notna()
    train = merged.loc[labeled_mask].copy()
    train_labels = train[label_column].astype(int)
    observed_classes = sorted(pd.unique(train_labels).tolist())

    if train.empty or len(observed_classes) < 2:
        _warn_degenerate_proxy_training(label_column, observed_classes, len(train))
        default_prob = float(observed_classes[0]) if observed_classes else 0.0
        positive_class = np.full(len(merged), default_prob, dtype=float)
    else:
        estimator = (
        _default_estimator_factory() if estimator_factory is None
        else estimator_factory()
    )
        estimator.fit(train[feature_columns], train_labels)
        positive_class = _extract_positive_class_probabilities(
            estimator.predict_proba(merged[feature_columns])
        )

    scores = pd.DataFrame(
        {
            "hadm_id": merged["hadm_id"],
            score_column: positive_class.astype(float),
        }
    )
    return (
        scores.sort_values("hadm_id")
        .drop_duplicates("hadm_id")
        .reset_index(drop=True)
    )


def build_noncompliance_mistrust_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    return build_proxy_probability_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column=PROXY_LABEL_COLUMNS["noncompliance"],
        estimator_factory=estimator_factory,
    )


def build_autopsy_mistrust_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    return build_proxy_probability_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column=PROXY_LABEL_COLUMNS["autopsy"],
        estimator_factory=estimator_factory,
    )


def build_negative_sentiment_mistrust_scores(
    note_corpus: pd.DataFrame,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Compute negative-sentiment mistrust from whitespace-tokenized note text.

    Empty notes (after whitespace normalization) always score 0.0 without
    invoking the scorer, regardless of whether the default backend or a custom
    ``sentiment_fn`` is used.
    """

    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")

    cleaned = note_corpus.copy()
    cleaned["note_text"] = cleaned["note_text"].map(_prepare_note_text_for_sentiment)
    if sentiment_fn is None:
        sentiment_scores = _default_sentiment_batch_backend(
            cleaned["note_text"].tolist()
        )
    else:
        empty_mask = cleaned["note_text"] == ""
        sentiment_scores = [(0.0, 0.0)] * len(cleaned)
        non_empty_indices = [
            index for index, is_empty in enumerate(empty_mask) if not is_empty
        ]
        for index in non_empty_indices:
            sentiment_scores[index] = sentiment_fn(cleaned["note_text"].iloc[index])

    cleaned["negative_sentiment_score"] = [
        float(-1.0 * score[0]) for score in sentiment_scores
    ]
    return (
        cleaned[["hadm_id", "negative_sentiment_score"]]
        .sort_values("hadm_id")
        .reset_index(drop=True)
    )


def z_normalize_scores(
    score_table: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply independent z-score normalization to the requested score columns."""

    _require_columns(score_table, ["hadm_id"], "score_table")
    normalized = score_table.copy()
    if columns is None:
        score_columns = [
            column
            for column in normalized.columns
            if column != "hadm_id"
            and (column.endswith("_score") or column.endswith("_score_z"))
        ]
    else:
        score_columns = list(columns)

    for column in score_columns:
        _require_columns(normalized, [column], "score_table")
        values = pd.to_numeric(normalized[column], errors="coerce").astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if pd.isna(std) or std == 0:
            normalized[column] = 0.0
        else:
            normalized[column] = (values - mean) / std
    return normalized


def build_mistrust_score_table(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    note_corpus: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build the three normalized mistrust metrics."""

    _require_columns(
        note_labels, ["hadm_id", *PROXY_LABEL_COLUMNS.values()], "note_labels"
    )
    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")

    proxy_scores: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for proxy_name, label_column in PROXY_LABEL_COLUMNS.items():
        proxy_scores[proxy_name] = build_proxy_probability_scores(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            label_column=label_column,
            estimator_factory=estimator_factory,
        )
    sentiment = build_negative_sentiment_mistrust_scores(
        note_corpus=note_corpus,
        sentiment_fn=sentiment_fn,
    )

    merged = None
    for score_table in list(proxy_scores.values()) + [sentiment]:
        if merged is None:
            merged = score_table
            continue
        merged = merged.merge(
            score_table, on="hadm_id", how="inner", validate="one_to_one"
        )
    assert merged is not None
    merged = merged.sort_values("hadm_id")

    raw_score_columns = [
        _score_column_name(label_column)
        for label_column in PROXY_LABEL_COLUMNS.values()
    ]
    rename_map = {
        _score_column_name(label_column): f"{proxy_name}_score_z"
        for proxy_name, label_column in PROXY_LABEL_COLUMNS.items()
    }
    raw_score_columns.append("negative_sentiment_score")
    rename_map["negative_sentiment_score"] = "negative_sentiment_score_z"

    normalized = z_normalize_scores(
        merged,
        columns=raw_score_columns,
    )
    normalized = normalized.rename(columns=rename_map)
    return normalized.reset_index(drop=True)


def summarize_feature_weights(
    estimator,
    feature_columns: Sequence[str],
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Summarize model coefficients into positive and negative rankings."""

    if not hasattr(estimator, "coef_"):
        raise ValueError("Estimator must expose `coef_` for weight summarization.")
    coefficients = np.asarray(estimator.coef_)
    if coefficients.ndim != 2 or coefficients.shape[0] == 0:
        raise ValueError("Estimator `coef_` must have shape (n_classes, n_features).")
    weights = coefficients[0]
    if len(weights) != len(feature_columns):
        raise ValueError("Feature columns must align with estimator coefficients.")

    summary = pd.DataFrame(
        {"feature": list(feature_columns), "weight": weights.astype(float)}
    )
    summary = summary.sort_values(
        ["weight", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    positive = summary.head(top_n).reset_index(drop=True)
    negative = (
        summary.sort_values(["weight", "feature"], ascending=[True, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    return {"all": summary, "positive": positive, "negative": negative}


def build_proxy_feature_weight_summary(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Fit a proxy model and summarize the learned coefficient weights."""

    _, feature_columns = _prepare_proxy_training_frame(
        feature_matrix, note_labels, label_column
    )
    estimator = fit_proxy_mistrust_model(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column=label_column,
        estimator_factory=estimator_factory,
    )
    return summarize_feature_weights(estimator, feature_columns, top_n=top_n)


def build_noncompliance_feature_weight_summary(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    return build_proxy_feature_weight_summary(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column=PROXY_LABEL_COLUMNS["noncompliance"],
        estimator_factory=estimator_factory,
        top_n=top_n,
    )


def build_autopsy_feature_weight_summary(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    return build_proxy_feature_weight_summary(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column=PROXY_LABEL_COLUMNS["autopsy"],
        estimator_factory=estimator_factory,
        top_n=top_n,
    )


def run_race_gap_analysis(
    mistrust_scores: pd.DataFrame,
    demographics: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    race_column: str = "race",
) -> pd.DataFrame:
    """Compare White and Black mistrust score distributions by Mann-Whitney U."""

    _require_columns(mistrust_scores, ["hadm_id"], "mistrust_scores")
    _require_columns(demographics, ["hadm_id", race_column], "demographics")

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = mistrust_scores.merge(
        demographics[["hadm_id", race_column]],
        on="hadm_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged[race_column].isin({RACE_WHITE, RACE_BLACK})].copy()

    rows: list[dict[str, float | int | str | bool]] = []
    for column in columns:
        black = merged.loc[merged[race_column] == RACE_BLACK, column]
        white = merged.loc[merged[race_column] == RACE_WHITE, column]
        (
            statistic,
            pvalue,
            median_black,
            median_white,
            n_black,
            n_white,
        ) = _make_metric_result(
            black, white
        )
        rows.append(
            {
                "metric": column,
                "n_black": n_black,
                "n_white": n_white,
                "median_black": median_black,
                "median_white": median_white,
                "median_gap_black_minus_white": median_black - median_white
                if not (pd.isna(median_black) or pd.isna(median_white))
                else float("nan"),
                "statistic": statistic,
                "pvalue": pvalue,
                "black_median_higher": bool(
                    not (pd.isna(median_black) or pd.isna(median_white))
                    and median_black > median_white
                ),
            }
        )
    return pd.DataFrame(rows)


def run_race_based_treatment_analysis(
    eol_cohort: pd.DataFrame,
    treatment_totals: pd.DataFrame,
    race_column: str = "race",
    treatment_columns: Sequence[str] = ("total_vent_min", "total_vaso_min"),
) -> pd.DataFrame:
    """Compare Black and White treatment durations within the EOL cohort."""

    _require_columns(eol_cohort, ["hadm_id", race_column], "eol_cohort")
    _require_columns(
        treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals"
    )

    merged = eol_cohort.merge(
        treatment_totals, on="hadm_id", how="left", validate="one_to_one"
    )
    merged = merged.loc[merged[race_column].isin({RACE_WHITE, RACE_BLACK})].copy()

    rows: list[dict[str, float | int | str]] = []
    for column in treatment_columns:
        usable = merged.loc[merged[column].notna()].copy()
        black = usable.loc[usable[race_column] == RACE_BLACK, column]
        white = usable.loc[usable[race_column] == RACE_WHITE, column]
        (
            statistic,
            pvalue,
            median_black,
            median_white,
            n_black,
            n_white,
        ) = _make_metric_result(
            black, white
        )
        rows.append(
            {
                "treatment": column,
                "n_black": n_black,
                "n_white": n_white,
                "median_black": median_black,
                "median_white": median_white,
                "median_gap_black_minus_white": median_black - median_white
                if not (pd.isna(median_black) or pd.isna(median_white))
                else float("nan"),
                "statistic": statistic,
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def run_race_based_treatment_analysis_by_acuity(
    eol_cohort: pd.DataFrame,
    treatment_totals: pd.DataFrame,
    acuity_scores: pd.DataFrame,
    race_column: str = "race",
    treatment_columns: Sequence[str] = ("total_vent_min", "total_vaso_min"),
    acuity_column: str = "oasis",
) -> pd.DataFrame:
    """Compare Black and White treatment duration within OASIS severity terciles."""

    _require_columns(eol_cohort, ["hadm_id", race_column], "eol_cohort")
    _require_columns(
        treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals"
    )
    _require_columns(acuity_scores, ["hadm_id", acuity_column], "acuity_scores")

    merged = (
        eol_cohort.merge(
            treatment_totals, on="hadm_id", how="left", validate="one_to_one"
        )
        .merge(
            acuity_scores[["hadm_id", acuity_column]],
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
    )
    merged = merged.loc[merged[race_column].isin({RACE_WHITE, RACE_BLACK})].copy()
    merged = _assign_severity_bins(merged, acuity_column=acuity_column)

    rows: list[dict[str, float | int | str]] = []
    for treatment in treatment_columns:
        for severity_bin in ("low", "medium", "high"):
            usable = merged.loc[
                (merged["severity_bin"] == severity_bin) & merged[treatment].notna()
            ].copy()
            black = usable.loc[usable[race_column] == RACE_BLACK, treatment]
            white = usable.loc[usable[race_column] == RACE_WHITE, treatment]
            (
            statistic,
            pvalue,
            median_black,
            median_white,
            n_black,
            n_white,
        ) = _make_metric_result(
                black,
                white,
            )
            rows.append(
                {
                    "severity_bin": severity_bin,
                    "treatment": treatment,
                    "n_black": n_black,
                    "n_white": n_white,
                    "median_black": median_black,
                    "median_white": median_white,
                    "median_gap_black_minus_white": median_black - median_white
                    if not (pd.isna(median_black) or pd.isna(median_white))
                    else float("nan"),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def build_race_based_treatment_cdf_plot_data(
    eol_cohort: pd.DataFrame,
    treatment_totals: pd.DataFrame,
    race_column: str = "race",
    treatment_columns: Sequence[str] = ("total_vent_min", "total_vaso_min"),
) -> dict[str, pd.DataFrame]:
    """Build plot-ready CDF curves and medians for race-based treatment analysis."""

    _require_columns(eol_cohort, ["hadm_id", race_column], "eol_cohort")
    _require_columns(
        treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals"
    )

    merged = eol_cohort.merge(
        treatment_totals, on="hadm_id", how="left", validate="one_to_one"
    )
    merged = merged.loc[merged[race_column].isin({RACE_WHITE, RACE_BLACK})].copy()

    curves: list[dict[str, float | str]] = []
    medians: list[dict[str, float | str]] = []
    for treatment in treatment_columns:
        usable = merged.loc[merged[treatment].notna()].copy()
        for race_value, label in ((RACE_WHITE, "White"), (RACE_BLACK, "Black")):
            values = usable.loc[usable[race_column] == race_value, treatment]
            cdf = build_empirical_cdf_curve(values)
            for row in cdf.itertuples(index=False):
                curves.append(
                    {
                        "treatment": treatment,
                        "group": label,
                        "x": float(row.x),
                        "cdf": float(row.cdf),
                    }
                )
            median = (
                pd.to_numeric(values, errors="coerce").dropna().astype(float).median()
            )
            medians.append(
                {
                    "treatment": treatment,
                    "group": label,
                    "median": float(median) if not pd.isna(median) else float("nan"),
                    "line_style": "dotted",
                }
            )
    return {"curves": pd.DataFrame(curves), "medians": pd.DataFrame(medians)}


def run_trust_based_treatment_analysis(
    eol_cohort: pd.DataFrame,
    mistrust_scores: pd.DataFrame,
    treatment_totals: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    treatment_columns: Sequence[str] = ("total_vent_min", "total_vaso_min"),
    group_sizes: Mapping[str, int] | None = None,
    race_column: str = "race",
) -> pd.DataFrame:
    """Compare high-vs-low mistrust groups on treatment duration within EOL."""

    _require_columns(eol_cohort, ["hadm_id"], "eol_cohort")
    _require_columns(mistrust_scores, ["hadm_id"], "mistrust_scores")
    _require_columns(
        treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals"
    )

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = (
        eol_cohort.merge(
            treatment_totals, on="hadm_id", how="left", validate="one_to_one"
        )
        .merge(
            mistrust_scores[["hadm_id", *columns]],
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
    )
    groups = dict(group_sizes or {})

    if race_column in merged.columns:
        race_based = run_race_based_treatment_analysis(
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
            race_column=race_column,
            treatment_columns=treatment_columns,
        )
        for row in race_based.itertuples(index=False):
            groups.setdefault(str(row.treatment), int(row.n_black))

    rows: list[dict[str, float | int | str]] = []
    for treatment in treatment_columns:
        for metric in columns:
            usable = merged.loc[
                merged[treatment].notna() & merged[metric].notna()
            ].copy()
            usable = usable.sort_values(
                [metric, "hadm_id"], ascending=[False, True]
            ).reset_index(drop=True)
            group_size = int(groups.get(treatment, 0))

            if group_size <= 0 or group_size >= len(usable):
                rows.append(
                    {
                        "metric": metric,
                        "treatment": treatment,
                        "stratification_n": group_size,
                        "n_high": min(group_size, len(usable)),
                        "n_low": max(len(usable) - group_size, 0),
                        "median_high": float("nan"),
                        "median_low": float("nan"),
                        "median_gap": float("nan"),
                        "statistic": float("nan"),
                        "pvalue": float("nan"),
                    }
                )
                continue

            high = usable.iloc[:group_size][treatment]
            low = usable.iloc[group_size:][treatment]
            (
                statistic,
                pvalue,
                median_high,
                median_low,
                n_high,
                n_low,
            ) = _make_metric_result(
                high, low
            )
            rows.append(
                {
                    "metric": metric,
                    "treatment": treatment,
                    "stratification_n": group_size,
                    "n_high": n_high,
                    "n_low": n_low,
                    "median_high": median_high,
                    "median_low": median_low,
                    "median_gap": median_high - median_low
                    if not (pd.isna(median_high) or pd.isna(median_low))
                    else float("nan"),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def run_trust_based_treatment_analysis_by_acuity(
    eol_cohort: pd.DataFrame,
    mistrust_scores: pd.DataFrame,
    treatment_totals: pd.DataFrame,
    acuity_scores: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    treatment_columns: Sequence[str] = ("total_vent_min", "total_vaso_min"),
    group_sizes: Mapping[str, int] | None = None,
    race_column: str = "race",
    acuity_column: str = "oasis",
) -> pd.DataFrame:
    """Compare high-vs-low mistrust groups within OASIS severity terciles."""

    _require_columns(eol_cohort, ["hadm_id"], "eol_cohort")
    _require_columns(mistrust_scores, ["hadm_id"], "mistrust_scores")
    _require_columns(
        treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals"
    )
    _require_columns(acuity_scores, ["hadm_id", acuity_column], "acuity_scores")

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = (
        eol_cohort.merge(
            treatment_totals, on="hadm_id", how="left", validate="one_to_one"
        )
        .merge(
            mistrust_scores[["hadm_id", *columns]],
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            acuity_scores[["hadm_id", acuity_column]],
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
    )
    merged = _assign_severity_bins(merged, acuity_column=acuity_column)
    explicit_groups = dict(group_sizes or {})

    derived_groups: dict[tuple[str, str], int] = {}
    if race_column in merged.columns:
        race_based = run_race_based_treatment_analysis_by_acuity(
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
            acuity_scores=acuity_scores,
            race_column=race_column,
            treatment_columns=treatment_columns,
            acuity_column=acuity_column,
        )
        for row in race_based.itertuples(index=False):
            derived_groups[(str(row.severity_bin), str(row.treatment))] = int(
                row.n_black
            )

    rows: list[dict[str, float | int | str]] = []
    for metric in columns:
        for treatment in treatment_columns:
            for severity_bin in ("low", "medium", "high"):
                usable = merged.loc[
                    (merged["severity_bin"] == severity_bin)
                    & merged[treatment].notna()
                    & merged[metric].notna()
                ].copy()
                usable = usable.sort_values(
                    [metric, "hadm_id"], ascending=[False, True]
                ).reset_index(drop=True)
                group_size = int(
                    explicit_groups.get(
                        treatment,
                        derived_groups.get((severity_bin, treatment), 0),
                    )
                )

                if group_size <= 0 or group_size >= len(usable):
                    rows.append(
                        {
                            "severity_bin": severity_bin,
                            "metric": metric,
                            "treatment": treatment,
                            "stratification_n": group_size,
                            "n_high": min(group_size, len(usable)),
                            "n_low": max(len(usable) - group_size, 0),
                            "median_high": float("nan"),
                            "median_low": float("nan"),
                            "median_gap": float("nan"),
                            "statistic": float("nan"),
                            "pvalue": float("nan"),
                        }
                    )
                    continue

                high = usable.iloc[:group_size][treatment]
                low = usable.iloc[group_size:][treatment]
                (
                statistic,
                pvalue,
                median_high,
                median_low,
                n_high,
                n_low,
            ) = _make_metric_result(
                    high,
                    low,
                )
                rows.append(
                    {
                        "severity_bin": severity_bin,
                        "metric": metric,
                        "treatment": treatment,
                        "stratification_n": group_size,
                        "n_high": n_high,
                        "n_low": n_low,
                        "median_high": median_high,
                        "median_low": median_low,
                        "median_gap": median_high - median_low
                        if not (pd.isna(median_high) or pd.isna(median_low))
                        else float("nan"),
                        "statistic": statistic,
                        "pvalue": pvalue,
                    }
                )
    return pd.DataFrame(rows)


def build_trust_based_treatment_cdf_plot_data(
    eol_cohort: pd.DataFrame,
    mistrust_scores: pd.DataFrame,
    treatment_totals: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    treatment_columns: Sequence[str] = ("total_vent_min", "total_vaso_min"),
    group_sizes: Mapping[str, int] | None = None,
    race_column: str = "race",
) -> dict[str, pd.DataFrame]:
    """Build plot-ready CDF curves and medians for trust-based treatment analysis."""

    _require_columns(eol_cohort, ["hadm_id"], "eol_cohort")
    _require_columns(mistrust_scores, ["hadm_id"], "mistrust_scores")
    _require_columns(
        treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals"
    )

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = (
        eol_cohort.merge(
            treatment_totals, on="hadm_id", how="left", validate="one_to_one"
        )
        .merge(
            mistrust_scores[["hadm_id", *columns]],
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
    )
    groups = dict(group_sizes or {})
    if race_column in merged.columns:
        race_based = run_race_based_treatment_analysis(
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
            race_column=race_column,
            treatment_columns=treatment_columns,
        )
        for row in race_based.itertuples(index=False):
            groups.setdefault(str(row.treatment), int(row.n_black))

    curves: list[dict[str, float | str]] = []
    medians: list[dict[str, float | str]] = []
    for treatment in treatment_columns:
        for metric in columns:
            usable = merged.loc[
                merged[treatment].notna() & merged[metric].notna()
            ].copy()
            usable = usable.sort_values(
                [metric, "hadm_id"], ascending=[False, True]
            ).reset_index(drop=True)
            group_size = int(groups.get(treatment, 0))
            if group_size <= 0 or group_size >= len(usable):
                continue

            grouped_values = {
                "High Mistrust": usable.iloc[:group_size][treatment],
                "Low Mistrust": usable.iloc[group_size:][treatment],
            }
            for label, values in grouped_values.items():
                cdf = build_empirical_cdf_curve(values)
                for row in cdf.itertuples(index=False):
                    curves.append(
                        {
                            "metric": metric,
                            "treatment": treatment,
                            "group": label,
                            "x": float(row.x),
                            "cdf": float(row.cdf),
                        }
                    )
                median = (
                pd.to_numeric(values, errors="coerce").dropna().astype(float).median()
            )
                medians.append(
                    {
                        "metric": metric,
                        "treatment": treatment,
                        "group": label,
                        "median": (
                            float(median) if not pd.isna(median) else float("nan")
                        ),
                        "line_style": "dotted",
                    }
                )
    return {"curves": pd.DataFrame(curves), "medians": pd.DataFrame(medians)}


def run_acuity_control_analysis(
    mistrust_scores: pd.DataFrame,
    acuity_scores: pd.DataFrame,
    score_columns: Sequence[str] | None = None,
    acuity_columns: Sequence[str] = ("oasis", "sapsii"),
) -> pd.DataFrame:
    """Compute pairwise Pearson correlations across mistrust and acuity scores."""

    _require_columns(mistrust_scores, ["hadm_id"], "mistrust_scores")
    _require_columns(acuity_scores, ["hadm_id", *acuity_columns], "acuity_scores")

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = mistrust_scores.merge(
        acuity_scores[["hadm_id", *acuity_columns]],
        on="hadm_id",
        how="inner",
        validate="one_to_one",
    )

    analysis_columns = columns + list(acuity_columns)
    rows: list[dict[str, float | int | str]] = []
    for left, right in combinations(analysis_columns, 2):
        correlation, pvalue, n = _pearson_with_pvalue(merged[left], merged[right])
        rows.append(
            {
                "feature_a": left,
                "feature_b": right,
                "correlation": correlation,
                "pvalue": pvalue,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def evaluate_downstream_average_weights(
    final_model_table: pd.DataFrame,
    feature_configurations: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, str] | None = None,
    estimator_factory: Callable[[], object] | None = None,
    downstream_estimator_factory_resolver: (
        DownstreamEstimatorFactoryResolver | None
    ) = None,
    split_fn: Callable[..., tuple] | None = None,
    repetitions: int = 100,
    test_size: float = 0.4,
) -> pd.DataFrame:
    """Average downstream regularized model weights across repeated 60/40 splits.

    Table 6 in the reference notebook reports coefficients from models trained on
    the already-prepared baseline/mistrust feature space without an additional
    sklearn ``StandardScaler`` pass. Age/LOS and mistrust features are already
    standardized earlier in the pipeline, so we preserve those raw columns here
    instead of re-scaling the one-hot baseline indicators.
    """

    rows: list[dict[str, float | int | str]] = []

    for (
        task_name,
        target_column,
        config_name,
        feature_columns,
        usable,
        X,
        y,
        estimator_factory_for_job,
    ) in _iter_downstream_jobs_with_estimators(
        final_model_table,
        feature_configurations=feature_configurations,
        task_map=task_map,
        estimator_factory=estimator_factory,
        downstream_estimator_factory_resolver=downstream_estimator_factory_resolver,
    ):
        collected_weights: list[np.ndarray] = []

        for split_result in _iter_downstream_repetition_splits(
            X,
            y,
            usable,
            repetitions=repetitions,
            test_size=test_size,
            split_fn=split_fn,
        ):
            if split_result is None:
                continue

            X_train, X_test, y_train, y_test = split_result
            del X_test  # coefficients come from the fitted train-side model only

            estimator = estimator_factory_for_job()
            estimator.fit(X_train, y_train.astype(int))
            coefficients = np.asarray(getattr(estimator, "coef_", None), dtype=float)
            if coefficients.ndim != 2 or coefficients.shape[0] == 0:
                raise ValueError(
                    "Downstream estimator must expose `coef_` with shape "
                    "(n_classes, n_features)."
                )
            weights = coefficients[0]
            if len(weights) != len(feature_columns):
                raise ValueError(
                    "Downstream feature columns must align with estimator "
                    "coefficients."
                )
            collected_weights.append(weights.astype(float))

        if collected_weights:
            weight_matrix = np.vstack(collected_weights)
            weight_mean = weight_matrix.mean(axis=0)
            weight_std = weight_matrix.std(axis=0, ddof=0)
            n_valid = weight_matrix.shape[0]
        else:
            weight_mean = np.full(len(feature_columns), np.nan, dtype=float)
            weight_std = np.full(len(feature_columns), np.nan, dtype=float)
            n_valid = 0

        for index, feature in enumerate(feature_columns):
            rows.append(
                {
                    "task": task_name,
                    "configuration": config_name,
                    "target_column": target_column,
                    "feature": feature,
                    "n_repeats": int(repetitions),
                    "n_valid_weights": int(n_valid),
                    "weight_mean": (
                        float(weight_mean[index])
                        if not np.isnan(weight_mean[index])
                        else float("nan")
                    ),
                    "weight_std": (
                        float(weight_std[index])
                        if not np.isnan(weight_std[index])
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_downstream_predictions(
    final_model_table: pd.DataFrame,
    feature_configurations: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, str] | None = None,
    estimator_factory: Callable[[], object] | None = None,
    downstream_estimator_factory_resolver: (
        DownstreamEstimatorFactoryResolver | None
    ) = None,
    split_fn: Callable[..., tuple] | None = None,
    auc_fn: Callable[[Iterable[int], Iterable[float]], float] | None = None,
    repetitions: int = 100,
    test_size: float = 0.4,
) -> pd.DataFrame:
    """Run repeated 60/40 downstream AUC evaluation across all tasks/configs."""

    metric = roc_auc_score if auc_fn is None else auc_fn

    rows: list[dict[str, float | int | str]] = []
    for (
        task_name,
        target_column,
        config_name,
        feature_columns,
        usable,
        X,
        y,
        estimator_factory_for_job,
    ) in _iter_downstream_jobs_with_estimators(
        final_model_table,
        feature_configurations=feature_configurations,
        task_map=task_map,
        estimator_factory=estimator_factory,
        downstream_estimator_factory_resolver=downstream_estimator_factory_resolver,
    ):
        auc_values: list[float] = []

        for split_result in _iter_downstream_repetition_splits(
            X,
            y,
            usable,
            repetitions=repetitions,
            test_size=test_size,
            split_fn=split_fn,
        ):
            if split_result is None:
                auc_values.append(float("nan"))
                continue

            X_train, X_test, y_train, y_test = split_result
            X_train, X_test = _standardize_downstream_features(
                X_train,
                X_test,
                feature_columns,
            )

            estimator = estimator_factory_for_job()
            estimator.fit(X_train, y_train.astype(int))
            probabilities = estimator.predict_proba(X_test)
            positive_class = _extract_positive_class_probabilities(probabilities)
            auc_values.append(float(metric(y_test.astype(int), positive_class)))

        auc_series = pd.Series(auc_values, dtype=float)
        rows.append(
            {
                "task": task_name,
                "configuration": config_name,
                "target_column": target_column,
                "n_rows": int(len(usable)),
                "n_features": int(len(feature_columns)),
                "n_repeats": int(repetitions),
                "n_valid_auc": int(auc_series.notna().sum()),
                "auc_mean": (
                    float(auc_series.mean())
                    if auc_series.notna().any()
                    else float("nan")
                ),
                "auc_std": (
                    float(auc_series.std(ddof=0))
                    if auc_series.notna().any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_treatment_cdf(
    curves: pd.DataFrame,
    medians: pd.DataFrame,
    group_column: str = "group",
    x_column: str = "x",
    y_column: str = "cdf",
    median_column: str = "median",
    ax=None,
):
    """Optional reporting helper for grouped empirical CDF visualization.

    This helper is intentionally isolated from the core modeling pipeline and
    exists only for lightweight plotting of already-computed CDF data.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "matplotlib is required for EOL mistrust CDF plotting."
        ) from exc

    if ax is None:
        _, ax = plt.subplots()

    ordered_curves = curves.copy()
    if not ordered_curves.empty:
        ordered_curves = ordered_curves.sort_values(
            [group_column, x_column]
        ).reset_index(drop=True)
    for group_value, group_df in ordered_curves.groupby(group_column, sort=False):
        ax.plot(group_df[x_column], group_df[y_column], label=str(group_value))

    for row in medians.itertuples(index=False):
        ax.axvline(
            getattr(row, median_column),
            linestyle=getattr(row, "line_style", "dotted"),
            label=f"{getattr(row, group_column)} median",
        )
    return ax


class EOLMistrustModelOutputs(TypedDict, total=False):
    """Typed contract for the dict returned by ``run_full_eol_mistrust_modeling``.

    All keys except ``mistrust_scores`` and ``feature_weight_summaries`` are
    optional because they require their corresponding input tables to be
    provided.
    """

    mistrust_scores: pd.DataFrame
    feature_weight_summaries: dict[str, dict[str, pd.DataFrame]]
    race_gap_results: pd.DataFrame
    race_treatment_results: pd.DataFrame
    race_treatment_by_acuity_results: pd.DataFrame
    race_treatment_cdf_plot_data: pd.DataFrame
    trust_treatment_results: pd.DataFrame
    trust_treatment_by_acuity_results: pd.DataFrame
    trust_treatment_cdf_plot_data: pd.DataFrame
    acuity_correlations: pd.DataFrame
    downstream_auc_results: pd.DataFrame
    downstream_weight_results: pd.DataFrame


def run_full_eol_mistrust_modeling(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    note_corpus: pd.DataFrame,
    demographics: pd.DataFrame | None = None,
    eol_cohort: pd.DataFrame | None = None,
    treatment_totals: pd.DataFrame | None = None,
    acuity_scores: pd.DataFrame | None = None,
    final_model_table: pd.DataFrame | None = None,
    estimator_factory: Callable[[], object] | None = None,
    downstream_estimator_factory_resolver: (
        DownstreamEstimatorFactoryResolver | None
    ) = None,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
    split_fn: Callable[..., tuple] | None = None,
    auc_fn: Callable[[Iterable[int], Iterable[float]], float] | None = None,
    repetitions: int = 100,
    include_downstream_weight_summary: bool = False,
    include_cdf_plot_data: bool = False,
    precomputed_mistrust_scores: pd.DataFrame | None = None,
    score_columns: Sequence[str] | None = None,
    feature_configurations: Mapping[str, Sequence[str]] | None = None,
) -> EOLMistrustModelOutputs:
    """Run the end-to-end model-stage workflow and collect its outputs."""

    if precomputed_mistrust_scores is not None:
        mistrust_scores = precomputed_mistrust_scores
    else:
        mistrust_scores = build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=estimator_factory,
            sentiment_fn=sentiment_fn,
        )
    feature_weight_summaries: OrderedDict[str, dict[str, pd.DataFrame]] = OrderedDict()
    for proxy_name, label_column in PROXY_LABEL_COLUMNS.items():
        feature_weight_summaries[proxy_name] = build_proxy_feature_weight_summary(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            label_column=label_column,
            estimator_factory=estimator_factory,
        )
    outputs: dict[str, pd.DataFrame | dict[str, pd.DataFrame]] = {
        "mistrust_scores": mistrust_scores,
        "feature_weight_summaries": feature_weight_summaries,
    }
    selected_score_columns = list(
        MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns
    )

    if demographics is not None:
        outputs["race_gap_results"] = run_race_gap_analysis(
            mistrust_scores,
            demographics,
            score_columns=selected_score_columns,
        )

    if eol_cohort is not None and treatment_totals is not None:
        outputs["race_treatment_results"] = run_race_based_treatment_analysis(
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
        )
        outputs["trust_treatment_results"] = run_trust_based_treatment_analysis(
            eol_cohort=eol_cohort,
            mistrust_scores=mistrust_scores,
            treatment_totals=treatment_totals,
            score_columns=selected_score_columns,
        )
        if acuity_scores is not None:
            outputs["race_treatment_by_acuity_results"] = (
                run_race_based_treatment_analysis_by_acuity(
                    eol_cohort=eol_cohort,
                    treatment_totals=treatment_totals,
                    acuity_scores=acuity_scores,
                )
            )
            outputs["trust_treatment_by_acuity_results"] = (
                run_trust_based_treatment_analysis_by_acuity(
                    eol_cohort=eol_cohort,
                    mistrust_scores=mistrust_scores,
                    treatment_totals=treatment_totals,
                    acuity_scores=acuity_scores,
                    score_columns=selected_score_columns,
                )
            )
        if include_cdf_plot_data:
            outputs["race_treatment_cdf_plot_data"] = (
                build_race_based_treatment_cdf_plot_data(
                    eol_cohort=eol_cohort,
                    treatment_totals=treatment_totals,
                )
            )
            outputs["trust_treatment_cdf_plot_data"] = (
                build_trust_based_treatment_cdf_plot_data(
                    eol_cohort=eol_cohort,
                    mistrust_scores=mistrust_scores,
                    treatment_totals=treatment_totals,
                    score_columns=selected_score_columns,
                )
            )

    if acuity_scores is not None:
        outputs["acuity_correlations"] = run_acuity_control_analysis(
            mistrust_scores=mistrust_scores,
            acuity_scores=acuity_scores,
            score_columns=selected_score_columns,
        )

    if final_model_table is not None:
        downstream = final_model_table.copy()
        if not set(MISTRUST_SCORE_COLUMNS).issubset(downstream.columns):
            downstream = downstream.merge(mistrust_scores, on="hadm_id", how="left")
        outputs["downstream_auc_results"] = evaluate_downstream_predictions(
            final_model_table=downstream,
            feature_configurations=feature_configurations,
            estimator_factory=estimator_factory,
            downstream_estimator_factory_resolver=downstream_estimator_factory_resolver,
            split_fn=split_fn,
            auc_fn=auc_fn,
            repetitions=repetitions,
        )
        if include_downstream_weight_summary:
            outputs["downstream_weight_results"] = evaluate_downstream_average_weights(
                final_model_table=downstream,
                feature_configurations=feature_configurations,
                estimator_factory=estimator_factory,
                downstream_estimator_factory_resolver=(
                    downstream_estimator_factory_resolver
                ),
                split_fn=split_fn,
                repetitions=repetitions,
            )

    return outputs


class EOLMistrustModel:
    """Thin object wrapper around the functional EOL mistrust model pipeline."""

    def __init__(
        self,
        estimator_factory: Callable[[], object] | None = None,
        sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
        split_fn: Callable[..., tuple] | None = None,
        auc_fn: Callable[[Iterable[int], Iterable[float]], float] | None = None,
        repetitions: int = 100,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.sentiment_fn = sentiment_fn
        self.split_fn = split_fn
        self.auc_fn = auc_fn
        self.repetitions = repetitions

    def build_mistrust_scores(
        self,
        feature_matrix: pd.DataFrame,
        note_labels: pd.DataFrame,
        note_corpus: pd.DataFrame,
    ) -> pd.DataFrame:
        return build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=self.estimator_factory,
            sentiment_fn=self.sentiment_fn,
        )

    def evaluate_downstream(
        self,
        final_model_table: pd.DataFrame,
        downstream_estimator_factory_resolver: (
        DownstreamEstimatorFactoryResolver | None
    ) = None,
    ) -> pd.DataFrame:
        return evaluate_downstream_predictions(
            final_model_table=final_model_table,
            estimator_factory=self.estimator_factory,
            downstream_estimator_factory_resolver=downstream_estimator_factory_resolver,
            split_fn=self.split_fn,
            auc_fn=self.auc_fn,
            repetitions=self.repetitions,
        )

    def run(
        self,
        feature_matrix: pd.DataFrame,
        note_labels: pd.DataFrame,
        note_corpus: pd.DataFrame,
        demographics: pd.DataFrame | None = None,
        eol_cohort: pd.DataFrame | None = None,
        treatment_totals: pd.DataFrame | None = None,
        acuity_scores: pd.DataFrame | None = None,
        final_model_table: pd.DataFrame | None = None,
        include_downstream_weight_summary: bool = False,
        include_cdf_plot_data: bool = False,
        precomputed_mistrust_scores: pd.DataFrame | None = None,
        score_columns: Sequence[str] | None = None,
        feature_configurations: Mapping[str, Sequence[str]] | None = None,
        downstream_estimator_factory_resolver: (
        DownstreamEstimatorFactoryResolver | None
    ) = None,
    ) -> EOLMistrustModelOutputs:
        """Return model-stage outputs only.

        For the full end-to-end pipeline including dataset-layer artifacts
        (base_admissions, all_cohort, eol_cohort, etc.) use
        ``build_eol_mistrust_outputs`` in ``examples/eol_mistrust.py``
        instead — that is the canonical single entry point.
        """
        return run_full_eol_mistrust_modeling(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            demographics=demographics,
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
            acuity_scores=acuity_scores,
            final_model_table=final_model_table,
            estimator_factory=self.estimator_factory,
            downstream_estimator_factory_resolver=downstream_estimator_factory_resolver,
            sentiment_fn=self.sentiment_fn,
            split_fn=self.split_fn,
            auc_fn=self.auc_fn,
            repetitions=self.repetitions,
            include_downstream_weight_summary=include_downstream_weight_summary,
            include_cdf_plot_data=include_cdf_plot_data,
            precomputed_mistrust_scores=precomputed_mistrust_scores,
            score_columns=score_columns,
            feature_configurations=feature_configurations,
        )


def _default_eol_mistrust_data_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "EOL_Workspace"
        / "eol_mistrust_required_combined"
    )


def _default_eol_mistrust_slice_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path(__file__).resolve().parents[2]
        / "EOL_Workspace"
        / "eol_mistrust_runs"
        / f"e2e_1pct_gpu_{timestamp}"
    )


def _log_eol_mistrust_runner(start_time: float, message: str) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def _write_optional_runner_csv(output_dir: Path, name: str, value) -> None:
    if isinstance(value, pd.DataFrame):
        value.to_csv(output_dir / f"{name}.csv", index=False)


def run_eol_mistrust_gpu_slice(
    *,
    root: Path | str | None = None,
    sample_fraction: float = 0.01,
    slice_seed: int = 5,
    repetitions: int = 100,
    note_chunksize: int = 100_000,
    chartevent_chunksize: int = 500_000,
    output_dir: Path | str | None = None,
    allow_online_hf: bool = False,
) -> dict[str, object]:
    """Run the EOL mistrust pipeline on a deterministic GPU-backed cohort slice."""

    start_time = time.time()
    resolved_root = _default_eol_mistrust_data_root() if root is None else Path(root)

    if not allow_online_hf:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    output_path = (
        _default_eol_mistrust_slice_output_dir()
        if output_dir is None
        else Path(output_dir)
    )
    output_path.mkdir(parents=True, exist_ok=True)

    cuda_available = False
    gpu_name = None
    cuda_peak_mb = None

    try:
        torch_module = importlib.import_module("torch")
        cuda_available = bool(
            getattr(torch_module, "cuda", None)
            and torch_module.cuda.is_available()
        )
        gpu_name = torch_module.cuda.get_device_name(0) if cuda_available else None
        if cuda_available:
            torch_module.cuda.empty_cache()
    except ModuleNotFoundError:  # pragma: no cover
        torch_module = None

    _log_eol_mistrust_runner(start_time, "Loading cached sentiment model")
    warmup_started = time.time()
    warmup_sentiment = _load_transformers_sentiment()
    _ = warmup_sentiment("patient is calm and cooperative.")
    warmup_seconds = round(time.time() - warmup_started, 2)

    if (
        cuda_available
        and torch_module is not None
        and hasattr(torch_module.cuda, "reset_peak_memory_stats")
    ):
        torch_module.cuda.reset_peak_memory_stats()

    example_module = importlib.import_module("examples.eol_mistrust")
    load_eol_mistrust_tables = getattr(example_module, "load_eol_mistrust_tables")

    _log_eol_mistrust_runner(start_time, f"Loading tables from {resolved_root}")
    raw_tables, materialized_views = load_eol_mistrust_tables(resolved_root)

    from pyhealth.datasets.eol_mistrust import (
        build_acuity_scores,
        build_all_cohort,
        build_base_admissions,
        build_chartevent_artifacts_from_csv,
        build_demographics_table,
        build_eol_cohort,
        build_final_model_table_from_code_status_targets,
        build_note_artifacts_from_csv,
        build_treatment_totals,
        write_minimal_deliverables,
    )

    admissions = raw_tables["admissions"]
    patients = raw_tables["patients"]
    icustays = raw_tables["icustays"]
    d_items = raw_tables["d_items"]

    base_full = build_base_admissions(admissions, patients)
    all_cohort_full = build_all_cohort(base_full, icustays)
    sample_n = max(1, int(len(all_cohort_full) * sample_fraction))
    sampled_hadm = (
        all_cohort_full[["hadm_id"]]
        .sample(n=sample_n, random_state=slice_seed)
        .sort_values("hadm_id")
        .reset_index(drop=True)
    )
    sampled_hadm_ids = set(
        pd.to_numeric(sampled_hadm["hadm_id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    admissions_slice = admissions.loc[
        admissions["hadm_id"].isin(sampled_hadm_ids)
    ].copy()
    subject_ids = set(
        pd.to_numeric(admissions_slice["subject_id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    patients_slice = patients.loc[patients["subject_id"].isin(subject_ids)].copy()
    icustays_slice = icustays.loc[icustays["hadm_id"].isin(sampled_hadm_ids)].copy()
    icustay_ids = set(
        pd.to_numeric(icustays_slice["icustay_id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    ventdurations_slice = materialized_views["ventdurations"].loc[
        materialized_views["ventdurations"]["icustay_id"].isin(icustay_ids)
    ].copy()
    vasopressordurations_slice = materialized_views["vasopressordurations"].loc[
        materialized_views["vasopressordurations"]["icustay_id"].isin(icustay_ids)
    ].copy()
    oasis_slice = materialized_views["oasis"].loc[
        materialized_views["oasis"]["hadm_id"].isin(sampled_hadm_ids)
    ].copy()
    sapsii_slice = materialized_views["sapsii"].loc[
        materialized_views["sapsii"]["hadm_id"].isin(sampled_hadm_ids)
    ].copy()

    _log_eol_mistrust_runner(
        start_time,
        (
            "Prepared slice with "
            f"{len(sampled_hadm_ids)} admissions, "
            f"{len(subject_ids)} patients, "
            f"{len(icustay_ids)} ICU stays"
        ),
    )

    base_admissions = build_base_admissions(admissions_slice, patients_slice)
    demographics = build_demographics_table(base_admissions)
    all_cohort = build_all_cohort(base_admissions, icustays_slice)
    eol_cohort = build_eol_cohort(base_admissions, demographics)
    treatment_totals = build_treatment_totals(
        icustays=icustays_slice,
        ventdurations=ventdurations_slice,
        vasopressordurations=vasopressordurations_slice,
    )
    acuity_scores = build_acuity_scores(oasis_slice, sapsii_slice)

    noteevents_csv_path = resolved_root / "mimiciii_notes" / "noteevents.csv"
    chartevents_csv_path = resolved_root / "mimiciii_clinical" / "chartevents.csv"

    notes_started = time.time()
    _log_eol_mistrust_runner(
        start_time,
        "Streaming notes to build sentiment corpus and note-derived labels",
    )
    note_corpus, note_labels = build_note_artifacts_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_cohort["hadm_id"],
        corpus_categories=None,
        label_categories=None,
        chunksize=note_chunksize,
    )
    note_present_hadm_ids = _note_present_hadm_ids(note_corpus)
    all_cohort = all_cohort.loc[
        all_cohort["hadm_id"].isin(note_present_hadm_ids)
    ].copy()
    note_corpus = note_corpus.loc[
        note_corpus["hadm_id"].isin(note_present_hadm_ids)
    ].copy()
    note_labels = note_labels.loc[
        note_labels["hadm_id"].isin(note_present_hadm_ids)
    ].copy()
    _log_eol_mistrust_runner(
        start_time,
        f"Retained {len(note_present_hadm_ids)} ALL-cohort admissions "
        "with at least one non-error note",
    )
    note_stage_seconds = round(time.time() - notes_started, 2)

    chartevents_started = time.time()
    _log_eol_mistrust_runner(
        start_time,
        "Streaming chartevents to build feature matrix and code-status targets",
    )
    feature_matrix, code_status_targets = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        all_hadm_ids=note_present_hadm_ids,
        chunksize=chartevent_chunksize,
    )
    chartevent_stage_seconds = round(time.time() - chartevents_started, 2)

    model_started = time.time()
    _log_eol_mistrust_runner(start_time, "Running EOL mistrust model pipeline")
    model = EOLMistrustModel(repetitions=repetitions)
    mistrust_scores = model.build_mistrust_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
    )
    final_model_table = build_final_model_table_from_code_status_targets(
        demographics=demographics,
        all_cohort=all_cohort,
        admissions=admissions_slice,
        code_status_targets=code_status_targets,
        mistrust_scores=mistrust_scores,
    )
    model_outputs = model.run(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
        demographics=demographics,
        eol_cohort=eol_cohort,
        treatment_totals=treatment_totals,
        acuity_scores=acuity_scores,
        final_model_table=final_model_table,
        include_downstream_weight_summary=False,
        include_cdf_plot_data=False,
    )
    model_stage_seconds = round(time.time() - model_started, 2)

    artifacts: dict[str, object] = {
        "base_admissions": base_admissions,
        "demographics": demographics,
        "all_cohort": all_cohort,
        "eol_cohort": eol_cohort,
        "treatment_totals": treatment_totals,
        "note_corpus": note_corpus,
        "note_labels": note_labels,
        "chartevent_feature_matrix": feature_matrix,
        "acuity_scores": acuity_scores,
        "mistrust_scores": mistrust_scores,
        "final_model_table": final_model_table,
        **model_outputs,
    }

    write_minimal_deliverables(
        {
            "base_admissions": base_admissions,
            "eol_cohort": eol_cohort,
            "all_cohort": all_cohort,
            "treatment_totals": treatment_totals,
            "chartevent_feature_matrix": feature_matrix,
            "note_labels": note_labels,
            "mistrust_scores": mistrust_scores,
            "acuity_scores": acuity_scores,
            "final_model_table": final_model_table,
        },
        output_dir=output_path,
    )

    for key in (
        "downstream_auc_results",
        "race_gap_results",
        "race_treatment_results",
        "race_treatment_by_acuity_results",
        "trust_treatment_results",
        "trust_treatment_by_acuity_results",
        "acuity_correlations",
    ):
        _write_optional_runner_csv(output_path, key, artifacts.get(key))

    feature_weight_summaries = artifacts.get("feature_weight_summaries", {})
    if isinstance(feature_weight_summaries, dict):
        summary_dir = output_path / "feature_weight_summaries"
        summary_dir.mkdir(exist_ok=True)
        for model_name, tables in feature_weight_summaries.items():
            if not isinstance(tables, dict):
                continue
            for table_name, table in tables.items():
                if isinstance(table, pd.DataFrame):
                    table.to_csv(
                        summary_dir / f"{model_name}_{table_name}.csv", index=False
                    )

    if cuda_available and torch_module is not None:
        cuda_peak_mb = round(
            torch_module.cuda.max_memory_allocated() / (1024 * 1024), 2
        )

    downstream_results = artifacts["downstream_auc_results"]
    if not isinstance(downstream_results, pd.DataFrame):
        raise ValueError("Expected downstream_auc_results to be a pandas DataFrame.")

    target_positives = {
        "left_ama_positive": int(
            pd.to_numeric(final_model_table["left_ama"], errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "code_status_positive": int(
            pd.to_numeric(final_model_table["code_status_dnr_dni_cmo"], errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "mortality_positive": int(
            pd.to_numeric(final_model_table["in_hospital_mortality"], errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        ),
    }
    summary = {
        "root": str(resolved_root.resolve()),
        "output_dir": str(output_path.resolve()),
        "sample_fraction": sample_fraction,
        "slice_seed": slice_seed,
        "repetitions": repetitions,
        "offline_hf": not allow_online_hf,
        "gpu": {
            "cuda_available": cuda_available,
            "device_name": gpu_name,
            "cuda_peak_memory_mb": cuda_peak_mb,
        },
        "counts": {
            "base_full_rows": int(len(base_full)),
            "all_cohort_full_rows": int(len(all_cohort_full)),
            "slice_rows": int(len(all_cohort)),
            "eol_slice_rows": int(len(eol_cohort)),
        },
        "note_label_positives": {
            "noncompliance_label": int(
                pd.to_numeric(note_labels["noncompliance_label"], errors="coerce")
                .fillna(0)
                .astype(int)
                .sum()
            ),
            "autopsy_label": int(
                pd.to_numeric(note_labels["autopsy_label"], errors="coerce")
                .fillna(0)
                .astype(int)
                .sum()
            ),
        },
        "target_positives": target_positives,
        "artifact_shapes": {
            key: list(value.shape)
            for key, value in artifacts.items()
            if isinstance(value, pd.DataFrame)
        },
        "stage_seconds": {
            "sentiment_warmup": warmup_seconds,
            "note_streaming": note_stage_seconds,
            "chartevent_streaming": chartevent_stage_seconds,
            "model_pipeline": model_stage_seconds,
            "total": round(time.time() - start_time, 2),
        },
        "downstream_auc_results": downstream_results.to_dict(orient="records"),
    }

    (output_path / "run_summary.json").write_text(json.dumps(summary, indent=2))
    _log_eol_mistrust_runner(
        start_time,
        f"Run complete; artifacts written to {output_path.resolve()}",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parse_eol_mistrust_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EOL mistrust pipeline on a deterministic "
            "GPU-backed cohort slice."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_eol_mistrust_data_root(),
        help="Root directory containing the combined EOL mistrust CSV exports.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.01,
        help="Fraction of the ICU-linked ALL cohort to sample.",
    )
    parser.add_argument(
        "--slice-seed",
        type=int,
        default=5,
        help="Deterministic pandas sample seed for the ALL-cohort slice.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
        help="Number of downstream repeated 60/40 evaluations.",
    )
    parser.add_argument(
        "--note-chunksize",
        type=int,
        default=100_000,
        help="Chunk size for noteevents streaming.",
    )
    parser.add_argument(
        "--chartevent-chunksize",
        type=int,
        default=500_000,
        help="Chunk size for chartevents streaming.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Defaults to EOL_Workspace/eol_mistrust_runs/<timestamp>."
        ),
    )
    parser.add_argument(
        "--allow-online-hf",
        action="store_true",
        help=(
            "Allow Hugging Face network access instead of forcing "
            "offline cached model loading."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_eol_mistrust_cli_args()
    run_eol_mistrust_gpu_slice(
        root=args.root,
        sample_fraction=args.sample_fraction,
        slice_seed=args.slice_seed,
        repetitions=args.repetitions,
        note_chunksize=args.note_chunksize,
        chartevent_chunksize=args.chartevent_chunksize,
        output_dir=args.output_dir,
        allow_online_hf=args.allow_online_hf,
    )


__all__ = [
    "EOLMistrustModelOutputs",
    "BASELINE_FEATURE_COLUMNS",
    "DOWNSTREAM_FEATURE_CONFIGS",
    "DOWNSTREAM_TASK_MAP",
    "EOLMistrustModel",
    "MISTRUST_SCORE_COLUMNS",
    "RACE_FEATURE_COLUMNS",
    "build_autopsy_feature_weight_summary",
    "build_autopsy_mistrust_scores",
    "build_empirical_cdf_curve",
    "build_logistic_estimator_factory",
    "build_logistic_cv_estimator_factory",
    "build_mistrust_score_table",
    "build_negative_sentiment_mistrust_scores",
    "build_noncompliance_feature_weight_summary",
    "build_noncompliance_mistrust_scores",
    "build_proxy_feature_weight_summary",
    "build_proxy_probability_scores",
    "build_race_based_treatment_cdf_plot_data",
    "build_trust_based_treatment_cdf_plot_data",
    "evaluate_downstream_average_weights",
    "evaluate_downstream_predictions",
    "fit_proxy_mistrust_model",
    "get_downstream_feature_configurations",
    "get_downstream_task_map",
    "run_acuity_control_analysis",
    "run_eol_mistrust_gpu_slice",
    "run_full_eol_mistrust_modeling",
    "run_race_based_treatment_analysis",
    "run_race_based_treatment_analysis_by_acuity",
    "run_race_gap_analysis",
    "run_trust_based_treatment_analysis",
    "run_trust_based_treatment_analysis_by_acuity",
    "summarize_feature_weights",
    "z_normalize_scores",
]


if __name__ == "__main__":
    main()
