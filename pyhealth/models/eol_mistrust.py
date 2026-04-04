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

import importlib
from collections import OrderedDict
from itertools import combinations
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pyhealth.tasks.eol_mistrust import get_eol_mistrust_task_map

try:
    from scipy.stats import mannwhitneyu, pearsonr  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mannwhitneyu = None
    pearsonr = None

try:
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-error
    from sklearn.metrics import roc_auc_score  # pylint: disable=import-error
    from sklearn.model_selection import train_test_split  # pylint: disable=import-error
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

    def roc_auc_score(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise ModuleNotFoundError(
            "scikit-learn is required for downstream AUC evaluation."
        )


RACE_WHITE = "WHITE"
RACE_BLACK = "BLACK"
DEFAULT_LOGISTIC_C = 0.1

MISTRUST_SCORE_COLUMNS = [
    "noncompliance_score_z",
    "autopsy_score_z",
    "negative_sentiment_score_z",
]

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
        ("Baseline + Noncompliant", list(BASELINE_FEATURE_COLUMNS + ["noncompliance_score_z"])),
        ("Baseline + Autopsy", list(BASELINE_FEATURE_COLUMNS + ["autopsy_score_z"])),
        (
            "Baseline + Neg-Sentiment",
            list(BASELINE_FEATURE_COLUMNS + ["negative_sentiment_score_z"]),
        ),
        (
            "Baseline + ALL",
            list(BASELINE_FEATURE_COLUMNS + RACE_FEATURE_COLUMNS + MISTRUST_SCORE_COLUMNS),
        ),
    ]
)


_SENTIMENT_BACKEND: Callable[[str], tuple[float, float]] | None = None


def _load_transformers_sentiment() -> Callable[[str], tuple[float, float]]:
    """Load the project-standard transformers sentiment pipeline.

    GPU is used first when CUDA is available; otherwise the backend falls back
    to CPU without changing the public scorer interface.
    """

    transformers_module = importlib.import_module("transformers")
    torch_module = importlib.import_module("torch")

    pipeline_factory = getattr(transformers_module, "pipeline", None)
    if not callable(pipeline_factory):
        raise ModuleNotFoundError("transformers.pipeline is unavailable in the current environment.")

    try:  # pragma: no cover - logging surface depends on transformers version
        transformers_logging = importlib.import_module("transformers.utils.logging")
        set_verbosity_error = getattr(transformers_logging, "set_verbosity_error", None)
        if callable(set_verbosity_error):
            set_verbosity_error()
    except Exception:
        pass

    use_cuda = bool(getattr(torch_module, "cuda", None) and torch_module.cuda.is_available())
    device = 0 if use_cuda else -1
    classifier = pipeline_factory(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

    def _transformers_sentiment(text: str) -> tuple[float, float]:
        cleaned = " ".join(str(text).split())
        if not cleaned:
            return (0.0, 0.0)
        result = classifier(cleaned[:2048], truncation=True)[0]
        label = str(result.get("label", "")).upper()
        score = float(result.get("score", 0.0))
        polarity = score if "POS" in label else -score
        return (polarity, 0.0)

    return _transformers_sentiment


def _default_sentiment_backend(text: str) -> tuple[float, float]:
    """Resolve and cache the default transformers sentiment backend lazily."""

    global _SENTIMENT_BACKEND
    if _SENTIMENT_BACKEND is None:
        _SENTIMENT_BACKEND = _load_transformers_sentiment()
    return _SENTIMENT_BACKEND(text)


pattern_sentiment = _default_sentiment_backend


def _require_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{df_name} is missing required columns: {missing_str}")


def _prepare_note_text_for_sentiment(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).split())


def _default_estimator_factory() -> object:
    return LogisticRegression(
        penalty="l1",
        C=DEFAULT_LOGISTIC_C,
        solver="liblinear",
        max_iter=1000,
    )


def _extract_positive_class_probabilities(probabilities) -> np.ndarray:
    """Validate predict_proba output and return the positive-class column."""

    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 2 or probability_array.shape[1] < 2:
        raise IndexError(
            "Estimator `predict_proba` output must have shape (n_samples, n_classes>=2)."
        )
    return probability_array[:, 1]


def _score_column_name(label_column: str) -> str:
    if label_column.endswith("_label"):
        return f"{label_column[:-6]}_score"
    return f"{label_column}_score"


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
            usable = final_model_table[["hadm_id", target_column, *feature_columns]].dropna().copy()
            usable = usable.sort_values("hadm_id").reset_index(drop=True)
            y = pd.to_numeric(usable[target_column], errors="coerce")
            X = usable[feature_columns]
            yield task_name, target_column, config_name, feature_columns, usable, X, y


def _prepare_proxy_training_frame(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    _require_columns(feature_matrix, ["hadm_id"], "feature_matrix")
    _require_columns(note_labels, ["hadm_id", label_column], "note_labels")

    feature_columns = [column for column in feature_matrix.columns if column != "hadm_id"]
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
        return float("nan"), float("nan"), float("nan"), float("nan"), len(left), len(right)

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

    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(float)
    series = series.sort_values().reset_index(drop=True)
    if series.empty:
        return pd.DataFrame(columns=["x", "cdf"])
    cdf = (np.arange(1, len(series) + 1) / len(series)).astype(float)
    return pd.DataFrame({"x": series, "cdf": cdf})


def get_downstream_feature_configurations() -> OrderedDict[str, list[str]]:
    """Return the six required downstream feature configurations."""

    return OrderedDict((name, list(columns)) for name, columns in DOWNSTREAM_FEATURE_CONFIGS.items())


def get_downstream_task_map() -> OrderedDict[str, str]:
    """Return the three required downstream prediction targets."""

    return OrderedDict(DOWNSTREAM_TASK_MAP)


def fit_proxy_mistrust_model(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
):
    """Fit the L1 logistic proxy model on the full ALL cohort."""

    merged, feature_columns = _prepare_proxy_training_frame(feature_matrix, note_labels, label_column)
    estimator = _default_estimator_factory() if estimator_factory is None else estimator_factory()
    estimator.fit(merged[feature_columns], merged[label_column].astype(int))
    return estimator


def build_proxy_probability_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Fit a proxy logistic model and return positive-class probabilities."""

    merged, feature_columns = _prepare_proxy_training_frame(feature_matrix, note_labels, label_column)
    estimator = _default_estimator_factory() if estimator_factory is None else estimator_factory()
    estimator.fit(merged[feature_columns], merged[label_column].astype(int))
    probabilities = estimator.predict_proba(merged[feature_columns])
    positive_class = _extract_positive_class_probabilities(probabilities)

    scores = pd.DataFrame(
        {
            "hadm_id": merged["hadm_id"],
            _score_column_name(label_column): positive_class.astype(float),
        }
    )
    return scores.sort_values("hadm_id").drop_duplicates("hadm_id").reset_index(drop=True)


def build_noncompliance_mistrust_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    return build_proxy_probability_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column="noncompliance_label",
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
        label_column="autopsy_label",
        estimator_factory=estimator_factory,
    )


def build_negative_sentiment_mistrust_scores(
    note_corpus: pd.DataFrame,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Compute negative-sentiment mistrust from whitespace-tokenized note text."""

    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")
    scorer = pattern_sentiment if sentiment_fn is None else sentiment_fn

    cleaned = note_corpus.copy()
    cleaned["note_text"] = cleaned["note_text"].map(_prepare_note_text_for_sentiment)
    cleaned["negative_sentiment_score"] = cleaned["note_text"].map(
        lambda text: float(-1.0 * scorer(text)[0])
    )
    return cleaned[["hadm_id", "negative_sentiment_score"]].sort_values("hadm_id").reset_index(drop=True)


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
            if column != "hadm_id" and (column.endswith("_score") or column.endswith("_score_z"))
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

    _require_columns(note_labels, ["hadm_id", "noncompliance_label", "autopsy_label"], "note_labels")
    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")

    noncompliance = build_noncompliance_mistrust_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        estimator_factory=estimator_factory,
    )
    autopsy = build_autopsy_mistrust_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        estimator_factory=estimator_factory,
    )
    sentiment = build_negative_sentiment_mistrust_scores(
        note_corpus=note_corpus,
        sentiment_fn=sentiment_fn,
    )

    merged = (
        noncompliance.merge(autopsy, on="hadm_id", how="inner", validate="one_to_one")
        .merge(sentiment, on="hadm_id", how="inner", validate="one_to_one")
        .sort_values("hadm_id")
    )

    normalized = z_normalize_scores(
        merged,
        columns=["noncompliance_score", "autopsy_score", "negative_sentiment_score"],
    )
    normalized = normalized.rename(
        columns={
            "noncompliance_score": "noncompliance_score_z",
            "autopsy_score": "autopsy_score_z",
            "negative_sentiment_score": "negative_sentiment_score_z",
        }
    )
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

    summary = pd.DataFrame({"feature": list(feature_columns), "weight": weights.astype(float)})
    summary = summary.sort_values(["weight", "feature"], ascending=[False, True]).reset_index(drop=True)
    positive = summary.head(top_n).reset_index(drop=True)
    negative = summary.sort_values(["weight", "feature"], ascending=[True, True]).head(top_n).reset_index(drop=True)
    return {"all": summary, "positive": positive, "negative": negative}


def build_proxy_feature_weight_summary(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Fit a proxy model and summarize the learned coefficient weights."""

    merged, feature_columns = _prepare_proxy_training_frame(feature_matrix, note_labels, label_column)
    estimator = _default_estimator_factory() if estimator_factory is None else estimator_factory()
    estimator.fit(merged[feature_columns], merged[label_column].astype(int))
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
        label_column="noncompliance_label",
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
        label_column="autopsy_label",
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
        statistic, pvalue, median_black, median_white, n_black, n_white = _make_metric_result(
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
    _require_columns(treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals")

    merged = eol_cohort.merge(treatment_totals, on="hadm_id", how="left", validate="one_to_one")
    merged = merged.loc[merged[race_column].isin({RACE_WHITE, RACE_BLACK})].copy()

    rows: list[dict[str, float | int | str]] = []
    for column in treatment_columns:
        usable = merged.loc[merged[column].notna()].copy()
        black = usable.loc[usable[race_column] == RACE_BLACK, column]
        white = usable.loc[usable[race_column] == RACE_WHITE, column]
        statistic, pvalue, median_black, median_white, n_black, n_white = _make_metric_result(
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
    _require_columns(treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals")
    _require_columns(acuity_scores, ["hadm_id", acuity_column], "acuity_scores")

    merged = (
        eol_cohort.merge(treatment_totals, on="hadm_id", how="left", validate="one_to_one")
        .merge(acuity_scores[["hadm_id", acuity_column]], on="hadm_id", how="inner", validate="one_to_one")
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
            statistic, pvalue, median_black, median_white, n_black, n_white = _make_metric_result(
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
    """Build plot-ready CDF curves and median markers for race-based treatment analysis."""

    _require_columns(eol_cohort, ["hadm_id", race_column], "eol_cohort")
    _require_columns(treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals")

    merged = eol_cohort.merge(treatment_totals, on="hadm_id", how="left", validate="one_to_one")
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
            median = pd.to_numeric(values, errors="coerce").dropna().astype(float).median()
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
    _require_columns(treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals")

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = (
        eol_cohort.merge(treatment_totals, on="hadm_id", how="left", validate="one_to_one")
        .merge(mistrust_scores[["hadm_id", *columns]], on="hadm_id", how="inner", validate="one_to_one")
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
            usable = merged.loc[merged[treatment].notna() & merged[metric].notna()].copy()
            usable = usable.sort_values([metric, "hadm_id"], ascending=[False, True]).reset_index(drop=True)
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
            statistic, pvalue, median_high, median_low, n_high, n_low = _make_metric_result(
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
    _require_columns(treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals")
    _require_columns(acuity_scores, ["hadm_id", acuity_column], "acuity_scores")

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = (
        eol_cohort.merge(treatment_totals, on="hadm_id", how="left", validate="one_to_one")
        .merge(
            mistrust_scores[["hadm_id", *columns]],
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(acuity_scores[["hadm_id", acuity_column]], on="hadm_id", how="inner", validate="one_to_one")
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
            derived_groups[(str(row.severity_bin), str(row.treatment))] = int(row.n_black)

    rows: list[dict[str, float | int | str]] = []
    for metric in columns:
        for treatment in treatment_columns:
            for severity_bin in ("low", "medium", "high"):
                usable = merged.loc[
                    (merged["severity_bin"] == severity_bin)
                    & merged[treatment].notna()
                    & merged[metric].notna()
                ].copy()
                usable = usable.sort_values([metric, "hadm_id"], ascending=[False, True]).reset_index(
                    drop=True
                )
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
                statistic, pvalue, median_high, median_low, n_high, n_low = _make_metric_result(
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
    """Build plot-ready CDF curves and median markers for trust-based treatment analysis."""

    _require_columns(eol_cohort, ["hadm_id"], "eol_cohort")
    _require_columns(mistrust_scores, ["hadm_id"], "mistrust_scores")
    _require_columns(treatment_totals, ["hadm_id", *treatment_columns], "treatment_totals")

    columns = list(MISTRUST_SCORE_COLUMNS if score_columns is None else score_columns)
    _require_columns(mistrust_scores, columns, "mistrust_scores")

    merged = (
        eol_cohort.merge(treatment_totals, on="hadm_id", how="left", validate="one_to_one")
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
            usable = merged.loc[merged[treatment].notna() & merged[metric].notna()].copy()
            usable = usable.sort_values([metric, "hadm_id"], ascending=[False, True]).reset_index(drop=True)
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
                median = pd.to_numeric(values, errors="coerce").dropna().astype(float).median()
                medians.append(
                    {
                        "metric": metric,
                        "treatment": treatment,
                        "group": label,
                        "median": float(median) if not pd.isna(median) else float("nan"),
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
    split_fn: Callable[..., tuple] | None = None,
    repetitions: int = 100,
    test_size: float = 0.4,
) -> pd.DataFrame:
    """Average downstream regularized model weights across repeated 60/40 splits."""

    splitter = train_test_split if split_fn is None else split_fn
    rows: list[dict[str, float | int | str]] = []

    for task_name, target_column, config_name, feature_columns, usable, X, y in _iter_downstream_jobs(
        final_model_table,
        feature_configurations=feature_configurations,
        task_map=task_map,
    ):
        collected_weights: list[np.ndarray] = []

        for random_state in range(repetitions):
            if usable.empty or y.nunique(dropna=True) < 2:
                continue

            X_train, X_test, y_train, y_test = splitter(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            del X_test  # coefficients come from the fitted train-side model only
            y_train = pd.Series(y_train)
            y_test = pd.Series(y_test)
            if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
                continue

            estimator = _default_estimator_factory() if estimator_factory is None else estimator_factory()
            estimator.fit(X_train, y_train.astype(int))
            coefficients = np.asarray(getattr(estimator, "coef_", None), dtype=float)
            if coefficients.ndim != 2 or coefficients.shape[0] == 0:
                raise ValueError(
                    "Downstream estimator must expose `coef_` with shape (n_classes, n_features)."
                )
            weights = coefficients[0]
            if len(weights) != len(feature_columns):
                raise ValueError("Downstream feature columns must align with estimator coefficients.")
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
                    "weight_mean": float(weight_mean[index]) if not np.isnan(weight_mean[index]) else float("nan"),
                    "weight_std": float(weight_std[index]) if not np.isnan(weight_std[index]) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def evaluate_downstream_predictions(
    final_model_table: pd.DataFrame,
    feature_configurations: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, str] | None = None,
    estimator_factory: Callable[[], object] | None = None,
    split_fn: Callable[..., tuple] | None = None,
    auc_fn: Callable[[Iterable[int], Iterable[float]], float] | None = None,
    repetitions: int = 100,
    test_size: float = 0.4,
) -> pd.DataFrame:
    """Run repeated 60/40 downstream AUC evaluation across all tasks/configs."""

    splitter = train_test_split if split_fn is None else split_fn
    metric = roc_auc_score if auc_fn is None else auc_fn

    rows: list[dict[str, float | int | str]] = []
    for task_name, target_column, config_name, feature_columns, usable, X, y in _iter_downstream_jobs(
        final_model_table,
        feature_configurations=feature_configurations,
        task_map=task_map,
    ):
        auc_values: list[float] = []

        for random_state in range(repetitions):
            if usable.empty or y.nunique(dropna=True) < 2:
                auc_values.append(float("nan"))
                continue

            X_train, X_test, y_train, y_test = splitter(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            y_train = pd.Series(y_train)
            y_test = pd.Series(y_test)

            if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
                auc_values.append(float("nan"))
                continue

            estimator = _default_estimator_factory() if estimator_factory is None else estimator_factory()
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
                "auc_mean": float(auc_series.mean()) if auc_series.notna().any() else float("nan"),
                "auc_std": float(auc_series.std(ddof=0)) if auc_series.notna().any() else float("nan"),
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
    """Plot grouped empirical CDF curves with dotted median lines."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("matplotlib is required for EOL mistrust CDF plotting.") from exc

    if ax is None:
        _, ax = plt.subplots()

    ordered_curves = curves.copy()
    if not ordered_curves.empty:
        ordered_curves = ordered_curves.sort_values([group_column, x_column]).reset_index(drop=True)
    for group_value, group_df in ordered_curves.groupby(group_column, sort=False):
        ax.plot(group_df[x_column], group_df[y_column], label=str(group_value))

    for row in medians.itertuples(index=False):
        ax.axvline(
            getattr(row, median_column),
            linestyle=getattr(row, "line_style", "dotted"),
            label=f"{getattr(row, group_column)} median",
        )
    return ax


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
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
    split_fn: Callable[..., tuple] | None = None,
    auc_fn: Callable[[Iterable[int], Iterable[float]], float] | None = None,
    repetitions: int = 100,
    include_downstream_weight_summary: bool = False,
    include_cdf_plot_data: bool = False,
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
    """Run the end-to-end model-stage workflow and collect its outputs."""

    mistrust_scores = build_mistrust_score_table(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
        estimator_factory=estimator_factory,
        sentiment_fn=sentiment_fn,
    )
    outputs: dict[str, pd.DataFrame | dict[str, pd.DataFrame]] = {
        "mistrust_scores": mistrust_scores,
        "feature_weight_summaries": {
            "noncompliance": build_noncompliance_feature_weight_summary(
                feature_matrix=feature_matrix,
                note_labels=note_labels,
                estimator_factory=estimator_factory,
            ),
            "autopsy": build_autopsy_feature_weight_summary(
                feature_matrix=feature_matrix,
                note_labels=note_labels,
                estimator_factory=estimator_factory,
            ),
        },
    }

    if demographics is not None:
        outputs["race_gap_results"] = run_race_gap_analysis(mistrust_scores, demographics)

    if eol_cohort is not None and treatment_totals is not None:
        outputs["race_treatment_results"] = run_race_based_treatment_analysis(
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
        )
        outputs["trust_treatment_results"] = run_trust_based_treatment_analysis(
            eol_cohort=eol_cohort,
            mistrust_scores=mistrust_scores,
            treatment_totals=treatment_totals,
        )
        if acuity_scores is not None:
            outputs["race_treatment_by_acuity_results"] = run_race_based_treatment_analysis_by_acuity(
                eol_cohort=eol_cohort,
                treatment_totals=treatment_totals,
                acuity_scores=acuity_scores,
            )
            outputs["trust_treatment_by_acuity_results"] = run_trust_based_treatment_analysis_by_acuity(
                eol_cohort=eol_cohort,
                mistrust_scores=mistrust_scores,
                treatment_totals=treatment_totals,
                acuity_scores=acuity_scores,
            )
        if include_cdf_plot_data:
            outputs["race_treatment_cdf_plot_data"] = build_race_based_treatment_cdf_plot_data(
                eol_cohort=eol_cohort,
                treatment_totals=treatment_totals,
            )
            outputs["trust_treatment_cdf_plot_data"] = build_trust_based_treatment_cdf_plot_data(
                eol_cohort=eol_cohort,
                mistrust_scores=mistrust_scores,
                treatment_totals=treatment_totals,
            )

    if acuity_scores is not None:
        outputs["acuity_correlations"] = run_acuity_control_analysis(
            mistrust_scores=mistrust_scores,
            acuity_scores=acuity_scores,
        )

    if final_model_table is not None:
        downstream = final_model_table.copy()
        if not set(MISTRUST_SCORE_COLUMNS).issubset(downstream.columns):
            downstream = downstream.merge(mistrust_scores, on="hadm_id", how="left")
        outputs["downstream_auc_results"] = evaluate_downstream_predictions(
            final_model_table=downstream,
            estimator_factory=estimator_factory,
            split_fn=split_fn,
            auc_fn=auc_fn,
            repetitions=repetitions,
        )
        if include_downstream_weight_summary:
            outputs["downstream_weight_results"] = evaluate_downstream_average_weights(
                final_model_table=downstream,
                estimator_factory=estimator_factory,
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

    def evaluate_downstream(self, final_model_table: pd.DataFrame) -> pd.DataFrame:
        return evaluate_downstream_predictions(
            final_model_table=final_model_table,
            estimator_factory=self.estimator_factory,
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
    ) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
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
            sentiment_fn=self.sentiment_fn,
            split_fn=self.split_fn,
            auc_fn=self.auc_fn,
            repetitions=self.repetitions,
            include_downstream_weight_summary=include_downstream_weight_summary,
            include_cdf_plot_data=include_cdf_plot_data,
        )


normalize_mistrust_scores = z_normalize_scores
run_racial_gap_validation = run_race_gap_analysis
run_acuity_correlation_analysis = run_acuity_control_analysis
run_downstream_prediction_experiments = evaluate_downstream_predictions
build_mistrust_metrics = build_mistrust_score_table


__all__ = [
    "BASELINE_FEATURE_COLUMNS",
    "DOWNSTREAM_FEATURE_CONFIGS",
    "DOWNSTREAM_TASK_MAP",
    "EOLMistrustModel",
    "MISTRUST_SCORE_COLUMNS",
    "RACE_FEATURE_COLUMNS",
    "build_autopsy_feature_weight_summary",
    "build_autopsy_mistrust_scores",
    "build_empirical_cdf_curve",
    "build_mistrust_metrics",
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
    "normalize_mistrust_scores",
    "plot_grouped_treatment_cdf",
    "run_acuity_control_analysis",
    "run_acuity_correlation_analysis",
    "run_downstream_prediction_experiments",
    "run_full_eol_mistrust_modeling",
    "run_race_based_treatment_analysis",
    "run_race_based_treatment_analysis_by_acuity",
    "run_race_gap_analysis",
    "run_racial_gap_validation",
    "run_trust_based_treatment_analysis",
    "run_trust_based_treatment_analysis_by_acuity",
    "summarize_feature_weights",
    "z_normalize_scores",
]
