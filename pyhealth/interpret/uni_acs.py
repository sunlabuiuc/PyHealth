"""Convert SHAP feature contributions into integer clinical scores."""

import numpy as np
import pandas as pd
import shap
from csaps import csaps
from scipy.special import expit
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression


def find_novel_breakpoints(
    values: np.ndarray,
    shap_values: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """Find feature boundaries from zero crossings of smoothed SHAP values.

    Args:
        values: Feature values for training samples.
        shap_values: SHAP contributions for the same samples.
        n_bins: Number of bins used to smooth the SHAP relationship.

    Returns:
        Sorted breakpoints separating different risk regions.
    """
    values = np.asarray(values, dtype=float)
    shap_values = np.asarray(shap_values, dtype=float)

    if values.ndim != 1 or values.shape != shap_values.shape:
        raise ValueError("Values and SHAP must be 1D and aligned.")
    if not np.isfinite(values).all() or not np.isfinite(shap_values).all():
        raise ValueError("Inputs must be finite.")
    if n_bins < 3:
        raise ValueError("n_bins must be at least 3.")

    if np.unique(values).size < 3:
        return np.array([])

    # Calculate average SHAP contributions within equal-width bins
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    groups = np.searchsorted(edges[1:-1], values, side="right")
    counts = np.bincount(groups, minlength=n_bins)
    totals = np.bincount(groups, weights=shap_values, minlength=n_bins)

    valid = counts > 0
    means = np.full(n_bins, np.nan)
    means[valid] = totals[valid] / counts[valid]
    means = np.interp(centers, centers[valid], means[valid])

    # Smooth the SHAP curve and locate zero crossings
    xs = np.linspace(centers[0], centers[-1], 100)
    ys = np.asarray(csaps(centers, means, xs, smooth=0.5)).ravel()

    changes = np.where(np.signbit(ys[:-1]) != np.signbit(ys[1:]))[0]
    breakpoints = []

    for i in changes:
        x1, x2 = xs[i], xs[i + 1]
        y1, y2 = ys[i], ys[i + 1]
        crossing = x1 - y1 * (x2 - x1) / (y2 - y1)

        if np.any(values <= crossing) and np.any(values > crossing):
            breakpoints.append(crossing)

    return np.unique(breakpoints)


def build_score_table(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    top_k: int = 5,
    threshold: float = 0.05,
    method: str = "quantile",
) -> tuple[pd.DataFrame, float]:
    """Build an integer scoring table from feature SHAP contributions.

    Args:
        X: Training feature matrix.
        shap_values: SHAP contributions aligned with X.
        top_k: Number of important features to include.
        threshold: Minimum absolute SHAP value for selecting score units.
        method: Feature grouping method, either quantile or novel.

    Returns:
        Scoring table and SHAP unit used for integer conversion.
    """
    if method not in ("quantile", "novel"):
        raise ValueError("method must be 'quantile' or 'novel'")

    importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(importance)[::-1][:top_k]
    rows = []

    for i in top_indices:
        feature = X.columns[i]
        values = X[feature].to_numpy()
        contributions = shap_values[:, i]

        if method == "quantile":
            breaks = np.unique(np.quantile(values, [0.2, 0.5, 0.8]))
        else:
            breaks = find_novel_breakpoints(values, contributions)

        groups = np.digitize(values, breaks, right=True)

        for group in range(len(breaks) + 1):
            mask = groups == group
            if not mask.any():
                continue

            rows.append({
                "feature": feature,
                "group": group,
                "upper_bound": (
                    breaks[group] if group < len(breaks) else np.inf
                ),
                "mean_shap": contributions[mask].mean(),
            })

    table = pd.DataFrame(rows)

    valid = np.abs(table["mean_shap"]) >= threshold
    if not valid.any():
        raise ValueError("No SHAP values exceed the threshold.")

    unit = float(np.abs(table.loc[valid, "mean_shap"]).min())
    table["points"] = np.rint(table["mean_shap"] / unit).astype(int)

    return table, unit


def predict_score(X: pd.DataFrame, score_table: pd.DataFrame) -> np.ndarray:
    """Calculate integer scores using a fitted scoring table."""
    scores = np.zeros(len(X), dtype=int)

    for feature in score_table["feature"].unique():
        feature_table = score_table[
            score_table["feature"] == feature
        ].sort_values("group")

        bounds = feature_table["upper_bound"].to_numpy()
        points = feature_table["points"].to_numpy()
        values = X[feature].to_numpy()

        group_indices = np.searchsorted(bounds, values, side="left")
        scores += points[group_indices]

    return scores


def predict_proba_from_score(
    scores: np.ndarray,
    unit: float,
    expected_value: float,
) -> np.ndarray:
    """Convert raw integer scores to approximate risk probabilities."""
    log_odds = expected_value + scores * unit
    return expit(log_odds)


def make_clinical_score_table(score_table: pd.DataFrame) -> pd.DataFrame:
    """Shift each feature's points so its minimum clinical score is zero."""
    table = score_table.copy()
    min_points = table.groupby("feature")["points"].transform("min")
    table["clinical_points"] = table["points"] - min_points

    return table


class UniACS:
    """Build and apply an interpretable integer clinical scoring system."""

    def __init__(
        self,
        top_k: int = 5,
        threshold: float = 0.05,
        method: str = "quantile",
    ) -> None:
        self.top_k = top_k
        self.threshold = threshold
        self.method = method
        self.score_table = None
        self.unit = None
        self.expected_value = None
        self.model = None
        self.score_calibrator = None

    def fit(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: float,
    ) -> "UniACS":
        """Fit the score table using training features and SHAP values."""
        self.score_table, self.unit = build_score_table(
            X,
            shap_values,
            top_k=self.top_k,
            threshold=self.threshold,
            method=self.method,
        )
        self.expected_value = expected_value
        self.score_calibrator = None
        return self

    def fit_from_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        random_state: int = 42,
    ) -> "UniACS":
        """Train a classifier and build a clinical scoring table."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Only binary labels 0 and 1 are supported.")

        # Keep score construction separate from model training
        X_model, X_remaining, y_model, y_remaining = train_test_split(
            X, y, test_size=0.4, stratify=y,
            random_state=random_state,
        )
        X_score, X_cal, _, y_cal = train_test_split(
            X_remaining, y_remaining,
            test_size=0.5,
            stratify=y_remaining,
            random_state=random_state,
        )

        self.model = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            method="isotonic",
            cv=3,
        )
        self.model.fit(X_model, y_model)

        shap_list = []
        expected_values = []

        for clf in self.model.calibrated_classifiers_:
            explainer = shap.TreeExplainer(clf.estimator)
            values = explainer.shap_values(X_score)

            if isinstance(values, list):
                values = values[1]
            elif values.ndim == 3:
                values = values[:, :, 1]

            base_prob = float(explainer.expected_value[1])
            base_prob = np.clip(base_prob, 0.0001, 0.9999)
            adjusted = np.clip(
                values + base_prob, 0.0001, 0.9999
            )

            shap_list.append(logit(adjusted) - logit(base_prob))
            expected_values.append(logit(base_prob))

        logodds_shap = np.mean(shap_list, axis=0)
        expected_value = float(np.mean(expected_values))

        self.fit(X_score, logodds_shap, expected_value)
        self.calibrate_scores(X_cal, y_cal)
        return self

    def calibrate_scores(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> "UniACS":
        """Calibrate integer scores using separate labeled data."""
        scores = self.predict_score(X)
        y = np.asarray(y)

        if len(scores) != len(y):
            raise ValueError("Scores and labels must have equal length.")

        self.score_calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        self.score_calibrator.fit(scores, y)
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate raw integer scores for new samples."""
        if self.score_table is None:
            raise ValueError("UniACS must be fitted first.")

        return predict_score(X, self.score_table)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate probabilities from integer scores."""
        scores = self.predict_score(X)

        if self.score_calibrator is not None:
            return self.score_calibrator.predict(scores)

        return predict_proba_from_score(
            scores, self.unit, self.expected_value
        )

    def get_score_table(self) -> pd.DataFrame:
        """Return the scoring table with non-negative clinical points."""
        if self.score_table is None:
            raise ValueError("UniACS must be fitted first.")

        return make_clinical_score_table(self.score_table)
