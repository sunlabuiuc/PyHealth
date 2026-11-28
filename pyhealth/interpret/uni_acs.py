"""Convert SHAP feature contributions into integer clinical scores.

Implements the Unified Auto Clinical Scoring (Uni-ACS) framework for
converting complex, black-box machine learning predictions and their
SHAP attributions into transparent, additive integer clinical scoring
tables.

Reference:
    Li, A., Ong, M. L., Oei, C. W., Lian, W., Phua, H. P., Htet, L. H.,
    Lim, W. Y., & Motani, M. (2022). Unified Auto Clinical Scoring
    (Uni-ACS) with Interpretable ML models. Proceedings of Machine Learning for
    Healthcare, PMLR 182:26–53.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.special import expit, logit

try:
    from csaps import csaps as _csaps
except ImportError:  # pragma: no cover - optional dependency
    _csaps = None

__all__ = [
    "UniACS",
    "build_score_table",
    "find_novel_breakpoints",
    "make_clinical_score_table",
    "predict_proba_from_score",
    "predict_score",
]


def find_novel_breakpoints(
    values: np.ndarray,
    shap_values: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """Find breakpoints from zero crossings of a smoothed SHAP curve.

    Bins the sorted feature values into ``n_bins - 1`` uniform intervals,
    interpolates empty bins, fits a cubic smoothing spline (using
    ``csaps`` when installed and otherwise using SciPy), and locates zero
    crossings. The two backends can produce different breakpoints.

    Args:
        values: One-dimensional array of feature values.
        shap_values: One-dimensional array of SHAP contributions aligned
            with ``values``.
        n_bins: Number of grid points used to build the moving-average
            curve. Must be at least 3.

    Returns:
        Sorted one-dimensional array of unique breakpoint values where
        the smoothed SHAP curve crosses zero.

    Raises:
        ValueError: If ``values`` and ``shap_values`` are not aligned 1D
            finite arrays, or if ``n_bins < 3``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.interpret.uni_acs import find_novel_breakpoints
        >>> vals = np.linspace(-3.0, 3.0, 100)
        >>> breaks = find_novel_breakpoints(vals, vals.copy())
        >>> len(breaks) >= 1
        True
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
        return np.array([], dtype=float)

    order = np.argsort(values)
    values = values[order]
    shap_values = shap_values[order]

    # Follow the original moving-average construction
    xs = np.linspace(values[0], values[-1], n_bins)
    x = [xs[0]]
    y = [shap_values[0]]

    for i in range(1, n_bins):
        left, right = xs[i - 1], xs[i]

        if i == n_bins - 1:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)

        x.append((left + right) / 2.0)
        y.append(shap_values[mask].mean() if mask.any() else np.nan)

    x.append(xs[-1])
    y.append(shap_values[-1])

    # Interpolate missing bins using array positions
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    indices = np.arange(len(y_arr))
    valid = np.isfinite(y_arr)
    y_arr = np.interp(indices, indices[valid], y_arr[valid])

    # Automatically select spline smoothing
    spline_x = np.linspace(x_arr[0], x_arr[-1], 100)
    if _csaps is not None:
        result = _csaps(x_arr, y_arr, spline_x)
        spline_y = np.asarray(result.values, dtype=float).ravel()
    elif len(x_arr) >= 5:
        spline = make_smoothing_spline(x_arr, y_arr)
        spline_y = np.asarray(spline(spline_x), dtype=float).ravel()
    else:
        spline = CubicSpline(x_arr, y_arr)
        spline_y = np.asarray(spline(spline_x), dtype=float).ravel()

    # Find crossings of the zero-SHAP baseline
    changes = np.where(np.sign(spline_y[1:]) != np.sign(spline_y[:-1]))[0]

    breakpoints = []
    start = spline_x[0]

    for i in changes:
        x1, x2 = spline_x[i], spline_x[i + 1]
        y1, y2 = spline_y[i], spline_y[i + 1]

        if y2 == y1:
            continue

        crossing = x1 - y1 * (x2 - x1) / (y2 - y1)

        if np.any((values >= start) & (values <= crossing)):
            breakpoints.append(crossing)
            start = crossing

    return np.unique(breakpoints)


def build_score_table(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    top_k: int = 5,
    threshold: float = 0.05,
    method: str = "quantile",
    n_bins: int = 50,
) -> tuple[pd.DataFrame, float]:
    """Build an integer scoring table from feature SHAP contributions.

    Selects the ``top_k`` features with the largest mean absolute SHAP
    values, discretizes each selected feature into risk intervals using
    either empirical quantiles or spline zero-crossings, and rounds bin
    mean SHAP values to integer points scaled by the smallest bin mean
    exceeding ``threshold``.

    Args:
        X: Training feature matrix.
        shap_values: Two-dimensional array of SHAP contributions aligned
            with ``X`` (shape ``(n_samples, n_features)``).
        top_k: Number of most important features to include. Must be at
            least 1.
        threshold: Minimum absolute mean SHAP value used when choosing
            the integer scaling unit. Must be positive.
        method: Feature discretization method, either ``"quantile"`` or
            ``"novel"``.
        n_bins: Number of bins passed to :func:`find_novel_breakpoints`
            when ``method="novel"``.

    Returns:
        Tuple ``(table, unit)`` where ``table`` is a DataFrame with
        columns ``["feature", "group", "upper_bound", "mean_shap",
        "points"]`` and ``unit`` is the positive float SHAP unit used
        for integer scaling.

    Raises:
        ValueError: If inputs are invalid, shapes do not match, or no
            bin mean SHAP value exceeds ``threshold``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyhealth.interpret.uni_acs import build_score_table
        >>> X = pd.DataFrame({"age": [20, 40, 60, 80], "hr": [60, 80, 90, 110]})
        >>> shap_vals = np.array([[-0.3, -0.1], [-0.1, 0.0], [0.1, 0.1], [0.3, 0.2]])
        >>> table, unit = build_score_table(X, shap_vals, top_k=2, threshold=0.01)
        >>> unit > 0 and len(table) > 0
        True
    """
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    if method not in ("quantile", "novel"):
        raise ValueError("method must be 'quantile' or 'novel'")

    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.shape != X.shape:
        raise ValueError("shap_values must have the same shape as X.")
    if not np.isfinite(shap_array).all():
        raise ValueError("shap_values must contain only finite values.")

    importance = np.abs(shap_array).mean(axis=0)
    k = min(top_k, X.shape[1])
    top_indices = np.argsort(importance)[::-1][:k]
    rows = []

    for i in top_indices:
        feature = X.columns[i]
        values = X[feature].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature '{feature}' contains non-finite values.")
        contributions = shap_array[:, i]

        if method == "quantile":
            breaks = np.unique(np.quantile(values, [0.2, 0.5, 0.8]))
        else:
            breaks = find_novel_breakpoints(values, contributions, n_bins=n_bins)

        groups = np.digitize(values, breaks, right=True)
        feature_rows = []

        for group in range(len(breaks) + 1):
            mask = groups == group
            if not mask.any():
                continue

            feature_rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "upper_bound": (
                        float(breaks[group]) if group < len(breaks) else np.inf
                    ),
                    "mean_shap": float(contributions[mask].mean()),
                }
            )

        # Guarantee the final retained interval covers (-inf, +inf) cleanly
        if feature_rows:
            feature_rows[-1]["upper_bound"] = np.inf
            rows.extend(feature_rows)

    table = pd.DataFrame(rows)

    valid = np.abs(table["mean_shap"]) >= threshold
    if not valid.any():
        raise ValueError("No SHAP values exceed the threshold.")

    unit = float(np.abs(table.loc[valid, "mean_shap"]).min())
    table["points"] = np.rint(table["mean_shap"] / unit).astype(int)

    return table, unit


def predict_score(
    X: pd.DataFrame,
    score_table: pd.DataFrame,
    clinical: bool = False,
) -> np.ndarray:
    """Calculate integer scores using a fitted scoring table.

    Args:
        X: Feature matrix containing all features present in
            ``score_table``.
        score_table: Scoring table returned by :func:`build_score_table`
            or :func:`make_clinical_score_table`.
        clinical: If ``True``, sum non-negative per-feature clinical
            points (shifted so each feature's minimum is zero). If
            ``False``, sum raw signed integer points.

    Returns:
        One-dimensional integer array of total scores with length
        ``len(X)``.

    Raises:
        ValueError: If ``X`` is missing required features or contains
            non-finite values in scored columns.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyhealth.interpret.uni_acs import predict_score
        >>> table = pd.DataFrame({
        ...     "feature": ["age", "age"],
        ...     "group": [0, 1],
        ...     "upper_bound": [50.0, np.inf],
        ...     "points": [-1, 2],
        ... })
        >>> X = pd.DataFrame({"age": [35, 65]})
        >>> predict_score(X, table).tolist()
        [-1, 2]
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    required = list(score_table["feature"].unique())
    missing = [col for col in required if col not in X.columns]
    if missing:
        raise ValueError(f"Missing required features in X: {missing}")

    if clinical:
        working_table = (
            score_table
            if "clinical_points" in score_table.columns
            else make_clinical_score_table(score_table)
        )
        point_col = "clinical_points"
    else:
        working_table = score_table
        point_col = "points"

    scores = np.zeros(len(X), dtype=int)

    for feature in required:
        feature_table = working_table[
            working_table["feature"] == feature
        ].sort_values("group")

        bounds = feature_table["upper_bound"].to_numpy(dtype=float)
        points = feature_table[point_col].to_numpy(dtype=int)
        values = X[feature].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Feature '{feature}' contains non-finite values.")

        group_indices = np.searchsorted(bounds, values, side="left")
        group_indices = np.clip(group_indices, 0, len(points) - 1)
        scores += points[group_indices]

    return scores


def predict_proba_from_score(
    scores: np.ndarray,
    unit: float,
    expected_value: float,
) -> np.ndarray:
    """Convert raw integer scores to approximate risk probabilities.

    Applies the logistic sigmoid to ``expected_value + scores * unit``.

    Args:
        scores: One-dimensional array of raw integer scores.
        unit: Positive SHAP scaling unit from :func:`build_score_table`.
        expected_value: Baseline log-odds expected value.

    Returns:
        One-dimensional float array of risk probabilities in ``(0, 1)``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.interpret.uni_acs import predict_proba_from_score
        >>> probs = predict_proba_from_score(np.array([-2, 0, 2]), 0.5, 0.0)
        >>> round(float(probs[1]), 4)
        0.5
    """
    log_odds = expected_value + np.asarray(scores, dtype=float) * unit
    return expit(log_odds)


def make_clinical_score_table(score_table: pd.DataFrame) -> pd.DataFrame:
    """Shift each feature's points so its minimum clinical score is zero.

    Args:
        score_table: Scoring table containing ``"feature"`` and
            ``"points"`` columns.

    Returns:
        Copy of ``score_table`` with an added ``"clinical_points"``
        integer column where each feature's minimum point value is 0.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyhealth.interpret.uni_acs import make_clinical_score_table
        >>> raw = pd.DataFrame({
        ...     "feature": ["age", "age"],
        ...     "group": [0, 1],
        ...     "upper_bound": [50.0, np.inf],
        ...     "points": [-2, 3],
        ... })
        >>> make_clinical_score_table(raw)["clinical_points"].tolist()
        [0, 5]
    """
    table = score_table.copy()
    min_points = table.groupby("feature")["points"].transform("min")
    table["clinical_points"] = (table["points"] - min_points).astype(int)

    return table


class UniACS:
    """Unified Auto Clinical Scoring (Uni-ACS) interpretable risk scorer.

    Converts SHAP feature attributions from any binary classification
    model into an additive, integer-valued clinical scoring table, with
    optional score-level isotonic calibration for probability outputs.

    Integer points are summed to produce a score. When using
    :meth:`fit_from_data`, the per-feature logit transformation of
    probability-scale SHAP values is an approximation. The summed
    contributions generally differ from the model's log-odds change.

    Args:
        top_k: Number of top features selected by mean absolute SHAP.
        threshold: Minimum absolute bin mean SHAP value used to determine
            the integer scaling unit.
        method: Feature discretization method, either ``"quantile"`` or
            ``"novel"`` (spline zero-crossing breakpoints).
        n_bins: Number of bins used by :func:`find_novel_breakpoints`
            when ``method="novel"``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyhealth.interpret.uni_acs import UniACS
        >>> X = pd.DataFrame({
        ...     "age": [20, 35, 50, 65, 80, 90],
        ...     "hr": [65, 72, 85, 95, 105, 115],
        ... })
        >>> shap_vals = np.array([
        ...     [-0.4, -0.2], [-0.2, -0.1], [-0.1, 0.0],
        ...     [0.1, 0.1], [0.2, 0.2], [0.4, 0.3],
        ... ])
        >>> scorer = UniACS(top_k=2, threshold=0.01, method="quantile")
        >>> _ = scorer.fit(X, shap_vals, expected_value=0.0)
        >>> len(scorer.predict_score(X)) == len(X)
        True
    """

    def __init__(
        self,
        top_k: int = 5,
        threshold: float = 0.05,
        method: str = "quantile",
        n_bins: int = 50,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if method not in ("quantile", "novel"):
            raise ValueError("method must be 'quantile' or 'novel'")
        if n_bins < 3:
            raise ValueError("n_bins must be at least 3.")

        self.top_k = top_k
        self.threshold = threshold
        self.method = method
        self.n_bins = n_bins
        self.score_table: pd.DataFrame | None = None
        self.unit: float | None = None
        self.expected_value: float | None = None
        self.model = None
        self.score_calibrator = None

    def fit(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: float,
    ) -> UniACS:
        """Fit the score table from precomputed SHAP attributions.

        Args:
            X: Feature matrix aligned with ``shap_values``.
            shap_values: SHAP contributions of shape
                ``(n_samples, n_features)``.
            expected_value: Baseline log-odds expected value.

        Returns:
            Fitted ``UniACS`` instance.
        """
        self.score_table, self.unit = build_score_table(
            X,
            shap_values,
            top_k=self.top_k,
            threshold=self.threshold,
            method=self.method,
            n_bins=self.n_bins,
        )
        self.expected_value = float(expected_value)
        self.score_calibrator = None
        return self

    def fit_from_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        random_state: int = 42,
    ) -> UniACS:
        """Train a calibrated forest, extract SHAP, and fit Uni-ACS.

        Splits ``(X, y)`` into 60% model training, 20% score table
        construction, and 20% score-level isotonic calibration to avoid
        leakage across stages. Requires the optional ``shap`` package.

        Each probability-scale SHAP value is transformed separately via
        the logit difference from the baseline. Because the logit is
        nonlinear, these contributions generally do not sum to the
        model's log-odds change. Held-out isotonic calibration maps the
        resulting integer score to probabilities.

        Args:
            X: Feature DataFrame.
            y: Binary target labels containing ``0`` and ``1``.
            random_state: Random seed for splitting and forest training.

        Returns:
            Fitted and calibrated ``UniACS`` instance.

        Raises:
            ImportError: If ``shap`` or ``scikit-learn`` is not
                installed.
            ValueError: If ``X`` is not a DataFrame, lengths differ, or
                ``y`` is not binary ``{0, 1}``.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "The 'shap' package is required for UniACS.fit_from_data(). "
                "Install it with `pip install shap`, or pass precomputed "
                "attributions to UniACS.fit()."
            ) from exc

        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Only binary labels 0 and 1 are supported.")

        # Keep score construction separate from model training
        X_model, X_remaining, y_model, y_remaining = train_test_split(
            X,
            y,
            test_size=0.4,
            stratify=y,
            random_state=random_state,
        )
        X_score, X_cal, _, y_cal = train_test_split(
            X_remaining,
            y_remaining,
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

            base_prob = float(np.asarray(explainer.expected_value)[1])
            base_prob = np.clip(base_prob, 0.0001, 0.9999)
            adjusted = np.clip(values + base_prob, 0.0001, 0.9999)

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
    ) -> UniACS:
        """Calibrate integer scores using held-out labeled data.

        Fits an :class:`sklearn.isotonic.IsotonicRegression` mapping raw
        integer scores to calibrated probabilities in ``[0, 1]``.

        Args:
            X: Calibration feature DataFrame.
            y: Binary target labels aligned with ``X``.

        Returns:
            Self with fitted ``score_calibrator``.

        Raises:
            ValueError: If ``UniACS`` is not fitted or lengths mismatch.
        """
        from sklearn.isotonic import IsotonicRegression

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

    def predict_score(
        self,
        X: pd.DataFrame,
        clinical: bool = False,
    ) -> np.ndarray:
        """Calculate integer scores for new samples.

        Args:
            X: Feature DataFrame containing the scored features.
            clinical: If ``True``, return non-negative shifted clinical
                scores. Defaults to ``False`` (raw signed scores).

        Returns:
            One-dimensional integer array of scores.

        Raises:
            ValueError: If ``UniACS`` has not been fitted yet.
        """
        if self.score_table is None:
            raise ValueError("UniACS must be fitted first.")

        return predict_score(X, self.score_table, clinical=clinical)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate risk probabilities from integer scores.

        Uses the fitted isotonic score calibrator when available;
        otherwise applies :func:`predict_proba_from_score` using the
        baseline expected log-odds and scaling unit for an approximate
        probability.

        Args:
            X: Feature DataFrame containing the scored features.

        Returns:
            One-dimensional float array of estimated probabilities.

        Raises:
            ValueError: If ``UniACS`` has not been fitted yet.
        """
        scores = self.predict_score(X)

        if self.score_calibrator is not None:
            return self.score_calibrator.predict(scores)

        return predict_proba_from_score(scores, self.unit, self.expected_value)

    def get_score_table(self) -> pd.DataFrame:
        """Return the scoring table with non-negative clinical points.

        Returns:
            DataFrame containing feature intervals, mean SHAP values,
            raw integer points, and shifted non-negative clinical points.

        Raises:
            ValueError: If ``UniACS`` has not been fitted yet.
        """
        if self.score_table is None:
            raise ValueError("UniACS must be fitted first.")

        return make_clinical_score_table(self.score_table)
