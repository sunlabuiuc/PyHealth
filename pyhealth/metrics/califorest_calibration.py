# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Six calibration metrics for binary risk prediction
"""CaliForest calibration metrics (Park and Ho, 2020).

Six commonly published calibration metrics for binary risk
prediction models, as described in Table 1 of the CaliForest
paper.

Paper: Y. Park and J. C. Ho. "CaliForest: Calibrated Random
Forest for Health Data." ACM CHIL, 2020.
https://doi.org/10.1145/3368555.3384461
"""

import numpy as np
import pandas as pd


def hosmer_lemeshow(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_groups: int = 10,
) -> float:
    """Hosmer-Lemeshow goodness-of-fit test p-value.

    Divides predictions into ``n_groups`` equal-sized bins and
    tests whether observed event rates match predicted rates
    using a chi-squared statistic.

    A p-value close to 1 indicates good calibration.

    Args:
        y_true: Binary ground-truth labels of shape ``(N,)``.
        y_score: Predicted probabilities of shape ``(N,)``.
        n_groups: Number of quantile groups. Default ``10``.

    Returns:
        The Hosmer-Lemeshow p-value (higher is better).

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.2, 0.8, 0.9])
        >>> p = hosmer_lemeshow(y_true, y_score)
    """
    from scipy.stats import chi2

    df = pd.DataFrame({"score": y_score, "target": y_true})
    df = df.sort_values("score")
    df["score"] = np.clip(df["score"], 1e-8, 1 - 1e-8)
    df["rank"] = range(len(df))
    df["decile"] = pd.qcut(
        df["rank"], n_groups, duplicates="raise"
    )

    grp = df.groupby("decile", observed=False)
    obs_pos = grp["target"].sum()
    obs_neg = grp["target"].count() - obs_pos
    exp_pos = grp["score"].sum()
    exp_neg = grp["score"].count() - exp_pos

    hl = (
        (obs_pos - exp_pos) ** 2 / exp_pos
        + (obs_neg - exp_neg) ** 2 / exp_neg
    ).sum()
    return float(1 - chi2.cdf(hl, n_groups - 2))


def spiegelhalter(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Spiegelhalter z-test p-value for calibration.

    Tests whether the Brier score is extreme under the null
    hypothesis that predicted probabilities equal true
    probabilities.

    A p-value close to 1 indicates good calibration.

    Args:
        y_true: Binary ground-truth labels of shape ``(N,)``.
        y_score: Predicted probabilities of shape ``(N,)``.

    Returns:
        The two-sided Spiegelhalter p-value (higher is better).

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.2, 0.8, 0.9])
        >>> p = spiegelhalter(y_true, y_score)
    """
    from scipy.stats import norm

    top = np.sum((y_true - y_score) * (1 - 2 * y_score))
    bot = np.sum(
        (1 - 2 * y_score) ** 2 * y_score * (1 - y_score)
    )
    z = top / np.sqrt(bot)
    return float(norm.sf(np.abs(z)) * 2)


def reliability(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_groups: int = 10,
) -> tuple:
    """Reliability-in-the-small and reliability-in-the-large.

    Reliability-in-the-small measures the average squared
    difference between observed and predicted event rates across
    quantile bins.  Reliability-in-the-large measures the squared
    difference between the overall mean prediction and the
    overall observed prevalence.

    Values close to 0 indicate good calibration.

    Args:
        y_true: Binary ground-truth labels of shape ``(N,)``.
        y_score: Predicted probabilities of shape ``(N,)``.
        n_groups: Number of quantile groups. Default ``10``.

    Returns:
        A tuple ``(rel_small, rel_large)``.

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.2, 0.8, 0.9])
        >>> rel_s, rel_l = reliability(y_true, y_score)
    """
    df = pd.DataFrame({"score": y_score, "target": y_true})
    df = df.sort_values("score")
    df["rank"] = range(len(df))
    df["decile"] = pd.qcut(
        df["rank"], n_groups, duplicates="raise"
    )

    grp = df.groupby("decile", observed=False)
    obs = grp["target"].mean()
    exp = grp["score"].mean()
    rel_small = float(np.mean((obs - exp) ** 2))
    rel_large = float(
        (np.mean(y_true) - np.mean(y_score)) ** 2
    )
    return rel_small, rel_large


def scaled_brier_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple:
    """Brier score and scaled (prevalence-adjusted) Brier score.

    The scaled Brier score normalises by the maximum Brier score
    achievable by always predicting the prevalence.  A scaled
    score of 1 indicates perfect calibration.

    Args:
        y_true: Binary ground-truth labels of shape ``(N,)``.
        y_score: Predicted probabilities of shape ``(N,)``.

    Returns:
        A tuple ``(brier, scaled_brier)``.

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.2, 0.8, 0.9])
        >>> brier, scaled = scaled_brier_score(y_true, y_score)
    """
    brier = float(np.mean((y_true - y_score) ** 2))
    p = np.mean(y_true)
    denom = p * (1 - p)
    scaled = 1 - brier / denom if denom > 0 else 0.0
    return brier, float(scaled)
