"""
Fairness metrics for risk scores used to prioritise care under a fixed capacity.

When a care team can act on only a fixed share of patients, it treats the
highest-scoring ones first, so the decision threshold follows from that capacity
rather than from a probability cut-off. These metrics compare a protected group
with the rest at such a threshold, with bootstrapped spread, as in the paper
below.

Paper:
    Vincent Jeanselme, Maria De-Arteaga, Zhe Zhang, Jessica Barrett and Brian
    Tom. "Imputation Strategies Under Clinical Presence: Impact on Algorithmic
    Fairness." Machine Learning for Health (ML4H) 2022, PMLR 193:12-34.

Paper link:
    https://proceedings.mlr.press/v193/jeanselme22a.html

Reference code (``evaluate`` in ``mimic/analysis groups.ipynb``):
    https://github.com/Jeanselme/ClinicalPresenceFairness

Authors:
    Chunyi Yu (cyu410@gatech.edu)
    John Nguyen (jnguyen461@gatech.edu)
"""

import numpy as np
from sklearn.metrics import roc_auc_score

_METRICS = ("auc", "prioritization_rate", "fnr", "fpr")


def _bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    rng: np.random.RandomState,
    group_name: str,
) -> dict[str, np.ndarray]:
    """Computes each metric on ``n_bootstrap`` resamples of one group.

    Args:
        y_true: Binary labels of the group, of shape (n_group,).
        y_prob: Scores of the group, of shape (n_group,).
        threshold: Samples scoring at least this are prioritised.
        n_bootstrap: Number of resamples.
        rng: Random state shared by all groups, advanced in place.
        group_name: Group name, used only in the error message.

    Returns:
        Dictionary from metric name to an array of shape (n_bootstrap,).

    Raises:
        ValueError: If a resample has only positives or only negatives.
    """
    n = len(y_true)
    out = {m: np.empty(n_bootstrap) for m in _METRICS}
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yb, pb = y_true[idx], y_prob[idx]
        n_pos = yb.sum()
        if n_pos in (0, n):
            raise ValueError(
                f"A bootstrap resample of the {group_name} group has only one "
                "class; use a larger evaluation set."
            )
        selected = pb >= threshold
        out["auc"][b] = roc_auc_score(yb, pb)
        out["prioritization_rate"][b] = selected.mean()
        out["fnr"][b] = yb[~selected].sum() / n_pos
        out["fpr"][b] = (1 - yb[selected]).sum() / (n - n_pos)
    return out


def capacity_fairness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attributes: np.ndarray | None = None,
    capacity: float = 0.3,
    n_bootstrap: int = 100,
    seed: int | None = 42,
) -> dict[str, float]:
    """Computes group metrics when only the top ``capacity`` share is prioritised.

    The threshold is the k-th highest score in ``y_prob``, with
    k = int(capacity * n_samples). Every sample scoring at least the threshold
    is prioritised, so all ties at the threshold are included. One threshold,
    computed over all samples, is shared by the groups.

    Each group is then resampled with replacement ``n_bootstrap`` times, the
    protected group first, from one ``np.random.RandomState(seed)``. On each
    resample the function computes:

        - ``auc``: ROC AUC of ``y_prob``;
        - ``prioritization_rate``: share of samples prioritised;
        - ``fnr``: positives not prioritised / positives;
        - ``fpr``: negatives prioritised / negatives.

    The ``difference`` is protected minus unprotected, paired resample by
    resample. With the same seed this reproduces the reference code's
    ``evaluate`` exactly.

    Args:
        y_true: True binary labels (0 or 1) of shape (n_samples,).
        y_prob: Predicted scores of shape (n_samples,). Only their order
            matters.
        sensitive_attributes: Sensitive attributes of shape (n_samples,) where
            1 is the protected group and 0 is the unprotected group. If None,
            the metrics are computed for the whole population.
        capacity: Share of samples that can be prioritised, in (0, 1].
            Default is 0.3.
        n_bootstrap: Number of bootstrap resamples per group. Default is 100.
        seed: Seed of the resampling. Default is 42, the reference's seed.
            None gives different resamples on every call.

    Returns:
        Dictionary with ``threshold`` and, for each metric in ``auc``,
            ``prioritization_rate``, ``fnr`` and ``fpr``, the keys
            ``{metric}_{group}`` (mean over resamples) and
            ``{metric}_{group}_std`` (standard deviation over resamples).
            The groups are ``protected``, ``unprotected`` and ``difference``,
            or only ``population`` if ``sensitive_attributes`` is None.

    Raises:
        ValueError: If the inputs differ in length; if ``y_true`` or
            ``sensitive_attributes`` holds values other than 0 and 1; if
            ``y_prob`` is not finite; if ``capacity`` is outside (0, 1] or
            too small to prioritise one sample; if ``n_bootstrap`` is below 1;
            if a group is empty or lacks positives or negatives; or if a
            resample lacks positives or negatives. The metrics are then
            undefined, and no NaN is returned.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.fairness_utils import capacity_fairness_metrics
        >>> rng = np.random.default_rng(0)
        >>> y_true = rng.binomial(1, 0.2, size=1000)
        >>> sensitive_attributes = rng.binomial(1, 0.3, size=1000)
        >>> logits = 2 * y_true - sensitive_attributes + rng.normal(size=1000)
        >>> y_prob = 1 / (1 + np.exp(-logits))
        >>> metrics = capacity_fairness_metrics(
        ...     y_true, y_prob, sensitive_attributes, capacity=0.3
        ... )
        >>> round(metrics["threshold"], 3)
        0.685
        >>> round(metrics["prioritization_rate_protected"], 3)
        0.163
        >>> round(metrics["prioritization_rate_unprotected"], 3)
        0.355
        >>> round(metrics["auc_difference"], 3), round(metrics["fnr_difference"], 3)
        (-0.009, 0.261)
        >>> population = capacity_fairness_metrics(y_true, y_prob, capacity=0.3)
        >>> round(population["prioritization_rate_population"], 3)
        0.301
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    y_true = y_true.astype(int)
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob must not contain NaN or infinite values.")
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1].")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    # int() truncation matches the reference
    k = int(len(y_prob) * capacity)
    if k < 1:
        raise ValueError(
            "capacity * n_samples is below 1, so no sample can be prioritised."
        )
    threshold = float(np.sort(y_prob)[-k])

    if sensitive_attributes is None:
        groups = {"population": np.ones(len(y_true), dtype=bool)}
    else:
        s = np.asarray(sensitive_attributes).ravel()
        if len(s) != len(y_true):
            raise ValueError(
                "sensitive_attributes must have the same length as y_true."
            )
        if not np.isin(s, (0, 1)).all():
            raise ValueError("sensitive_attributes must contain only 0 and 1.")
        groups = {"protected": s == 1, "unprotected": s == 0}
    for name, mask in groups.items():
        if not mask.any():
            raise ValueError(f"The {name} group has no samples.")
        if y_true[mask].min() == y_true[mask].max():
            raise ValueError(f"The {name} group needs both positives and negatives.")

    # Legacy RandomState reproduces the reference's np.random.seed stream
    rng = np.random.RandomState(seed)
    draws = {
        name: _bootstrap(y_true[mask], y_prob[mask], threshold, n_bootstrap, rng, name)
        for name, mask in groups.items()
    }
    if sensitive_attributes is not None:
        draws["difference"] = {
            m: draws["protected"][m] - draws["unprotected"][m] for m in _METRICS
        }

    result = {"threshold": threshold}
    for name, out in draws.items():
        for m in _METRICS:
            result[f"{m}_{name}"] = float(np.mean(out[m]))
            result[f"{m}_{name}_std"] = float(np.std(out[m]))
    return result
