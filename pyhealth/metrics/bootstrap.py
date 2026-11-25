"""Bootstrap confidence intervals for evaluation metrics.

Clinical evaluations usually report a metric with a confidence interval and,
when comparing models, the interval of their difference on the same data.
Samples from one patient are correlated, so resample whole patients
(``groups=patient_ids``) rather than individual samples.
"""

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from .binary import binary_metrics_fn

Metric = str | Callable[[np.ndarray, np.ndarray], float]


def _metric_fn(metric: Metric) -> Callable[[np.ndarray, np.ndarray], float]:
    if callable(metric):
        return metric
    return lambda y_true, y_prob: binary_metrics_fn(y_true, y_prob, metrics=[metric])[
        metric
    ]


def _resample_indices(
    n: int, groups: np.ndarray | None, n_boot: int, seed: int
) -> Iterator[np.ndarray]:
    """Yields n_boot index arrays, resampling samples or whole groups."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, size=n)
        return
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(counts)[:-1])
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        yield np.concatenate([members[i] for i in picked])


def _summarise(estimate: float, stats: list, n_skipped: int, alpha: float) -> dict:
    if stats:
        lower, upper = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    else:
        lower = upper = float("nan")
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "n_boot": len(stats),
        "n_skipped": n_skipped,
    }


def _mode_metrics_fn(mode: str) -> Callable:
    """Returns the mode's metrics function (imported lazily).

    ``multilabel.py`` itself imports from ``pyhealth.metrics``, so these
    imports cannot live at module level without a circular import.
    """
    if mode == "binary":
        return binary_metrics_fn
    if mode == "multiclass":
        from .multiclass import multiclass_metrics_fn

        return multiclass_metrics_fn
    if mode == "multilabel":
        from .multilabel import multilabel_metrics_fn

        return multilabel_metrics_fn
    raise ValueError(
        f"mode must be one of 'binary', 'multiclass', 'multilabel', "
        f"got {mode!r}"
    )


def _usable(
    y_true: np.ndarray, idx: np.ndarray, mode: str, n_classes: int
) -> bool:
    """Whether a resample defines every class the metric needs."""
    truth = y_true[idx]
    if mode == "multilabel":
        # Each class column needs at least one positive, as the multilabel
        # benchmark's ``get_appropriate_bootstrap_samples`` requires.
        return truth.sum(axis=0).min() > 0
    if mode == "multiclass":
        # Every class present in the full truth must appear, so macro F1
        # and one-vs-rest AUC stay defined on the resample.
        return np.unique(truth).size == n_classes
    return np.unique(truth).size >= 2


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Metric,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap confidence interval for a binary metric.

    Args:
        y_true: True binary labels, shape (n,).
        y_prob: Predicted probabilities, shape (n,).
        metric: A name accepted by ``binary_metrics_fn`` (e.g. ``"roc_auc"``,
            ``"brier"``) or a callable ``metric(y_true, y_prob) -> float``.
        groups: Optional cluster ids, shape (n,), e.g. patient ids. Whole
            groups are resampled with replacement, keeping all their samples.
        n_boot: Number of resamples drawn.
        seed: Random seed; results are deterministic given it.
        alpha: 1 - confidence level (0.05 gives a 95% interval).

    Returns:
        dict with ``estimate`` (metric on all data), ``lower`` and ``upper``
        (percentile bounds), ``n_boot`` (resamples used) and ``n_skipped``
        (resamples with a single class, which are skipped).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.bootstrap import bootstrap_ci
        >>> y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        >>> y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.4, 0.6, 0.2, 0.9])
        >>> patients = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        >>> ci = bootstrap_ci(y_true, y_prob, "brier", groups=patients, n_boot=200)
        >>> sorted(ci)
        ['estimate', 'lower', 'n_boot', 'n_skipped', 'upper']
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    fn = _metric_fn(metric)
    stats, n_skipped = [], 0
    for idx in _resample_indices(len(y_true), groups, n_boot, seed):
        if np.unique(y_true[idx]).size < 2:
            n_skipped += 1
            continue
        stats.append(fn(y_true[idx], y_prob[idx]))
    return _summarise(fn(y_true, y_prob), stats, n_skipped, alpha)


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric: Metric,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap interval for metric(model A) - metric(model B) on the same data.

    Both models are scored on identical resamples, so the interval reflects
    their difference rather than two independent uncertainties. An interval
    that excludes 0 indicates a difference at level ``alpha``.

    Args:
        y_true: True binary labels, shape (n,).
        y_prob_a: Model A's predicted probabilities, shape (n,).
        y_prob_b: Model B's predicted probabilities, shape (n,).
        metric: A ``binary_metrics_fn`` name or a callable, as in
            :func:`bootstrap_ci`.
        groups: Optional cluster ids (e.g. patient ids) for a cluster bootstrap.
        n_boot: Number of resamples drawn.
        seed: Random seed; results are deterministic given it.
        alpha: 1 - confidence level.

    Returns:
        dict with ``estimate`` (A - B on all data), ``lower``, ``upper``,
        ``n_boot`` and ``n_skipped``, as in :func:`bootstrap_ci`.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.bootstrap import paired_bootstrap_diff
        >>> y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        >>> model_a = np.array([0.1, 0.3, 0.7, 0.8, 0.4, 0.6, 0.2, 0.9])
        >>> model_b = np.array([0.4, 0.5, 0.5, 0.6, 0.5, 0.4, 0.3, 0.7])
        >>> diff = paired_bootstrap_diff(
        ...     y_true, model_a, model_b, "roc_auc", n_boot=200
        ... )
        >>> diff["estimate"] > 0
        True
    """
    y_true = np.asarray(y_true)
    y_prob_a, y_prob_b = np.asarray(y_prob_a), np.asarray(y_prob_b)
    fn = _metric_fn(metric)
    stats, n_skipped = [], 0
    for idx in _resample_indices(len(y_true), groups, n_boot, seed):
        if np.unique(y_true[idx]).size < 2:
            n_skipped += 1
            continue
        stats.append(fn(y_true[idx], y_prob_a[idx]) - fn(y_true[idx], y_prob_b[idx]))
    estimate = fn(y_true, y_prob_a) - fn(y_true, y_prob_b)
    return _summarise(estimate, stats, n_skipped, alpha)


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Sequence[str],
    mode: str,
    n_boot: int = 100,
    seed: int = 0,
    alpha: float = 0.10,
    groups: np.ndarray | None = None,
) -> dict[str, dict]:
    """Percentile bootstrap intervals for multiclass / multilabel metrics.

    Mirrors the evaluation protocol of the ECG benchmarking benchmark
    (*Nonaka & Seita, 2021*, *In-depth Benchmarking of Deep Neural Network
    Architectures for ECG Diagnosis*): resample the test set with
    replacement, **discard resamples that do not define every class** (a
    multilabel class with no positive, or a multiclass class with no
    member), and report the metric's bootstrap mean and percentile bounds.
    Label shapes follow the ``metrics_fn`` conventions: multiclass
    ``y_true`` is ``(n,)`` class indices with ``y_prob`` ``(n, K)``;
    multilabel ``y_true`` / ``y_prob`` are both ``(n, C)``.

    Note:
        Defaults match the released benchmark code: ``n_boot=100`` and
        ``alpha=0.10`` (bounds at the 5% and 95% quantiles, a central 90%
        interval), unlike :func:`bootstrap_ci`'s 95% interval. The original
        drew resamples until 100 were usable; here exactly ``n_boot`` are
        drawn and unusable ones are counted in ``n_skipped`` instead.
        Resampling is per test sample (as in the original); pass
        ``groups=patient_ids`` for a patient-level cluster bootstrap.

    Args:
        y_true: True labels, layout per ``mode`` (see above).
        y_prob: Predicted probabilities, shape ``(n,)`` for binary,
            ``(n, K)`` / ``(n, C)`` otherwise.
        metrics: Metric names accepted by the ``mode``'s metrics function
            (e.g. ``["roc_auc_macro"]`` for multilabel, ``["f1_macro"]``
            for multiclass).
        mode: One of ``"binary"``, ``"multiclass"``, ``"multilabel"``.
        n_boot: Number of resamples drawn.
        seed: Random seed; results are deterministic given it.
        alpha: ``1 - confidence level``; 0.10 gives the benchmark's
            5% / 95% bounds.
        groups: Optional cluster ids, shape (n,), for a patient-level
            bootstrap.

    Returns:
        dict: ``{metric: {...}}`` where each entry has ``estimate`` (the
        metric on the full test set), ``mean`` (bootstrap mean),
        ``lower`` / ``upper`` (percentile bounds), ``n_boot`` (usable
        resamples) and ``n_skipped`` (unusable resamples).

    Raises:
        ValueError: If ``mode`` is unknown; metric names are validated by
        the mode's metrics function.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics import bootstrap_metrics_ci
        >>> y_true = np.array([0, 0, 1, 1, 2, 2])
        >>> y_prob = np.eye(3)[y_true] + 0.1
        >>> result = bootstrap_metrics_ci(
        ...     y_true, y_prob, ["f1_macro"], "multiclass", n_boot=20, seed=1)
        >>> sorted(result)
        ['f1_macro']
        >>> sorted(result["f1_macro"])
        ['estimate', 'lower', 'mean', 'n_boot', 'n_skipped', 'upper']
        >>> result["f1_macro"]["estimate"]
        1.0
    """
    metrics_fn = _mode_metrics_fn(mode)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    names = list(metrics)
    full = metrics_fn(y_true, y_prob, metrics=names)
    n_classes = len(np.unique(y_true))
    stats = {name: [] for name in names}
    n_skipped = 0
    for idx in _resample_indices(len(y_true), groups, n_boot, seed):
        if not _usable(y_true, idx, mode, n_classes):
            n_skipped += 1
            continue
        scores = metrics_fn(y_true[idx], y_prob[idx], metrics=names)
        for name in names:
            stats[name].append(scores[name])
    out = {}
    for name in names:
        values = stats[name]
        if values:
            lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            mean = float(np.mean(values))
        else:
            lower = upper = mean = float("nan")
        out[name] = {
            "estimate": float(full[name]),
            "mean": mean,
            "lower": float(lower),
            "upper": float(upper),
            "n_boot": len(values),
            "n_skipped": n_skipped,
        }
    return out
