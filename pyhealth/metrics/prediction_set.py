"""Prediction-set measurements. Undefined measurements return NaN."""

import numpy as np
import torch

# --- Set size and singleton predictions ---


def size(y_pred: np.ndarray) -> float | np.floating:
    """Mean prediction-set size per sample, or NaN for an empty batch.

    Examples:
        >>> float(size(np.array([[1, 1, 0], [0, 1, 0]])))
        1.5
    """
    _validate_prediction_sets(y_pred)

    return _mean_or_nan(y_pred.sum(axis=1))


def rejection_rate(y_pred: np.ndarray) -> float | np.floating:
    """Fraction of samples with set size other than one; NaN for an empty batch.

    Examples:
        >>> float(rejection_rate(np.array([[1, 1, 0], [0, 1, 0]])))
        0.5
    """
    _validate_prediction_sets(y_pred)

    return _mean_or_nan(y_pred.sum(axis=1) != 1)


# --- Per-class miscoverage and singleton error ---


def _missrate(y_pred: np.ndarray, y_true: np.ndarray, ignore_rejected=False):
    """Return one miss rate per true class, optionally restricting to singletons.

    Accept multiclass label indices or multilabel masks. A class with no true
    labels among the selected samples has an undefined rate, represented by NaN.
    """
    _validate_prediction_targets(y_pred, y_true)
    class_count = y_pred.shape[1]
    if y_true.ndim == 1:
        labels = y_true
        y_true = np.zeros((len(labels), class_count), dtype=bool)
        y_true[np.arange(len(labels)), labels] = True

    y_true = y_true.astype(bool)
    keep = (
        y_pred.sum(axis=1) == 1 if ignore_rejected else np.ones(len(y_true), dtype=bool)
    )
    missed = []

    for label in range(class_count):
        selected = keep & y_true[:, label]
        if selected.any():
            missed.append(1 - np.mean(y_pred[selected, label]))

        else:
            missed.append(float("nan"))

    return np.asarray(missed)


def miscoverage_ps(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Miss rate per true class for multiclass indices or multilabel masks.

    For class k, divide the number of samples omitting k from their prediction
    set by the number whose true labels contain k. This gives one result per
    column of y_pred. Multiclass labels must be integers: label k refers to
    column k of y_pred. Return NaN for classes with no applicable true labels.

    Examples:
        >>> y_pred = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> y_true = np.array([1, 0, 1, 2])
        >>> miscoverage_ps(y_pred, y_true)
        array([0. , 0.5, 1. ])
    """
    return _missrate(y_pred, y_true, False)


def miscoverage_mean_ps(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float | np.floating:
    """Equal-weight average of class miss rates.

    Exclude classes with no true labels and return NaN if every class is
    undefined. Retain genuine zero rates. This differs from averaging each
    sample's missed fraction.

    Examples:
        >>> y_pred = np.array([[1, 0, 0], [1, 0, 0]])
        >>> float(miscoverage_mean_ps(y_pred, np.array([0, 1])))
        0.5
    """
    return _mean_defined(miscoverage_ps(y_pred, y_true))


def error_ps(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Miss rate per true class among samples whose prediction set has size one.

    Multiclass labels must be integers: label k refers to column k of y_pred.
    Return NaN for classes with no true labels among singleton samples.

    Examples:
        >>> y_pred = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> y_true = np.array([1, 0, 1, 2])
        >>> error_ps(y_pred, y_true)
        array([0., 1., 1.])
    """
    return _missrate(y_pred, y_true, True)


def error_mean_ps(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float | np.floating:
    """Equal-weight average of per-class singleton error rates.

    Exclude classes with no true labels among singleton predictions. Return
    NaN if every class is undefined; retain genuine zero error rates.

    Examples:
        >>> y_pred = np.array([[1, 0, 0], [1, 0, 0]])
        >>> float(error_mean_ps(y_pred, np.array([0, 1])))
        0.5
    """
    return _mean_defined(error_ps(y_pred, y_true))


# --- Overall miscoverage and singleton error ---


def miscoverage_overall_ps(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float | np.floating:
    """Mean fraction of true labels omitted from prediction sets.

    A one-dimensional y_true contains multiclass label indices: each sample
    contributes 0 if its true label is included, or 1 otherwise. A
    two-dimensional y_true contains multilabel masks: each sample contributes
    ``|Y minus S| / |Y|``. Average these fractions with equal weight per sample,
    regardless of the number of true labels.

    Multilabel samples with no true labels are excluded because their fraction
    is undefined. Return NaN if none remain, or if a multiclass batch is empty.

    Args:
        y_pred: Binary prediction-set masks, shape (n_samples, n_labels).
        y_true: Multiclass indices, shape (n_samples,), or binary multilabel
            masks with the same shape as y_pred. Boolean and numeric 0/1
            multilabel masks are accepted.

    Returns:
        Mean per-sample miscoverage, or NaN when it is undefined.

    Raises:
        ValueError: If the shapes do not match the supported formats, or the
            multilabel inputs are not binary membership masks.

    Examples:
        >>> y_pred = np.array([[1, 0, 0], [1, 0, 0]])
        >>> y_true = np.array([1, 0])
        >>> float(miscoverage_overall_ps(y_pred, y_true))
        0.5
        >>> y_pred = np.array([[1, 0, 1], [0, 0, 1]])
        >>> y_true = np.array([[1, 1, 0], [0, 0, 1]])
        >>> float(miscoverage_overall_ps(y_pred, y_true))
        0.25
    """
    _validate_prediction_targets(y_pred, y_true)
    if y_true.ndim == 1:
        truth_pred = y_pred[np.arange(len(y_true)), y_true]

        if not len(truth_pred):
            return float("nan")

        return 1 - truth_pred.mean()

    true_counts = y_true.sum(axis=1)
    valid = true_counts > 0
    if not valid.any():
        return float("nan")

    covered_counts = (y_pred[valid] * y_true[valid]).sum(axis=1)

    return np.mean(1 - covered_counts / true_counts[valid])


def error_overall_ps(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float | np.floating:
    """Multiclass miss rate among predictions whose set size is exactly one.

    The denominator is the number of singleton predictions. Return NaN if
    there are none, including an empty batch.

    Examples:
        >>> y_pred = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> y_true = np.array([1, 0, 1])
        >>> float(error_overall_ps(y_pred, y_true))
        0.5
    """
    _validate_prediction_targets(y_pred, y_true)
    if y_true.ndim != 1:
        raise ValueError("singleton overall error requires multiclass label indices")

    truth_pred = y_pred[np.arange(len(y_true)), y_true]
    truth_pred = truth_pred[y_pred.sum(axis=1) == 1]

    if not len(truth_pred):
        return float("nan")

    return 1 - truth_pred.mean()


# --- Weighted multilabel cost and value ---


def fp_cost(y_pred: np.ndarray, y_true: np.ndarray, cost_weights=1.0) -> float:
    """Mean weighted false-positive cost per sample, in the supplied weight units.

    Compute ``sum(cost_weights * S * (1 - Y))`` per sample, then average.
    cost_weights may be a finite non-negative scalar or a one-dimensional
    array/list with one weight per encoded class. Unit weights give ordinary
    mean false-positive counts. Return NaN for an empty batch.

    Examples:
        >>> y_pred = np.array([[1, 1, 1], [1, 1, 0]])
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
        >>> fp_cost(y_pred, y_true, cost_weights=[2, 3, 5])
        2.5
    """
    counts = _weighted_label_counts(y_pred, 1 - y_true, cost_weights)

    return float(counts.mean()) if len(counts) else float("nan")


def tp_value(y_pred: np.ndarray, y_true: np.ndarray, value_weights=1.0) -> float:
    """Mean weighted true-positive value per sample, in the supplied weight units.

    Compute ``sum(value_weights * S * Y)`` per sample, then average.
    value_weights may be a finite non-negative scalar or a one-dimensional
    array/list with one weight per encoded class. Unit weights give ordinary
    mean true-positive counts. Return NaN for an empty batch.

    Examples:
        >>> y_pred = np.array([[1, 1, 1], [1, 1, 0]])
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
        >>> tp_value(y_pred, y_true, value_weights=[2, 3, 5])
        5.0
    """
    counts = _weighted_label_counts(y_pred, y_true, value_weights)

    return float(counts.mean()) if len(counts) else float("nan")


def fp_cost_violation_rate(
    y_pred: np.ndarray, y_true: np.ndarray, target_cost: float, cost_weights=1.0
) -> float:
    """Fraction of samples whose weighted false-positive cost exceeds a limit.

    Compute each sample's cost in the supplied weight units, then return the
    fraction with cost strictly greater than target_cost. Equal costs do not
    count as violations. The relative tolerance is the larger of 1e-12 and
    the floating-point precision of cost_weights and target_cost; the absolute
    tolerance is zero. The denominator includes all samples, including those
    with no true labels. Return NaN for an empty batch.

    target_cost must be a finite non-negative scalar; zero-dimensional NumPy
    arrays and Torch tensors are accepted. cost_weights follows
    fp_cost: a finite non-negative scalar or one weight per encoded class.

    Examples:
        >>> y_true = np.zeros((2, 2), dtype=int)
        >>> y_pred = np.array([[0, 0], [1, 1]])
        >>> fp_cost(y_pred, y_true)
        1.0
        >>> fp_cost_violation_rate(y_pred, y_true, target_cost=1)
        0.5
        >>> y_pred = np.array([[1, 0], [0, 1]])
        >>> fp_cost(y_pred, y_true)
        1.0
        >>> fp_cost_violation_rate(y_pred, y_true, target_cost=1)
        0.0
    """
    rtol = _comparison_rtol(target_cost, cost_weights)
    target_cost = _validate_target_cost(target_cost)
    costs = _weighted_label_counts(y_pred, 1 - y_true, cost_weights)
    if not len(costs):
        return float("nan")

    at_limit = np.isclose(costs, target_cost, rtol=rtol, atol=0.0)

    return float(np.mean((costs > target_cost) & ~at_limit))


def _weighted_label_counts(y_pred, y_true, weights):
    """Validate masks/weights and return one weighted label count per sample."""
    _validate_multilabel_masks(y_pred, y_true)
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim > 1
        or (weights.ndim == 1 and weights.shape != (y_pred.shape[1],))
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("weights must be a finite non-negative scalar or class vector")

    return (y_pred * y_true * weights).sum(axis=1)


# --- Input validation and undefined class averages ---


def _mean_defined(rates):
    observed = rates[~np.isnan(rates)]

    return _mean_or_nan(observed)


def _mean_or_nan(values):
    """Keep NumPy's mean dtype without warning on an empty denominator."""
    if len(values):
        return values.mean()

    dtype = values.dtype if np.issubdtype(values.dtype, np.inexact) else np.dtype(float)

    return dtype.type(np.nan)


def _validate_prediction_sets(y_pred):
    if y_pred.ndim != 2:
        raise ValueError("prediction-set masks must have (n_samples, n_labels) shapes")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain binary membership masks")


def _validate_prediction_targets(y_pred, y_true):
    if y_true.ndim == 1:
        if y_pred.ndim != 2 or len(y_pred) != len(y_true):
            raise ValueError("y_pred must have one prediction-set row per true label")

        _validate_prediction_sets(y_pred)
        if not np.issubdtype(y_true.dtype, np.integer):
            raise ValueError("multiclass labels must be integer class indices")

        if np.any((y_true < 0) | (y_true >= y_pred.shape[1])):
            raise ValueError("multiclass labels must index prediction-set columns")

    else:
        _validate_multilabel_masks(y_pred, y_true)


def _validate_multilabel_masks(y_pred, y_true):
    """Require aligned binary matrices, including boolean and numeric 0/1 masks."""
    if y_true.ndim != 2 or y_pred.shape != y_true.shape:
        raise ValueError(
            "multilabel masks must have matching (n_samples, n_labels) shapes"
        )

    _validate_prediction_sets(y_pred)
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("multilabel y_pred and y_true must be binary membership masks")


def _validate_target_cost(target_cost):
    if getattr(target_cost, "ndim", 0) == 0:
        # .item() accepts scalar containers without depending on Torch here.
        if hasattr(target_cost, "item"):
            target_cost = target_cost.item()

        if (
            not isinstance(target_cost, (bool, np.bool_))
            and isinstance(target_cost, (int, float, np.integer, np.floating))
            and np.isfinite(target_cost)
            and target_cost >= 0
        ):
            return target_cost

    raise ValueError("target_cost must be a finite non-negative scalar")


def _comparison_rtol(target_cost, cost_weights):
    """Allow rounding at the precision of the original limit and weight dtypes."""
    rtol = 1e-12
    for values in (target_cost, cost_weights):
        if isinstance(values, torch.Tensor):
            if values.is_floating_point():
                rtol = max(rtol, torch.finfo(values.dtype).eps)
        else:
            dtype = np.asarray(values).dtype
            if np.issubdtype(dtype, np.floating):
                rtol = max(rtol, np.finfo(dtype).eps)

    return rtol
