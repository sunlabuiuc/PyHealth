"""
Maximum likelihood label shift estimation (MLLS) with expectation
maximization (EM).

MLLS estimates the label distribution of a target distribution from a
black box classifier's predicted probabilities, without target labels
(Alexandari et al. 2020). Under label shift, a classifier trained on the
source distribution still estimates the source class probabilities
P(y | x), so the target data are a mixture of the source
class-conditional distributions with the target label distribution as
the mixture weights. MLLS fits that mixture by maximum likelihood: the
E-step reweights each target point's predicted probabilities by the
current importance weights q(y) / p(y), and the M-step averages those
posteriors to form the next estimate of the target label distribution q.

Paper:
    Alexandari, Amr, Anshul Kundaje, and Avanti Shrikumar.
    "Maximum Likelihood with Bias-Corrected Calibration is
    Hard-To-Beat at Label Shift Adaptation."
    International Conference on Machine Learning (ICML), 2020.
    https://arxiv.org/abs/1901.06852
"""

import warnings

import numpy as np

from pyhealth.calib.predictionset.label_shift.bbse import _as_label_array

__all__ = ["em"]


def _as_probability_matrix(y_proba_target, num_classes: int) -> np.ndarray:
    """Validate the target predicted probabilities.

    Returns them as a float64 array of shape (n_target, num_classes).
    """
    proba = np.asarray(y_proba_target, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] == 0:
        raise ValueError(
            "y_proba_target must be a non-empty two-dimensional array of "
            "predicted probabilities with one row per target example."
        )
    if not np.all(np.isfinite(proba)):
        raise ValueError(
            "y_proba_target must contain finite predicted probabilities."
        )
    if np.any(proba < 0):
        raise ValueError(
            "y_proba_target must contain non-negative predicted "
            "probabilities."
        )
    if proba.shape[1] != num_classes:
        raise ValueError(
            "y_proba_target has "
            f"{proba.shape[1]} probability columns but num_classes is "
            f"{num_classes}."
        )
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-6):
        raise ValueError(
            "Each row of y_proba_target must be a probability "
            "distribution summing to 1."
        )
    return proba


def em(
    y_true_source,
    y_proba_target,
    num_classes=None,
    max_iter=1000,
    tol=1e-6,
):
    """Estimate the target label distribution with MLLS via EM.

    The estimate can be passed to
    :meth:`pyhealth.calib.predictionset.LabelShift.calibrate` as
    ``target_label_dist`` to build prediction sets that account for the
    shift. The classifier only needs to supply predicted probabilities
    on the unlabeled target data; its argmax predictions are not used.

    Args:
        y_true_source: True labels of a labeled source dataset, used to
            estimate the source label distribution. One entry per source
            example.
        y_proba_target: The model's predicted class probabilities on
            unlabeled target examples, an array of shape (n_target,
            num_classes) with rows summing to 1. No target labels are
            needed.
        num_classes: Number of classes. If None, it is inferred from the
            width of y_proba_target.
        max_iter: Maximum number of EM iterations. Must be a positive
            integer.
        tol: The EM loop stops when the largest absolute change in the
            estimated distribution between iterations falls below tol.
            Must be non-negative; with tol=0 the loop runs until the
            estimate stops changing exactly or max_iter is reached.

    Returns:
        The estimated target label distribution, an array of shape
        (num_classes,) that is non-negative and sums to 1.

    Raises:
        ValueError: If the inputs are not valid, if a class never
            appears in the source labels (its importance weight
            q(y) / p(y) is undefined), if max_iter is not a positive
            integer, or if tol is not positive.

    Note:
        The same fixed classifier must produce the target probabilities,
        and the labeled source data should be held out from that
        classifier's training data: MLLS assumes the classifier
        estimates the source class probabilities P(y | x), which training
        data would violate. The probabilities should be calibrated:
        miscalibrated probabilities bias the estimate (Alexandari et al.
        pair EM with a calibration step for this reason). The estimate is
        a plug-in input for LabelShift, so coverage guarantees stated
        for known label ratios become approximate. The EM estimate is
        only meaningful if the classifier separates the classes well:
        with an uninformative classifier the likelihood is flat and the
        estimate can be unstable. A UserWarning is issued if the EM loop
        does not converge within max_iter iterations.

    Examples:
        >>> import numpy as np
        >>> y_true_source = np.array([0, 0, 1, 1])
        >>> # A perfect classifier: each row is one-hot on the true class.
        >>> y_proba_target = np.array(
        ...     [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ... )
        >>> em(y_true_source, y_proba_target)
        array([0.75, 0.25])
    """
    y_true_source = _as_label_array(y_true_source, "y_true_source")
    if num_classes is None:
        if np.ndim(y_proba_target) != 2:
            raise ValueError(
                "num_classes could not be inferred: y_proba_target must "
                "be a two-dimensional array when num_classes is None."
            )
        num_classes = np.shape(y_proba_target)[1]
    else:
        if isinstance(num_classes, bool) or not isinstance(
            num_classes, (int, np.integer)
        ):
            raise ValueError(
                f"num_classes must be a positive integer, got "
                f"{num_classes!r}."
            )
        num_classes = int(num_classes)
        if num_classes < 1:
            raise ValueError(
                f"num_classes must be a positive integer, got "
                f"{num_classes}."
            )
    if y_true_source.max() >= num_classes:
        raise ValueError(
            f"y_true_source contains label {y_true_source.max()}, which "
            f"is outside [0, {num_classes})."
        )
    proba = _as_probability_matrix(y_proba_target, num_classes)

    if isinstance(max_iter, bool) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError(
            f"max_iter must be a positive integer, got {max_iter!r}."
        )
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(
            f"max_iter must be a positive integer, got {max_iter}."
        )
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be a non-negative number, got {tol}.")

    source_prior = np.bincount(y_true_source, minlength=num_classes) / len(
        y_true_source
    )
    if np.any(source_prior == 0):
        missing = np.flatnonzero(source_prior == 0).tolist()
        raise ValueError(
            "MLLS cannot estimate the target proportion of classes that "
            f"never appear in the source labels: {missing}."
        )

    q = source_prior.copy()
    converged = False
    for _ in range(max_iter):
        # E-step: reweight each target point's predicted probabilities
        # by the current importance weights, then normalize each row to
        # a posterior over classes.
        importance = q / source_prior
        numerator = proba * importance
        denominators = numerator.sum(axis=1, keepdims=True)
        posterior = np.divide(
            numerator,
            denominators,
            out=np.zeros_like(numerator),
            where=denominators > 0,
        )
        # M-step: the next estimate is the average posterior.
        q_new = posterior.mean(axis=0)
        if np.max(np.abs(q_new - q)) <= tol:
            q = q_new
            converged = True
            break
        q = q_new
    if not converged:
        warnings.warn(
            "The EM loop did not converge within "
            f"{max_iter} iterations; the returned estimate may be "
            "unstable. Consider increasing max_iter or relaxing tol.",
            UserWarning,
            stacklevel=2,
        )

    total = q.sum()
    if total <= 0:
        raise ValueError(
            "MLLS produced a degenerate estimate from the given inputs: "
            "no target example has positive weight under the fitted "
            "mixture."
        )
    return q / total
