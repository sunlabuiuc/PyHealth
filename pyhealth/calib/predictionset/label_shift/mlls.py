"""
Maximum likelihood label shift estimation with expectation maximization.

MLLS estimates the label distribution of a target distribution from a
black box classifier's predicted probabilities, without target labels
(Alexandari et al. 2020). Under label shift, a classifier that estimates
the source class probabilities P(y | x) turns the target data into a
mixture of the source class-conditional distributions, with the target
label distribution as the mixture weights. MLLS fits those weights by
maximum likelihood: the E-step reweights each target point's predicted
probabilities by the current importance weights q(y) / p(y), and the
M-step averages the resulting posteriors into the next estimate.

This module implements the basic EM algorithm for the mixture
proportions (Saerens et al. 2002), as analyzed for label shift in
Alexandari et al. It does not include that paper's bias-corrected
calibration step, so the predicted probabilities should already be
calibrated.

Papers:
    Alexandari, Amr, Anshul Kundaje, and Avanti Shrikumar. "Maximum
    Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label
    Shift Adaptation." International Conference on Machine Learning
    (ICML), 2020. https://arxiv.org/abs/1901.06852
    Saerens, Marco, Patrice Latinne, and Christine Decaestecker.
    "Adjusting the Outputs of a Classifier to New a Priori
    Probabilities: A Simple Procedure." Neural Computation, 2002.
"""

import warnings

import numpy as np

from pyhealth.calib.predictionset.label_shift.black_box_shift import _as_label_array

__all__ = ["em"]


def _as_probability_matrix(y_prob_target, num_classes: int) -> np.ndarray:
    """Validate the target predicted probabilities.

    Returns them as a float64 array of shape (n_target, num_classes)
    with rows renormalized to sum to exactly 1.
    """
    proba = np.asarray(y_prob_target, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] == 0:
        raise ValueError(
            "y_prob_target must be a non-empty two-dimensional array of "
            "predicted probabilities with one row per target example."
        )
    if proba.shape[1] != num_classes or num_classes < 1:
        raise ValueError(
            "y_prob_target must have one probability column per class."
        )
    if not np.all(np.isfinite(proba)):
        raise ValueError(
            "y_prob_target must contain finite predicted probabilities."
        )
    if np.any(proba < 0) or np.any(proba > 1):
        raise ValueError(
            "y_prob_target entries must lie in [0, 1]."
        )
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-6):
        raise ValueError(
            "Each row of y_prob_target must be a probability distribution "
            "summing to 1."
        )
    return proba / row_sums[:, np.newaxis]


def _posteriors(
    proba: np.ndarray, q: np.ndarray, source_prior: np.ndarray
) -> np.ndarray:
    """One E-step: reweight each target row by the current importance
    weights q(y) / p(y) and normalize to a posterior over classes.

    Classes with zero source prior keep zero importance; validation
    guarantees such classes receive no predicted probability, so their
    columns are all zero here.
    """
    importance = np.zeros_like(q)
    positive = source_prior > 0
    importance[positive] = q[positive] / source_prior[positive]
    numerator = proba * importance
    denominators = numerator.sum(axis=1)
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0):
        bad = np.flatnonzero(
            ~np.isfinite(denominators) | (denominators <= 0)
        ).tolist()
        raise ValueError(
            "EM encountered a target example with zero or nonfinite "
            f"total weight under the current fit (rows {bad}); its "
            "posterior is undefined."
        )
    return numerator / denominators[:, np.newaxis]


def _validate_source_prior(
    source_label_dist, y_true_source: np.ndarray, num_classes: int
) -> np.ndarray:
    """Return the source label distribution, either supplied or estimated
    from the source labels. A supplied distribution is used as given,
    even when it differs from the empirical source proportions, since a
    known population prior almost never matches a finite sample exactly.
    It must still give positive probability to every class observed in
    the source labels; a zero there would contradict the source data
    itself, rather than just differing from its proportions."""
    empirical = np.bincount(y_true_source, minlength=num_classes) / len(
        y_true_source
    )
    if source_label_dist is None:
        return empirical
    prior = np.asarray(source_label_dist, dtype=np.float64)
    if prior.ndim != 1 or len(prior) != num_classes:
        raise ValueError(
            "source_label_dist must be a one-dimensional array with one "
            f"entry per class (expected {num_classes})."
        )
    if not np.all(np.isfinite(prior)) or np.any(prior < 0):
        raise ValueError(
            "source_label_dist must contain finite non-negative entries."
        )
    if not np.isclose(prior.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError("source_label_dist must sum to 1.")
    if np.any((empirical > 0) & (prior == 0)):
        raise ValueError(
            "source_label_dist gives zero probability to a class "
            "observed in y_true_source."
        )
    return prior


def em(
    y_true_source,
    y_prob_target,
    source_label_dist=None,
    max_iter=1000,
    tol=1e-8,
):
    """Estimate the target label distribution with EM (MLLS).

    The estimate can be passed to
    :meth:`pyhealth.calib.predictionset.LabelShift.calibrate` as
    ``target_label_dist`` to build prediction sets that account for the
    shift. The classifier only needs to supply predicted probabilities
    on the unlabeled target data; its argmax predictions are not used.

    Args:
        y_true_source: True labels of a labeled source dataset, used to
            estimate the source label distribution. One entry per source
            example.
        y_prob_target: The model's predicted class probabilities on
            unlabeled target examples, an array of shape (n_target,
            n_classes) with rows summing to 1. No target labels are
            needed.
        source_label_dist: Optional known source label distribution. If
            given, it is used as the source prior in place of the
            empirical distribution of y_true_source, so it may differ
            from the sample proportions. It must still give positive
            probability to every class observed in y_true_source.
        max_iter: Maximum number of EM iterations. Must be a positive
            integer.
        tol: The EM loop stops when the largest absolute change in the
            estimated distribution between iterations falls below tol.
            Must be positive and finite.

    Returns:
        The estimated target label distribution, an array of shape
        (n_classes,) that is finite, non-negative, and sums to 1, in the
        same class order as the columns of y_prob_target.

    Raises:
        ValueError: If the inputs are not valid probability data, if a
            class never appears in the source data but receives positive
            predicted probability on the target data, if
            source_label_dist gives zero probability to a class observed
            in y_true_source, if max_iter is not a positive integer, or
            if tol is not positive and finite.

    Note:
        The same fixed classifier must produce the target probabilities,
        and the labeled source data should be held out from that
        classifier's training data: MLLS assumes the classifier
        estimates the source class probabilities P(y | x), which training
        data would violate. The probabilities should be calibrated:
        miscalibrated probabilities bias the estimate (Alexandari et al.
        pair EM with a bias-corrected calibration step for this reason;
        this function implements the basic EM only, without that step).
        The estimate is a plug-in input for LabelShift, so coverage
        guarantees stated for known label ratios become approximate.
        Convergence does not establish identifiability: when the target
        probabilities are nearly constant across examples (the largest
        per-class spread across rows is at most 1e-3), they carry
        almost no information about a prevalence change, and a
        UserWarning is issued in that case. The estimate may then be
        driven by the source prior, but the source prior is an exact
        fixed point only when the constant row equals it. More
        generally, an estimate that equals the source prior is not
        evidence on its own that no shift occurred.

    Examples:
        >>> import numpy as np
        >>> y_true_source = np.array([0, 0, 1, 1])
        >>> # A perfect classifier: each row is one-hot on the true class.
        >>> y_prob_target = np.array(
        ...     [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ... )
        >>> em(y_true_source, y_prob_target)
        array([0.75, 0.25])
    """
    y_true_source = _as_label_array(y_true_source, "y_true_source")
    proba = np.asarray(y_prob_target, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] == 0 or proba.shape[1] < 1:
        raise ValueError(
            "y_prob_target must be a non-empty two-dimensional array of "
            "predicted probabilities with one row per target example."
        )
    num_classes = proba.shape[1]
    if y_true_source.max() >= num_classes:
        raise ValueError(
            f"y_true_source contains label {y_true_source.max()}, which "
            f"is outside [0, {num_classes})."
        )
    proba = _as_probability_matrix(y_prob_target, num_classes)
    source_prior = _validate_source_prior(
        source_label_dist, y_true_source, num_classes
    )

    # A class with no source mass can only keep a zero target proportion;
    # if the target data assign it positive probability, the shift is
    # inconsistent with the source support.
    zero_prior = source_prior == 0
    if np.any(zero_prior):
        supported = np.any(proba[:, zero_prior] > 0, axis=0)
        bad = np.flatnonzero(zero_prior)[supported].tolist()
        if bad:
            raise ValueError(
                "EM cannot estimate the target proportion of classes "
                "that never appear in the source data but receive "
                f"positive predicted probability: {bad}."
            )

    # Every invalid input raises ValueError (not TypeError), including
    # a wrongly typed max_iter.
    if isinstance(max_iter, bool) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError(  # noqa: TRY004
            f"max_iter must be a positive integer, got {max_iter!r}."
        )
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(
            f"max_iter must be a positive integer, got {max_iter}."
        )
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be positive and finite, got {tol}.")

    # Nearly constant target rows carry almost no information about a
    # prevalence change: with a spread of at most 1e-3 in any class,
    # the rows cannot distinguish target proportions, so the estimate
    # may be driven by the source prior. The source prior is an exact
    # fixed point only when the constant row equals it. Warn instead
    # of raising, since the inputs are valid, only uninformative.
    # This catches clearly uninformative inputs; it does not diagnose
    # every unidentifiable case, and convergence alone never
    # establishes identifiability.
    if proba.shape[0] > 1 and (
        proba.max(axis=0) - proba.min(axis=0)
    ).max() <= 1e-3:
        warnings.warn(
            "Target predicted probabilities are nearly constant "
            "across examples (largest per-class spread at most 0.001), "
            "so they carry little or no information about a prevalence "
            "change; the EM estimate may simply reflect the source "
            "prior.",
            UserWarning,
            stacklevel=2,
        )

    q = source_prior.copy()
    converged = False
    for _ in range(max_iter):
        q_new = _posteriors(proba, q, source_prior).mean(axis=0)
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            converged = True
            break
        q = q_new
    if not converged:
        warnings.warn(
            f"EM did not converge within {max_iter} iterations; "
            "returning the last estimate, which may not have reached "
            "the fixed point.",
            UserWarning,
            stacklevel=2,
        )

    # The M-step averages valid posteriors, so q must already lie on the
    # simplex; validate that invariant instead of renormalizing, which
    # would hide a broken computation.
    if (
        not np.all(np.isfinite(q))
        or np.any(q < 0)
        or not np.isclose(q.sum(), 1.0, rtol=0, atol=1e-8)
    ):
        raise RuntimeError(
            "Internal error: the EM estimate left the probability simplex."
        )
    return q
