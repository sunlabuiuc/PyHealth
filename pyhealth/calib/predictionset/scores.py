"""Shared, pluggable conformity/nonconformity score functions.

This module separates the *score* used by a conformal-prediction-set method
from the *calibration/thresholding procedure* it's plugged into. These are
two independent axes: the choice of score ("threshold"/LAC vs "aps") does
not depend on how the resulting scores get turned into a threshold (marginal
quantile, per-class quantile, per-cluster quantile, weighted quantile for
covariate shift, or localized weighted quantile for neighborhood methods).

Supported score types:

    - "threshold" (a.k.a. LAC, Sadinle, Lei, and Wasserman 2019): the score
      for class k is simply based on the model's predicted probability for
      k. Simple, but not adaptive to how "peaked" or "flat" the predicted
      distribution is.

    - "aps" (Adaptive Prediction Sets, Romano, Sesia, and Candes 2020;
      score definition as restated in Angelopoulos, Bates, Malik, and
      Jordan 2021, "Uncertainty Sets for Image Classifiers using Conformal
      Prediction", Algorithm 2): the score for class k is the cumulative
      sum of predicted probabilities for all classes ranked strictly above
      k, plus a Uniform(0,1)-randomized fraction of k's own probability::

          E(x, k) = sum_{j : pi(x,j) > pi(x,k)} pi(x,j) + U * pi(x,k)

      where pi(x, ·) are the model's predicted class probabilities and
      U ~ Uniform(0,1) is drawn once per example and reused across every
      candidate class k for that example (so the resulting prediction sets
      are "nested": the set of included classes is always a prefix of the
      classes sorted by decreasing probability). This adapts the set size
      to the model's confidence for each individual input, which the
      "threshold" score does not.

    - "raps" (Regularized Adaptive Prediction Sets, Angelopoulos, Bates,
      Malik, and Jordan 2021, "Uncertainty Sets for Image Classifiers
      using Conformal Prediction", Algorithm 3): the APS score plus a
      regularization term that penalizes low-ranked classes, encouraging
      smaller prediction sets::

          S(x, k) = E(x, k) + lambda * (rank(x, k) - k_reg)^+

      where rank(x, k) is the 1-indexed rank of class k in descending
      probability order, lambda >= 0 is the penalty weight, k_reg >= 0 is
      the rank at which penalizing starts, and (z)^+ = max(0, z). lambda
      and k_reg are tuned on a validation split in the paper; here they
      default to 0.1 and 1 as convenience defaults and are settable via
      the ``raps_lambda`` and ``raps_kreg`` arguments of the score
      functions (and of the predictors that accept ``score_type``).

All scores are computed here in *nonconformity* convention (higher = less
conforming, i.e. 1 minus a probability-like quantity) since that's the
convention BaseConformal/LABEL/ClusterLabel use internally. A *conformity*
(higher = more conforming) variant is also provided for CovariateLabel/
NeighborhoodLabel, which use the opposite sign convention internally; it is
simply `1 - nonconformity`, preserving the same ranking of examples either
way.
"""

import numpy as np

__all__ = [
    "SUPPORTED_SCORE_TYPES",
    "all_class_conformity_scores",
    "all_class_nc_scores",
    "true_class_conformity_scores",
    "true_class_nc_scores",
]

SUPPORTED_SCORE_TYPES = ("threshold", "aps", "raps")


def _validate_score_type(score_type: str) -> None:
    if score_type not in SUPPORTED_SCORE_TYPES:
        raise ValueError(
            f"Unknown score_type: {score_type!r}. Supported: "
            f"{SUPPORTED_SCORE_TYPES}."
        )


def _validate_raps_lambda(raps_lambda) -> float:
    """Validate the RAPS penalty-weight hyperparameter.

    Only called when score_type="raps" is selected, so other score types
    never fail on ``raps_lambda``.
    """
    if isinstance(raps_lambda, bool) or not isinstance(
        raps_lambda, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "raps_lambda must be a non-negative float, "
            f"got {raps_lambda!r}."
        )
    raps_lambda = float(raps_lambda)
    if not np.isfinite(raps_lambda) or raps_lambda < 0:
        raise ValueError(
            "raps_lambda must be a non-negative finite float, "
            f"got {raps_lambda!r}."
        )
    return raps_lambda


def _validate_raps_kreg(raps_kreg) -> int:
    """Validate the RAPS rank-cutoff hyperparameter.

    Only called when score_type="raps" is selected, so other score types
    never fail on ``raps_kreg``. A larger kreg is mathematically valid
    and simply disables regularization for that task; no upper bound is
    enforced.
    """
    if isinstance(raps_kreg, bool) or not isinstance(
        raps_kreg, (int, np.integer)
    ):
        raise TypeError(
            "raps_kreg must be a non-negative integer, "
            f"got {raps_kreg!r}."
        )
    raps_kreg = int(raps_kreg)
    if raps_kreg < 0:
        raise ValueError(
            "raps_kreg must be a non-negative integer, "
            f"got {raps_kreg!r}."
        )
    return raps_kreg


def _aps_all_class_nc_scores_with_ranks(
    y_prob: np.ndarray,
    rng: np.random.Generator,
    randomize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the APS nonconformity score for every class, every row,
    together with each class's 1-indexed rank in descending probability
    order.

    E(x, k) = [sum of predicted probabilities for classes ranked strictly
    above k] + U * p(x, k), with U ~ Uniform(0,1) drawn once per row and
    reused across all classes in that row (Angelopoulos et al. 2021,
    Algorithm 2/3, with the regularization term lambda=0, i.e. plain APS
    rather than RAPS). The scores and the ranks come from one shared sort
    and one per-row uniform draw: tied probabilities are assigned distinct
    ranks at random, and the same assigned rank is used consistently for
    both the APS ordering and the RAPS regularization.

    Args:
        y_prob: Predicted probabilities, shape (N, K).
        rng: Random generator used to draw the per-row U ~ Uniform(0,1)
            tie-breaking/adaptivity term.
        randomize: If False, uses U=1 for every row (the conservative,
            non-randomized variant: ties are broken by always including the
            full probability mass of a class's own rank). If True (the
            variant the APS/RAPS papers use for their reported results),
            draws a genuine U ~ Uniform(0,1) per row.

    Returns:
        (scores, ranks), both of shape (N, K) in original class-index
        order; higher scores mean less conforming, ranks are 1-indexed.
    """
    n, k = y_prob.shape
    # Ties in probability are broken randomly by perturbing the sort key
    # infinitesimally, per the APS paper's note that "label-ordering ties
    # should be broken randomly" when probabilities aren't all distinct.
    tie_break = rng.uniform(0.0, 1e-12, size=y_prob.shape)
    order = np.argsort(-(y_prob + tie_break), axis=1)  # descending, per row
    sorted_probs = np.take_along_axis(y_prob, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    # Sum of all classes ranked strictly above each rank r (0-indexed):
    # cumsum up to and including r, minus r's own probability.
    cumsum_excl_own = cumsum - sorted_probs

    if randomize:
        u = rng.uniform(0.0, 1.0, size=(n, 1))
    else:
        u = np.ones((n, 1))

    sorted_scores = cumsum_excl_own + u * sorted_probs  # (N, K), sorted order

    # Undo the sort to get back to original class-index order.
    inverse_order = np.argsort(order, axis=1)
    scores = np.take_along_axis(sorted_scores, inverse_order, axis=1)
    ranks_sorted = np.broadcast_to(np.arange(1, k + 1), (n, k))
    ranks = np.take_along_axis(ranks_sorted, inverse_order, axis=1)
    return scores, ranks


def _aps_all_class_nc_scores(
    y_prob: np.ndarray,
    rng: np.random.Generator,
    randomize: bool,
) -> np.ndarray:
    """Computes the APS nonconformity score for every class, every row.

    See :func:`_aps_all_class_nc_scores_with_ranks` for the definition.
    """
    scores, _ = _aps_all_class_nc_scores_with_ranks(y_prob, rng, randomize)
    return scores


def _raps_all_class_nc_scores(
    y_prob: np.ndarray,
    rng: np.random.Generator,
    randomize: bool,
    raps_lambda: float,
    raps_kreg: int,
) -> np.ndarray:
    """Computes the RAPS nonconformity score for every class, every row.

    S(x, k) = E(x, k) + lambda * (rank(x, k) - k_reg)^+, with E the APS
    score, rank the 1-indexed rank in descending probability order, and
    lambda, k_reg validated non-negative by the caller (Angelopoulos et
    al. 2021, Algorithm 3). The APS part and the penalty share one rank
    ordering and one per-row uniform draw, so tied probabilities cannot
    receive one rank under APS and a different rank under the penalty.

    Args:
        y_prob: Predicted probabilities, shape (N, K).
        rng: Random generator used for the APS tie-breaking and per-row
            U draws (same convention as the APS implementation above).
        randomize: If True, draws a genuine U ~ Uniform(0,1) per row. If
            False, uses U=1 for every row.
        raps_lambda: Penalty weight, validated non-negative and finite by
            the caller.
        raps_kreg: Rank at which penalizing starts, validated a
            non-negative integer by the caller.

    Returns:
        Nonconformity scores of shape (N, K); higher means less conforming.
    """
    aps_scores, ranks = _aps_all_class_nc_scores_with_ranks(
        y_prob, rng, randomize
    )
    penalty = raps_lambda * np.maximum(ranks - raps_kreg, 0)
    return aps_scores + penalty


def all_class_nc_scores(
    y_prob: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    raps_lambda: float = 0.1,
    raps_kreg: int = 1,
) -> np.ndarray:
    """Nonconformity score (higher = less conforming) for every class.

    Args:
        y_prob: Predicted probabilities, shape (N, K).
        score_type: "threshold" (Sadinle, Lei, and Wasserman 2019), "aps"
            (Romano, Sesia, and Candes 2020), or "raps" (Angelopoulos et
            al. 2021). Default "threshold".
        rng: Random generator, required (and only used) if score_type is
            "aps" or "raps" and randomize=True.
        randomize: Whether to use the randomized ("exact coverage") variant
            of APS/RAPS. Ignored for score_type="threshold".
        raps_lambda: Penalty weight for score_type="raps" (default 0.1, a
            convenience default; the paper tunes it on a validation
            split). Ignored for other score types.
        raps_kreg: Rank at which penalizing starts for score_type="raps"
            (default 1, a convenience default). Ignored for other score
            types.

    Returns:
        Nonconformity scores of shape (N, K).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import all_class_nc_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> all_class_nc_scores(y_prob, score_type="threshold")
        array([[0.3, 0.8, 0.9],
               [0.7, 0.5, 0.8]])
        >>> rng = np.random.default_rng(0)
        >>> scores = all_class_nc_scores(y_prob, score_type="aps", rng=rng)
        >>> np.round(scores, 2)
        array([[0.42, 0.82, 0.96],
               [0.72, 0.36, 0.95]])
        >>> all_class_nc_scores(
        ...     np.array([[0.7, 0.2, 0.1]]), score_type="raps",
        ...     randomize=False, raps_lambda=0.1, raps_kreg=1)
        array([[0.7, 1. , 1.2]])
    """
    _validate_score_type(score_type)
    if score_type == "threshold":
        return 1.0 - y_prob
    if rng is None:
        rng = np.random.default_rng()
    if score_type == "aps":
        return _aps_all_class_nc_scores(y_prob, rng, randomize)
    # score_type == "raps"
    return _raps_all_class_nc_scores(
        y_prob,
        rng,
        randomize,
        _validate_raps_lambda(raps_lambda),
        _validate_raps_kreg(raps_kreg),
    )


def all_class_conformity_scores(
    y_prob: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    raps_lambda: float = 0.1,
    raps_kreg: int = 1,
) -> np.ndarray:
    """Conformity score (higher = more conforming) for every class.

    Equivalent to ``1 - all_class_nc_scores(...)``: same ranking of
    examples, just the sign convention used by CovariateLabel and
    NeighborhoodLabel (which threshold with ``score >= t`` rather than
    ``nc_score <= t``).

    Args:
        raps_lambda: Penalty weight for score_type="raps" (default 0.1,
            validated non-negative and finite; ignored for other score
            types).
        raps_kreg: Rank at which penalizing starts for score_type="raps"
            (default 1, validated a non-negative integer; ignored for
            other score types).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import all_class_conformity_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> all_class_conformity_scores(y_prob, score_type="threshold")
        array([[0.7, 0.2, 0.1],
               [0.3, 0.5, 0.2]])
    """
    return 1.0 - all_class_nc_scores(
        y_prob, score_type, rng, randomize, raps_lambda, raps_kreg
    )


def true_class_nc_scores(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    raps_lambda: float = 0.1,
    raps_kreg: int = 1,
) -> np.ndarray:
    """Nonconformity score of the true class only, shape (N,). Used during
    calibration, where only the true label's score is needed.

    Args:
        raps_lambda: Penalty weight for score_type="raps" (default 0.1,
            validated non-negative and finite; ignored for other score
            types).
        raps_kreg: Rank at which penalizing starts for score_type="raps"
            (default 1, validated a non-negative integer; ignored for
            other score types).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import true_class_nc_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> y_true = np.array([0, 1])
        >>> true_class_nc_scores(y_prob, y_true, score_type="threshold")
        array([0.3, 0.5])
    """
    scores = all_class_nc_scores(
        y_prob, score_type, rng, randomize, raps_lambda, raps_kreg
    )
    n = len(y_true)
    return scores[np.arange(n), y_true]


def true_class_conformity_scores(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    raps_lambda: float = 0.1,
    raps_kreg: int = 1,
) -> np.ndarray:
    """Conformity score of the true class only, shape (N,).

    Args:
        raps_lambda: Penalty weight for score_type="raps" (default 0.1,
            validated non-negative and finite; ignored for other score
            types).
        raps_kreg: Rank at which penalizing starts for score_type="raps"
            (default 1, validated a non-negative integer; ignored for
            other score types).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import true_class_conformity_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> y_true = np.array([0, 1])
        >>> true_class_conformity_scores(y_prob, y_true, score_type="threshold")
        array([0.7, 0.5])
    """
    return 1.0 - true_class_nc_scores(
        y_prob, y_true, score_type, rng, randomize, raps_lambda, raps_kreg
    )
