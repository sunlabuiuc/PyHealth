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

    - "saps" (Sorted Adaptive Prediction Sets, Huang, Xi, Zhang, Yao, Qiu,
      and Wei 2024, "Conformal Prediction for Deep Classifier via Label
      Ranking", Eq. 10): discards every probability value except the
      maximum softmax probability, keeping only the label ranking. With
      pi_max(x) the row's maximum probability, o(x, k) the 1-indexed rank
      of class k in descending probability order, U ~ Uniform(0,1) drawn
      once per example, and lambda >= 0 the ranking-weight hyperparameter::

          S(x, k) = U * pi_max(x)                      if o(x, k) == 1
          S(x, k) = pi_max(x) + (o(x, k) - 2 + U) * lambda   otherwise

      Like APS, the scores are monotone in rank, so prediction sets are
      nested prefixes of the probability-sorted classes. lambda is tuned
      on a validation split in the paper (grid {0.02, ..., 0.6}); here it
      defaults to 0.1 as a convenience default and is settable via the
      ``saps_lambda`` argument of the score functions (and of the
      predictors that accept ``score_type``).

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

SUPPORTED_SCORE_TYPES = ("threshold", "aps", "saps")


def _validate_score_type(score_type: str) -> None:
    if score_type not in SUPPORTED_SCORE_TYPES:
        raise ValueError(
            f"Unknown score_type: {score_type!r}. Supported: "
            f"{SUPPORTED_SCORE_TYPES}."
        )


def _validate_saps_lambda(saps_lambda) -> float:
    """Validate the SAPS ranking-weight hyperparameter.

    Only called when score_type="saps" is selected, so other score types
    never fail on ``saps_lambda``.
    """
    if isinstance(saps_lambda, bool) or not isinstance(
        saps_lambda, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "saps_lambda must be a non-negative float, "
            f"got {saps_lambda!r}."
        )
    saps_lambda = float(saps_lambda)
    if not np.isfinite(saps_lambda) or saps_lambda < 0:
        raise ValueError(
            "saps_lambda must be a non-negative finite float, "
            f"got {saps_lambda!r}."
        )
    return saps_lambda


def _aps_all_class_nc_scores(
    y_prob: np.ndarray,
    rng: np.random.Generator,
    randomize: bool,
) -> np.ndarray:
    """Computes the APS nonconformity score for every class, every row.

    E(x, k) = [sum of predicted probabilities for classes ranked strictly
    above k] + U * p(x, k), with U ~ Uniform(0,1) drawn once per row and
    reused across all classes in that row (Angelopoulos et al. 2021,
    Algorithm 2/3, with the regularization term lambda=0, i.e. plain APS
    rather than RAPS).

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
        Nonconformity scores of shape (N, K); higher means less conforming.
    """
    n = y_prob.shape[0]
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
    return scores


def _saps_all_class_nc_scores(
    y_prob: np.ndarray,
    rng: np.random.Generator,
    randomize: bool,
    saps_lambda: float,
) -> np.ndarray:
    """Computes the SAPS nonconformity score for every class, every row.

    S(x, k) = U * pi_max(x) if rank(x, k) == 1, else
              pi_max(x) + (rank(x, k) - 2 + U) * lambda,
    with pi_max(x) the row's maximum probability, rank the 1-indexed rank
    in descending probability order, and U ~ Uniform(0,1) drawn once per
    row and reused across all classes in that row (Huang et al. 2024,
    Eq. 10). Only pi_max is used; every other probability value is
    discarded, so rows with the same max and rank structure score
    identically regardless of their tail probabilities.

    Args:
        y_prob: Predicted probabilities, shape (N, K).
        rng: Random generator used for the infinitesimal tie-breaking of
            the rank sort (same convention as the APS implementation
            above) and, when randomize=True, the per-row U draws.
        randomize: If True, draws a genuine U ~ Uniform(0,1) per row. If
            False, uses U=1 for every row (the conservative,
            non-randomized variant).
        saps_lambda: Ranking-weight hyperparameter, validated
            non-negative and finite by the caller.

    Returns:
        Nonconformity scores of shape (N, K); higher means less conforming.
        Scores are non-decreasing in rank, so thresholded prediction sets
        are nested prefixes of the probability-sorted classes.
    """
    n, k = y_prob.shape
    # Same tie-breaking convention as APS: perturb the sort key
    # infinitesimally so tied probabilities get a random rank order.
    tie_break = rng.uniform(0.0, 1e-12, size=y_prob.shape)
    order = np.argsort(-(y_prob + tie_break), axis=1)  # descending, per row
    # 1-indexed ranks in sorted order, broadcast over rows.
    ranks_sorted = np.arange(1, k + 1)
    pi_max = y_prob.max(axis=1, keepdims=True)

    if randomize:
        u = rng.uniform(0.0, 1.0, size=(n, 1))
    else:
        u = np.ones((n, 1))

    sorted_scores = np.where(
        ranks_sorted == 1,
        u * pi_max,
        pi_max + (ranks_sorted - 2 + u) * saps_lambda,
    )

    # Undo the sort to get back to original class-index order.
    inverse_order = np.argsort(order, axis=1)
    scores = np.take_along_axis(sorted_scores, inverse_order, axis=1)
    return scores


def all_class_nc_scores(
    y_prob: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    saps_lambda: float = 0.1,
) -> np.ndarray:
    """Nonconformity score (higher = less conforming) for every class.

    Args:
        y_prob: Predicted probabilities, shape (N, K).
        score_type: "threshold" (Sadinle, Lei, and Wasserman 2019), "aps"
            (Romano, Sesia, and Candes 2020), or "saps" (Huang et al. 2024).
            Default "threshold".
        rng: Random generator, required (and only used) if score_type is
            "aps" or "saps" and randomize=True.
        randomize: Whether to use the randomized ("exact coverage") variant
            of APS/SAPS. Ignored for score_type="threshold".
        saps_lambda: Ranking-weight hyperparameter for score_type="saps"
            (default 0.1, a convenience default; the paper tunes it on a
            validation split). Ignored for other score types.

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
        ...     np.array([[0.7, 0.2, 0.1]]), score_type="saps",
        ...     randomize=False, saps_lambda=0.1)
        array([[0.7, 0.8, 0.9]])
    """
    _validate_score_type(score_type)
    if score_type == "threshold":
        return 1.0 - y_prob
    if rng is None:
        rng = np.random.default_rng()
    if score_type == "aps":
        return _aps_all_class_nc_scores(y_prob, rng, randomize)
    # score_type == "saps"
    return _saps_all_class_nc_scores(
        y_prob, rng, randomize, _validate_saps_lambda(saps_lambda)
    )


def all_class_conformity_scores(
    y_prob: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    saps_lambda: float = 0.1,
) -> np.ndarray:
    """Conformity score (higher = more conforming) for every class.

    Equivalent to ``1 - all_class_nc_scores(...)``: same ranking of
    examples, just the sign convention used by CovariateLabel and
    NeighborhoodLabel (which threshold with ``score >= t`` rather than
    ``nc_score <= t``).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import all_class_conformity_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> all_class_conformity_scores(y_prob, score_type="threshold")
        array([[0.7, 0.2, 0.1],
               [0.3, 0.5, 0.2]])
    """
    return 1.0 - all_class_nc_scores(
        y_prob, score_type, rng, randomize, saps_lambda
    )


def true_class_nc_scores(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    saps_lambda: float = 0.1,
) -> np.ndarray:
    """Nonconformity score of the true class only, shape (N,). Used during
    calibration, where only the true label's score is needed.

    Args:
        saps_lambda: Ranking-weight hyperparameter for score_type="saps"
            (default 0.1, validated non-negative and finite; ignored for
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
        y_prob, score_type, rng, randomize, saps_lambda
    )
    n = len(y_true)
    return scores[np.arange(n), y_true]


def true_class_conformity_scores(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
    saps_lambda: float = 0.1,
) -> np.ndarray:
    """Conformity score of the true class only, shape (N,).

    Args:
        saps_lambda: Ranking-weight hyperparameter for score_type="saps"
            (default 0.1, validated non-negative and finite; ignored for
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
        y_prob, y_true, score_type, rng, randomize, saps_lambda
    )
