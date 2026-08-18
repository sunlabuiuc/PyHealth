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

Both scores are computed here in *nonconformity* convention (higher = less
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

SUPPORTED_SCORE_TYPES = ("threshold", "aps")


def _validate_score_type(score_type: str) -> None:
    if score_type not in SUPPORTED_SCORE_TYPES:
        raise ValueError(
            f"Unknown score_type: {score_type!r}. Supported: "
            f"{SUPPORTED_SCORE_TYPES}."
        )


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


def all_class_nc_scores(
    y_prob: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
) -> np.ndarray:
    """Nonconformity score (higher = less conforming) for every class.

    Args:
        y_prob: Predicted probabilities, shape (N, K).
        score_type: "threshold" (Sadinle, Lei, and Wasserman 2019) or "aps"
            (Romano, Sesia, and Candes 2020). Default "threshold".
        rng: Random generator, required (and only used) if score_type="aps"
            and randomize=True.
        randomize: Whether to use the randomized ("exact coverage") variant
            of APS. Ignored for score_type="threshold".

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
    """
    _validate_score_type(score_type)
    if score_type == "threshold":
        return 1.0 - y_prob
    # score_type == "aps"
    if rng is None:
        rng = np.random.default_rng()
    return _aps_all_class_nc_scores(y_prob, rng, randomize)


def all_class_conformity_scores(
    y_prob: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
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
    return 1.0 - all_class_nc_scores(y_prob, score_type, rng, randomize)


def true_class_nc_scores(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
) -> np.ndarray:
    """Nonconformity score of the true class only, shape (N,). Used during
    calibration, where only the true label's score is needed.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import true_class_nc_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> y_true = np.array([0, 1])
        >>> true_class_nc_scores(y_prob, y_true, score_type="threshold")
        array([0.3, 0.5])
    """
    scores = all_class_nc_scores(y_prob, score_type, rng, randomize)
    n = len(y_true)
    return scores[np.arange(n), y_true]


def true_class_conformity_scores(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    score_type: str = "threshold",
    rng: np.random.Generator | None = None,
    randomize: bool = True,
) -> np.ndarray:
    """Conformity score of the true class only, shape (N,).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.calib.predictionset.scores import true_class_conformity_scores
        >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> y_true = np.array([0, 1])
        >>> true_class_conformity_scores(y_prob, y_true, score_type="threshold")
        array([0.7, 0.5])
    """
    return 1.0 - true_class_nc_scores(y_prob, y_true, score_type, rng, randomize)
