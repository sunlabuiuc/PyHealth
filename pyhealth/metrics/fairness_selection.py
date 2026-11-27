"""Fairness-aware model selection from MEDFAIR (Zong et al., ICLR 2023).

Two alternatives to picking the candidate with the best overall AUC:

    - distance to optimal (DTO): normalise each subgroup's score by the best
      value any candidate reached for it (the utopia point) and pick the
      candidate closest to it;
    - minimax Pareto: among the Pareto-optimal candidates, pick the one with
      the best worst-group score.

Each function takes per-subgroup validation scores, one row per candidate
and one column per subgroup, higher is better, such as the ``"auc_{group}"``
values of :func:`~pyhealth.metrics.group_fairness.group_fairness_metrics_fn`.
"""

import numpy as np


def _check(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError(
            f"scores must be a non-empty (n_candidates, n_groups) matrix, "
            f"got shape {scores.shape}."
        )
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN; drop or impute those groups first.")
    return scores


def dto_select(scores: np.ndarray) -> int:
    """Index of the candidate closest to the utopia point.

    ``DTO_i = || 1 - scores_i / max_j scores_j ||_2``, the utopia point being
    the best score of each subgroup over all candidates.

    Args:
        scores: Per-subgroup scores of shape (n_candidates, n_groups).

    Returns:
        Row index of the candidate with the smallest DTO.

    Examples:
        >>> dto_select(np.array([[0.80, 0.70], [0.78, 0.76], [0.85, 0.60]]))
        1
    """
    scores = _check(scores)
    return int(np.argmin(np.linalg.norm(1.0 - scores / scores.max(axis=0), axis=1)))


def pareto_front(scores: np.ndarray) -> np.ndarray:
    """Indices of the candidates no other candidate dominates.

    Candidate ``j`` dominates ``i`` if it scores at least as well on every
    subgroup and strictly better on one. Identical candidates are all kept.
    The comparison is all-pairs, which suits a hyper-parameter grid rather
    than thousands of candidates.

    Args:
        scores: Per-subgroup scores of shape (n_candidates, n_groups).

    Returns:
        Sorted row indices of the Pareto front.

    Examples:
        >>> pareto_front(np.array([[0.8, 0.7], [0.75, 0.65], [0.6, 0.9]]))
        array([0, 2])
    """
    scores = _check(scores)
    geq = (scores[:, None, :] >= scores[None, :, :]).all(axis=2)
    gt = (scores[:, None, :] > scores[None, :, :]).any(axis=2)
    dominated = (geq & gt).any(axis=0)
    return np.flatnonzero(~dominated)


def minimax_pareto_select(scores: np.ndarray, group: int | None = None) -> int:
    """Pareto-optimal candidate with the best worst-group score.

    Args:
        scores: Per-subgroup scores of shape (n_candidates, n_groups).
        group: Column of a fixed reference group to maximise instead of each
            candidate's own minimum. MEDFAIR fixes it to the group with the
            lowest best AUC under ERM, so every method is judged on the same
            group. Defaults to None.

    Returns:
        Row index of the selected candidate.

    Examples:
        >>> scores = np.array([[0.80, 0.70], [0.78, 0.76], [0.85, 0.60]])
        >>> minimax_pareto_select(scores), minimax_pareto_select(scores, group=0)
        (1, 2)
    """
    scores = _check(scores)
    if group is not None and not 0 <= group < scores.shape[1]:
        raise ValueError(f"group must be in [0, {scores.shape[1] - 1}], got {group}.")
    front = pareto_front(scores)
    target = scores[front].min(axis=1) if group is None else scores[front, group]
    return int(front[np.argmax(target)])
