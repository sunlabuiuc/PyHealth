"""Plausibility metrics: do token explanations match human evidence spans?

Contributors: Juri Moon (jmoon326)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958

Each explanation is one (note, code) pair: a vector of token scores that
sums to 1, and the indices of the tokens a human coder marked as evidence
for that code (for example from MDACE, Cheng et al. 2023). Definitions
follow the authors' evaluation code:

* precision / recall / F1: tokens scoring above ``threshold`` are the
  predicted evidence; counts are pooled over all pairs (micro average).
* empty: share of pairs where no token scores above ``threshold``.
* recall@k: evidence tokens found among each pair's top-k tokens, divided by
  all evidence tokens.

``threshold`` is chosen on a validation split with
:func:`best_plausibility_threshold`.
"""

from typing import Sequence

import numpy as np


def spans_to_token_ids(
    offsets: Sequence[Sequence[int]], spans: Sequence[Sequence[int]]
) -> list[int]:
    """Indices of tokens that overlap any character span.

    Args:
        offsets: ``(start, end)`` character offsets per token, as returned by
            a fast tokenizer with ``return_offsets_mapping=True``. Special
            tokens have ``(0, 0)`` and never match.
        spans: ``(start, end)`` character spans of the evidence (end
            exclusive).

    Returns:
        Sorted token indices. Spans past the truncated text give no tokens.

    Example:
        >>> spans_to_token_ids([(0, 0), (0, 5), (6, 10), (0, 0)], [(6, 8)])
        [2]
    """
    hits = set()
    for start, end in spans:
        for index, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > tok_start and tok_start < end and tok_end > start:
                hits.add(index)
    return sorted(hits)


def plausibility_metrics(
    attributions: Sequence[np.ndarray],
    evidence: Sequence[Sequence[int]],
    threshold: float,
    ks: Sequence[int] = (5,),
) -> dict[str, float]:
    """Token-level agreement between explanations and evidence.

    Args:
        attributions: One score vector per (note, code) pair.
        evidence: Evidence token indices for each pair.
        threshold: Tokens scoring strictly above it are predicted evidence.
        ks: Cut-offs for ``recall@k``.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``empty`` and one
        ``recall@k`` per ``k``.

    Raises:
        ValueError: If the two inputs differ in length or are empty.

    Example:
        >>> scores = [np.array([0.7, 0.2, 0.1])]
        >>> plausibility_metrics(scores, [[0]], threshold=0.5)["f1"]
        1.0
    """
    if len(attributions) != len(evidence) or not attributions:
        raise ValueError("Need the same, non-zero number of explanations and evidence.")
    tp = fp = fn = empty = 0
    found_at = {k: 0 for k in ks}
    for scores, gold in zip(attributions, evidence):
        scores = np.asarray(scores)
        gold = set(gold)
        predicted = set(np.flatnonzero(scores > threshold).tolist())
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        empty += not predicted
        order = np.argsort(-scores, kind="stable")
        for k in ks:
            found_at[k] += len(gold & set(order[:k].tolist()))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_gold = sum(len(set(g)) for g in evidence)
    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "empty": empty / len(attributions),
    }
    for k in ks:
        result[f"recall@{k}"] = found_at[k] / total_gold if total_gold else 0.0
    return result


def best_plausibility_threshold(
    attributions: Sequence[np.ndarray],
    evidence: Sequence[Sequence[int]],
    grid: Sequence[float] = tuple(np.linspace(0, 0.1, 101)),
) -> float:
    """Threshold from ``grid`` with the highest token F1 (paper: 0 to 0.1).

    Args:
        attributions: Validation explanations.
        evidence: Validation evidence token indices.
        grid: Candidate thresholds.

    Returns:
        The best threshold; the smallest one on ties.
    """
    f1s = [plausibility_metrics(attributions, evidence, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])
