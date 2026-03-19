"""Metrics for generative text outputs in medical VQA tasks."""

from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Optional


def normalize_text_for_exact_match(text: str) -> str:
    """Normalizes text for strict exact-match comparison.

    The normalization lowercases text, removes punctuation and articles, and
    collapses repeated whitespace.
    """
    normalized = text.lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def exact_match_score(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Computes normalized exact-match accuracy for generated text."""
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true_list:
        return 0.0

    matches = 0
    for truth, pred in zip(y_true_list, y_pred_list):
        if normalize_text_for_exact_match(str(truth)) == normalize_text_for_exact_match(
            str(pred)
        ):
            matches += 1
    return matches / len(y_true_list)


def bertscore_f1_score(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    **kwargs,
) -> float:
    """Computes BERTScore F1 for generated text.

    This metric lazily imports ``bert_score`` and raises an actionable
    ``ImportError`` if the dependency is unavailable.
    """
    y_true_list = [str(item) for item in y_true]
    y_pred_list = [str(item) for item in y_pred]
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true_list:
        return 0.0

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError(
            "bertscore_f1 requested but `bert_score` is not installed. "
            "Install with `pip install bert-score`."
        ) from exc

    _, _, f1 = bert_score(y_pred_list, y_true_list, **kwargs)
    return float(f1.mean().item())


def generative_metrics_fn(
    y_true: List[str],
    y_pred: List[str],
    metrics: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Computes a collection of text-generation metrics.

    Supported metrics:
    - ``exact_match``
    - ``bertscore_f1``
    """
    if metrics is None:
        metrics = ["exact_match"]

    metric_list = [metric.strip().lower() for metric in metrics]
    results: Dict[str, float] = {}

    for metric in metric_list:
        if metric == "exact_match":
            results[metric] = exact_match_score(y_true, y_pred)
        elif metric == "bertscore_f1":
            results[metric] = bertscore_f1_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown generative metric: {metric}")

    return results
