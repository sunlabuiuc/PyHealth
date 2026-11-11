"""Interpretability metrics for evaluating feature attribution methods."""

from .base import RemovalBasedMetric
from .comprehensiveness import ComprehensivenessMetric
from .evaluator import Evaluator, evaluate_attribution
from .plausibility import (
    best_plausibility_threshold,
    plausibility_metrics,
    spans_to_token_ids,
)
from .sufficiency import SufficiencyMetric
from .utils import (
    SampleClass,
    SampleFilterFn,
    get_model_predictions,
    threshold_sample_filter,
)

__all__ = [
    "ComprehensivenessMetric",
    "SufficiencyMetric",
    "RemovalBasedMetric",
    "Evaluator",
    "evaluate_attribution",
    # Plausibility against human evidence spans
    "plausibility_metrics",
    "best_plausibility_threshold",
    "spans_to_token_ids",
    # Sample classification
    "SampleClass",
    "SampleFilterFn",
    "threshold_sample_filter",
    # Utility functions
    "get_model_predictions",
]
