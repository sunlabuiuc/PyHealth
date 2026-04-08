"""Interpretability metrics for evaluating feature attribution methods."""

from .base import RemovalBasedMetric
from .comprehensiveness import ComprehensivenessMetric
from .evaluator import Evaluator, evaluate_attribution
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
    # Sample classification
    "SampleClass",
    "SampleFilterFn",
    "threshold_sample_filter",
    # Utility functions
    "get_model_predictions",
]
