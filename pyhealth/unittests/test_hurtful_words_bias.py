# =============================================================================
# Tests for HurtfulWordsBiasTask
# Author: Ritul Soni (rsoni27)
# Description: Unit tests for log_bias and precision_gap metrics of
# HurtfulWordsBiasTask in PyHealth.
# =============================================================================

import pytest
import numpy as np
from pyhealth.tasks.hurtful_words_bias import HurtfulWordsBiasTask


class DummyModel:
    """
    Dummy model that returns predetermined log-probability scores.
    """
    def __init__(self, scores):
        self.scores = scores
        self.idx = 0

    def get_log_prob(self, text):
        # Return the next score in the list
        val = self.scores[self.idx]
        self.idx += 1
        return val


def test_log_bias_and_precision_gap():
    # Prepare synthetic data
    genders = ["female", "female", "male", "male"]
    # Scores: female->[3.0,1.0], male->[2.0,0.0]
    scores = [3.0, 1.0, 2.0, 0.0]

    data = [{"text": "", "gender": g} for g in genders]
    model = DummyModel(scores)

    task = HurtfulWordsBiasTask(positive_group="female", negative_group="male")
    results = task.evaluate(data, model, metrics=["log_bias", "precision_gap"])

    # log_bias = mean(female)-mean(male) = (3+1)/2 - (2+0)/2 = 2 - 1 = 1
    assert pytest.approx(results["log_bias"], rel=1e-6) == 1.0

    # precision_gap = 1.0 (privileged precision 1.0 vs unprivileged 0.0)
    assert pytest.approx(results["precision_gap"], rel=1e-6) == 1.0


def test_empty_data():
    # Edge case: no data
    data = []
    model = DummyModel([])
    task = HurtfulWordsBiasTask()

    # Should return empty dict or zeros without raising error
    results = task.evaluate(data, model, metrics=["log_bias", "precision_gap"])
    assert isinstance(results, dict)
    assert results.get("log_bias", 0) == 0 or results.get("log_bias") is None
    assert results.get("precision_gap", 0) == 0 or results.get("precision_gap") is None


def test_single_group_data():
    # Edge case: all records belong to positive_group
    genders = ["female", "female"]
    scores = [0.5, 0.7]
    data = [{"text": "", "gender": g} for g in genders]
    model = DummyModel(scores)

    task = HurtfulWordsBiasTask(positive_group="female", negative_group="male")
    results = task.evaluate(data, model, metrics=["precision_gap"])

    # Unprivileged group missing; precision_gap should be computed as difference with zero or None
    # privileged precision = 1.0 (all predicted positive), unprivileged = 0.0
    assert pytest.approx(results["precision_gap"], rel=1e-6) == pytest.approx(1.0)
