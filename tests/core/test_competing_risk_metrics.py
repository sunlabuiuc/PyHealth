"""Tests for competing-risk calibration and discrimination metrics."""

import math

import numpy as np
import pytest

from pyhealth.metrics import (
    competing_risk_brier_score,
    competing_risk_concordance_index,
)


def test_competing_brier_perfect_predictions_without_censoring():
    duration = [1.0, 2.0, 3.0, 4.0]
    event = [1, 2, 1, 0]
    prediction = [1.0, 0.0, 0.0, 0.0]
    assert competing_risk_brier_score(duration, event, prediction, 2.5) == 0.0


def test_competing_brier_excludes_early_censoring():
    duration = [0.5, 1.0, 3.0, 4.0]
    event = [0, 1, 2, 0]
    perfect = [0.9, 1.0, 0.0, 0.0]
    score = competing_risk_brier_score(duration, event, perfect, 2.0)
    assert score == pytest.approx(0.0)


def test_competing_concordance_perfect_reverse_and_ties():
    duration = [1.0, 2.0, 3.0, 4.0]
    event = [1, 1, 2, 0]
    assert competing_risk_concordance_index(
        duration, event, [0.9, 0.7, 0.2, 0.1], 3.0
    ) == pytest.approx(1.0)
    assert competing_risk_concordance_index(
        duration, event, [0.1, 0.2, 0.7, 0.9], 3.0
    ) == pytest.approx(0.0)
    assert competing_risk_concordance_index(
        duration, event, [0.5, 0.5, 0.5, 0.5], 3.0
    ) == pytest.approx(0.5)


def test_competing_concordance_nan_without_comparable_pairs():
    score = competing_risk_concordance_index([1, 2], [0, 2], [0.2, 0.1], 2)
    assert math.isnan(score)


@pytest.mark.parametrize(
    "duration,event,prediction",
    [
        ([1, 2], [0], [0.1, 0.2]),
        ([1, -2], [0, 1], [0.1, 0.2]),
        ([1, 2], [0, 1.5], [0.1, 0.2]),
        ([1, 2], [0, 1], [0.1, np.nan]),
        ([1, 2], [0, 1], [0.1, 1.2]),
    ],
)
def test_invalid_competing_metric_inputs(duration, event, prediction):
    with pytest.raises(ValueError):
        competing_risk_brier_score(duration, event, prediction, 1.5)
    with pytest.raises(ValueError):
        competing_risk_concordance_index(duration, event, prediction, 1.5)
