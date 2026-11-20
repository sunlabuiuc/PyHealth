"""Tests of the NumPy helpers of dynamic survival analysis (no torch)."""

import numpy as np
import pytest

from pyhealth.models.dynamic_survival import (
    hazard_prior,
    risk_from_hazards,
    survival_nll,
    survival_targets,
)


def test_targets_of_an_event_inside_the_horizon():
    y, w = survival_targets([3], [1], horizon=5)
    assert w.tolist() == [[1, 1, 1, 0, 0]]
    assert y.tolist() == [[0, 0, 1, 0, 0]]


def test_targets_of_a_censored_step():
    y, w = survival_targets([2], [0], horizon=5)
    assert w.tolist() == [[1, 1, 0, 0, 0]]
    assert y.sum() == 0


def test_targets_without_observed_hours():
    y, w = survival_targets([0], [0], horizon=3)
    assert w.sum() == 0 and y.sum() == 0


def test_event_beyond_the_horizon_is_truncated():
    y, w = survival_targets([30], [1], horizon=4)
    assert w.tolist() == [[1, 1, 1, 1]]
    assert y.sum() == 0  # no event is seen inside the horizon


def test_event_on_the_last_hour_of_the_horizon():
    y, _ = survival_targets([4], [1], horizon=4)
    assert y.tolist() == [[0, 0, 0, 1]]


def test_hazard_prior_counts_only_observed_hours():
    # hour 1: counted by 3 steps, 1 event; hour 2: 2 steps, 1 event;
    # hour 3: 1 step, no event
    prior = hazard_prior([2, 1, 3], [1, 1, 0], horizon=3, eps=1e-6)
    np.testing.assert_allclose(prior, [1 / 3, 1 / 2, 1e-6])


def test_hazard_prior_fills_unseen_horizons_with_the_mean():
    prior = hazard_prior([1, 1], [1, 0], horizon=3, eps=1e-6)
    np.testing.assert_allclose(prior, [0.5, 0.5, 0.5])


def test_hazard_prior_needs_observed_hours():
    with pytest.raises(ValueError):
        hazard_prior([0, 0], [0, 0], horizon=3)


def test_prior_is_the_base_rate_of_the_cumulative_risk():
    # a constant hazard p over h hours gives the cumulative risk 1 - (1 - p)^h
    p, h = 0.01, 24
    np.testing.assert_allclose(
        risk_from_hazards(np.full(h, p)), 1 - (1 - p) ** h
    )


def test_risk_from_hazards():
    assert float(risk_from_hazards(np.array([0.5, 0.5]))) == 0.75
    batch = risk_from_hazards(np.array([[0.0, 0.0], [1.0, 0.3]]))
    np.testing.assert_allclose(batch, [0.0, 1.0])


def test_survival_nll_single_entry():
    # two counted hours at logit 0: each costs log 2
    assert survival_nll(np.zeros((1, 2)), [2], [1]) == pytest.approx(np.log(2))


def test_survival_nll_ignores_uncounted_hours():
    logits = np.array([[0.0, 50.0, -50.0]])  # hours 2 and 3 are not counted
    assert survival_nll(logits, [1], [0]) == pytest.approx(np.log(2))


def test_survival_nll_event_is_the_positive_class():
    # event in hour 2; a confident hazard there is cheap, elsewhere costly
    good = np.array([[-20.0, 20.0]])
    bad = np.array([[20.0, -20.0]])
    assert survival_nll(good, [2], [1]) < 1e-6
    assert survival_nll(bad, [2], [1]) > 10


def test_survival_nll_without_counted_hours():
    assert survival_nll(np.zeros((2, 3)), [0, 0], [0, 0]) == 0.0
