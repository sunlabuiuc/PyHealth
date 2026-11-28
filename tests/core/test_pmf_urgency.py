"""Synthetic regression tests for the PMF urgency extension."""

import numpy as np
import pytest

from examples.mortality_prediction.pmf_urgency import pmf_urgency_score


def test_equal_risk_earlier_event_has_higher_urgency():
    pmf = np.zeros((3, 24))
    pmf[0, 0] = pmf[1, 11] = pmf[2, 23] = 0.5
    scores = pmf_urgency_score(pmf)
    np.testing.assert_allclose(scores, [0.5, 0.25 * (1 - 11 / 23), 0.0])
    assert scores[0] > scores[1] > scores[2]


def test_mixed_event_time_matches_hand_calculation():
    pmf = np.zeros((1, 24))
    pmf[0, 2] = 0.2
    pmf[0, 11] = 0.4
    expected = 0.6 * (0.5 * (0.2 / 0.6) + 0.5 * (1 - (9 - 1) / 23))
    np.testing.assert_allclose(pmf_urgency_score(pmf), [expected])


def test_zero_and_negligible_risk():
    pmf = np.zeros((2, 24))
    pmf[1, 0] = 1e-8
    np.testing.assert_array_equal(pmf_urgency_score(pmf), [0.0, 0.0])


def test_alpha_and_near_horizon_configuration():
    pmf = np.zeros((1, 24))
    pmf[0, 5] = 0.4
    np.testing.assert_allclose(pmf_urgency_score(pmf, alpha=1.0), [0.4])
    np.testing.assert_allclose(
        pmf_urgency_score(pmf, near_horizon=5, alpha=1.0), [0.0]
    )
    np.testing.assert_allclose(
        pmf_urgency_score(pmf, alpha=0.0), [0.4 * (1 - 5 / 23)]
    )


def test_score_bounded_by_total_event_risk():
    pmf = np.random.default_rng(42).dirichlet(np.ones(25), size=10)[:, :24]
    scores = pmf_urgency_score(pmf)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= pmf.sum(axis=1) + 1e-12)


def test_reject_invalid_pmf_and_parameters():
    with pytest.raises(ValueError):
        pmf_urgency_score(np.zeros((1, 23)))
    with pytest.raises(ValueError):
        pmf_urgency_score(np.full((1, 24), -0.1))
    with pytest.raises(ValueError):
        pmf_urgency_score(np.full((1, 24), np.nan))
    with pytest.raises(ValueError):
        pmf_urgency_score(np.ones((1, 24)))
    with pytest.raises(ValueError):
        pmf_urgency_score(np.zeros((1, 24)), alpha=1.1)
    with pytest.raises(ValueError):
        pmf_urgency_score(np.zeros((1, 24)), near_horizon=0)
