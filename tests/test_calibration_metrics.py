# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Tests for CaliForest calibration metrics
"""Tests for CaliForest calibration metrics."""

import numpy as np
import pytest

from pyhealth.metrics.califorest_calibration import (
    hosmer_lemeshow,
    reliability,
    scaled_brier_score,
    spiegelhalter,
)


def _perfect_data():
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([0.0] * 50 + [1.0] * 50)
    return y_true, y_score


def _random_data(seed=42):
    np.random.seed(seed)
    y_true = np.random.randint(0, 2, 200)
    y_score = np.clip(
        y_true + np.random.randn(200) * 0.2, 0.01, 0.99
    )
    return y_true, y_score


class TestScaledBrierScore:
    def test_perfect(self):
        brier, scaled = scaled_brier_score(*_perfect_data())
        assert brier == pytest.approx(0.0, abs=1e-7)
        assert scaled == pytest.approx(1.0, abs=1e-7)

    def test_returns_tuple(self):
        result = scaled_brier_score(*_random_data())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_brier_range(self):
        brier, scaled = scaled_brier_score(*_random_data())
        assert 0 <= brier <= 1
        assert scaled <= 1


class TestHosmerLemeshow:
    def test_returns_float(self):
        p = hosmer_lemeshow(*_random_data())
        assert isinstance(p, float)

    def test_range(self):
        p = hosmer_lemeshow(*_random_data())
        assert 0 <= p <= 1

    def test_well_calibrated_high_pvalue(self):
        y_true, y_score = _random_data()
        p = hosmer_lemeshow(y_true, y_score)
        # Well-calibrated data should have a reasonably high p-value
        assert p > 0.01


class TestSpiegelhalter:
    def test_returns_float(self):
        p = spiegelhalter(*_random_data())
        assert isinstance(p, float)

    def test_range(self):
        p = spiegelhalter(*_random_data())
        assert 0 <= p <= 1


class TestReliability:
    def test_returns_tuple(self):
        result = reliability(*_random_data())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_perfect_calibration(self):
        rel_s, rel_l = reliability(*_perfect_data())
        assert rel_l == pytest.approx(0.0, abs=1e-7)

    def test_non_negative(self):
        rel_s, rel_l = reliability(*_random_data())
        assert rel_s >= 0
        assert rel_l >= 0
