"""Tests authors'-code-aligned reference metrics against slow direct formulas."""
import numpy as np
import pytest

from examples.neuralfg.metrics import CensoringKM, brier_score_ipcw, time_dependent_cindex
from examples.neuralfg.metrics_reference import reference_cindex, reference_brier


def _oracle_cindex(t, e, p, risk, horizon, km, tol=1e-8):
    numer = denom = 0.0
    g = np.maximum(km.survival(t), 1e-4)
    for i in range(len(t)):
        if e[i] != risk or t[i] > horizon:
            continue
        for j in range(len(t)):
            if t[j] > t[i]:
                w = 1.0 / g[i] ** 2
            elif t[j] <= t[i] and e[j] > 0 and e[j] != risk:
                w = 1.0 / (g[i] * g[j])
            else:
                continue
            concord = 0.5 if abs(p[i] - p[j]) <= tol else float(p[i] > p[j])
            numer += w * concord
            denom += w
    return numer / denom if denom else float('nan')


def _oracle_brier(t, e, p, risk, horizon, km):
    g = np.maximum(km.survival(t), 1e-4)
    gh = max(float(km.survival(horizon)), 1e-4)
    tot = 0.
    for i in range(len(t)):
        if t[i] > horizon:
            tot += p[i] ** 2 / gh
        elif e[i] != 0:
            tot += (float(e[i] == risk) - p[i]) ** 2 / g[i]
    return tot / len(t)


def test_competing_events_before_cases_are_comparable():
    t = np.array([1., 2., 3., 4.])
    e = np.array([2, 1, 0, 2])
    p = np.array([0.1, 0.8, 0.4, 0.3])
    km = CensoringKM.fit(t, e)
    expected = _oracle_cindex(t, e, p, risk=1, horizon=2, km=km)
    got = reference_cindex(t, e, p, risk=1, horizon=2, censoring_km=km)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(1.)


def test_tie_time_competing_control():
    t = np.array([1., 1., 2.])
    e = np.array([1, 2, 0])
    p = np.array([0.6, 0.9, 0.2])
    km = CensoringKM.fit(t, e)
    got = reference_cindex(t, e, p, risk=1, horizon=1, censoring_km=km)
    assert got == pytest.approx(0.5)
    original = time_dependent_cindex(t, e, p, risk=1, horizon=1, censoring_km=km)
    assert original == pytest.approx(1.)


def test_brier_includes_competing_events_and_right_km():
    t = np.array([1., 1., 2., 4., 7.])
    e = np.array([0, 1, 2, 0, 1])
    p = np.array([0.8, 0.8, 0.2, 0.1, 0.3])
    km = CensoringKM.fit(t, e)
    assert reference_brier(t, e, p, risk=1, horizon=3, censoring_km=km) == pytest.approx(
        _oracle_brier(t,e,p,1,3,km)
    )
    b_old = brier_score_ipcw(t, e, p, risk=1, horizon=3, censoring_km=km,
                             include_competing_events=True)
    assert reference_brier(t, e, p, risk=1, horizon=3, censoring_km=km) != pytest.approx(b_old)


def test_random_concordance_matches_quadratic_oracle():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        train_t = rng.integers(1, 15, size=30).astype(float)
        train_e = rng.choice([0, 1, 2], size=30, p=[0.3,0.4,0.3])
        km = CensoringKM.fit(train_t, train_e)
        t = rng.integers(1, 12, size=40).astype(float)
        e = rng.choice([0, 1, 2], size=40, p=[0.3,0.4,0.3])
        p = rng.choice([0.2,0.3,0.5,0.7,0.9], size=40)
        for risk in [1, 2]:
            for horizon in [2., 6., 11.]:
                got = reference_cindex(t,e,p,risk=risk,horizon=horizon,censoring_km=km)
                exp = _oracle_cindex(t,e,p,risk,horizon,km)
                if np.isnan(exp):
                    assert np.isnan(got)
                else:
                    assert got == pytest.approx(exp, abs=1e-10)
                b = reference_brier(t,e,p,risk=risk,horizon=horizon,censoring_km=km)
                if not np.any((t <= horizon) & (e == risk)):
                    assert np.isnan(b)
                else:
                    assert b == pytest.approx(_oracle_brier(t,e,p,risk,horizon,km), abs=1e-10)
