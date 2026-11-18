"""Fast, deterministic tests for examples/neuralfg/metrics.py."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from examples.neuralfg.metrics import (
    CensoringKM,
    brier_score_ipcw,
    evaluate_neuralfg,
    time_dependent_cindex,
)


def test_km_censoring_left_and_right_limits():
    km = CensoringKM.fit([1, 1, 2, 4], [0, 1, 2, 0])
    assert km.survival(1.0, left_limit=True) == pytest.approx(1.0)
    assert km.survival(1.0) == pytest.approx(0.75)
    assert km.survival(2.0) == pytest.approx(0.75)


def test_brier_paper_vs_competing_known_negative():
    times = [1, 3, 5, 6]
    events = [1, 2, 0, 1]
    predicted = [0.8, 0.2, 0.3, 0.1]
    km = CensoringKM.fit(times, events)
    score_paper = brier_score_ipcw(
        times, events, predicted, risk=1, horizon=4, censoring_km=km,
    )
    score_competing = brier_score_ipcw(
        times, events, predicted, risk=1, horizon=4, censoring_km=km,
        include_competing_events=True,
    )
    assert score_paper == pytest.approx(0.035)
    assert score_competing == pytest.approx(0.045)


def test_concordance_ignores_tied_observation_times():
    times = [1, 1, 2]
    events = [1, 2, 0]
    predicted = [0.7, 0.1, 0.4]
    km = CensoringKM.fit(times, events)
    c = time_dependent_cindex(
        times, events, predicted, risk=1, horizon=1, censoring_km=km,
    )
    assert c == pytest.approx(1.0)


def test_concordance_half_credit_for_tied_predicted_risk():
    km = CensoringKM.fit([1, 2], [1, 2])
    c = time_dependent_cindex(
        [1, 2], [1, 2], [0.5, 0.5], risk=1, horizon=1,
        censoring_km=km,
    )
    assert c == pytest.approx(0.5)


def test_concordance_nan_without_comparable_pairs():
    km = CensoringKM.fit([2, 3], [2, 1])
    c = time_dependent_cindex(
        [2, 3], [2, 1], [0.2, 0.8], risk=1, horizon=1,
        censoring_km=km,
    )
    assert math.isnan(c)


def test_censoring_before_horizon_is_not_known_negative():
    km = CensoringKM.fit([1, 3, 10, 12], [0, 1, 1, 2])
    val = brier_score_ipcw(
        [1, 3, 10, 12], [0, 1, 1, 2], [0.9, 0.8, 0.3, 0.4],
        risk=1, horizon=5, censoring_km=km,
        include_competing_events=True,
    )
    # At time 5: patient 1 censored (no contribution), patient 2 a
    # positive, and patients 3/4 known negatives; weights reflect G.
    assert val == pytest.approx(((1 - 0.8)**2 / .75 + (.3**2 + .4**2) / .75) / 4)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.num_risks = 2
        self.register_buffer("time_grid", torch.tensor([1.0, 3.0]))

    @torch.no_grad()
    def predict_cif(self, *, features, times):
        risk_1 = (0.2 + 0.1 * torch.sigmoid(features[:, :1])) * (
            1 - torch.exp(-times[None, :] / 10)
        )
        risk_2 = (0.25 + 0.05 * torch.sigmoid(features[:, :1])) * (
            1 - torch.exp(-times[None, :] / 10)
        )
        return torch.stack([risk_1, risk_2], dim=-1)


def test_evaluate_model_shapes_and_disjoint_split():
    train = pd.DataFrame({
        "time": [1, 2, 3, 4, 5, 6],
        "label": [1, 0, 2, 1, 2, 0],
        "feature1": [0, 1, 2, 3, 4, 5],
    })
    test = pd.DataFrame({
        "time": [1, 2, 3, 4, 5, 6],
        "label": [1, 2, 0, 1, 2, 1],
        "feature1": [0, 1, 2, 3, 4, 5],
    }, index=range(6, 12))
    result = evaluate_neuralfg(_ToyModel(), train, test, batch_size=2)
    assert result.shape[0] == 4
    assert set(result.risk) == {1, 2}
    assert set(result.time) == {1.0, 3.0}
    assert set(result.columns) == {
        "risk", "time", "c_index", "brier_paper", "brier_competing",
        "mean_predicted_cif", "n_test",
    }
    with pytest.raises(ValueError, match="overlap"):
        evaluate_neuralfg(_ToyModel(), train, train)
