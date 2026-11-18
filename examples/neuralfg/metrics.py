"""Competing-risk evaluation for the DeepHit / NeuralFG example.

Implements the time-horizon cause-specific concordance and IPCW Brier scores
specified in Appendix A.2 of Jeanselme et al., Neural Fine-Gray (2023).

The paper's printed Brier formula omits competing-event subjects observed
before the horizon. We report that formula as ``brier_paper`` and a standard
competing-risk CIF Brier variant (those subjects are known negatives) as
``brier_competing``. Do not interchange these values when comparing papers.

All censoring weights are estimated ONLY on the training split.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class CensoringKM:
    """Kaplan-Meier estimate G(t) = P(C > t), using 0 as censoring label."""

    times: np.ndarray
    survival_after: np.ndarray

    @classmethod
    def fit(cls, time, event) -> "CensoringKM":
        time = np.asarray(time, dtype=float).reshape(-1)
        event = np.asarray(event).reshape(-1)
        _validate_observations(time, event)

        unique_times, inverse = np.unique(time, return_inverse=True)
        at_time = np.bincount(inverse, minlength=len(unique_times))
        censored_at_time = np.bincount(
            inverse, weights=(event == 0).astype(int),
            minlength=len(unique_times),
        )
        at_risk = len(time) - np.r_[0, np.cumsum(at_time[:-1])]
        survival_after = np.cumprod(1.0 - censored_at_time / at_risk)
        return cls(unique_times, survival_after)

    def survival(self, query, *, left_limit: bool = False) -> np.ndarray:
        """G(t) normally; G(t-) for event-time IPCW weights.

        Left-limit G(t-) avoids penalizing a true event for censoring
        events recorded at the same discrete time.
        """
        query = np.asarray(query, dtype=float)
        side = "left" if left_limit else "right"
        indices = np.searchsorted(self.times, query, side=side) - 1
        result = np.ones_like(query, dtype=float)
        valid = indices >= 0
        result[valid] = self.survival_after[indices[valid]]
        return result


def _validate_observations(time: np.ndarray, event: np.ndarray) -> None:
    if time.ndim != 1 or event.ndim != 1 or time.size != event.size:
        raise ValueError("time and event must be matching 1-D arrays")
    if time.size == 0 or not np.all(np.isfinite(time)) or np.any(time < 0):
        raise ValueError("time must contain nonnegative finite observations")
    if not np.all(np.isfinite(event)) or np.any(event != np.floor(event)):
        raise ValueError("event codes must be nonnegative integers")
    if np.any(event < 0):
        raise ValueError("event codes must be nonnegative integers")


def _inverse_survival(km: CensoringKM, query, *, left_limit=False):
    g = km.survival(query, left_limit=left_limit)
    if np.any(g <= 1e-8):
        raise ValueError(
            "Censoring KM survival is near zero at an evaluation/event time; "
            "restrict the time horizon or reconsider the censoring estimate."
        )
    return 1.0 / g


class _Fenwick:
    """Prefix counts supporting O(n log n) concordance calculations."""

    def __init__(self, n: int):
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, position: int) -> None:
        while position < len(self.tree):
            self.tree[position] += 1
            position += position & -position

    def prefix(self, count: int) -> int:
        total = 0
        while count:
            total += int(self.tree[count])
            count -= count & -count
        return total


def time_dependent_cindex(
    time,
    event,
    predicted_cif,
    *,
    risk: int,
    horizon: float,
    censoring_km: CensoringKM,
    tied_tol: float = 1e-8,
) -> float:
    """Paper-style cause-specific, IPCW time-horizon concordance.

    Comparable pairs: i has event ``risk`` at t_i <= horizon; j has
    observed follow-up time t_j > t_i. At the horizon, a concordant pair
    has predicted CIF_i > predicted CIF_j. Tied scores earn 0.5.
    Each case is weighted by 1/G(t_i-); G is fit on training data.
    Returns NaN if no comparable pairs exist.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1)
    score = np.asarray(predicted_cif, dtype=float).reshape(-1)
    _validate_observations(time, event)
    if len(score) != len(time) or not np.all(np.isfinite(score)):
        raise ValueError("predicted_cif must be finite and match time length")
    if risk < 1 or horizon < 0 or tied_tol < 0:
        raise ValueError("risk >= 1, horizon >= 0, and tied_tol >= 0 required")

    levels = np.unique(score)
    ranks = np.searchsorted(levels, score) + 1
    tree = _Fenwick(len(levels))
    order = np.argsort(-time, kind="stable")
    num_later = 0
    numerator = 0.0
    denominator = 0.0

    start = 0
    while start < len(time):
        end = start + 1
        current_time = time[order[start]]
        while end < len(time) and time[order[end]] == current_time:
            end += 1
        group = order[start:end]

        # Don't add current-time subjects until comparisons are complete:
        # ties in observed time are NOT comparable here.
        if current_time <= horizon and num_later > 0:
            cases = group[event[group] == risk]
            if len(cases):
                weights = _inverse_survival(
                    censoring_km, time[cases], left_limit=True
                )
                for patient, weight in zip(cases, weights):
                    value = score[patient]
                    # Query by the NUMBER of unique score levels inside the bound.
                    below = np.searchsorted(levels, value - tied_tol, side="left")
                    through = np.searchsorted(levels, value + tied_tol, side="right")
                    n_below = tree.prefix(int(below))
                    n_through = tree.prefix(int(through))
                    numerator += weight * (
                        n_below + 0.5 * (n_through - n_below)
                    )
                    denominator += weight * num_later

        for patient in group:
            tree.add(int(ranks[patient]))
        num_later += len(group)
        start = end

    return float(numerator / denominator) if denominator else float("nan")


def brier_score_ipcw(
    time,
    event,
    predicted_cif,
    *,
    risk: int,
    horizon: float,
    censoring_km: CensoringKM,
    include_competing_events: bool = False,
) -> float:
    """IPCW CIF Brier score at a fixed time.

    ``include_competing_events=False`` matches the displayed formula in
    NeuralFG Appendix A.2 (``brier_paper``). It scores risk-r events by
    horizon as cases and event-free-at-horizon as controls; competing
    events observed earlier contribute zero.

    ``True`` adds earlier competing-event patients as known negatives
    (``brier_competing``), the natural binary CIF Brier target. Patients
    censored on/before the horizon never count as known negatives.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1)
    probability = np.asarray(predicted_cif, dtype=float).reshape(-1)
    _validate_observations(time, event)
    if len(probability) != len(time) or not np.all(np.isfinite(probability)):
        raise ValueError("predicted_cif must be finite and match time length")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("predicted CIF must be within [0,1]")
    if risk < 1 or horizon < 0:
        raise ValueError("risk and horizon must be nonnegative (risk >= 1)")

    positives = (time <= horizon) & (event == risk)
    still_observed = time > horizon
    score_sum = 0.0

    if np.any(positives):
        weights = _inverse_survival(
            censoring_km, time[positives], left_limit=True
        )
        score_sum += np.sum(weights * (1 - probability[positives]) ** 2)

    if np.any(still_observed):
        horizon_weight = float(_inverse_survival(censoring_km, horizon))
        score_sum += horizon_weight * np.sum(probability[still_observed] ** 2)

    if include_competing_events:
        competing = (time <= horizon) & (event > 0) & (event != risk)
        if np.any(competing):
            weights = _inverse_survival(
                censoring_km, time[competing], left_limit=True
            )
            score_sum += np.sum(weights * probability[competing] ** 2)

    return float(score_sum / len(time))


def evaluate_neuralfg(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    batch_size: int = 512,
    evaluation_times=None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate a trained NeuralFG model only on held-out test patients.

    Returns one row per (horizon, cause). No training is performed here.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if "time" not in train_df or "label" not in train_df:
        raise ValueError("train_df requires time and label columns")
    if "time" not in test_df or "label" not in test_df:
        raise ValueError("test_df requires time and label columns")
    if not set(train_df.index).isdisjoint(test_df.index):
        raise ValueError("Training and test rows overlap")

    if feature_columns is None:
        feature_columns = sorted(
            (col for col in train_df if re.fullmatch(r"feature\d+", col)),
            key=lambda name: int(name[7:]),
        )
    if not feature_columns or not set(feature_columns).issubset(test_df.columns):
        raise ValueError("Unable to find matching feature columns")

    time_train = train_df["time"].to_numpy(dtype=float)
    event_train = train_df["label"].to_numpy(dtype=int)
    time_test = test_df["time"].to_numpy(dtype=float)
    event_test = test_df["label"].to_numpy(dtype=int)
    _validate_observations(time_train, event_train)
    _validate_observations(time_test, event_test)
    km = CensoringKM.fit(time_train, event_train)

    if evaluation_times is None:
        evaluation_times = model.time_grid.detach().cpu().numpy()
    times = np.asarray(evaluation_times, dtype=float).reshape(-1)
    if not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("evaluation_times must be nonempty and finite")
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("evaluation_times must be nonnegative and increasing")

    device = next(model.parameters()).device
    features = test_df[feature_columns].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("Test features contain NaN or infinite values")
    times_torch = torch.tensor(times, dtype=torch.float32, device=device)
    predictions = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start:start + batch_size]).to(device)
        with torch.no_grad():
            cif = model.predict_cif(features=values, times=times_torch)
        predictions.append(cif.detach().cpu().numpy())
    predicted = np.concatenate(predictions, axis=0)

    if predicted.shape != (len(test_df), len(times), model.num_risks):
        raise ValueError(f"Unexpected model CIF shape: {predicted.shape}")
    if not np.all(np.isfinite(predicted)) or np.any(predicted < -1e-6):
        raise ValueError("Predicted CIFs contain invalid probabilities")
    if np.any(predicted > 1 + 1e-6):
        raise ValueError("Predicted CIF exceeds one")
    if np.any(predicted.sum(axis=-1) > 1 + 1e-6):
        raise ValueError("Sum of competing-risk CIFs exceeds one")
    if np.any(np.diff(predicted, axis=1) < -1e-6):
        raise ValueError("A predicted CIF decreases with time")
    predicted = np.clip(predicted, 0.0, 1.0)

    results = []
    for time_idx, horizon in enumerate(times):
        for risk in range(1, model.num_risks + 1):
            score = predicted[:, time_idx, risk - 1]
            results.append({
                "risk": risk,
                "time": float(horizon),
                "c_index": time_dependent_cindex(
                    time_test, event_test, score, risk=risk,
                    horizon=float(horizon), censoring_km=km,
                ),
                "brier_paper": brier_score_ipcw(
                    time_test, event_test, score, risk=risk,
                    horizon=float(horizon), censoring_km=km,
                ),
                "brier_competing": brier_score_ipcw(
                    time_test, event_test, score, risk=risk,
                    horizon=float(horizon), censoring_km=km,
                    include_competing_events=True,
                ),
                "mean_predicted_cif": float(score.mean()),
                "n_test": len(test_df),
            })
    return pd.DataFrame(results)
