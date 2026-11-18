"""Fine-Gray competing-risk regression for the DeepHit / NeuralFG example.

This is an independent, pure-Python implementation of proportional
subdistribution hazards regression using IPCW Fine-Gray risk sets, a Breslow
approximation for tied events, and L-BFGS optimization.

Fit ONLY on the training split. Evaluate on a disjoint validation/test split.
For an exact benchmark, additionally compare estimates with R cmprsk::crr().

Dependencies: numpy, pandas, scipy, and neuralfg.metrics (the project's
existing evaluation implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import (
    CensoringKM,
    brier_score_ipcw,
    time_dependent_cindex,
)


@dataclass
class _RiskFit:
    cause: int
    coef: np.ndarray
    event_times: np.ndarray
    cumulative_baseline_hazard: np.ndarray
    iterations: int
    objective: float


class FineGrayBaseline:
    """Cause-wise Fine-Gray subdistribution proportional hazards models.

    Event labels: 0=censored, 1..num_risks=competing event causes.
    Input covariates are standardized using training-set moments only.
    Tied target events use the Breslow approximation.

    Separate cause-specific subdistribution fits do not jointly constrain
    their CIF sum to <=1. Never silently renormalize predictions.
    """

    def __init__(
        self,
        num_risks: int = 2,
        feature_columns: list[str] | None = None,
        maxiter: int = 250,
        tolerance: float = 1e-9,
    ) -> None:
        if num_risks < 1 or maxiter < 1 or tolerance <= 0:
            raise ValueError("num_risks and maxiter must be positive")
        self.num_risks = int(num_risks)
        self.feature_columns = feature_columns
        self.maxiter = int(maxiter)
        self.tolerance = float(tolerance)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.models_: dict[int, _RiskFit] = {}

    def _check_dataframe(self, df: pd.DataFrame, *, require_outcomes: bool) -> None:
        needed = list(self.feature_columns or [])
        if require_outcomes:
            needed += ["time", "label"]
        missing = set(needed) - set(df.columns)
        if missing:
            raise ValueError(f"Missing dataframe columns: {sorted(missing)}")
        if not len(df):
            raise ValueError("Dataframe is empty")
        if not np.all(np.isfinite(df[self.feature_columns].to_numpy(dtype=float))):
            raise ValueError("Feature matrix has NaN or infinite values")
        if require_outcomes:
            t = df["time"].to_numpy(dtype=float)
            events = df["label"].to_numpy(dtype=float)
            if (not np.all(np.isfinite(t)) or np.any(t < 0)
                    or not np.all(np.isfinite(events))
                    or np.any(events != np.floor(events))
                    or np.any((events < 0) | (events > self.num_risks))):
                raise ValueError("Invalid times or event labels")

    def fit(self, train_df: pd.DataFrame) -> "FineGrayBaseline":
        """Fit independently for each observed cause using training data."""
        if self.feature_columns is None:
            self.feature_columns = sorted(
                (col for col in train_df if re.fullmatch(r"feature\d+", col)),
                key=lambda name: int(name[7:]),
            )
        if not self.feature_columns:
            raise ValueError("No feature columns were provided or detected")
        self._check_dataframe(train_df, require_outcomes=True)

        raw = train_df[self.feature_columns].to_numpy(dtype=np.float64)
        self.mean_ = raw.mean(axis=0)
        self.scale_ = raw.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        x = (raw - self.mean_) / self.scale_
        t = train_df["time"].to_numpy(dtype=np.float64)
        event = train_df["label"].to_numpy(dtype=np.int64)
        km = CensoringKM.fit(t, event)

        sort_index = np.argsort(t, kind="mergesort")
        t_sorted = t[sort_index]
        x_sorted = x[sort_index]
        event_sorted = event[sort_index]
        n, d = x.shape
        self.models_ = {}

        # G(T_i-) is needed for those having competing events before t.
        g_subject = km.survival(t_sorted, left_limit=True)
        if np.any(g_subject <= 0):
            raise ValueError("Censoring KM vanishes at a training observation time")

        for cause in range(1, self.num_risks + 1):
            cases = event == cause
            if not np.any(cases):
                raise ValueError(f"No observed events for cause {cause}")
            event_times, counts = np.unique(t[cases], return_counts=True)
            counts = counts.astype(np.float64)
            starts = np.searchsorted(t_sorted, event_times, side="left")
            g_at_events = km.survival(event_times, left_limit=True)
            competing = (event_sorted != 0) & (event_sorted != cause)
            other_weights = np.where(competing, 1.0 / g_subject, 0.0)
            event_x_sum = x[cases].sum(axis=0)

            def denominators(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
                """Weighted risk totals and first moments at each cause event time.

                Risk sets contain all T_i >= t plus earlier competing failures,
                the latter weighted by G(t-)/G(T_i-). Suffix/prefix sums
                avoid constructing a huge person-by-event-time matrix.
                """
                eta = x_sorted @ beta
                shift = float(np.max(eta))
                exp_eta = np.exp(eta - shift)

                # Everyone still observed at t (including event tied at t).
                suffix0 = np.r_[np.cumsum(exp_eta[::-1])[::-1], 0.0]
                suffix1 = np.vstack((
                    np.cumsum((exp_eta[:, None] * x_sorted)[::-1], axis=0)[::-1],
                    np.zeros((1, d)),
                ))

                # Earlier competing events retained in the Fine-Gray risk set.
                weighted_other = exp_eta * other_weights
                prefix0 = np.r_[0.0, np.cumsum(weighted_other)]
                prefix1 = np.vstack((
                    np.zeros((1, d)),
                    np.cumsum(weighted_other[:, None] * x_sorted, axis=0),
                ))

                den0 = suffix0[starts] + g_at_events * prefix0[starts]
                den1 = suffix1[starts] + g_at_events[:, None] * prefix1[starts]
                if np.any(den0 <= 0):
                    raise FloatingPointError("Nonpositive Fine-Gray risk denominator")
                return den0, den1, shift

            def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
                den0, den1, shift = denominators(beta)
                loss = (np.dot(counts, np.log(den0) + shift)
                        - np.dot(event_x_sum, beta))
                gradient = ((counts[:, None] * (den1 / den0[:, None])).sum(axis=0)
                            - event_x_sum)
                return float(loss), gradient

            optimization = minimize(
                objective,
                np.zeros(d, dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.maxiter, "ftol": self.tolerance},
            )
            if not optimization.success:
                raise RuntimeError(
                    f"Fine-Gray optimization for cause {cause} did not converge: "
                    f"{optimization.message}; try larger maxiter or inspect features"
                )

            den0, _, shift = denominators(optimization.x)
            # Breslow-type baseline subdistribution hazard increments.
            hazard_increments = counts * np.exp(-shift) / den0
            self.models_[cause] = _RiskFit(
                cause=cause,
                coef=optimization.x.copy(),
                event_times=event_times,
                cumulative_baseline_hazard=np.cumsum(hazard_increments),
                iterations=int(optimization.nit),
                objective=float(optimization.fun),
            )
        return self

    def predict_cif(self, df: pd.DataFrame, evaluation_times) -> np.ndarray:
        """Return CIF array [patients, evaluation_times, risks].

        Each cause is independently fitted, so the sum across causes may
        exceed 1; this is a known limitation of separate Fine-Gray fits.
        """
        if len(self.models_) != self.num_risks or self.mean_ is None:
            raise RuntimeError("Call fit(train_df) before predicting")
        self._check_dataframe(df, require_outcomes=False)
        horizons = np.asarray(evaluation_times, dtype=np.float64).reshape(-1)
        if (not len(horizons) or not np.all(np.isfinite(horizons))
                or np.any(horizons < 0) or np.any(np.diff(horizons) <= 0)):
            raise ValueError("Evaluation times must be nonnegative and increasing")

        x = (df[self.feature_columns].to_numpy(dtype=np.float64) - self.mean_) / self.scale_
        predicted = np.zeros((len(df), len(horizons), self.num_risks), dtype=np.float64)
        for risk in range(1, self.num_risks + 1):
            fit = self.models_[risk]
            positions = np.searchsorted(fit.event_times, horizons, side="right") - 1
            cumulative_baseline = np.zeros(len(horizons))
            has_event = positions >= 0
            cumulative_baseline[has_event] = fit.cumulative_baseline_hazard[positions[has_event]]
            # 1-exp(-H0(t)*exp(x @ beta)) is the Fine-Gray CIF.
            relative_risk = np.exp(np.clip(x @ fit.coef, -50.0, 50.0))
            predicted[:, :, risk - 1] = -np.expm1(
                -relative_risk[:, None] * cumulative_baseline[None, :]
            )
        return np.clip(predicted, 0.0, 1.0)

    def coefficients(self) -> pd.DataFrame:
        """Regression coefficients per risk (on standardized feature scale)."""
        if len(self.models_) != self.num_risks:
            raise RuntimeError("Fit model before accessing coefficients")
        return pd.DataFrame({
            "feature": self.feature_columns,
            **{f"risk_{risk}": self.models_[risk].coef for risk in self.models_},
        })

    def fit_summary(self) -> pd.DataFrame:
        """Convergence summary of the two separately fitted regressions."""
        if len(self.models_) != self.num_risks:
            raise RuntimeError("Fit model before accessing summary")
        return pd.DataFrame([
            {"risk": risk, "iterations": fitted.iterations,
             "neg_partial_log_likelihood": fitted.objective}
            for risk, fitted in self.models_.items()
        ])


def evaluate_finegray(
    model: FineGrayBaseline,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    evaluation_times=None,
) -> pd.DataFrame:
    """Calculate the EXACT same C-index and Brier formulas as NeuralFG.

    Supply validation data during model selection; save a separate test
    fold for the final chosen model. Censoring KM is fit only on train_df.
    """
    if not set(train_df.index).isdisjoint(eval_df.index):
        raise ValueError("Training and evaluation rows overlap")
    if evaluation_times is None:
        observed = train_df.loc[train_df["label"] > 0, "time"]
        evaluation_times = observed.quantile([0.25, 0.50, 0.75]).to_numpy()
    horizons = np.asarray(evaluation_times, dtype=float).reshape(-1)
    cif = model.predict_cif(eval_df, horizons)
    km = CensoringKM.fit(train_df["time"].to_numpy(), train_df["label"].to_numpy())
    time = eval_df["time"].to_numpy(dtype=float)
    events = eval_df["label"].to_numpy(dtype=int)

    max_cif_sum = float(cif.sum(axis=-1).max())
    if max_cif_sum > 1 + 1e-6:
        warnings.warn(
            "Independently fitted Fine-Gray CIFs sum to more than 1 "
            f"(maximum {max_cif_sum:.4f}); this is possible for separately "
            "fitted competing-risk models. Results were not renormalized.",
            RuntimeWarning,
            stacklevel=2,
        )

    records = []
    for time_idx, horizon in enumerate(horizons):
        for risk in range(1, model.num_risks + 1):
            predicted = cif[:, time_idx, risk - 1]
            records.append({
                "risk": risk,
                "time": float(horizon),
                "c_index": time_dependent_cindex(
                    time, events, predicted, risk=risk, horizon=float(horizon),
                    censoring_km=km,
                ),
                "brier_paper": brier_score_ipcw(
                    time, events, predicted, risk=risk, horizon=float(horizon),
                    censoring_km=km,
                ),
                "brier_competing": brier_score_ipcw(
                    time, events, predicted, risk=risk, horizon=float(horizon),
                    censoring_km=km, include_competing_events=True,
                ),
                "mean_predicted_cif": float(predicted.mean()),
                "n_test": len(eval_df),
            })
    return pd.DataFrame(records)
