"""Deep Cox Mixtures reproduction and ablation on SUPPORT2.

Compares Cox proportional hazards, DeepSurv and Deep Cox
Mixtures using five-fold cross-validation.

Reference:
    Nagpal et al. (2021).
    Deep Cox Mixtures for Survival Regression.
    https://proceedings.mlr.press/v149/nagpal21a.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from pyhealth.datasets import Support2Dataset
from pyhealth.models import DeepCoxMixtures
from pyhealth.tasks import SurvivalTimeToEventSupport2

QUANTILES = [25, 50, 75]


def load_support2(
    data_root: str,
) -> tuple:
    """Load SUPPORT2 features, outcomes and demographic groups."""

    dataset = Support2Dataset(
        root=data_root,
        tables=["support2"],
    )

    samples = dataset.set_task(SurvivalTimeToEventSupport2())

    raw = pd.read_csv(Path(data_root) / "support2.csv")

    race_lookup = {
        str(pid): str(race).strip().lower()
        for pid, race in zip(raw["sno"], raw["race"])
    }

    features = []
    durations = []
    events = []
    races = []

    for i in range(len(samples)):
        sample = samples[i]

        features.append(np.asarray(sample["features"]).reshape(-1))

        durations.append(np.asarray(sample["duration"]).reshape(-1)[0])

        events.append(np.asarray(sample["event"]).reshape(-1)[0])

        patient_id = str(sample["patient_id"])

        races.append(race_lookup.get(patient_id, "unknown"))

    X = np.asarray(features, dtype=np.float32)
    T = np.asarray(durations, dtype=np.float32)
    E = np.asarray(events, dtype=np.float32)
    A = np.asarray(races)

    # Exclude missing race only from demographic evaluation.
    groups = np.where(
        np.isin(A, ["nan", "", "unknown", "missing", "none"]),
        "unknown",
        np.where(A == "white", "white", "non_white"),
    )

    print("\nSUPPORT2 Dataset Summary")
    print("-" * 50)
    print("Patients:", len(X))
    print("Features:", X.shape[1])
    print("Deaths:", int(E.sum()))
    print("Censored:", int((E == 0).sum()))
    print("Censoring rate:", round(1 - E.mean(), 4))
    print("White patients:", int((groups == "white").sum()))
    print("Non-White patients:", int((groups == "non_white").sum()))
    print("Unknown race:", int((groups == "unknown").sum()))

    return samples, X, T, E, groups


def calibration_error(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> float:
    """Compute an IPCW-weighted binned calibration error."""

    censoring = CensoringDistributionEstimator()
    censoring.fit(y_train)

    time = y_test["time"]
    event = y_test["event"]

    survival = np.asarray(survival).reshape(-1)
    observed = (time > horizon).astype(float)

    weights = np.zeros(len(time), dtype=float)

    deaths = (time <= horizon) & event
    survivors = time > horizon

    if deaths.any():
        g_deaths = censoring.predict_proba(time[deaths])

        weights[deaths] = 1.0 / np.maximum(g_deaths, 1e-8)

    if survivors.any():
        g_horizon = censoring.predict_proba(np.array([horizon]))[0]

        weights[survivors] = 1.0 / max(g_horizon, 1e-8)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_indices = np.digitize(
        survival,
        bin_edges[1:-1],
    )

    total_weight = weights.sum()

    if total_weight == 0:
        return float("nan")

    ece = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        bin_weight = weights[mask].sum()

        if bin_weight == 0:
            continue

        observed_survival = np.average(
            observed[mask],
            weights=weights[mask],
        )

        predicted_survival = np.average(
            survival[mask],
            weights=weights[mask],
        )

        ece += bin_weight / total_weight * abs(observed_survival - predicted_survival)

    return float(ece)


def evaluate_survival(
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    predictions: np.ndarray,
    horizons: np.ndarray,
) -> dict:
    """Evaluate discrimination and calibration."""

    y_train = Surv.from_arrays(
        event=E_train.astype(bool),
        time=T_train,
    )

    y_test = Surv.from_arrays(
        event=E_test.astype(bool),
        time=T_test,
    )

    predictions = np.asarray(predictions)

    ctd_scores = []
    ece_scores = []

    for j, horizon in enumerate(horizons):
        risk = 1.0 - predictions[:, j]

        ctd = concordance_index_ipcw(
            y_train,
            y_test,
            risk,
            tau=float(horizon),
        )[0]

        ece = calibration_error(
            y_train,
            y_test,
            predictions[:, j],
            float(horizon),
        )

        ctd_scores.append(ctd)
        ece_scores.append(ece)

    _, brier_scores = brier_score(
        y_train,
        y_test,
        predictions,
        horizons,
    )

    return {
        "ctd": np.asarray(ctd_scores),
        "ece": np.asarray(ece_scores),
        "brier": np.asarray(brier_scores),
    }


def train_cph(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """Fit a conventional Cox proportional hazards model."""

    y_train = Surv.from_arrays(
        event=E_train.astype(bool),
        time=T_train,
    )

    model = CoxPHSurvivalAnalysis(
        alpha=0.01,
        ties="breslow",
    )

    model.fit(X_train, y_train)

    survival_functions = model.predict_survival_function(X_test)

    return np.asarray([fn(horizons) for fn in survival_functions])


def train_dcm(
    samples,
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    horizons: np.ndarray,
    components: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    """Train a single-component or mixture deep Cox model."""

    torch.manual_seed(seed)

    model = DeepCoxMixtures(
        dataset=samples,
        input_dim=X_train.shape[1],
        num_components=components,
        hidden_dims=(64, 32),
    )

    model.fit_em(
        features=torch.tensor(X_train, dtype=torch.float32),
        duration=torch.tensor(T_train, dtype=torch.float32),
        event=torch.tensor(E_train, dtype=torch.float32),
        epochs=epochs,
        learning_rate=0.001,
        seed=seed,
        verbose=False,
    )

    predictions = model.predict_survival(
        features=torch.tensor(X_test, dtype=torch.float32),
        horizons=horizons,
    )

    return predictions.detach().cpu().numpy()


def record_results(
    records: list,
    model_name: str,
    fold: int,
    population: str,
    horizons: np.ndarray,
    results: dict,
    n_patients: int,
) -> None:
    """Store individual fold and horizon results."""

    for j, quantile in enumerate(QUANTILES):
        records.append(
            {
                "model": model_name,
                "fold": fold,
                "population": population,
                "quantile": quantile,
                "horizon": float(horizons[j]),
                "n_patients": n_patients,
                "ctd": float(results["ctd"][j]),
                "ece": float(results["ece"][j]),
                "brier": float(results["brier"][j]),
            }
        )


def main(
    data_root: str,
    epochs: int,
    folds: int,
    components: list[int],
    output_dir: str,
) -> None:
    """Execute cross-validation and mixture ablation."""

    torch.manual_seed(42)
    np.random.seed(42)

    samples, X, T, E, groups = load_support2(data_root)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    strata = np.asarray([f"{int(event)}_{group}" for event, group in zip(E, groups)])

    _, counts = np.unique(strata, return_counts=True)

    if counts.min() < folds:
        strata = E.astype(int)

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    records = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, strata),
        start=1,
    ):
        print("\n" + "=" * 65)
        print(f"FOLD {fold}/{folds}")
        print("=" * 65)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        T_train = T[train_idx]
        E_train = E[train_idx]

        T_test = T[test_idx]
        E_test = E[test_idx]

        A_test = groups[test_idx]

        horizons = np.percentile(
            T_train[E_train == 1],
            QUANTILES,
        )

        print("Training patients:", len(train_idx))
        print("Testing patients:", len(test_idx))
        print("Horizons:", horizons)

        predictions = {}

        # Conventional Cox benchmark.
        print("\nTraining CPH...")

        predictions["CPH"] = train_cph(
            X_train,
            T_train,
            E_train,
            X_test,
            horizons,
        )

        # DeepSurv-equivalent control and DCM ablations.
        for k in sorted(set([1] + components)):

            if k == 1:
                name = "DeepSurv / DCM K=1"
            else:
                name = f"DCM K={k}"

            print(f"Training {name}...")

            predictions[name] = train_dcm(
                samples,
                X_train,
                T_train,
                E_train,
                X_test,
                horizons,
                components=k,
                epochs=epochs,
                seed=42 + fold,
            )

        # Evaluate overall and demographic performance.
        population_masks = {
            "Overall": np.ones(len(T_test), dtype=bool),
            "Non-White": A_test == "non_white",
            "White": A_test == "white",
        }

        for name, survival in predictions.items():

            for population, mask in population_masks.items():

                if not mask.any():
                    continue

                results = evaluate_survival(
                    T_train,
                    E_train,
                    T_test[mask],
                    E_test[mask],
                    survival[mask],
                    horizons,
                )

                record_results(
                    records,
                    name,
                    fold,
                    population,
                    horizons,
                    results,
                    int(mask.sum()),
                )

                if population == "Overall":
                    print(f"\n{name}")
                    print("Ctd:", np.round(results["ctd"], 4))
                    print("ECE:", np.round(results["ece"], 4))
                    print("Brier:", np.round(results["brier"], 4))

        # Save intermediate results after every fold.
        pd.DataFrame(records).to_csv(
            output / "fold_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(records)

    summary = results_df.groupby(
        ["population", "model", "quantile"],
        as_index=False,
    ).agg(
        horizon_mean=("horizon", "mean"),
        ctd_mean=("ctd", "mean"),
        ctd_std=("ctd", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
    )

    summary.to_csv(
        output / "summary_results.csv",
        index=False,
    )

    for population in ["Overall", "Non-White", "White"]:
        print("\n" + "=" * 75)
        print(f"FINAL RESULTS: {population}")
        print("=" * 75)

        table = summary[summary["population"] == population]

        print(
            table[
                [
                    "model",
                    "quantile",
                    "ctd_mean",
                    "ece_mean",
                    "brier_mean",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print("\nResults saved to:", output.resolve())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DCM reproduction and ablation.")

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=[1, 3, 4, 6],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dcm",
    )

    args = parser.parse_args()

    if not (Path(args.data_root) / "support2.csv").exists():
        raise FileNotFoundError("Directory must contain support2.csv")

    main(
        data_root=args.data_root,
        epochs=args.epochs,
        folds=args.folds,
        components=args.components,
        output_dir=args.output_dir,
    )
