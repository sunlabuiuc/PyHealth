"""Train and evaluate discrete-time survival models on processed SUPPORT2 data.

Author: Shuxian Xu (NetID: sx30).
Paper: `Let the Experts Speak: Improving Survival Prediction & Calibration
via Mixture-of-Experts Heads <https://proceedings.mlr.press/v297/morrill26a.html>`_.

Run ``python support2_experiment.py`` after preparing ``data/support2/support2.csv``.
From a notebook, call ``run_experiment(r"data/support2/support2.csv")``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.metrics import brier_score, concordance_index_censored
from sksurv.util import Surv
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from pyhealth.models.survival_moe import ExpertType, SurvivalMoE

LOGGER = logging.getLogger("support2_experiment")
TARGET_DURATION = "d.time"
TARGET_EVENT = "death"
PATIENT_ID = "sno"
OUTCOME_OR_POST_OUTCOME_COLUMNS = {
    PATIENT_ID,
    "id",
    TARGET_DURATION,
    TARGET_EVENT,
    "hospdead",
    "surv2m",
    "surv6m",
    "prg2m",
    "prg6m",
    "sfdm2",
    "slos",
    "charges",
    "totcst",
    "totmcst",
    "dnrday",
    "dnr",  # includes DNR orders written after study entry
    "avtisst",  # averaged over days 3-25 after study entry
}
NEURAL_MODELS: tuple[ExpertType, ...] = ("personalized", "fixed", "mtlr")


@dataclass(frozen=True)
class DataSplit:
    """Preprocessed data and train-fitted survival time grid for one seed."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    x_all: np.ndarray
    duration_train: np.ndarray
    duration_validation: np.ndarray
    duration_test: np.ndarray
    event_train: np.ndarray
    event_validation: np.ndarray
    event_test: np.ndarray
    duration_bins_train: np.ndarray
    duration_bins_validation: np.ndarray
    time_grid: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def load_support2(csv_path: str | Path) -> pd.DataFrame:
    """Load an already processed SUPPORT2 CSV and validate its survival labels."""
    data = pd.read_csv(csv_path)
    required = {PATIENT_ID, TARGET_DURATION, TARGET_EVENT}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"SUPPORT2 CSV is missing required columns: {sorted(missing)}")
    if data[PATIENT_ID].isna().any() or data[PATIENT_ID].duplicated().any():
        raise ValueError("Patient IDs must be present and unique.")
    numeric_duration = pd.to_numeric(data[TARGET_DURATION], errors="coerce")
    numeric_event = pd.to_numeric(data[TARGET_EVENT], errors="coerce")
    if not np.isfinite(numeric_duration.to_numpy()).all():
        raise ValueError("Duration values must be finite numbers.")
    if (numeric_duration <= 0).any():
        raise ValueError("All follow-up durations must be positive.")
    if not np.isfinite(numeric_event.to_numpy()).all():
        raise ValueError("Event values must be finite numbers.")
    event_values = set(numeric_event.unique())
    if not event_values.issubset({0, 1}):
        raise ValueError(f"Event labels must be binary 0/1; found {event_values}.")
    data[TARGET_DURATION] = numeric_duration
    data[TARGET_EVENT] = numeric_event.astype(np.int64)
    data[PATIENT_ID] = data[PATIENT_ID].astype(str)
    return data


def _bin_durations(
    durations: np.ndarray,
    events: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Map events to the first grid time >= t and censored times to the first
    grid time > t, so the censored likelihood equals the evaluated S(t)."""
    bins = np.where(
        events.astype(bool),
        np.searchsorted(time_grid, durations, side="left"),
        np.searchsorted(time_grid, durations, side="right"),
    )
    return bins.clip(0, len(time_grid) - 1).astype(np.int64)


def prepare_split(
    data: pd.DataFrame,
    seed: int,
    num_time_bins: int = 100,
) -> DataSplit:
    """Split data and fit preprocessing plus a distinct quantile grid on training only."""
    if num_time_bins < 2:
        raise ValueError("num_time_bins must be at least 2.")
    durations = data[TARGET_DURATION].to_numpy(dtype=np.float64)
    events = data[TARGET_EVENT].to_numpy(dtype=np.int64)
    indices = np.arange(len(data))

    train_indices, remainder_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=events,
    )
    validation_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=0.5,
        random_state=seed,
        stratify=events[remainder_indices],
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in OUTCOME_OR_POST_OUTCOME_COLUMNS
    ]
    features = data.loc[:, feature_columns]
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    if not numeric_columns and not categorical_columns:
        raise ValueError("No patient features remain after excluding outcome columns.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    x_train = np.asarray(
        preprocessor.fit_transform(features.iloc[train_indices]), dtype=np.float32
    )
    x_validation = np.asarray(
        preprocessor.transform(features.iloc[validation_indices]), dtype=np.float32
    )
    x_test = np.asarray(
        preprocessor.transform(features.iloc[test_indices]), dtype=np.float32
    )
    x_all = np.asarray(preprocessor.transform(features), dtype=np.float32)

    train_duration = durations[train_indices]
    train_quantiles = np.quantile(
        train_duration, np.linspace(0.0, 1.0, num_time_bins)
    )
    unique_train_times = np.unique(train_duration)
    if len(unique_train_times) < num_time_bins:
        raise ValueError(
            "The training split has fewer distinct durations than requested time bins."
        )
    time_grid = np.empty(num_time_bins, dtype=np.float64)
    previous_position = -1
    for bin_index, target_time in enumerate(train_quantiles):
        minimum_position = previous_position + 1
        maximum_position = len(unique_train_times) - (num_time_bins - bin_index)
        insertion_position = int(
            np.searchsorted(unique_train_times, target_time, side="left")
        )
        lower_position = int(
            np.clip(insertion_position - 1, minimum_position, maximum_position)
        )
        upper_position = int(
            np.clip(insertion_position, minimum_position, maximum_position)
        )
        if abs(unique_train_times[lower_position] - target_time) <= abs(
            unique_train_times[upper_position] - target_time
        ):
            previous_position = lower_position
        else:
            previous_position = upper_position
        time_grid[bin_index] = unique_train_times[previous_position]

    return DataSplit(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        x_all=x_all,
        duration_train=train_duration,
        duration_validation=durations[validation_indices],
        duration_test=durations[test_indices],
        event_train=events[train_indices].astype(bool),
        event_validation=events[validation_indices].astype(bool),
        event_test=events[test_indices].astype(bool),
        duration_bins_train=_bin_durations(
            train_duration, events[train_indices], time_grid
        ),
        duration_bins_validation=_bin_durations(
            durations[validation_indices], events[validation_indices], time_grid
        ),
        time_grid=time_grid,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def _train_neural_model(
    split: DataSplit,
    expert_type: ExpertType,
    seed: int,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    num_layers: int,
    num_experts: int,
    top_k: int | None,
    load_balance_weight: float,
) -> SurvivalMoE:
    torch.manual_seed(seed)
    model = SurvivalMoE(
        input_dim=split.x_train.shape[1],
        num_time_bins=len(split.time_grid),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        expert_type=expert_type,
        top_k=top_k,
        load_balance_weight=load_balance_weight,
    ).to(device)
    training = TensorDataset(
        torch.from_numpy(split.x_train),
        torch.from_numpy(split.duration_bins_train),
        torch.from_numpy(split.event_train),
    )
    loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_validation = torch.from_numpy(split.x_validation).to(device)
    bins_validation = torch.from_numpy(split.duration_bins_validation).to(device)
    events_validation = torch.from_numpy(split.event_validation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 10
    epochs_without_improvement = 0
    for _ in range(max_epochs):
        model.train()
        for x_batch, bins_batch, events_batch in loader:
            x_batch = x_batch.to(device)
            bins_batch = bins_batch.to(device)
            events_batch = events_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch, bins_batch, events_batch)
            output["loss"].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_output = model(
                x_validation, bins_validation, events_validation
            )
            validation_loss = float(validation_output["nll"].item())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")
    model.load_state_dict(best_state)
    return model


def _survival_from_pmf(
    pmf: np.ndarray,
    time_grid: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Evaluate the piecewise-constant discrete survival curve at query times."""
    tail_probability = np.flip(
        np.cumsum(np.flip(pmf, axis=1), axis=1),
        axis=1,
    )
    indices = np.searchsorted(time_grid, query_times, side="right")
    clipped_indices = np.minimum(indices, len(time_grid) - 1)
    survival = tail_probability[:, clipped_indices]
    survival[:, indices >= len(time_grid)] = 0.0
    return survival


def _censoring_kaplan_meier(
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the censoring survival function from the training split."""
    order = np.argsort(durations)
    sorted_durations = durations[order]
    sorted_censored = ~events[order]
    times, first_indices = np.unique(sorted_durations, return_index=True)
    at_risk = len(sorted_durations) - first_indices
    censored_at_time = np.add.reduceat(
        sorted_censored.astype(np.int64), first_indices
    )
    survival_after = np.cumprod(1.0 - censored_at_time / at_risk)
    return times, survival_after


def _censoring_probability(
    times: np.ndarray,
    survival_after: np.ndarray,
    query: np.ndarray,
    left_limit: bool,
) -> np.ndarray:
    side = "left" if left_limit else "right"
    indices = np.searchsorted(times, query, side=side) - 1
    result = np.ones(query.shape, dtype=np.float64)
    has_previous = indices >= 0
    result[has_previous] = survival_after[indices[has_previous]]
    return np.maximum(result, 1e-6)


def _ipcw_equal_mass_ece(
    durations_train: np.ndarray,
    events_train: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    prediction_times: np.ndarray,
    predicted_survival: np.ndarray,
    num_groups: int = 10,
) -> float:
    """Compute IPCW-adjusted equal-frequency calibration error over time."""
    censor_times, censor_survival = _censoring_kaplan_meier(
        durations_train, events_train
    )
    errors: list[float] = []
    for column, time in enumerate(prediction_times):
        observed = events_test | (durations_test > time)
        if not np.any(observed):
            continue
        outcomes = (durations_test > time).astype(np.float64)
        weights = np.zeros(len(durations_test), dtype=np.float64)
        survived_past_time = durations_test > time
        weights[survived_past_time] = 1.0 / _censoring_probability(
            censor_times,
            censor_survival,
            np.full(np.count_nonzero(survived_past_time), time),
            left_limit=False,
        )
        observed_events = events_test & (durations_test <= time)
        weights[observed_events] = 1.0 / _censoring_probability(
            censor_times,
            censor_survival,
            durations_test[observed_events],
            left_limit=True,
        )
        weights[~observed] = 0.0

        order = np.argsort(predicted_survival[:, column], kind="stable")
        for group in np.array_split(order, min(num_groups, len(order))):
            group_weight = weights[group]
            weight_sum = group_weight.sum()
            if weight_sum > 0:
                predicted = np.average(
                    predicted_survival[group, column], weights=group_weight
                )
                observed_rate = np.average(outcomes[group], weights=group_weight)
                errors.append(float(abs(predicted - observed_rate)))
    if not errors:
        raise ValueError("IPCW ECE could not be estimated for this validation/test split.")
    return float(np.mean(errors))


def _evaluation_times(split: DataSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose quartile Brier horizons and ECE time points from training durations.

    Neither depends on the time grid, so all A2 settings are scored at the
    same time points.
    """
    brier_times = np.quantile(split.duration_train, [0.25, 0.5, 0.75])
    if len(np.unique(brier_times)) != 3:
        raise ValueError("Training durations have tied quartiles; Brier horizons are not distinct.")
    if (
        brier_times[0] < split.duration_test.min()
        or brier_times[-1] > split.duration_test.max()
    ):
        raise ValueError(
            "Training quartile horizons fall outside the test duration range; "
            "use a different split seed."
        )
    ece_times = np.unique(
        np.quantile(split.duration_train, np.linspace(0.05, 0.95, 100))
    )
    all_times = np.unique(np.concatenate((brier_times, ece_times)))
    return brier_times, ece_times, all_times


def _score_predictions(
    split: DataSplit,
    risk: np.ndarray,
    predicted_survival: np.ndarray,
    all_times: np.ndarray,
    brier_times: np.ndarray,
    ece_times: np.ndarray,
) -> dict[str, float]:
    y_train = Surv.from_arrays(split.event_train, split.duration_train)
    y_test = Surv.from_arrays(split.event_test, split.duration_test)
    c_index = concordance_index_censored(
        split.event_test, split.duration_test, risk
    )[0]
    brier_columns = np.searchsorted(all_times, brier_times)
    _, brier_scores = brier_score(
        y_train,
        y_test,
        predicted_survival[:, brier_columns],
        brier_times,
    )
    ece_columns = np.searchsorted(all_times, ece_times)
    ece = _ipcw_equal_mass_ece(
        split.duration_train,
        split.event_train,
        split.duration_test,
        split.event_test,
        ece_times,
        predicted_survival[:, ece_columns],
    )
    return {
        "ece": ece,
        "c_index": float(c_index * 100.0),
        "brier_25": float(brier_scores[0]),
        "brier_50": float(brier_scores[1]),
        "brier_75": float(brier_scores[2]),
    }


def _fit_classical_models(
    split: DataSplit,
    seed: int,
    allow_rsf: bool,
    rf_n_estimators: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit CoxPH and, when its optional dependency works, random survival forest."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    y_train = Surv.from_arrays(split.event_train, split.duration_train)
    models: dict[str, Any] = {
        "CoxPH": CoxnetSurvivalAnalysis(
            alphas=[0.01],
            l1_ratio=0.01,
            fit_baseline_model=True,
        )
    }
    if allow_rsf:
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as error:
            warnings.warn(
                "Skipping RSF: scikit-survival could not import "
                "RandomSurvivalForest in this environment "
                f"({error}). CoxPH and neural models will still run.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            models["RSF"] = RandomSurvivalForest(
                n_estimators=rf_n_estimators,
                max_features="sqrt",
                min_samples_split=200,
                n_jobs=-1,
                random_state=seed,
            )

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, estimator in models.items():
        estimator.fit(split.x_train, y_train)
        risk = np.asarray(estimator.predict(split.x_test), dtype=np.float64).reshape(-1)
        survival_functions = estimator.predict_survival_function(split.x_test)
        results[name] = (risk, np.asarray(survival_functions, dtype=object))
    return results


def _evaluate_classical_survival(
    survival_functions: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [function(query_times) for function in survival_functions.tolist()]
    )


def _cluster_stability(assignments: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Adjusted Rand Index across model seeds."""
    pivot = assignments.pivot(index=PATIENT_ID, columns="seed", values="expert")
    records: list[dict[str, float | int]] = []
    seeds = pivot.columns.to_list()
    for first_position, first_seed in enumerate(seeds):
        for second_seed in seeds[first_position + 1 :]:
            records.append(
                {
                    "seed_a": int(first_seed),
                    "seed_b": int(second_seed),
                    "ari": float(
                        adjusted_rand_score(
                            pivot[first_seed].to_numpy(),
                            pivot[second_seed].to_numpy(),
                        )
                    ),
                }
            )
    return pd.DataFrame(records, columns=["seed_a", "seed_b", "ari"])


def _cluster_profiles(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    first_seed = int(assignments["seed"].min())
    representative = assignments.loc[
        assignments["seed"] == first_seed, [PATIENT_ID, "expert"]
    ]
    merged = data.merge(representative, on=PATIENT_ID, validate="one_to_one")
    profiles: list[dict[str, Any]] = []
    for expert, group in merged.groupby("expert", sort=True):
        profile: dict[str, Any] = {
            "expert": int(expert),
            "n_patients": len(group),
            "mortality_rate": float(group[TARGET_EVENT].mean()),
            "median_followup_days": float(group[TARGET_DURATION].median()),
        }
        if "age" in group:
            profile["median_age"] = float(pd.to_numeric(group["age"]).median())
        if "dzgroup" in group:
            mode = group["dzgroup"].mode(dropna=True)
            profile["most_common_diagnosis_group"] = (
                str(mode.iloc[0]) if not mode.empty else "unknown"
            )
        profiles.append(profile)
    return pd.DataFrame(profiles)


def run_experiment(
    csv_path: str | Path,
    output_dir: str | Path = "outputs/support2_survival_moe",
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55),
    max_epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    num_experts: int = 8,
    num_time_bins: int = 100,
    load_balance_weight: float = 0.01,
    top_k: int | None = None,
    rf_n_estimators: int = 200,
    device_name: str = "auto",
    include_rsf: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the five-seed SUPPORT2 comparison and write metrics/cluster outputs."""
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if max_epochs < 1 or batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive.")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    data = load_support2(csv_path)
    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float | int | str]] = []
    routing_records: list[dict[str, int | str]] = []

    progress = tqdm(
        total=len(seeds) * (len(NEURAL_MODELS) + 1 + int(include_rsf)),
        desc="SUPPORT2 models",
        unit="model",
        dynamic_ncols=True,
    )
    for seed in seeds:
        progress.set_postfix(seed=seed, stage="preprocessing")
        split = prepare_split(data, seed, num_time_bins)
        brier_times, ece_times, all_times = _evaluation_times(split)
        for expert_type in NEURAL_MODELS:
            progress.set_postfix(seed=seed, model=expert_type)
            model_hidden_dim = hidden_dim
            model_num_layers = num_layers
            if model_hidden_dim is None:
                model_hidden_dim = {
                    "personalized": 128,
                    "fixed": 176,
                    "mtlr": 176,
                }[expert_type]
            if model_num_layers is None:
                model_num_layers = {
                    "personalized": 1,
                    "fixed": 2,
                    "mtlr": 2,
                }[expert_type]
            model = _train_neural_model(
                split=split,
                expert_type=expert_type,
                seed=seed,
                device=device,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dim=model_hidden_dim,
                num_layers=model_num_layers,
                num_experts=num_experts,
                top_k=top_k,
                load_balance_weight=load_balance_weight,
            )
            model.eval()
            with torch.no_grad():
                test_output = model(
                    torch.from_numpy(split.x_test).to(device)
                )
                test_pmf = test_output["pmf"].cpu().numpy()
                risk = -(
                    test_pmf * split.time_grid[np.newaxis, :]
                ).sum(axis=1)
                predicted_survival = _survival_from_pmf(
                    test_pmf, split.time_grid, all_times
                )
                all_output = model(torch.from_numpy(split.x_all).to(device))
                all_routing = all_output["routing"].cpu().numpy()

            score = _score_predictions(
                split,
                risk,
                predicted_survival,
                all_times,
                brier_times,
                ece_times,
            )
            metrics.append({"seed": seed, "model": expert_type, **score})
            progress.update()
            if expert_type == "personalized":
                patient_ids = data[PATIENT_ID].astype(str).to_numpy()
                routing_records.extend(
                    {
                        PATIENT_ID: patient_id,
                        "seed": seed,
                        "expert": int(expert),
                    }
                    for patient_id, expert in zip(
                        patient_ids, all_routing.argmax(axis=1)
                    )
                )

        progress.set_postfix(seed=seed, stage="CoxPH/RSF")
        classical = _fit_classical_models(
            split,
            seed,
            allow_rsf=include_rsf,
            rf_n_estimators=rf_n_estimators,
        )
        if include_rsf and "RSF" not in classical:
            progress.total -= 1
            progress.refresh()
        for model_name, (risk, functions) in classical.items():
            predicted_survival = _evaluate_classical_survival(functions, all_times)
            score = _score_predictions(
                split,
                risk,
                predicted_survival,
                all_times,
                brier_times,
                ece_times,
            )
            metrics.append({"seed": seed, "model": model_name, **score})
            progress.update()
        progress.write(f"Finished seed {seed}.")
    progress.close()

    metrics_by_seed = pd.DataFrame(metrics)
    metrics_by_seed.to_csv(result_dir / "metrics_by_seed.csv", index=False)
    metric_columns = ["ece", "c_index", "brier_25", "brier_50", "brier_75"]
    summary = metrics_by_seed.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(result_dir / "metrics_summary.csv", index=False)

    baseline = metrics_by_seed.loc[
        metrics_by_seed["model"] == "mtlr", ["seed", *metric_columns]
    ].rename(columns={column: f"{column}_mtlr" for column in metric_columns})
    paired = metrics_by_seed.merge(baseline, on="seed", how="left")
    for column in metric_columns:
        paired[f"{column}_difference_vs_mtlr"] = (
            paired[column] - paired[f"{column}_mtlr"]
        )
    paired.to_csv(result_dir / "paired_differences_vs_mtlr.csv", index=False)

    assignments = pd.DataFrame(routing_records)
    assignments.to_csv(result_dir / "personalized_moe_assignments.csv", index=False)
    stability = _cluster_stability(assignments)
    stability.to_csv(result_dir / "routing_stability_ari.csv", index=False)
    _cluster_profiles(data, assignments).to_csv(
        result_dir / "cluster_profiles.csv", index=False
    )

    return metrics_by_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SUPPORT2 discrete-time survival MoE experiment."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/support2/support2.csv"),
        help="Path to the SUPPORT2 CSV that has already been prepared.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/support2_survival_moe"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--top-k", type=int, choices=[1, 2])
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--without-rsf", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, summary = run_experiment(
        csv_path=args.csv,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_time_bins=args.time_bins,
        top_k=args.top_k,
        load_balance_weight=args.load_balance_weight,
        rf_n_estimators=args.rf_trees,
        device_name=args.device,
        include_rsf=not args.without_rsf,
    )
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
