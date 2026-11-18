"""Controlled NeuralFG extension experiments for the CSE6250 proposal.

Copy to PyHealth/examples/deephit_extension_experiments.py.
Run from PyHealth repository root after the main random search completes.

Examples:
    python examples/deephit_extension_experiments.py --experiment dropout \
        --split-dir examples/neuralfg/results/paper_reproduction_search_long/search_nfull_k5_trials5_ep1000_pat3_seed0_cfg4a34f27f --smoke

    python examples/deephit_extension_experiments.py --experiment censoring \
        --split-dir examples/neuralfg/results/paper_reproduction_search_long/search_nfull_k5_trials5_ep1000_pat3_seed0_cfg4a34f27f --folds 1 --epochs 300

This script does NOT modify the completed 100-trial random search or re-select
hyperparameters using held-out test observations. Running --folds 1 and then
--folds 5 with the same other options reuses Fold 1 results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from neuralfg.data import load_data
from neuralfg.paper_reproduction import (
    PaperRunConfig,
    fit_neuralfg_paper_style,
    make_training_config,
    neural_predictions,
    reference_rows,
    transform_for_network,
)
from neuralfg.finegray_baseline import FineGrayBaseline

DROPOUT_RATES = (0.0, 0.25, 0.5, 0.75)
CENSOR_RATES = (0.25, 0.50, 0.75)
# Fixed exploratory architecture from the completed 100-trial search:
# the most common embedding/monotonic design among the fold winners.
# Keep it constant for all censoring and dropout conditions.
# Do not adjust it after viewing extension test-set results.
BASE_CANDIDATE = {
    "embedding_shape": (50, 3),
    "monotonic_shape": (25, 2),
    "learning_rate": 1e-3,
    "batch_size": 100,
    "dropout": 0.0,
}


def validate_latent_outcomes(df: pd.DataFrame) -> None:
    """Check that true_time/true_label behave as latent event outcomes."""
    required = {"time", "label", "true_time", "true_label"}
    if not required.issubset(df):
        raise ValueError(f"Missing columns: {sorted(required - set(df))}")
    time = df["time"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    latent = df["true_time"].to_numpy(dtype=float)
    latent_labels = df["true_label"].to_numpy(dtype=int)
    if not (np.isfinite(time).all() and np.isfinite(latent).all()):
        raise ValueError("Nonfinite times")
    if np.any(latent < 0) or np.any(time < 0):
        raise ValueError("Times must be >=0")
    if not set(np.unique(latent_labels)).issubset({1, 2}):
        raise ValueError("true_label must identify competing cause 1 or 2")
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError("Observed label must be 0, 1 or 2")
    if np.any(time > latent + 1e-8):
        raise ValueError("Observed time exceeds latent event time; verify DeepHit columns")
    events = labels > 0
    if np.any(labels[events] != latent_labels[events]):
        raise ValueError("Observed event types disagree with true_label")
    if np.any(~np.isclose(time[events], latent[events])):
        raise ValueError("Uncensored observed time disagrees with true_time")


def generate_censoring_scenarios(df: pd.DataFrame, seed: int = 2026):
    """Create three nested independent exponential censoring regimes.

    A single subject-specific Exp(1) draw is shared by conditions. A scale
    calibrated on this fixed synthetic cohort makes observed censoring close
    to the intended percentage while preserving all latent event outcomes.
    """
    validate_latent_outcomes(df)
    latent = df["true_time"].to_numpy(dtype=float)
    labels = df["true_label"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    exp1 = rng.exponential(scale=1.0, size=len(df))
    ratios = latent / np.maximum(exp1, 1e-12)
    conditions = {}
    for rate in CENSOR_RATES:
        scale = float(np.quantile(ratios, 1.0 - rate))
        censor_time = exp1 * scale
        is_censored = censor_time < latent
        generated = df.copy()
        generated["time"] = np.where(is_censored, censor_time, latent)
        generated["label"] = np.where(is_censored, 0, labels).astype(int)
        achieved = float(is_censored.mean())
        if abs(achieved - rate) > 0.005:
            raise RuntimeError(f"Target censoring rate {rate:.2%} not achieved: {achieved:.2%}")
        conditions[rate] = (generated, achieved, scale)
    return conditions


def load_splits(split_dir: Path, fold: int, df: pd.DataFrame):
    path = split_dir / f"fold_{fold}" / "split_indices.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved paper-reproduction split: {path}")
    original = pd.read_csv(path)
    if not {"row_index", "split"}.issubset(original.columns):
        raise ValueError("split_indices.csv must contain row_index and split")
    if len(original) != len(df) or original["row_index"].duplicated().any():
        raise ValueError("Split rows are duplicated or missing")
    if set(original["row_index"].astype(int)) != set(df.index):
        raise ValueError("Split indices do not correspond to this dataset")
    if set(original["split"]) != {"train", "tune", "stop", "test"}:
        raise ValueError("Expected train, tune, stop, test split labels")
    return {
        label: original.loc[original["split"] == label, "row_index"].astype(int).to_numpy()
        for label in ("train", "tune", "stop", "test")
    }


def fit_condition(df: pd.DataFrame, indices: dict, fold: int, candidate: dict,
                  cfg: PaperRunConfig, horizons: np.ndarray,
                  condition_dir: Path, include_finegray: bool):
    # Explicitly exclude latent event columns: neither model may see them.
    columns = ["time", "label"] + [f"feature{i}" for i in range(1, 13)]
    train, tune, stop, test = [
        df.loc[indices[name], columns].copy()
        for name in ("train", "tune", "stop", "test")
    ]
    transformed, scaling = transform_for_network(train, [tune, stop, test])
    train_scaled, tune_scaled, stop_scaled, test_scaled = transformed
    neural_cfg = make_training_config(cfg, candidate, fold)
    normalized_horizons = (horizons + 1) / scaling["time_scale"]
    model, history, fit_info = fit_neuralfg_paper_style(
        train_scaled, stop_scaled, normalized_horizons, neural_cfg, fold=fold
    )
    prediction = neural_predictions(
        model, test_scaled, horizons, scaling["time_scale"]
    )
    rows = reference_rows(fold, "NeuralFG", prediction, train, test, horizons)
    condition_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), condition_dir / "best_model.pt")
    history.to_csv(condition_dir / "training_history.csv", index=False)
    (condition_dir / "training_summary.json").write_text(
        json.dumps({**fit_info, "candidate": candidate, "evaluation_times": horizons.tolist()}, indent=2)
    )
    if include_finegray:
        fg = FineGrayBaseline(num_risks=2, maxiter=250, tolerance=1e-9).fit(train)
        fg.coefficients().to_csv(condition_dir / "finegray_coefficients.csv", index=False)
        fg_prediction = fg.predict_cif(test, horizons)
        rows.extend(reference_rows(fold, "Fine-Gray", fg_prediction, train, test, horizons))
    pd.DataFrame(rows).to_csv(condition_dir / "test_metrics_reference.csv", index=False)
    return rows, fit_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["censoring", "dropout"], required=True)
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Existing completed paper_reproduction experiment folder")
    parser.add_argument("--folds", type=int, choices=range(1, 6), default=1,
                        help="Number of folds to include (1 for initial check, 5 for full extension)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--censor-seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite completed condition outputs instead of resuming")
    parser.add_argument("--output-root", type=Path,
                        default=Path("examples/neuralfg/results/extension_experiments"))
    args = parser.parse_args()
    if args.folds < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("folds, epochs and patience must be positive")
    split_dir = args.split_dir.resolve()
    original = load_data()
    validate_latent_outcomes(original)
    # Fix the same splits for all treatments; do NOT resplit by modified labels.
    scenarios = generate_censoring_scenarios(original, seed=args.censor_seed) if args.experiment == "censoring" else {}
    conditions = (
        [(f"censor_{int(k*100)}", scenarios[k][0], scenarios[k][1]) for k in CENSOR_RATES]
        if args.experiment == "censoring" else
        [(f"dropout_{int(k*100)}", original, float((original["label"] == 0).mean())) for k in DROPOUT_RATES]
    )
    max_epochs = 2 if args.smoke else args.epochs
    n_folds = 1 if args.smoke else args.folds
    model_tag = (f"emb{BASE_CANDIDATE['embedding_shape'][0]}x{BASE_CANDIDATE['embedding_shape'][1]}"
                 f"_mono{BASE_CANDIDATE['monotonic_shape'][0]}x{BASE_CANDIDATE['monotonic_shape'][1]}"
                 f"_lr{BASE_CANDIDATE['learning_rate']}_bs{BASE_CANDIDATE['batch_size']}")
    root = (args.output_root / args.experiment /
            f"{model_tag}_ep{max_epochs}_pat{args.patience}"
            f"_seed{args.seed}_cseed{args.censor_seed}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "experiment": args.experiment,
        "split_source": str(split_dir),
        "folds": n_folds,
        "max_epochs": max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "censor_seed": args.censor_seed,
        "base_candidate": BASE_CANDIDATE,
        "horizon_rule": "Original observed training event-time quantiles, fixed across conditions within fold",
    }, indent=2))
    all_rows = []
    info = []
    split_counts = []
    for fold in range(1, n_folds + 1):
        indices = load_splits(split_dir, fold, original)
        if args.smoke:
            sizes = {"train": 200, "tune": 60, "stop": 60, "test": 100}
            indices = {k: v[:sizes[k]] for k,v in indices.items()}
        # Evaluation horizons are IDENTICAL across censoring regimes within fold.
        reference_train = original.loc[indices["train"]]
        horizons = reference_train.loc[reference_train["label"] > 0, "time"].quantile(
            [0.25, 0.50, 0.75]).to_numpy(dtype=float)
        for name, frame, censor_fraction in conditions:
            if args.experiment == "censoring":
                for phase, row_indices in indices.items():
                    phase_labels = frame.loc[row_indices, "label"].to_numpy(dtype=int)
                    split_counts.append({
                        "fold": fold, "condition": name, "split": phase,
                        "n": int(len(row_indices)),
                        "censoring_fraction": float((phase_labels == 0).mean()),
                        "risk1_events": int((phase_labels == 1).sum()),
                        "risk2_events": int((phase_labels == 2).sum()),
                    })
            candidate = BASE_CANDIDATE.copy()
            if args.experiment == "dropout":
                candidate["dropout"] = int(name.split("_")[1]) / 100.0
            condition_dir = root / f"fold_{fold}" / name
            print(f"\nFold {fold}/{n_folds}, {name}; observed censoring={censor_fraction:.1%}; "
                  f"horizons={horizons.tolist()}", flush=True)
            metrics_file = condition_dir / "test_metrics_reference.csv"
            info_file = condition_dir / "training_summary.json"
            if not args.force and metrics_file.is_file() and info_file.is_file():
                print("  Reusing completed condition outputs", flush=True)
                rows = pd.read_csv(metrics_file).to_dict("records")
                fit = json.loads(info_file.read_text())
                fit = {k: fit[k] for k in ("best_epoch", "best_val_nll", "epochs_run",
                                             "training_seed", "stopping_rule")}
            else:
                rows, fit = fit_condition(
                    frame, indices, fold, candidate,
                    PaperRunConfig(
                        max_epochs=max_epochs,
                        patience=args.patience,
                        seed=args.seed,
                        mode="fixed",
                    ),
                    horizons, condition_dir,
                    include_finegray=(args.experiment == "censoring"),
                )
            for row in rows:
                row.update({"experiment": args.experiment, "condition": name,
                            "censoring_rate": censor_fraction,
                            "dropout": candidate["dropout"]})
            all_rows.extend(rows)
            info.append({"fold": fold, "condition": name, "censoring_rate": censor_fraction,
                         "dropout": candidate["dropout"], **fit})
            pd.DataFrame(all_rows).to_csv(root / "fold_metrics_reference.csv", index=False)
            pd.DataFrame(info).to_csv(root / "training_summary.csv", index=False)
    if split_counts:
        pd.DataFrame(split_counts).to_csv(root / "censoring_by_split.csv", index=False)
    metrics = pd.DataFrame(all_rows)
    summary = metrics.groupby(
        ["experiment", "condition", "model", "risk", "quantile"], as_index=False
    ).agg(
        mean_c_index=("c_index_reference", "mean"),
        sd_c_index=("c_index_reference", "std"),
        mean_brier=("brier_reference", "mean"),
        sd_brier=("brier_reference", "std"),
        n_folds=("fold", "nunique"),
    )
    summary.to_csv(root / "experiment_summary.csv", index=False)
    print("\nSaved extension results to:", root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
