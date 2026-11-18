"""Five-fold, paired NeuralFG versus Fine-Gray competing-risk evaluation.

Use the runner ``examples/deephit_competing_risks_cv.py`` from the PyHealth
repository root. Each outer fold is evaluated exactly once; a disjoint inner
validation subset is used only for NeuralFG checkpoint selection. Fine-Gray
and NeuralFG fit on the SAME inner-training observations within each fold.

Event labels: 0=censored, 1 and 2=observed competing-risk events.
All preprocessing, censoring-weight estimation, and evaluation horizons are
fit on the inner-training split; outer-fold outcomes are used only for scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class CVConfig:
    """Fixed, preselected hyperparameters for paired cross-validation."""

    # Dataset and split settings
    max_samples: int | None = None  # None = all 30,000 DeepHit observations
    n_splits: int = 5
    validation_fraction: float = 0.20  # fraction of outer development set
    seed: int = 42

    # NeuralFG: same hyperparameters as the leading single-split experiment
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_dim: int = 50
    embedding_dim: int = 50
    embedding_layers: int = 2
    monotonic_layers: int = 2
    dropout: float = 0.0
    patience: int = 20
    min_delta: float = 0.0001
    gradient_clip: float = 1.0
    device: str = "cpu"

    # Fine-Gray: match the previously tested baseline
    num_risks: int = 2
    fg_maxiter: int = 250
    fg_tolerance: float = 1e-9

    # Optional destination override; otherwise use
    # examples/neuralfg/results/cross_validation/<experiment_name>/
    output_root: str | None = None


def _validate_config(config: CVConfig) -> None:
    if config.n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if config.max_samples is not None and config.max_samples < 20:
        raise ValueError("max_samples must be at least 20")
    if min(config.epochs, config.batch_size, config.patience, config.num_risks) <= 0:
        raise ValueError("epochs, batch_size, patience, num_risks must be positive")
    if config.fg_maxiter <= 0 or config.fg_tolerance <= 0:
        raise ValueError("Fine-Gray optimization settings must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def experiment_name(config: CVConfig) -> str:
    """Different model/split configurations get separate output folders."""
    sample_tag = "full" if config.max_samples is None else str(config.max_samples)
    # A short settings hash prevents overwriting runs that differ in a
    # less-visible parameter (e.g., gradient clipping or device).
    config_for_hash = asdict(config)
    config_for_hash.pop("output_root", None)
    digest = hashlib.sha256(
        json.dumps(config_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"n{sample_tag}_k{config.n_splits}_val{config.validation_fraction:g}"
        f"_h{config.hidden_dim}_e{config.embedding_dim}"
        f"_el{config.embedding_layers}_ml{config.monotonic_layers}"
        f"_drop{config.dropout:g}_epochs{config.epochs}_bs{config.batch_size}"
        f"_lr{config.learning_rate:g}_wd{config.weight_decay:g}"
        f"_pat{config.patience}_mind{config.min_delta:g}"
        f"_fgiter{config.fg_maxiter}_fgtol{config.fg_tolerance:g}"
        f"_seed{config.seed}_cfg{digest}"
    )


def make_cv_splits(df: pd.DataFrame, config: CVConfig):
    """Yield (fold_number, train_df, val_df, test_df) with disjoint indices.

    Uses stratified outer K-fold and stratified inner holdout. Each row appears
    in exactly one outer test fold. No class labels from test are used for
    training, tuning, or computing evaluation-time quantiles.
    """
    _validate_config(config)
    if not df.index.is_unique:
        raise ValueError("DataFrame indices must be unique")
    if "label" not in df or "time" not in df:
        raise ValueError("DataFrame must contain 'time' and 'label' columns")
    if len(df) < config.n_splits * 5:
        raise ValueError("Dataset is too small for the requested folds")
    event = df["label"].to_numpy(dtype=int)
    labels, counts = np.unique(event, return_counts=True)
    if not np.array_equal(labels, np.arange(config.num_risks + 1)):
        raise ValueError("Expected event codes 0, 1, ..., num_risks")
    if counts.min() < config.n_splits:
        raise ValueError("Every event class needs at least n_splits samples")

    outer = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    for fold, (dev_pos, test_pos) in enumerate(outer.split(df, event), start=1):
        train_pos, val_pos = train_test_split(
            dev_pos,
            test_size=config.validation_fraction,
            random_state=config.seed + fold,
            stratify=event[dev_pos],
        )
        train_df = df.iloc[train_pos].copy()
        val_df = df.iloc[val_pos].copy()
        test_df = df.iloc[test_pos].copy()

        ids = [set(f.index) for f in (train_df, val_df, test_df)]
        assert ids[0].isdisjoint(ids[1])
        assert ids[0].isdisjoint(ids[2])
        assert ids[1].isdisjoint(ids[2])
        assert sum(map(len, ids)) == len(df)
        yield fold, train_df, val_df, test_df


def fit_neuralfg_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    evaluation_times: np.ndarray,
    config: CVConfig,
    *,
    fold: int,
):
    """Fit NeuralFG on a PROVIDED training set (no additional splitting).

    Returns the best-validation checkpoint and its epoch history. Crucially,
    this function never sees the outer test data.
    """
    import torch
    from pyhealth.datasets import get_dataloader
    from pyhealth.models.neural_fg import NeuralFG

    from .data import create_dataset, prepare_pyhealth_samples
    from .train import _run_epoch

    torch.manual_seed(config.seed + fold)
    np.random.seed(config.seed + fold)

    train_ds = create_dataset(prepare_pyhealth_samples(train_df))
    val_ds = create_dataset(prepare_pyhealth_samples(val_df))
    # PyHealth's get_dataloader does not support num_workers in all versions.
    train_loader = get_dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = NeuralFG(
        dataset=train_ds,
        num_risks=config.num_risks,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        embedding_layers=config.embedding_layers,
        monotonic_layers=config.monotonic_layers,
        dropout=config.dropout,
        time_grid=torch.tensor(evaluation_times, dtype=torch.float32),
    ).to(torch.device(config.device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    rows = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, config.gradient_clip)
        # _run_epoch retains autograd for dCIF/dt even in validation.
        val_loss = _run_epoch(model, val_loader, None, config.gradient_clip)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  Fold {fold} epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"  Fold {fold}: early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("NeuralFG failed to create a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    details = {
        "best_epoch": best_epoch,
        "best_val_nll": best_val_loss,
        "epochs_run": len(rows),
        "training_seed": config.seed + fold,
    }
    return model, pd.DataFrame(rows), details


def summarize_cv_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sample SD across held-out outer folds, grouped by quantile.

    Event-time quantiles can differ from fold to fold, hence grouping by
    quantile 'q25/q50/q75' instead of floating-point time values.
    """
    required = {"fold", "model", "risk", "quantile", "time", "c_index",
                "brier_paper", "brier_competing"}
    if not required.issubset(fold_metrics.columns):
        raise ValueError(f"fold_metrics missing fields: {sorted(required-set(fold_metrics.columns))}")
    records = []
    for (model, risk, quantile), group in fold_metrics.groupby(
        ["model", "risk", "quantile"], sort=True
    ):
        for metric in ("c_index", "brier_paper", "brier_competing"):
            values = group[metric].to_numpy(dtype=float)
            records.append({
                "model": model,
                "risk": int(risk),
                "quantile": quantile,
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"),
                "n_folds": int(np.isfinite(values).sum()),
                "mean_evaluation_time": float(group["time"].mean()),
            })
    return pd.DataFrame.from_records(records)


def run_cross_validation(df: pd.DataFrame, config: CVConfig | None = None):
    """Train both models on paired folds; save each fold and aggregate CSVs.

    Returns: (fold_metrics_df, summary_df, experiment_directory).
    """
    import torch
    from .finegray_baseline import FineGrayBaseline, evaluate_finegray
    from .metrics import evaluate_neuralfg

    config = config if config is not None else CVConfig()
    _validate_config(config)
    if config.max_samples is not None:
        df = df.sample(
            n=min(config.max_samples, len(df)), random_state=config.seed
        ).copy()

    root = (
        Path(config.output_root).expanduser().resolve()
        if config.output_root is not None
        else Path(__file__).resolve().parent / "results" / "cross_validation"
    )
    output_dir = root / experiment_name(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    rows = []
    for fold, train_df, val_df, test_df in make_cv_splits(df, config):
        fold_dir = output_dir / f"fold_{fold}"
        neural_dir = fold_dir / "neuralfg"
        fg_dir = fold_dir / "finegray"
        neural_dir.mkdir(parents=True, exist_ok=True)
        fg_dir.mkdir(parents=True, exist_ok=True)

        observed_times = train_df.loc[train_df["label"] > 0, "time"]
        if observed_times.empty:
            raise ValueError(f"Fold {fold} has no observed events in training")
        horizons = observed_times.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
        if np.any(np.diff(horizons) <= 0):
            raise ValueError(f"Fold {fold} has duplicate evaluation quantiles: {horizons}")
        print(
            f"\n===== Fold {fold}/{config.n_splits} =====\n"
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"horizons={horizons.tolist()}",
            flush=True,
        )

        # Record exact split membership (original DataFrame row indices).
        pd.concat([
            pd.DataFrame({"row_index": train_df.index, "split": "train"}),
            pd.DataFrame({"row_index": val_df.index, "split": "validation"}),
            pd.DataFrame({"row_index": test_df.index, "split": "test"}),
        ], ignore_index=True).to_csv(fold_dir / "split_indices.csv", index=False)

        # Model 1: train with inner validation only, score outer held-out fold.
        neural_model, history, details = fit_neuralfg_on_split(
            train_df, val_df, horizons, config, fold=fold
        )
        history.to_csv(neural_dir / "training_history.csv", index=False)
        torch.save(neural_model.state_dict(), neural_dir / "best_model.pt")
        (neural_dir / "fit_summary.json").write_text(
            json.dumps(details, indent=2), encoding="utf-8"
        )
        neural_scores = evaluate_neuralfg(
            neural_model, train_df, test_df,
            evaluation_times=horizons,
            batch_size=512,
        )
        neural_scores.to_csv(neural_dir / "test_metrics.csv", index=False)

        # Model 2: fit on exactly the same inner-training data as NeuralFG.
        fg_model = FineGrayBaseline(
            num_risks=config.num_risks,
            maxiter=config.fg_maxiter,
            tolerance=config.fg_tolerance,
        ).fit(train_df)
        fg_model.fit_summary().to_csv(fg_dir / "fit_summary.csv", index=False)
        fg_model.coefficients().to_csv(fg_dir / "coefficients.csv", index=False)
        fg_scores = evaluate_finegray(
            fg_model, train_df, test_df, evaluation_times=horizons
        )
        fg_scores.to_csv(fg_dir / "test_metrics.csv", index=False)

        for model_name, scores in (("NeuralFG", neural_scores), ("Fine-Gray", fg_scores)):
            scores = scores.copy()
            time_to_quantile = {
                float(horizon): quantile
                for horizon, quantile in zip(horizons, ("q25", "q50", "q75"))
            }
            scores["quantile"] = scores["time"].map(time_to_quantile)
            if scores["quantile"].isna().any():
                raise RuntimeError("Unable to associate metric times with quantiles")
            scores.insert(0, "model", model_name)
            scores.insert(0, "fold", fold)
            rows.append(scores)

        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(output_dir / "fold_metrics.csv", index=False)
        print(f"Fold {fold} completed; saved to {fold_dir}", flush=True)

    fold_metrics = pd.concat(rows, ignore_index=True)
    summary = summarize_cv_metrics(fold_metrics)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    print("\nFive-fold comparison (mean ± sample SD):")
    print(summary.to_string(index=False))
    print("\nResults saved to:", output_dir)
    return fold_metrics, summary, output_dir
