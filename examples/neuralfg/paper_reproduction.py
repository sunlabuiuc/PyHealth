"""Paper-aligned (not bit-for-bit CHIL) NeuralFG / Fine-Gray reproduction.

Place at PyHealth/examples/neuralfg/paper_reproduction.py.
Runner: PyHealth/examples/deephit_paper_reproduction.py.

Compared with the established CV script, this adds: an independent dev set
for selecting candidate hyperparameters; a distinct validation set for early
stopping; the authors' random-search grid; train-only standardization; and
(time + 1)/max(train_time + 1) time scaling, as in their released NFG API.

Intentionally differs from the author's exact CHIL implementation in model
architecture, parameter initialization, fold-specific feature standardization,
PyHealth training/early-stop mechanics, and the custom Python Fine-Gray model.
Report as a paper-aligned PyHealth reproduction, not a bitwise replication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

from .cross_validation import CVConfig
from .metrics import CensoringKM
from .metrics_reference import reference_cindex, reference_brier
from .finegray_baseline import FineGrayBaseline

QUANTILES = ("q25", "q50", "q75")


@dataclass
class PaperRunConfig:
    max_samples: int | None = None
    folds: int = 5
    seed: int = 0
    trials: int = 100
    max_epochs: int = 1000
    # Authors' nfg.utilities.train_nfg defaults to patience_max=3, but its
    # stopping logic differs from PyHealth's best-loss early stopping.
    patience: int = 3
    min_delta: float = 0.0
    gradient_clip: float = 1e9  # effectively disable PyHealth clipping
    weight_decay: float = 0.001  # authors' nfg.utilities.train_nfg default
    device: str = "cpu"
    # Choose --mode fixed first, then --mode search for random search.
    mode: str = "fixed"  # fixed or search
    output_root: str | None = None


def paper_hyperparameter_grid() -> dict:
    # Original author script: layers = [[i]*(j+1) for i in (25,50) for j in range(4)].
    # Their layers and layers_surv are sampled independently.
    shapes = [(width, depth) for width in (25, 50) for depth in range(1, 5)]
    return {
        "embedding_shape": shapes,
        "monotonic_shape": shapes,
        "learning_rate": [1e-3, 1e-4],
        "batch_size": [100, 250],
        "dropout": [0.0, 0.25, 0.5, 0.75],
    }


def candidate_configs(config: PaperRunConfig) -> list[dict]:
    if config.mode == "fixed":
        return [{
            "embedding_shape": (50, 2), "monotonic_shape": (50, 2),
            "learning_rate": 1e-3, "batch_size": 100, "dropout": 0.0,
        }]
    if config.mode != "search":
        raise ValueError("mode must be 'fixed' or 'search'")
    grid = paper_hyperparameter_grid()
    total = 1
    for choices in grid.values():
        total *= len(choices)
    if not (1 <= config.trials <= total):
        raise ValueError(f"trials must be between 1 and {total}")
    return list(ParameterSampler(grid, n_iter=config.trials, random_state=config.seed))


def split_paper_fold(dev_outer: pd.DataFrame, fold: int, config: PaperRunConfig):
    """80% inner train, 10% tune/dev, 10% early-stop of the 80% outer data.

    The outer fold's 20% test partition is supplied separately and is never
    involved in selection or early stopping.
    """
    if not dev_outer.index.is_unique:
        raise ValueError("Original row indices must be unique")
    train80, held20 = train_test_split(
        dev_outer.index.to_numpy(), test_size=0.2,
        random_state=config.seed, stratify=dev_outer["label"],
    )
    tune, stop = train_test_split(
        held20, test_size=0.5, random_state=config.seed,
        stratify=dev_outer.loc[held20, "label"],
    )
    return dev_outer.loc[train80].copy(), dev_outer.loc[tune].copy(), dev_outer.loc[stop].copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = sorted((c for c in frame if re.fullmatch(r"feature\d+", c)),
                  key=lambda c: int(c[7:]))
    if not cols:
        raise ValueError("No featureN columns found")
    return cols


def transform_for_network(train: pd.DataFrame, other_frames: list[pd.DataFrame]):
    """Training-only feature moments; author script used global scaling.

    Time normalization mirrors NeuralFineGray._normalise('minmax'):
    (time + 1) / max(training_time + 1).
    """
    columns = feature_columns(train)
    mu = train[columns].to_numpy(dtype=float).mean(axis=0)
    sigma = train[columns].to_numpy(dtype=float).std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    time_scale = float(train["time"].max() + 1.0)
    if time_scale <= 0:
        raise ValueError("Invalid training time scale")

    transformed = []
    for frame in [train, *other_frames]:
        new = frame.copy()
        new.loc[:, columns] = ((frame[columns].to_numpy(dtype=float) - mu) / sigma)
        new["time"] = (frame["time"].to_numpy(dtype=float) + 1.0) / time_scale
        transformed.append(new)
    return transformed, {"feature_mean": mu.tolist(), "feature_sd": sigma.tolist(),
                         "time_scale": time_scale, "feature_columns": columns}


def make_training_config(paper: PaperRunConfig, candidate: dict, fold: int) -> CVConfig:
    ew, ed = map(int, candidate["embedding_shape"])
    mw, md = map(int, candidate["monotonic_shape"])
    # Our PyHealth balancing-network width is tied to monotonic hidden_dim.
    # The original code exposes different networks; exact parity requires
    # a separate architecture audit.
    return CVConfig(
        n_splits=paper.folds,
        seed=paper.seed + 1000 * (fold - 1),
        epochs=paper.max_epochs,
        patience=paper.patience,
        min_delta=paper.min_delta,
        gradient_clip=paper.gradient_clip,
        weight_decay=paper.weight_decay,
        device=paper.device,
        learning_rate=float(candidate["learning_rate"]),
        batch_size=int(candidate["batch_size"]),
        dropout=float(candidate["dropout"]),
        embedding_dim=ew,
        embedding_layers=ed,
        hidden_dim=mw,
        monotonic_layers=md,
    )



def fit_neuralfg_paper_style(train_df: pd.DataFrame, stop_df: pd.DataFrame,
                            normalized_horizons: np.ndarray, config: CVConfig,
                            fold: int):
    """Fit PyHealth NeuralFG with authors' previous-loss patience convention.

    nfg.utilities.train_nfg resets patience on *any* validation loss decrease,
    and stops after consecutive non-improvements. It still restores the
    best-ever checkpoint, which can differ from the last epoch.
    """
    from pyhealth.models.neural_fg import NeuralFG
    from pyhealth.datasets import get_dataloader
    from .data import create_dataset, prepare_pyhealth_samples
    from .train import _run_epoch

    torch.manual_seed(config.seed + fold)
    np.random.seed(config.seed + fold)
    train_ds = create_dataset(prepare_pyhealth_samples(train_df))
    stop_ds = create_dataset(prepare_pyhealth_samples(stop_df))
    loader_train = get_dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
    loader_stop = get_dataloader(stop_ds, batch_size=config.batch_size, shuffle=False)
    model = NeuralFG(
        dataset=train_ds, num_risks=2,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        embedding_layers=config.embedding_layers,
        monotonic_layers=config.monotonic_layers,
        dropout=config.dropout,
        time_grid=torch.as_tensor(normalized_horizons, dtype=torch.float32),
    ).to(torch.device(config.device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, previous_loss, patience_count = float("inf"), float("inf"), 0
    best_epoch, best_state = 0, None
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, loader_train, optimizer, config.gradient_clip)
        val_loss = _run_epoch(model, loader_stop, None, config.gradient_clip)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"    Epoch {epoch:04d}: train={train_loss:.5f} stop={val_loss:.5f}", flush=True)

        # Match the authors' train_nfg() rule: reset patience after any
        # improvement relative to the immediately preceding validation loss.
        if val_loss < previous_loss:
            patience_count = 0
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience_count == config.patience:
            print(f"    Paper-style early stopping at epoch {epoch}", flush=True)
            break
        else:
            patience_count += 1
        previous_loss = val_loss

    if best_state is None:
        raise RuntimeError("No valid NeuralFG checkpoint was generated")
    model.load_state_dict(best_state)
    model.eval()
    details = {
        "best_epoch": int(best_epoch), "best_val_nll": float(best_loss),
        "epochs_run": len(history), "training_seed": config.seed + fold,
        "stopping_rule": "previous_validation_loss_patience",
    }
    return model, pd.DataFrame(history), details

def score_dev_nll(model, tune_df: pd.DataFrame, batch_size: int, clip: float) -> float:
    # NeuralFG density contains d(CIF)/dt; do NOT wrap this in no_grad().
    from pyhealth.datasets import get_dataloader
    from .data import create_dataset, prepare_pyhealth_samples
    from .train import _run_epoch
    loader = get_dataloader(
        create_dataset(prepare_pyhealth_samples(tune_df)),
        batch_size=batch_size, shuffle=False,
    )
    return float(_run_epoch(model, loader, None, clip))


def neural_predictions(model, test_scaled: pd.DataFrame,
                       horizons: np.ndarray, scale: float, batch_size: int = 512):
    cols = feature_columns(test_scaled)
    device = next(model.parameters()).device
    normalized_times = torch.as_tensor(
        (np.asarray(horizons, dtype=float) + 1.0) / scale,
        dtype=torch.float32, device=device,
    )
    arr = test_scaled[cols].to_numpy(dtype=np.float32)
    batches = []
    model.eval()
    for start in range(0, len(arr), batch_size):
        xs = torch.from_numpy(arr[start:start+batch_size]).to(device)
        with torch.no_grad():
            cif = model.predict_cif(features=xs, times=normalized_times)
        batches.append(cif.detach().cpu().numpy())
    pred = np.concatenate(batches, axis=0)
    if not np.isfinite(pred).all() or np.any(pred < -1e-6) or np.any(pred > 1 + 1e-6):
        raise ValueError("Invalid NeuralFG CIF")
    if np.any(pred.sum(axis=2) > 1 + 1e-6):
        raise ValueError("NeuralFG CIFs for competing risks sum to >1")
    return np.clip(pred, 0.0, 1.0)


def reference_rows(fold: int, name: str, prediction: np.ndarray,
                   training_df: pd.DataFrame, test_df: pd.DataFrame,
                   horizons: np.ndarray) -> list[dict]:
    km = CensoringKM.fit(training_df["time"].to_numpy(),
                         training_df["label"].to_numpy())
    t, e = test_df["time"].to_numpy(), test_df["label"].to_numpy()
    result = []
    for j, (tag, horizon) in enumerate(zip(QUANTILES, horizons)):
        for risk in (1, 2):
            p = prediction[:, j, risk - 1]
            result.append({
                "fold": fold, "model": name, "risk": risk, "quantile": tag,
                "time": float(horizon), "n_test": int(len(test_df)),
                "c_index_reference": reference_cindex(
                    t, e, p, risk=risk, horizon=float(horizon), censoring_km=km,
                ),
                "brier_reference": reference_brier(
                    t, e, p, risk=risk, horizon=float(horizon), censoring_km=km,
                ),
            })
    return result


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (model, risk, q), df in rows.groupby(["model", "risk", "quantile"], sort=True):
        for metric in ("c_index_reference", "brier_reference"):
            values = df[metric].to_numpy(dtype=float)
            output.append({
                "model": model, "risk": int(risk), "quantile": q,
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"),
                "n_folds": int(np.isfinite(values).sum()),
            })
    return pd.DataFrame(output)


def output_folder(config: PaperRunConfig) -> Path:
    root = (Path(config.output_root).expanduser().resolve()
            if config.output_root else
            Path(__file__).resolve().parent / "results" / "paper_reproduction")
    obj = asdict(config).copy()
    obj.pop("output_root", None)
    digest = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]
    ntag = "full" if config.max_samples is None else str(config.max_samples)
    return root / (f"{config.mode}_n{ntag}_k{config.folds}_trials"
                   f"{1 if config.mode == 'fixed' else config.trials}"
                   f"_ep{config.max_epochs}_pat{config.patience}_seed{config.seed}_cfg{digest}")


def run_paper_reproduction(df: pd.DataFrame, config: PaperRunConfig):
    if config.folds < 2 or config.max_epochs < 1 or config.patience < 1:
        raise ValueError("Invalid number of folds, epochs or patience")
    candidates = candidate_configs(config)
    if config.max_samples is not None:
        df = df.sample(n=min(config.max_samples, len(df)), random_state=config.seed).copy()
    out = output_folder(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    # Paper's original protocol: stratified outer folds with random_seed=0.
    outer = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    overall = []
    for fold, (dev_pos, test_pos) in enumerate(outer.split(df, df["label"]), start=1):
        outer_dev = df.iloc[dev_pos].copy()
        test = df.iloc[test_pos].copy()
        train, tune, stop = split_paper_fold(outer_dev, fold, config)
        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        index_record = pd.concat([
            pd.DataFrame({"row_index": frame.index, "split": key})
            for key, frame in (("train", train), ("tune", tune), ("stop", stop), ("test", test))
        ], ignore_index=True)
        index_record.to_csv(fold_dir / "split_indices.csv", index=False)
        if index_record.row_index.duplicated().any() or len(index_record) != len(df):
            raise AssertionError("Fold split overlap or missing observations")
        print(f"\nFold {fold}/{config.folds}: train={len(train)} tune={len(tune)} "
              f"stop={len(stop)} test={len(test)}", flush=True)

        horizons = train.loc[train["label"] > 0, "time"].quantile(
            [0.25, 0.50, 0.75]).to_numpy(dtype=float)
        transformed, scaler = transform_for_network(train, [tune, stop, test])
        train_scaled, tune_scaled, stop_scaled, test_scaled = transformed
        (fold_dir / "scaling.json").write_text(json.dumps(scaler, indent=2))
        normalized_horizons = (horizons + 1.0) / scaler["time_scale"]
        torch.manual_seed(config.seed + fold)
        np.random.seed(config.seed + fold)

        trial_results = []
        for trial, candidate in enumerate(candidates, start=1):
            trial_dir = fold_dir / f"trial_{trial:03d}"
            trial_dir.mkdir(exist_ok=True)
            trial_json = trial_dir / "selection.json"
            trial_pt = trial_dir / "best_model.pt"
            if trial_json.is_file() and trial_pt.is_file():
                details = json.loads(trial_json.read_text())
                if details.get("candidate") != json.loads(json.dumps(candidate)):
                    raise RuntimeError(f"Saved trial {trial} candidate mismatch")
                print(f"  Fold {fold}, trial {trial}: loading existing evaluation", flush=True)
                trial_results.append(details)
                continue

            neural_cfg = make_training_config(config, candidate, fold)
            print(f"  Fold {fold}, trial {trial}/{len(candidates)}: {candidate}", flush=True)
            model, hist, fit = fit_neuralfg_paper_style(
                train_scaled, stop_scaled, normalized_horizons,
                neural_cfg, fold=fold,
            )
            dev_loss = score_dev_nll(
                model, tune_scaled, neural_cfg.batch_size, neural_cfg.gradient_clip)
            hist.to_csv(trial_dir / "training_history.csv", index=False)
            # Save checkpoint before the trial metadata so a partial/crashed
            # evaluation isn't accidentally treated as a complete trial.
            torch.save(model.state_dict(), trial_pt)
            details = {"fold": fold, "trial": trial, "candidate": candidate,
                       "tune_nll": dev_loss, **fit}
            trial_json.write_text(json.dumps(details, indent=2))
            trial_results.append(details)
            del model

        search_table = pd.DataFrame([{**r, **{
            "embedding_width": r["candidate"]["embedding_shape"][0],
            "embedding_depth": r["candidate"]["embedding_shape"][1],
            "monotonic_width": r["candidate"]["monotonic_shape"][0],
            "monotonic_depth": r["candidate"]["monotonic_shape"][1],
            "learning_rate": r["candidate"]["learning_rate"],
            "batch_size": r["candidate"]["batch_size"],
            "dropout": r["candidate"]["dropout"],
        }} for r in trial_results])
        search_table.drop(columns=["candidate"]).to_csv(
            fold_dir / "hyperparameter_search.csv", index=False)
        winner = min(trial_results, key=lambda x: x["tune_nll"])
        (fold_dir / "best_hyperparameters.json").write_text(
            json.dumps(winner, indent=2))
        best_candidate = winner["candidate"]
        best_cfg = make_training_config(config, best_candidate, fold)

        # Rebuild architecture, restore best STOP-epoch for winner. Never refit
        # using outer test outcomes. Import happens here to keep split tests light.
        from pyhealth.models.neural_fg import NeuralFG
        from .data import create_dataset, prepare_pyhealth_samples
        train_ds = create_dataset(prepare_pyhealth_samples(train_scaled))
        model = NeuralFG(
            dataset=train_ds, num_risks=2,
            hidden_dim=best_cfg.hidden_dim,
            embedding_dim=best_cfg.embedding_dim,
            embedding_layers=best_cfg.embedding_layers,
            monotonic_layers=best_cfg.monotonic_layers,
            dropout=best_cfg.dropout,
            time_grid=torch.tensor(normalized_horizons, dtype=torch.float32),
        ).to(torch.device(config.device))
        checkpoint = fold_dir / f"trial_{int(winner['trial']):03d}" / "best_model.pt"
        try:
            state = torch.load(checkpoint, map_location=config.device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=config.device)
        model.load_state_dict(state)
        predictions = neural_predictions(model, test_scaled, horizons, scaler["time_scale"])
        rows = reference_rows(fold, "NeuralFG", predictions, train, test, horizons)

        # Custom Python FG is kept as a *separate* methodological baseline.
        fg = FineGrayBaseline(num_risks=2, maxiter=250, tolerance=1e-9).fit(train)
        fg.coefficients().to_csv(fold_dir / "finegray_coefficients.csv", index=False)
        fg.fit_summary().to_csv(fold_dir / "finegray_fit_summary.csv", index=False)
        fg_prediction = fg.predict_cif(test, horizons)
        rows.extend(reference_rows(fold, "Fine-Gray", fg_prediction, train, test, horizons))
        pd.DataFrame(rows).to_csv(fold_dir / "test_metrics_reference.csv", index=False)
        overall.extend(rows)
        pd.DataFrame(overall).to_csv(out / "fold_metrics_reference.csv", index=False)
        print(f"  Fold {fold} complete: best trial {winner['trial']}, "
              f"tune NLL={winner['tune_nll']:.5f}", flush=True)

    summary = summary_table(pd.DataFrame(overall))
    summary.to_csv(out / "cv_summary_reference.csv", index=False)
    print("\nPaper-aligned results (not bitwise CHIL):\n", summary.to_string(index=False))
    print("\nSaved to:", out)
    return summary, out
