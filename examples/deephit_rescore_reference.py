"""Re-evaluate saved five-fold CV checkpoints using original authors' metrics.

Run from the PyHealth repository root:

    python examples/deephit_rescore_reference.py \
        --results-dir examples/neuralfg/results/cross_validation/<full_run_folder>

IMPORTANT: NeuralFG is NOT retrained. Its saved state_dict is loaded and
predictions are regenerated. Fine-Gray *must be refitted* on the recorded
training fold, because original cross_validation.py saved coefficients but
not the baseline subdistribution hazard used for CIF prediction.

Original fold_metrics.csv, cv_summary.csv, and report/ are NOT modified.
New metrics are placed under <experiment>/reference_review/.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuralfg.cross_validation import CVConfig, make_cv_splits
from neuralfg.data import create_dataset, load_data, prepare_pyhealth_samples
from neuralfg.finegray_baseline import FineGrayBaseline
from neuralfg.metrics import CensoringKM
from neuralfg.metrics_reference import reference_brier, reference_cindex
from pyhealth.models.neural_fg import NeuralFG


QUANTILES = ("q25", "q50", "q75")


def saved_neuralfg_predictions(train_df, test_df, times, config, checkpoint):
    """Reconstruct architecture and load trained weights, no optimizer/training."""
    train_ds = create_dataset(prepare_pyhealth_samples(train_df))
    model = NeuralFG(
        dataset=train_ds,
        num_risks=config.num_risks,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        embedding_layers=config.embedding_layers,
        monotonic_layers=config.monotonic_layers,
        dropout=config.dropout,
        time_grid=torch.tensor(times, dtype=torch.float32),
    ).cpu()
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch without weights_only keyword
        weights = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(weights, strict=True)
    model.eval()

    feature_columns = sorted(
        (column for column in train_df if column.startswith("feature")
         and column[len("feature"):].isdigit()),
        key=lambda c: int(c[len("feature"):]),
    )
    values = test_df[feature_columns].to_numpy(dtype=np.float32)
    tt = torch.tensor(times, dtype=torch.float32)
    batches = []
    for start in range(0, len(values), 512):
        xx = torch.from_numpy(values[start:start + 512])
        with torch.no_grad():
            pp = model.predict_cif(features=xx, times=tt)
        batches.append(pp.detach().cpu().numpy())
    result = np.concatenate(batches, axis=0)
    expected = (len(test_df), len(times), config.num_risks)
    if result.shape != expected:
        raise ValueError(f"CIF shape {result.shape} != {expected}")
    if not np.isfinite(result).all() or (result < -1e-6).any() or (result > 1 + 1e-6).any():
        raise ValueError("NeuralFG CIF has invalid probabilities")
    return np.clip(result, 0, 1)


def _verify_fold_membership(fold_dir, train_df, val_df, test_df):
    saved = pd.read_csv(fold_dir / "split_indices.csv")
    for split, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        old = set(saved.loc[saved["split"] == split, "row_index"].tolist())
        if old != set(frame.index):
            raise ValueError(
                f"{fold_dir.name}: recorded {split} membership differs from "
                "the reconstructed original data split; refusing to rescore"
            )


def _evaluation_times_from_existing(fold_metrics, fold):
    subset = fold_metrics.loc[
        (fold_metrics["fold"] == fold) &
        (fold_metrics["model"] == "NeuralFG") &
        (fold_metrics["risk"] == 1)
    ]
    horizon_by_quantile = subset.set_index("quantile")["time"].to_dict()
    if set(QUANTILES) - horizon_by_quantile.keys():
        raise ValueError(f"Fold {fold} missing evaluation horizons")
    return np.array([float(horizon_by_quantile[q]) for q in QUANTILES])


def _summarize(metrics):
    rows = []
    for (model,risk,quantile), group in metrics.groupby(["model", "risk", "quantile"]):
        for name in ("c_index_reference", "brier_reference"):
            arr = group[name].to_numpy(dtype=float)
            vals = arr[np.isfinite(arr)]
            rows.append({
                "model": model,
                "risk": int(risk),
                "quantile": quantile,
                "metric": name,
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals)>1 else float("nan"),
                "n_folds": len(vals),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="Existing saved full CV experiment directory")
    args = parser.parse_args()
    experiment_dir = args.results_dir.resolve()
    cfg_path = experiment_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing original config: {cfg_path}")
    with cfg_path.open() as fh:
        saved_config = json.load(fh)
    supported = {f.name for f in fields(CVConfig)}
    config = CVConfig(**{k:v for k,v in saved_config.items() if k in supported})
    existing_metrics = pd.read_csv(experiment_dir / "fold_metrics.csv")

    original_data = load_data()
    if config.max_samples is not None:
        original_data = original_data.sample(
            n=min(config.max_samples, len(original_data)),random_state=config.seed
        ).copy()
    output_dir = experiment_dir / "reference_review"
    output_dir.mkdir(exist_ok=True)
    all_rows = []
    for fold, train_df, val_df, test_df in make_cv_splits(original_data, config):
        print(f"\nFold {fold}/{config.n_splits}: reconstructing saved predictions", flush=True)
        fold_dir = experiment_dir / f"fold_{fold}"
        _verify_fold_membership(fold_dir, train_df, val_df, test_df)
        times = _evaluation_times_from_existing(existing_metrics, fold)
        km = CensoringKM.fit(train_df["time"].to_numpy(),train_df["label"].to_numpy())

        neural_prediction = saved_neuralfg_predictions(
            train_df, test_df, times, config, fold_dir / "neuralfg" / "best_model.pt"
        )
        print("  NeuralFG checkpoint loaded; no neural training", flush=True)
        fg_model = FineGrayBaseline(
            num_risks=config.num_risks,
            maxiter=config.fg_maxiter,
            tolerance=config.fg_tolerance,
        ).fit(train_df)
        # Model was not serialized in prior CV, so use a deterministic refit.
        old_coefs = pd.read_csv(fold_dir / "finegray" / "coefficients.csv")
        new_coefs = fg_model.coefficients()
        for risk in range(1, config.num_risks + 1):
            key = f"risk_{risk}"
            np.testing.assert_allclose(
                new_coefs[key], old_coefs[key], rtol=1e-5, atol=1e-6,
                err_msg=f"FineGray coefficients in fold {fold} differ from saved run",
            )
        fg_prediction = fg_model.predict_cif(test_df, times)
        print("  Fine-Gray deterministic refit verified against saved coefficients",flush=True)

        t_test = test_df["time"].to_numpy(dtype=float)
        e_test = test_df["label"].to_numpy(dtype=int)
        for model_name, cif in (("NeuralFG", neural_prediction), ("Fine-Gray", fg_prediction)):
            for j,(quantile,horizon) in enumerate(zip(QUANTILES,times)):
                for risk in range(1,config.num_risks+1):
                    probabilities = cif[:,j,risk-1]
                    all_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "risk": risk,
                        "quantile": quantile,
                        "time": float(horizon),
                        "c_index_reference": reference_cindex(
                            t_test,e_test,probabilities,risk=risk,
                            horizon=float(horizon),censoring_km=km,
                        ),
                        "brier_reference": reference_brier(
                            t_test,e_test,probabilities,risk=risk,
                            horizon=float(horizon),censoring_km=km,
                        ),
                    })
        pd.DataFrame(all_rows).to_csv(output_dir / "fold_metrics_reference.csv",index=False)
        print("  Reference-style metrics saved", flush=True)

    result = pd.DataFrame(all_rows)
    summary = _summarize(result)
    summary.to_csv(output_dir / "cv_summary_reference.csv",index=False)
    print("\nReference-style five-fold comparison:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Compare reference scores against corresponding previously saved numbers.
    old_cols = existing_metrics[["fold","model","risk","quantile","c_index","brier_competing"]]
    comparison = result.merge(
        old_cols, on=["fold","model","risk","quantile"], validate="one_to_one"
    )
    comparison["delta_c_index"] = comparison["c_index_reference"] - comparison["c_index"]
    comparison["delta_brier"] = comparison["brier_reference"] - comparison["brier_competing"]
    comparison.to_csv(output_dir / "comparison_to_original.csv",index=False)
    print("\nMean absolute metric changes vs existing results:")
    print(comparison.groupby("model")[["delta_c_index","delta_brier"]].agg(
        lambda col: float(np.mean(np.abs(col)))
    ).to_string())
    print("\nNew results saved to:",output_dir)
    print("Your original cross-validation and report files remain unchanged.")


if __name__ == "__main__":
    main()
