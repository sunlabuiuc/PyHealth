"""Evaluate existing NeuralFG censoring checkpoints on the same latent test outcomes.

CSE6250 Neural Fine-Gray project, evaluation only (no NeuralFG retraining).
Run from the PyHealth repository root with this file saved in examples/.

Use original train/tune/stop/test splits, each scenario's training-set feature
standardization and time normalization, and the saved per-fold horizons. True
latent outcomes are used ONLY for the held-out evaluation, never for fitting.

Fine-Gray is optionally refit because its saved coefficients alone cannot
reconstruct the baseline subdistribution hazards needed for CIF predictions.
This does not refit NeuralFG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuralfg.data import load_data, create_dataset, prepare_pyhealth_samples
from neuralfg.finegray_baseline import FineGrayBaseline
from neuralfg.metrics import CensoringKM
from neuralfg.metrics_reference import reference_cindex, reference_brier
from neuralfg.paper_reproduction import neural_predictions, transform_for_network
from deephit_extension_experiments import (
    generate_censoring_scenarios,
    load_splits,
    validate_latent_outcomes,
)

CONDITIONS = ((25, 0.25), (50, 0.50), (75, 0.75))
LABELS = ("q25", "q50", "q75")


def load_neuralfg_model(
    train_scaled: pd.DataFrame,
    candidate: dict,
    horizons: np.ndarray,
    time_scale: float,
    checkpoint: Path,
):
    """Recreate the SAME PyHealth architecture used by fit_neuralfg_paper_style."""
    from pyhealth.models.neural_fg import NeuralFG

    ew, ed = (int(x) for x in candidate["embedding_shape"])
    mw, md = (int(x) for x in candidate["monotonic_shape"])
    train_ds = create_dataset(prepare_pyhealth_samples(train_scaled))
    model = NeuralFG(
        dataset=train_ds,
        num_risks=2,
        hidden_dim=mw,
        embedding_dim=ew,
        embedding_layers=ed,
        monotonic_layers=md,
        dropout=float(candidate["dropout"]),
        time_grid=torch.as_tensor(
            (horizons + 1.0) / time_scale, dtype=torch.float32
        ),
    )
    # Only accept state dictionaries, not arbitrary pickled model objects.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state_dict checkpoint: {checkpoint}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def latent_metric_rows(
    *, fold: int, condition: str, model_name: str,
    prediction: np.ndarray, test_latent: pd.DataFrame,
    horizons: np.ndarray, no_censor_km: CensoringKM,
) -> list[dict]:
    """Cause-specific, horizon-truncated concordance and uncensored Brier.

    For competing risks, reference_cindex still includes controls that had a
    competing event earlier. With all true event outcomes observed, G(t) = 1
    and the reference metric reduces to an unweighted concordance measure.
    The Brier target is 1 if a risk event happened by t, 0 otherwise.
    """
    t = test_latent["time"].to_numpy(dtype=float)
    e = test_latent["label"].to_numpy(dtype=int)
    if prediction.shape != (len(test_latent), len(horizons), 2):
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    if not np.isfinite(prediction).all() or np.any(prediction < -1e-7) or np.any(prediction > 1+1e-7):
        raise ValueError("Invalid CIF predictions")
    output = []
    for q, horizon in enumerate(horizons):
        for risk in (1, 2):
            score = prediction[:, q, risk - 1]
            cind = reference_cindex(
                t, e, score, risk=risk, horizon=float(horizon),
                censoring_km=no_censor_km,
            )
            brier = reference_brier(
                t, e, score, risk=risk, horizon=float(horizon),
                censoring_km=no_censor_km,
            )
            # Independently verify the uncensored Brier reduces to MSE.
            true_status = ((t <= horizon) & (e == risk)).astype(float)
            direct_brier = float(np.mean((true_status - score)**2))
            if not np.isclose(brier, direct_brier, atol=1e-10, rtol=1e-10):
                raise AssertionError("Uncensored Brier does not equal direct MSE")
            if not np.isfinite(cind) or not np.isfinite(brier):
                raise ValueError(f"Nonfinite metric: fold={fold}, {condition}, risk={risk}, q={q}")
            output.append({
                "fold": fold, "condition": condition,
                "censoring_target": int(condition.split("_")[1])/100.0,
                "model": model_name, "risk": risk, "quantile": LABELS[q],
                "time": float(horizon), "n_test": int(len(t)),
                "c_index": float(cind), "brier": float(brier),
            })
    return output


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Unweighted fold means with sample SD (five outer folds)."""
    if frame.empty:
        raise ValueError("No metrics to summarize")
    result = frame.groupby(
        ["condition", "censoring_target", "model", "risk", "quantile"],
        as_index=False,
    ).agg(
        mean_c_index=("c_index", "mean"),
        sd_c_index=("c_index", "std"),
        mean_brier=("brier", "mean"),
        sd_brier=("brier", "std"),
        n_folds=("fold", "nunique"),
    )
    return result.sort_values(["model", "risk", "quantile", "censoring_target"])


def paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-fold differences relative to 25%-censored training."""
    base = frame[frame["condition"] == "censor_25"].set_index(
        ["fold", "model", "risk", "quantile"]
    )
    out = []
    for condition in ("censor_50", "censor_75"):
        other = frame[frame["condition"] == condition].set_index(
            ["fold", "model", "risk", "quantile"]
        )
        if set(base.index) != set(other.index):
            raise ValueError(f"Misaligned fold/risk/horizon comparisons: {condition}")
        for key in base.index:
            fold, model, risk, quantile = key
            out.append({
                "fold": fold, "model": model, "risk": risk,
                "quantile": quantile, "comparison": f"{condition}_minus_censor_25",
                "delta_c_index": other.loc[key, "c_index"] - base.loc[key, "c_index"],
                "delta_brier": other.loc[key, "brier"] - base.loc[key, "brier"],
            })
    paired = pd.DataFrame(out)
    return paired.groupby(
        ["model", "risk", "quantile", "comparison"], as_index=False
    ).agg(
        mean_delta_c_index=("delta_c_index", "mean"),
        sd_delta_c_index=("delta_c_index", "std"),
        mean_delta_brier=("delta_brier", "mean"),
        sd_delta_brier=("delta_brier", "std"),
        n_folds=("fold", "nunique"),
    )


def save_plots(summary: pd.DataFrame, outdir: Path) -> None:
    """One figure per metric and risk; q50 primary analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for risk in (1, 2):
        for metric, ylabel in (("c_index", "Uncensored C-index (higher better)"),
                              ("brier", "Uncensored Brier score (lower better)")):
            fig, ax = plt.subplots(figsize=(7, 4.8))
            for model_name in sorted(summary["model"].unique()):
                series = summary[
                    (summary["risk"] == risk) & (summary["quantile"] == "q50") &
                    (summary["model"] == model_name)
                ].sort_values("censoring_target")
                if series.empty:
                    continue
                ax.errorbar(
                    series["censoring_target"].to_numpy() * 100,
                    series[f"mean_{metric}"],
                    yerr=series[f"sd_{metric}"].fillna(0),
                    marker="o", capsize=4, label=model_name,
                )
            ax.set_xticks([25, 50, 75])
            ax.set_xlabel("Censoring in training data (%)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Risk {risk}: common latent test outcomes, q50")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / f"latent_risk{risk}_{metric}_q50.png", dpi=180)
            plt.close(fig)

    # A separate figure for all three horizons, for each risk and metric.
    for risk in (1, 2):
        for metric, ylabel in (("c_index", "Uncensored C-index"),
                              ("brier", "Uncensored Brier score")):
            fig, ax = plt.subplots(figsize=(7, 4.8))
            subset = summary[(summary["risk"] == risk) & (summary["model"] == "NeuralFG")]
            for quantile in LABELS:
                s = subset[subset["quantile"] == quantile].sort_values("censoring_target")
                ax.errorbar(s["censoring_target"].to_numpy() * 100,
                            s[f"mean_{metric}"], yerr=s[f"sd_{metric}"].fillna(0),
                            marker="o", capsize=3, label=quantile)
            ax.set_xticks([25, 50, 75])
            ax.set_xlabel("Censoring in training data (%)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"NeuralFG Risk {risk}: three horizons, latent test")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / f"latent_neuralfg_risk{risk}_{metric}_allhorizons.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Completed emb50x3_mono25x2_..._ep500_... directory")
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="Original 100-trial results folder, if moved")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--neural-only", action="store_true",
                        help="Skip Fine-Gray baseline refits to finish faster")
    args = parser.parse_args()

    root = args.results_dir.expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.exists():
        parser.error(f"Missing {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("experiment") != "censoring":
        parser.error("Expected a censoring experiment directory")
    split_dir = (
        args.split_dir.expanduser().resolve() if args.split_dir else
        Path(config["split_source"]).expanduser().resolve()
    )
    output = (args.output_dir or (root / "latent_evaluation")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    original = load_data()
    validate_latent_outcomes(original)
    generated = generate_censoring_scenarios(original, seed=int(config["censor_seed"]))
    cols = ["time", "label"] + [f"feature{i}" for i in range(1, 13)]
    all_rows = []

    for fold in range(1, args.folds + 1):
        indices = load_splits(split_dir, fold, original)
        # Held-out latent outcomes: exactly the same test patients across regimes.
        test_latent = original.loc[indices["test"], cols].copy()
        test_latent.loc[:, "time"] = original.loc[indices["test"], "true_time"].to_numpy(float)
        test_latent.loc[:, "label"] = original.loc[indices["test"], "true_label"].to_numpy(int)
        assert (test_latent["label"] > 0).all(), "True labels must be event causes"
        censor_free_km = CensoringKM.fit(test_latent["time"], test_latent["label"])
        if not np.allclose(censor_free_km.survival(test_latent["time"].to_numpy()), 1.0):
            raise AssertionError("Latent test set is not fully uncensored")

        expected_horizons = None
        for percent, rate in CONDITIONS:
            name = f"censor_{percent}"
            path = root / f"fold_{fold}" / name
            info_path = path / "training_summary.json"
            checkpoint = path / "best_model.pt"
            if not info_path.exists() or not checkpoint.exists():
                raise FileNotFoundError(f"Missing saved run files: {path}")
            info = json.loads(info_path.read_text())
            horizons = np.array(info["evaluation_times"], dtype=float)
            if len(horizons) != 3 or np.any(np.diff(horizons) <= 0):
                raise ValueError(f"Invalid saved horizons: {horizons}")
            if expected_horizons is None:
                expected_horizons = horizons
                print(f"\nFold {fold}: same test n={len(test_latent)}, "
                      f"saved horizons={horizons.tolist()}", flush=True)
            elif not np.array_equal(expected_horizons, horizons):
                raise ValueError(f"Different evaluation horizons among conditions: fold {fold}")

            observed, actual, _ = generated[rate]
            train = observed.loc[indices["train"], cols].copy()
            (train_scaled, test_scaled), scales = transform_for_network(
                train, [test_latent]
            )
            model = load_neuralfg_model(train_scaled, info["candidate"],
                                        horizons, float(scales["time_scale"]), checkpoint)
            neural_prob = neural_predictions(
                model, test_scaled, horizons, float(scales["time_scale"])
            )
            all_rows.extend(latent_metric_rows(
                fold=fold, condition=name, model_name="NeuralFG",
                prediction=neural_prob, test_latent=test_latent,
                horizons=horizons, no_censor_km=censor_free_km,
            ))
            print(f"  {name}: NeuralFG evaluated (actual overall censoring={actual:.1%})",
                  flush=True)
            del model

            if not args.neural_only:
                # Coefficients CSV alone cannot reproduce Fine-Gray CIF predictions.
                # Refit baseline on SAME observed training data; no NeuralFG retraining.
                fg = FineGrayBaseline(
                    num_risks=2, maxiter=250, tolerance=1e-9
                ).fit(train)
                fg_prob = fg.predict_cif(test_latent, horizons)
                all_rows.extend(latent_metric_rows(
                    fold=fold, condition=name, model_name="Fine-Gray",
                    prediction=fg_prob, test_latent=test_latent,
                    horizons=horizons, no_censor_km=censor_free_km,
                ))
                saved_coef = pd.read_csv(path / "finegray_coefficients.csv")
                now_coef = fg.coefficients()
                if not np.allclose(
                    saved_coef[["risk_1", "risk_2"]].to_numpy(),
                    now_coef[["risk_1", "risk_2"]].to_numpy(),
                    atol=2e-5, rtol=1e-4,
                ):
                    raise AssertionError(
                        f"Fine-Gray refit differs from original saved coefficients: {path}"
                    )
                print(f"  {name}: Fine-Gray refit and evaluated", flush=True)

            # Save partial results in case evaluation is interrupted.
            pd.DataFrame(all_rows).to_csv(output / "latent_fold_metrics.csv", index=False)

    frame = pd.DataFrame(all_rows)
    summary = summarize(frame)
    changes = paired_differences(frame)
    frame.to_csv(output / "latent_fold_metrics.csv", index=False)
    summary.to_csv(output / "latent_cv_summary.csv", index=False)
    changes.to_csv(output / "latent_paired_changes.csv", index=False)

    # Include observed-data evaluation for context, but don't treat its
    # differences from latent evaluation as an isolated causal effect.
    observed_summary_path = root / "experiment_summary.csv"
    if observed_summary_path.exists():
        observed_summary = pd.read_csv(observed_summary_path).copy()
        observed_summary["censoring_target"] = (
            observed_summary["condition"].str.split("_").str[-1].astype(float) / 100.0
        )
        observed_summary["evaluation"] = "observed_IPCW"
        summary_compare = summary.copy()
        summary_compare["evaluation"] = "latent_uncensored"
        common = list(summary_compare.columns)
        pd.concat([summary_compare, observed_summary[common]], ignore_index=True).to_csv(
            output / "observed_vs_latent_summary.csv", index=False
        )

    save_plots(summary, output)
    print("\nLatent, five-fold means ± SD (q50):")
    print(summary.loc[summary["quantile"] == "q50"].round(4).to_string(index=False))
    print("\nPaired effects relative to 25% training censoring (q50):")
    print(changes.loc[changes["quantile"] == "q50"].round(4).to_string(index=False))
    print("\nSaved evaluation-only results to:", output)


if __name__ == "__main__":
    main()
