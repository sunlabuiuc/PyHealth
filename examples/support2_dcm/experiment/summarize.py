# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Summarize saved folds into RESULTS.md, CSV tables and diagnostic plots."""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402


def validation_table(records: list[dict[str, Any]]) -> list[str]:
    """Markdown rows: mean +- SD validation Brier per searched configuration.

    Args:
        records: Completed fold records (``fold_*.json``).

    Returns:
        Markdown table lines, one row per model family and configuration, with
        the number of folds that selected it.
    """
    rows = []
    for record in records:
        for entry in record["search"]:
            if entry["status"] != "ok":
                continue
            config = ", ".join(f"{k}={v}" for k, v in entry["config"].items())
            selected = record["selected"][entry["model"]]["config"] == entry["config"]
            rows.append(
                {
                    "model": entry["model"],
                    "config": config,
                    "brier": entry["mean_brier"],
                    "epochs": entry.get("audit", {}).get("epochs"),
                    "selected": selected,
                }
            )
    table = (
        pd.DataFrame(rows)
        .groupby(["model", "config"], sort=False)
        .agg(
            mean=("brier", "mean"),
            std=("brier", "std"),
            epochs=("epochs", "mean"),
            selected=("selected", "sum"),
        )
    )
    lines = [
        "| Model | Configuration | Validation mean IPCW Brier | Mean epochs "
        "| Folds selected |",
        "|---|---|---:|---:|---:|",
    ]
    for (model, config), row in table.iterrows():
        epochs = "" if pd.isna(row.epochs) else f"{row.epochs:.1f}"
        lines.append(
            f"| {model} | {config} | {row['mean']:.6f} ± {row['std']:.6f} "
            f"| {epochs} | {int(row.selected)} |"
        )
    return lines


def main(argv: Optional[list[str]] = None) -> None:
    """Read ``fold_*.json`` and write the summary files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="study output directory")
    parser.add_argument(
        "--output", help="where to write summaries; default: --results"
    )
    args = parser.parse_args(argv)
    root = Path(args.results)
    out = Path(args.output or args.results)
    out.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(path.read_text()) for path in sorted(root.glob("fold_*.json"))
    ]
    if len(records) != 5:
        raise RuntimeError(f"Expected all five completed folds; found {len(records)}")
    rows = []
    for record in records:
        for entry in record["evaluation"]:
            row = {
                k: entry[k] for k in ["model", "mask_rate", "mask_seed", "mean_brier"]
            }
            row["fold"] = record["fold"]
            row["mean_ipcw_concordance"] = np.mean(
                [v for v in entry["ipcw_concordance"] if v is not None]
            )
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "scores.csv", index=False)
    folded = (
        raw.groupby(["fold", "model", "mask_rate"])[
            ["mean_brier", "mean_ipcw_concordance"]
        ]
        .mean()
        .reset_index()
    )
    clean = folded[folded.mask_rate == 0].copy()
    clean.to_csv(out / "clean_per_fold.csv", index=False)
    summary = clean.groupby("model")[["mean_brier", "mean_ipcw_concordance"]].agg(
        ["mean", "std"]
    )
    summary.to_csv(out / "clean_summary.csv")
    clean_ref = clean.set_index(["fold", "model"]).mean_brier
    folded["brier_deterioration"] = [
        r.mean_brier - clean_ref.loc[(r.fold, r.model)] for r in folded.itertuples()
    ]
    folded.to_csv(out / "robustness_per_fold.csv", index=False)
    paired = []
    for rate in [0.1, 0.3]:
        for fold in range(5):
            group = folded[
                (folded.fold == fold) & (folded.mask_rate == rate)
            ].set_index("model")
            paired.append(
                {
                    "fold": fold,
                    "mask_rate": rate,
                    "plain_deterioration": float(
                        group.loc["dcm", "brier_deterioration"]
                    ),
                    "indicator_deterioration": float(
                        group.loc["dcm_indicators", "brier_deterioration"]
                    ),
                    "paired_delta": float(
                        group.loc["dcm_indicators", "brier_deterioration"]
                        - group.loc["dcm", "brier_deterioration"]
                    ),
                }
            )
    pair = pd.DataFrame(paired)
    pair.to_csv(out / "paired_differences.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    values = summary["mean_brier"]
    ax.bar(
        values.index,
        values["mean"],
        yerr=values["std"],
        capsize=4,
        color=["#75869a", "#447aab", "#bd7b43", "#699477"],
    )
    ax.set(
        ylabel="Mean IPCW Brier (lower is better)",
        title="Clean held-out data: mean ± SD across 5 folds",
    )
    fig.tight_layout()
    fig.savefig(out / "clean_brier.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ["cox", "rsf", "dcm", "dcm_indicators"]:
        s = (
            folded[folded.model == name]
            .groupby("mask_rate")
            .brier_deterioration.agg(["mean", "std"])
        )
        ax.errorbar(
            s.index * 100, s["mean"], yerr=s["std"], marker="o", capsize=3, label=name
        )
    ax.axhline(0, color="gray", lw=0.6)
    ax.legend(frameon=False)
    ax.set(
        xlabel="Additional masking of originally observed laboratory values (%)",
        ylabel="Change in mean IPCW Brier",
        title="Robustness: seeds averaged within each fold",
    )
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    bins = []
    # Representative fold 0 diagnostics; its exact training-derived times are labeled.
    for name in ["cox", "rsf", "dcm", "dcm_indicators"]:
        data = np.load(root / "predictions" / f"fold0_{name}_mask0_seed71.npz")
        for j, horizon in enumerate(data["horizons"]):
            points = []
            for b, ix in enumerate(
                np.array_split(np.argsort(data["survival"][:, j], kind="stable"), 5)
            ):
                t, s = kaplan_meier_estimator(data["event"][ix], data["time"][ix])
                if horizon > t[-1]:
                    continue
                pos = np.searchsorted(t, horizon, side="right") - 1
                observed = float(s[pos]) if pos >= 0 else 1.0
                predicted = float(data["survival"][ix, j].mean())
                points.append((predicted, observed))
                bins.append(
                    dict(
                        model=name,
                        fold=0,
                        horizon=float(horizon),
                        bin=b,
                        n=len(ix),
                        predicted_survival=predicted,
                        km_survival=observed,
                    )
                )
            a = np.asarray(points)
            axes[j].plot(a[:, 0], a[:, 1], marker="o", ms=3, label=name)
            axes[j].set_title(f"Fold 0: t = {horizon:g} days")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.7)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted survival")
    axes[0].set_ylabel("Kaplan–Meier survival within prediction bin")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "calibration_fold0.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(bins).to_csv(out / "calibration_bins.csv", index=False)
    failures = sum(len(r["failures"]) for r in records)
    warning_fits = sum(
        bool(e.get("audit", {}).get("warnings")) for r in records for e in r["search"]
    )
    repairs = sum(
        e.get("prediction_audit", {}).get("out_of_bounds_cells", 0)
        for r in records
        for e in r["evaluation"]
    )
    fits = sum(len(r["search"]) + len(r["selected"]) + 1 for r in records)
    lines = [
        "# SUPPORT2 reproduction results",
        "",
        "Five outer folds on all 9,105 SUPPORT2 records. Values are mean ± "
        "sample SD across folds (descriptive, not a confidence interval); the "
        "three mask seeds are averaged inside each fold. Lower Brier is better.",
        "",
        "## Held-out test results (clean data)",
        "",
        "| Model | Clean mean IPCW Brier | Mean IPCW concordance |",
        "|---|---:|---:|",
    ]
    for name, row in summary.iterrows():
        brier = row[("mean_brier", "mean")], row[("mean_brier", "std")]
        conc = (
            row[("mean_ipcw_concordance", "mean")],
            row[("mean_ipcw_concordance", "std")],
        )
        lines.append(
            f"| {name} | {brier[0]:.6f} ± {brier[1]:.6f} "
            f"| {conc[0]:.6f} ± {conc[1]:.6f} |"
        )
    lines += [
        "",
        "`dcm_indicators` is the selected DCM architecture refit with eight "
        "laboratory missing-value indicators. The indicators and the masking "
        "experiment below are an extension; they are not in the paper.",
        "",
        "## Validation search (hyperparameter ablation)",
        "",
        "Each fold selects one configuration per model family by mean IPCW "
        "Brier on a 20% inner validation split of its training data; the "
        "held-out fold is never used for selection. Mean ± SD over the five "
        "folds; DCM epochs are the early-stopped epoch counts.",
        "",
        *validation_table(records),
        "",
        "## Paired masking extension",
        "",
        "Additional MCAR masking of originally observed laboratory values in the "
        "test fold, without retraining. Primary comparison: indicator "
        "deterioration minus plain DCM deterioration (negative favors "
        "indicators). Descriptive over five fixed folds, not proof of benefit.",
        "",
        "| Additional mask | Plain deterioration | Indicator deterioration "
        "| Paired difference ± fold SD |",
        "|---|---:|---:|---:|",
    ]
    for rate, g in pair.groupby("mask_rate"):
        lines.append(
            f"| {rate:.0%} | {g.plain_deterioration.mean():.6f} "
            f"| {g.indicator_deterioration.mean():.6f} "
            f"| {g.paired_delta.mean():.6f} ± {g.paired_delta.std():.6f} |"
        )
    lines += [
        "",
        "## Audit and limits",
        "",
        f"{fits} completed protocol fits; {failures} failed search "
        f"configurations; {warning_fits} DCM search fits emitted warnings "
        "(including upstream numerical and dependency warnings). Held-out "
        f"evaluations had {repairs} out-of-bounds cells before the documented "
        "tiny projection. Each fold JSON records the exact warnings, epochs and "
        "settings; only these aggregates are included here.",
        "",
        "The authors' spline/posterior numerical handling is part of this "
        "reproduction, so warnings limit claims of exact agreement with the "
        "paper. Only SUPPORT2 is reproduced, with Cox/RSF controls; FLCHAIN, "
        "SEER, subgroup fairness and the full published tables are not.",
        "",
        "Calibration is a separate diagnostic: five equal-count prediction bins "
        "on held-out fold 0, Kaplan–Meier survival at its training-derived "
        "horizons. No calibration metric is substituted for Brier.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
