"""Audit and visualize paired NeuralFG vs Fine-Gray cross-validation results.

Place at: PyHealth/examples/neuralfg_cv_report.py
Run from the PyHealth repository root:
    python examples/neuralfg_cv_report.py
Or select a specific experiment:
    python examples/neuralfg_cv_report.py --results-dir examples/neuralfg/results/cross_validation/<experiment_folder>

No model retraining is performed. Uses only numpy, pandas, matplotlib.
Outputs are written to <experiment_folder>/report/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ("NeuralFG", "Fine-Gray")
QUANTILES = ("q25", "q50", "q75")
METRICS = ("c_index", "brier_paper", "brier_competing")
COMPARISON_COLS = ("c_index", "brier_paper", "brier_competing", "mean_predicted_cif")


def locate_results(supplied: str | None) -> Path:
    if supplied is not None:
        root = Path(supplied).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent / "neuralfg" / "results" / "cross_validation"
        if not root.is_dir():
            raise FileNotFoundError(f"Cross-validation results directory not found: {root}")
        candidates = sorted(p for p in root.glob("nfull_k5_*") if p.is_dir() and (p / "cv_summary.csv").exists())
        if len(candidates) != 1:
            formatted = "\n".join(f"  {p}" for p in candidates) or "  (none)"
            raise ValueError(
                "Expected exactly one completed full five-fold experiment, found "
                f"{len(candidates)}. Specify --results-dir explicitly.\n{formatted}"
            )
        root = candidates[0]
    for filename in ("config.json", "fold_metrics.csv", "cv_summary.csv"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Missing {root / filename}")
    return root


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def check_metrics_against_fold_files(root: Path, fold_data: pd.DataFrame, folds: list[int]) -> None:
    """Check that combined fold metrics match the separately saved model files."""
    for fold in folds:
        for model in MODELS:
            subfolder = "neuralfg" if model == "NeuralFG" else "finegray"
            file = root / f"fold_{fold}" / subfolder / "test_metrics.csv"
            require(file.is_file(), f"Missing model metrics: {file}")
            original = pd.read_csv(file).sort_values(["risk", "time"]).reset_index(drop=True)
            combined = (fold_data.loc[(fold_data["fold"] == fold) & (fold_data["model"] == model)]
                        .sort_values(["risk", "time"]).reset_index(drop=True))
            require(len(original) == len(combined), f"Row count mismatch: {file}")
            for field in ("risk", "time", *COMPARISON_COLS, "n_test"):
                require(field in original.columns and field in combined.columns,
                        f"Missing column {field} in {file} or fold_metrics.csv")
                require(np.allclose(original[field].to_numpy(dtype=float),
                                    combined[field].to_numpy(dtype=float), rtol=1e-8, atol=1e-10),
                        f"Data mismatch for {field}: {file}")


def check_split_membership(root: Path, folds: list[int], n_splits: int) -> dict:
    """Verify no train/val/test overlap and each observation is tested once."""
    all_test = []
    expected_total = None
    per_fold_sizes = []
    for fold in folds:
        path = root / f"fold_{fold}" / "split_indices.csv"
        require(path.is_file(), f"Missing split indices: {path}")
        splits = pd.read_csv(path)
        require(set(splits.columns) >= {"row_index", "split"}, f"Malformed split file {path}")
        require(set(splits.split) == {"train", "validation", "test"}, f"Incorrect splits in {path}")
        require(splits.row_index.is_unique, f"Overlapping train/validation/test rows in {path}")
        overall_ids = set(splits.row_index)
        if expected_total is None:
            expected_total = overall_ids
        require(overall_ids == expected_total, f"Different underlying observations in fold {fold}")
        counts = splits.split.value_counts().to_dict()
        per_fold_sizes.append({"fold": fold, **{s: int(counts[s]) for s in ("train", "validation", "test")}})
        all_test.extend(splits.loc[splits.split == "test", "row_index"].tolist())
    require(len(all_test) == len(expected_total), "Incorrect total held-out test observations")
    require(len(set(all_test)) == len(expected_total), "An observation appears in more than one outer test fold")
    require(len(folds) == n_splits, "Incorrect number of folds")
    return {"total_unique_observations": len(expected_total), "fold_sizes": per_fold_sizes}


def recompute_summary(fold_data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, risk, quantile), subset in fold_data.groupby(["model", "risk", "quantile"], sort=True):
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            require(np.all(np.isfinite(values)), f"Non-finite {metric} for {model} {risk} {quantile}")
            records.append({
                "model": model, "risk": int(risk), "quantile": quantile, "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "n_folds": len(values),
                "mean_evaluation_time": float(subset["time"].mean()),
            })
    return pd.DataFrame(records)


def audit_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    n_splits = int(config["n_splits"])
    require(n_splits >= 2, "At least two folds required")
    data = pd.read_csv(root / "fold_metrics.csv")
    summary = pd.read_csv(root / "cv_summary.csv")
    keys = ["fold", "model", "risk", "quantile"]
    expected_cols = set(keys) | {"time", "n_test", *METRICS}
    require(expected_cols <= set(data.columns), f"fold_metrics.csv missing {expected_cols - set(data.columns)}")
    require(not data.duplicated(keys).any(), "Duplicate metric rows for the same fold/model/risk/horizon")
    folds = sorted(data.fold.unique().tolist())
    require(folds == list(range(1, n_splits + 1)), f"Unexpected fold identifiers: {folds}")
    risks = list(range(1, int(config["num_risks"]) + 1))
    require(set(data.model) == set(MODELS), f"Model labels do not match {MODELS}")
    require(set(data["quantile"]) == set(QUANTILES), "Unexpected quantile labels")
    require(set(data.risk) == set(risks), "Unexpected risk identifiers")
    require(len(data) == n_splits * 2 * len(risks) * 3, "Missing model/risk/quantile/fold combinations")
    require((data.groupby(["model", "risk", "quantile"]).size() == n_splits).all(),
            "A model/risk/quantile combination is missing one or more folds")
    require(np.all(data[list(METRICS)].apply(pd.to_numeric, errors="coerce").notna()),
            "Metric columns contain NaN or non-numeric data")
    require(((data["c_index"] >= 0) & (data["c_index"] <= 1)).all(), "C-index out of range")
    require(((data["brier_paper"] >= 0) & (data["brier_competing"] >= 0)).all(),
            "Negative Brier score detected")
    check_metrics_against_fold_files(root, data, folds)
    split_info = check_split_membership(root, folds, n_splits)
    computed = recompute_summary(data)
    summary_cols = ["model", "risk", "quantile", "metric"]
    comparison = summary.merge(computed, on=summary_cols, how="outer", suffixes=("_saved", "_recomputed"), validate="one_to_one", indicator=True)
    require((comparison["_merge"] == "both").all(), "Mismatch in summary metric keys")
    for field in ("mean", "std", "mean_evaluation_time"):
        require(np.allclose(comparison[f"{field}_saved"], comparison[f"{field}_recomputed"],
                            atol=1e-8, rtol=1e-8), f"Incorrect {field} in cv_summary.csv")
    require((comparison.n_folds_saved == comparison.n_folds_recomputed).all(), "Incorrect fold counts")
    details = {"status": "PASS", "n_splits": n_splits, "models": list(MODELS),
               "metrics_verified": len(data), "summary_rows_verified": len(summary),
               **split_info,
               "checks": ["all fold/model/risk/quantile combinations present",
                          "per-model test CSVs match fold_metrics.csv",
                          "each observation appears in exactly one outer test fold",
                          "train/validation/test are disjoint within each fold",
                          "cv_summary.csv means and sample SDs match per-fold values"]}
    return data, computed, details


def paired_differences(fold_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (fold, risk, quantile), group in fold_data.groupby(["fold", "risk", "quantile"], sort=True):
        n = group[group["model"] == "NeuralFG"].iloc[0]
        f = group[group["model"] == "Fine-Gray"].iloc[0]
        for metric in METRICS:
            # Positive always means NeuralFG is better.
            improvement = float(n[metric] - f[metric]) if metric == "c_index" else float(f[metric] - n[metric])
            rows.append({"fold": fold, "risk": risk, "quantile": quantile,
                         "metric": metric, "neuralfg": float(n[metric]),
                         "finegray": float(f[metric]), "improvement": improvement})
    differences = pd.DataFrame(rows)
    summary = (differences.groupby(["risk", "quantile", "metric"], sort=True)
               .agg(mean_improvement=("improvement", "mean"),
                    sd_improvement=("improvement", "std"),
                    min_improvement=("improvement", "min"),
                    max_improvement=("improvement", "max"),
                    n_folds=("improvement", "size"))
               .reset_index())
    return differences, summary


def create_plots(summary: pd.DataFrame, output_dir: Path, risks: list[int]) -> list[str]:
    figures = []
    for risk in risks:
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(7, 4.7))
            for model in MODELS:
                subset = summary.loc[(summary["model"] == model) & (summary["risk"] == risk) & (summary["metric"] == metric)]
                values = subset.set_index("quantile").reindex(QUANTILES)
                require(values["mean"].notna().all(), f"Missing plot values: {model}, risk {risk}, {metric}")
                ax.errorbar(range(3), values["mean"], yerr=values["std"], label=model,
                            fmt="o-", capsize=4, linewidth=1.8)
            ax.set_xticks(range(3), ["25%", "50%", "75%"])
            ax.set_xlabel("Training event-time quantile (per fold)")
            names = {"c_index": "Time-dependent C-index", "brier_paper": "Paper-style IPCW Brier score",
                     "brier_competing": "Competing-risk IPCW Brier score"}
            ax.set_ylabel(names[metric])
            ax.set_title(f"Risk {risk}: {names[metric]} (5-fold CV)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            filename = f"risk{risk}_{metric}.png"
            fig.savefig(output_dir / filename, dpi=200)
            plt.close(fig)
            figures.append(filename)
    return figures


def create_markdown(summary: pd.DataFrame, improvements: pd.DataFrame,
                    audit: dict, files: list[str]) -> str:
    out = ["# NeuralFG vs. Fine-Gray — Five-fold cross-validation",
           "", f"Audit: **{audit['status']}**. {audit['total_unique_observations']:,} observations, "
           f"{audit['n_splits']} outer folds, {audit['metrics_verified']} held-out metric rows.",
           "", "Each entry is **mean ± sample SD** over outer test folds.", ""]
    for metric in METRICS:
        label = {"c_index": "C-index (higher is better)", "brier_paper": "Paper-style Brier (lower is better)",
                 "brier_competing": "Competing-event-aware Brier (lower is better)"}[metric]
        out.extend([f"## {label}", "", "| Risk | Model | q25 | q50 | q75 |",
                    "|---|---|---:|---:|---:|"])
        for risk in sorted(summary["risk"].unique()):
            for model in MODELS:
                vals = summary[(summary["risk"] == risk) & (summary["model"] == model) & (summary["metric"] == metric)].set_index("quantile")
                row = [f"{vals.loc[q, 'mean']:.4f} ± {vals.loc[q, 'std']:.4f}" for q in QUANTILES]
                out.append(f"| {risk} | {model} | " + " | ".join(row) + " |")
        out.append("")
    out.extend(["## Paired differences (positive favors NeuralFG)", "",
                "| Risk | Metric | q25 | q50 | q75 |", "|---|---|---:|---:|---:|"])
    for risk in sorted(improvements["risk"].unique()):
        for metric in METRICS:
            subset = improvements[(improvements["risk"] == risk) & (improvements["metric"] == metric)].set_index("quantile")
            row = [f"{subset.loc[q, 'mean_improvement']:+.4f} ± {subset.loc[q, 'sd_improvement']:.4f}" for q in QUANTILES]
            out.append(f"| {risk} | {metric} | " + " | ".join(row) + " |")
    out.extend(["", "## Validation notes", "",
                "- Identical held-out folds were used for both models. Inner validation rows were reserved for NeuralFG early stopping.",
                "- Evaluation horizons are training-only event-time quantiles and can differ slightly by fold.",
                "- These checks verify saved-file consistency and partitioning, not correctness of the statistical metric definitions.",
                "- Reference validation against the paper's original metric code and an established Fine-Gray estimator remains recommended.",
                "- Hyperparameters were selected during earlier experiments on the same dataset; results are not fully independent of that tuning.",
                "", "## Plots", ""])
    out.extend([f"![{filename}]({filename})" for filename in files])
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", help="Path to one completed cross-validation experiment")
    args = parser.parse_args()
    root = locate_results(args.results_dir)
    print(f"Checking experiment: {root}")
    fold_data, summary, audit = audit_results(root)
    details, improvement_summary = paired_differences(fold_data)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "audit_results.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    details.to_csv(report_dir / "paired_fold_differences.csv", index=False)
    improvement_summary.to_csv(report_dir / "paired_improvement_summary.csv", index=False)
    plots = create_plots(summary, report_dir, sorted(summary["risk"].unique()))
    (report_dir / "cv_results_summary.md").write_text(
        create_markdown(summary, improvement_summary, audit, plots), encoding="utf-8")
    print(f"PASS — verified {audit['metrics_verified']} rows and {audit['summary_rows_verified']} summary entries")
    print(f"PASS — each of {audit['total_unique_observations']:,} observations is held out exactly once")
    print("\nPaired model improvements (+ means NeuralFG better):")
    print(improvement_summary[["risk", "quantile", "metric", "mean_improvement", "sd_improvement"]].to_string(index=False))
    print(f"\nReports, CSVs, and {len(plots)} figures saved to: {report_dir}")


if __name__ == "__main__":
    main()
