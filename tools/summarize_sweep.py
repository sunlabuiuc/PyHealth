"""Summarize a completed E2E sweep.

Reads metrics_history.json and predictions_*.csv from each experiment directory
and prints a results table + saves plots.

Usage:
    python tools/summarize_sweep.py --sweep-dir ~/Downloads/e2e_sweep
    python tools/summarize_sweep.py --sweep-dir ~/Downloads/e2e_sweep --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics(exp_dir: Path) -> dict | None:
    p = exp_dir / "metrics_history.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_log_metrics(exp_dir: Path) -> dict | None:
    """Fallback: parse log.txt for best val metrics when metrics_history.json absent."""
    log = exp_dir / "log.txt"
    if not log.exists():
        return None

    best = {}
    current_epoch = {}
    for raw in log.read_text().splitlines():
        # strip optional "YYYY-MM-DD HH:MM:SS " timestamp prefix
        line = raw.strip()
        parts = line.split(" ", 2)
        if len(parts) == 3 and len(parts[0]) == 10 and parts[0][4] == "-":
            line = parts[2].strip()
        # detect epoch boundary
        if "--- Eval epoch-" in line:
            current_epoch = {}
        for metric in ("pr_auc", "roc_auc", "accuracy"):
            if line.startswith(metric + ":"):
                try:
                    current_epoch[metric] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        if "New best pr_auc" in line:
            best = dict(current_epoch)
    return best if best else None


def parse_exp_name(name: str) -> tuple[str, int]:
    # e.g. "bottleneck_transformer_seed42"
    parts = name.rsplit("_seed", 1)
    return parts[0], int(parts[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--plot", action="store_true", help="Save convergence plots")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir).expanduser()
    if not sweep_dir.exists():
        raise SystemExit(f"Directory not found: {sweep_dir}")

    rows = []
    histories = {}

    for exp_dir in sorted(sweep_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        try:
            model, seed = parse_exp_name(exp_dir.name)
        except (ValueError, IndexError):
            continue

        hist = load_metrics(exp_dir)
        if hist:
            # hist is a flat list of epoch dicts with keys like val_pr_auc, train_loss, etc.
            valid = [e for e in hist if e.get("val_pr_auc") is not None]
            if valid:
                best_e = max(valid, key=lambda e: e.get("val_pr_auc", -1))
                best = {
                    "pr_auc": best_e.get("val_pr_auc", float("nan")),
                    "roc_auc": best_e.get("val_roc_auc", float("nan")),
                    "accuracy": best_e.get("val_accuracy", float("nan")),
                }
                nan_epochs = [e["epoch"] for e in hist if np.isnan(e.get("train_loss", 0))]
            else:
                best, nan_epochs = {}, []
            histories[exp_dir.name] = hist
        else:
            # fallback to log parsing
            best = load_log_metrics(exp_dir) or {}
            nan_epochs = []
            # check log for nan
            log = exp_dir / "log.txt"
            if log.exists():
                nan_epochs = [i for i, l in enumerate(log.read_text().splitlines()) if "loss: nan" in l]

        rows.append({
            "model": model,
            "seed": seed,
            "pr_auc": best.get("pr_auc", float("nan")),
            "roc_auc": best.get("roc_auc", float("nan")),
            "accuracy": best.get("accuracy", float("nan")),
            "nan_epochs": len(nan_epochs),
            "status": "NaN" if nan_epochs else ("OK" if best else "missing"),
        })

    if not rows:
        raise SystemExit("No experiment directories found.")

    # ---- Print summary table ------------------------------------
    preferred_order = [
        "mlp",
        "rnn",
        "transformer",
        "bottleneck_transformer",
        "ehrmamba",
        "jambaehr",
    ]
    seen_models = sorted({r["model"] for r in rows})
    MODELS = [m for m in preferred_order if m in seen_models] + [
        m for m in seen_models if m not in preferred_order
    ]
    SEEDS = sorted({r["seed"] for r in rows})

    row_map = {(r["model"], r["seed"]): r for r in rows}

    col_w = 28
    seed_w = 32

    header = f"{'Model':<{col_w}}" + "".join(f"{'seed' + str(s):<{seed_w}}" for s in SEEDS) + f"{'Mean PR-AUC':>12}"
    print("\n" + "=" * len(header))
    print("SWEEP RESULTS — Best Val PR-AUC (roc_auc / accuracy)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model in MODELS:
        model_rows = [row_map.get((model, s)) for s in SEEDS]
        pr_aucs = [r["pr_auc"] for r in model_rows if r and not np.isnan(r["pr_auc"])]
        mean_pr = np.mean(pr_aucs) if pr_aucs else float("nan")

        cells = []
        for r in model_rows:
            if r is None:
                cells.append(f"{'—':<{seed_w}}")
            elif r["status"] == "missing" or np.isnan(r["pr_auc"]):
                cells.append(f"{'missing':<{seed_w}}")
            else:
                marker = "* " if r["status"] == "NaN" else ""
                cell = f"{marker}{r['pr_auc']:.4f} ({r['roc_auc']:.4f}/{r['accuracy']:.4f})"
                cells.append(f"{cell:<{seed_w}}")

        mean_str = f"{mean_pr:.4f}" if not np.isnan(mean_pr) else "—"
        print(f"{model:<{col_w}}" + "".join(cells) + f"{mean_str:>12}")

    print("=" * len(header))

    # ---- Per-model mean ± std -----------------------------------
    print("\nMean ± Std PR-AUC across seeds:")
    for model in MODELS:
        vals = [row_map[(model, s)]["pr_auc"] for s in SEEDS
                if (model, s) in row_map and not np.isnan(row_map[(model, s)]["pr_auc"])]
        if vals:
            print(f"  {model:<35} {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")
        else:
            print(f"  {model:<35} no valid runs")

    # ---- NaN warnings -------------------------------------------
    nan_runs = [r for r in rows if r["status"] == "NaN"]
    if nan_runs:
        print(f"\n⚠  NaN runs detected ({len(nan_runs)}):")
        for r in nan_runs:
            print(f"   {r['model']}_seed{r['seed']}")

    # ---- Plots --------------------------------------------------
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed — skipping plots (pip install matplotlib)")
            return

        plot_dir = sweep_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        for model in MODELS:
            fig, ax = plt.subplots(figsize=(8, 4))
            plotted = False
            for seed in SEEDS:
                key = f"{model}_seed{seed}"
                hist = histories.get(key)
                if not hist:
                    continue
                pr_aucs = [e.get("val_pr_auc", float("nan")) for e in hist if "val_pr_auc" in e]
                ax.plot(pr_aucs, label=f"seed {seed}")
                plotted = True
            if plotted:
                ax.set_title(f"{model} — Val PR-AUC over epochs")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("PR-AUC")
                ax.legend()
                ax.grid(True, alpha=0.3)
                out = plot_dir / f"{model}_pr_auc.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved {out}")
            plt.close(fig)

        # Combined final-epoch bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(MODELS))
        width = 0.25
        for i, seed in enumerate(SEEDS):
            vals = []
            for model in MODELS:
                r = row_map.get((model, seed))
                vals.append(r["pr_auc"] if r and not np.isnan(r["pr_auc"]) else 0)
            ax.bar(x + i * width, vals, width, label=f"seed {seed}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(MODELS, rotation=15, ha="right")
        ax.set_ylabel("Best Val PR-AUC")
        ax.set_title("Model Comparison — Best Val PR-AUC")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        out = plot_dir / "model_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
