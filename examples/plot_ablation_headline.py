"""Generate the headline ablation figure from ``ablation_results.json``.

Produces ``examples/figures/ablation_headline.png``, a two-panel layout
tuned to communicate the paper's central claim clearly:

    [ left, large ]  Val segmentation IoU per epoch, one line per λ.
                     The λ = 0 baseline (RetinaNet only) is flat near
                     0.09 across all epochs because the seg branch
                     receives no direct supervision; every λ > 0 line
                     climbs into the 0.28–0.32 band.

    [ right, small ] Bar chart of final-epoch values:
                     - Top:    seg IoU  (the clean result)
                     - Bottom: detection F1  (directional, noisy)

The three-panel per-epoch plot produced by the ablation script itself
(``ablation_results.png``) remains as the raw view; this figure is the
one intended for the video / slides.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = pathlib.Path(__file__).parent / "figures" / "ablation_results.json"
DEFAULT_OUT = pathlib.Path(__file__).parent / "figures" / "ablation_headline.png"


def load_histories(path: pathlib.Path):
    with open(path) as f:
        d = json.load(f)
    return {float(k): v for k, v in d["histories"].items()}, d.get("config", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--title", type=str, default="MSD Task04 Hippocampus")
    args = parser.parse_args()
    RESULTS = args.results
    OUT = args.out

    histories, config = load_histories(RESULTS)
    lambdas = sorted(histories.keys())

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1, 1],
                          hspace=0.6, wspace=0.35)

    ax_iou = fig.add_subplot(gs[:, 0])
    ax_bar_iou = fig.add_subplot(gs[0, 1:])
    ax_bar_ap = fig.add_subplot(gs[1, 1:])
    ax_bar_f1 = fig.add_subplot(gs[2, 1:])

    # Color palette — baseline gets a muted gray, others a sequential blue ramp.
    base_color = "#888888"
    cmap = plt.get_cmap("viridis")
    lam_colors = {}
    non_zero = [l for l in lambdas if l > 0]
    for i, lam in enumerate(non_zero):
        lam_colors[lam] = cmap(0.15 + 0.75 * (i / max(len(non_zero) - 1, 1)))
    lam_colors[0.0] = base_color

    # --- Left: seg IoU curves ---
    for lam in lambdas:
        hist = histories[lam]
        xs = [r["epoch"] + 1 for r in hist]
        ys = [r["val_seg_iou"] for r in hist]
        label = f"λ = {lam}" + (" (RetinaNet baseline)" if lam == 0 else "")
        ls = "--" if lam == 0 else "-"
        lw = 1.8 if lam == 0 else 2.2
        ax_iou.plot(xs, ys, ls=ls, lw=lw, color=lam_colors[lam], label=label, marker="o", markersize=3)

    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("Val segmentation IoU")
    ax_iou.set_title(
        "Segmentation supervision lifts seg IoU ~3× over the RetinaNet baseline",
        fontsize=11, loc="left",
    )
    ax_iou.grid(alpha=0.3)
    ax_iou.legend(loc="lower right", fontsize=9)
    ax_iou.set_ylim(0, max(0.4, max(r["val_seg_iou"] for h in histories.values() for r in h) * 1.15))

    # --- Top right: final seg IoU bar ---
    finals_iou = [histories[l][-1]["val_seg_iou"] for l in lambdas]
    bar_colors = [lam_colors[l] for l in lambdas]
    bars1 = ax_bar_iou.bar([str(l) for l in lambdas], finals_iou, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax_bar_iou.set_ylabel("Final seg IoU")
    ax_bar_iou.set_title("Final epoch — segmentation IoU", fontsize=10, loc="left")
    ax_bar_iou.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars1, finals_iou):
        ax_bar_iou.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                        f"{val:.3f}", ha="center", fontsize=8)
    ax_bar_iou.set_ylim(0, max(finals_iou) * 1.2)
    # Baseline dashed reference line
    ax_bar_iou.axhline(finals_iou[0], color=base_color, ls="--", lw=0.8, alpha=0.8)

    # --- Middle right: final AP@0.3 bar (threshold-independent) ---
    finals_ap = [histories[l][-1].get("val_ap_30", 0.0) for l in lambdas]
    bars_ap = ax_bar_ap.bar([str(l) for l in lambdas], finals_ap,
                            color=bar_colors, edgecolor="black", linewidth=0.6)
    ax_bar_ap.set_ylabel("Final AP @ IoU 0.3")
    ax_bar_ap.set_title("Final epoch — detection AP@0.3 (threshold-free)",
                        fontsize=10, loc="left")
    ax_bar_ap.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars_ap, finals_ap):
        ax_bar_ap.text(bar.get_x() + bar.get_width() / 2, val + max(finals_ap + [1e-3]) * 0.02,
                       f"{val:.3f}", ha="center", fontsize=8)
    ax_bar_ap.set_ylim(0, max(finals_ap + [1e-3]) * 1.25)
    ax_bar_ap.axhline(finals_ap[0], color=base_color, ls="--", lw=0.8, alpha=0.8)

    # --- Bottom right: final F1 bar (for reference; sensitive to threshold) ---
    finals_f1 = [histories[l][-1]["val_f1"] for l in lambdas]
    bars2 = ax_bar_f1.bar([str(l) for l in lambdas], finals_f1, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax_bar_f1.set_xlabel("seg_weight (λ)")
    ax_bar_f1.set_ylabel("Final F1 @ IoU 0.3")
    ax_bar_f1.set_title("Final epoch — detection F1 (score-threshold-dependent)", fontsize=10, loc="left")
    ax_bar_f1.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars2, finals_f1):
        ax_bar_f1.text(bar.get_x() + bar.get_width() / 2, val + max(finals_f1 + [1e-3]) * 0.02,
                       f"{val:.3f}", ha="center", fontsize=8)
    ax_bar_f1.set_ylim(0, max(finals_f1 + [1e-3]) * 1.3)
    ax_bar_f1.axhline(finals_f1[0], color=base_color, ls="--", lw=0.8, alpha=0.8)

    n_epochs = config.get("num_epochs", len(next(iter(histories.values()))))
    n_train = len(config.get("train_ids", []))
    n_val = len(config.get("val_ids", []))

    fig.suptitle(
        f"Retina U-Net λ ablation — {args.title} "
        f"({n_train} train patients / {n_val} val patient, {n_epochs} epochs, resnet18-FPN)",
        fontsize=11, y=1.00,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")

    # Also write a small text summary so callers can paste it.
    txt = OUT.with_suffix(".txt")
    lines = ["Final-epoch val metrics\n", "-" * 48 + "\n"]
    lines.append(
        f"{'λ':>5} | {'AP@0.3':>7} | {'AP@0.5':>7} | {'seg IoU':>8} | "
        f"{'F1':>6} | {'precision':>9} | {'recall':>6}\n"
    )
    for lam in lambdas:
        row = histories[lam][-1]
        lines.append(
            f"{lam:>5.2f} | {row.get('val_ap_30', 0):>7.3f} | "
            f"{row.get('val_ap_50', 0):>7.3f} | "
            f"{row['val_seg_iou']:>8.3f} | {row['val_f1']:>6.3f} | "
            f"{row['val_precision']:>9.3f} | {row['val_recall']:>6.3f}\n"
        )
    lines.append("\n")
    lines.append(f"train patients: {config.get('train_ids', [])}\n")
    lines.append(f"val patients:   {config.get('val_ids', [])}\n")
    lines.append(f"device:         {config.get('device', '?')}  epochs: {n_epochs}\n")
    txt.write_text("".join(lines))
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
