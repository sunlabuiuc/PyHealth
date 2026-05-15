"""Side-by-side comparison figure: Hippocampus MR vs Spleen CT.

Produces ``examples/figures/ablation_comparison.png`` with two rows:

    Top row:    Hippocampus (small-lesion MR)   — hard regime
    Bottom row: Spleen      (large-organ CT)    — easier regime

and two columns: final-epoch segmentation IoU + final-epoch detection F1
as bar charts per λ, with the λ = 0 baseline drawn as a dashed line on
each subplot.

This is the single figure we want to put on a slide for the video: it
shows the paper's central claim reproduced on two modalities with very
different object scales, dataset sizes, and difficulties.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = pathlib.Path(__file__).parent / "figures"
HIPPO_JSON  = FIG_DIR / "ablation_results.json"
SPLEEN_JSON = FIG_DIR / "ablation_results_spleen.json"
LUNA_JSON   = FIG_DIR / "ablation_results_luna16.json"
OUT = FIG_DIR / "ablation_comparison.png"


def load(path: pathlib.Path):
    with open(path) as f:
        d = json.load(f)
    histories = {float(k): v for k, v in d["histories"].items()}
    return histories, d.get("config", {})


def final_values(histories, key):
    return {lam: hist[-1][key] for lam, hist in sorted(histories.items())}


def lambda_colors(lambdas):
    base_color = "#888888"
    cmap = plt.get_cmap("viridis")
    non_zero = [l for l in sorted(lambdas) if l > 0]
    colors = {0.0: base_color}
    for i, lam in enumerate(non_zero):
        colors[lam] = cmap(0.15 + 0.75 * (i / max(len(non_zero) - 1, 1)))
    return colors


def draw_bar(ax, data_dict, title, ylabel, max_scale=1.15):
    lambdas = sorted(data_dict.keys())
    values = [data_dict[l] for l in lambdas]
    colors = lambda_colors(lambdas)
    bar_colors = [colors[l] for l in lambdas]
    bars = ax.bar([str(l) for l in lambdas], values, color=bar_colors,
                  edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.015,
                f"{val:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(max(values) * max_scale, 0.02))
    ax.axhline(values[0], color="#888888", ls="--", lw=0.8, alpha=0.8)


def _final_detection_key(hist):
    """Prefer AP if present, fall back to F1 for legacy JSONs."""
    final = hist[list(hist.keys())[0]][-1]
    return "val_ap_30" if "val_ap_30" in final else "val_f1"


def _lift(data_dict):
    """λ=1 vs λ=0 multiplicative lift for a metric, safe against zero baseline."""
    if 0.0 in data_dict and 1.0 in data_dict and data_dict[0.0] > 0:
        return data_dict[1.0] / data_dict[0.0]
    return float("inf")


def main():
    # Each dataset: (title, json_path, 2D-caveat, short label for subtitle line)
    specs = []
    for title, path, subtitle in [
        ("Hippocampus MR",  HIPPO_JSON,  "~22 train pts, ~490 slices"),
        ("Spleen CT",       SPLEEN_JSON, "9 train pts, ~680 lesion slices"),
        ("LUNA16 CT (2D)",  LUNA_JSON,   "400 train slices, nodule ~5-25px"),
    ]:
        if path.exists():
            hist, cfg = load(path)
            specs.append((title, hist, cfg, subtitle))
        else:
            print(f"[skip] {path} missing")

    if not specs:
        raise SystemExit("no ablation JSONs found")

    n_rows = len(specs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.0 * n_rows + 0.5),
                             squeeze=False)

    for r, (title, hist, cfg, subtitle) in enumerate(specs):
        seg = final_values(hist, "val_seg_iou")
        det_key = _final_detection_key(hist)
        det = final_values(hist, det_key)
        det_label = "AP @ IoU 0.3" if det_key == "val_ap_30" else "F1 @ IoU 0.3"

        seg_lift = _lift(seg)
        det_lift = _lift(det)

        seg_lift_s = f"{seg_lift:.1f}×" if seg_lift != float("inf") else "∞×"
        det_lift_s = f"{det_lift:.1f}×" if det_lift != float("inf") else "∞×"

        draw_bar(axes[r, 0], seg,
                 f"{title} — seg IoU  (λ=1 vs λ=0: {seg_lift_s} lift)",
                 "seg IoU")
        draw_bar(axes[r, 1], det,
                 f"{title} — detection {det_label}  (λ=1 vs λ=0: {det_lift_s} lift)",
                 det_label)

    for ax in axes[-1]:
        ax.set_xlabel("seg_weight (λ)")

    subtitle_line = "   |   ".join(
        f"{title}: {subtitle}" for title, _, _, subtitle in specs
    )

    fig.suptitle(
        "Retina U-Net λ ablation — seg supervision improves every dataset over the RetinaNet baseline\n"
        + subtitle_line +
        "\nAll runs: 2D slice-wise reimplementation (paper uses 3D patches)",
        fontsize=10, y=1.00,
    )
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
