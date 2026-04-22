"""λ ablation: Retina U-Net vs. RetinaNet-only, varying seg_weight.

This script answers the proposal's two open training-time questions in
one sweep:

1. **Baseline comparison** — ``seg_weight=0`` disables segmentation
   supervision entirely, reducing the model to vanilla RetinaNet.
2. **Segmentation-loss weight ablation** — ``seg_weight ∈ {0, 0.5, 1.0, 2.0}``
   sweeps the λ knob described in the paper.

For each λ value we train a fresh model on the MSD Task04 Hippocampus
patients laid out at ``examples/data/hippocampus/`` (leave-one-patient-out
cross-val is used for the validation split). After training, we report
detection F1 @ IoU 0.5 and binary segmentation IoU on the held-out
patient, plus save a JSON with the full per-epoch history and a
matplotlib figure.

Run locally (slow; ~10-20 min on CPU for the defaults):

    PYTHONPATH=. python examples/retina_unet_seg_weight_ablation.py

Colab / GPU usage: see ``examples/retina_unet_hippocampus_colab.ipynb``.
Override defaults via env vars::

    RUN_EPOCHS=2 RUN_LAMBDAS=0,1 python examples/retina_unet_seg_weight_ablation.py
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets.retina_unet_ct_dataset import RetinaUNetCTDataset
from pyhealth.models.retina_unet import RetinaUNet
from pyhealth.models.retina_unet_training import (
    RetinaUNetTorchDataset,
    collate_fn,
    evaluate,
    train_one_epoch,
)


DEFAULT_DATA_ROOT = Path(__file__).parent / "data" / "hippocampus"
DEFAULT_OUT_JSON = Path(__file__).parent / "figures" / "ablation_results.json"
DEFAULT_OUT_PNG = Path(__file__).parent / "figures" / "ablation_results.png"


def _parse_lambdas(env_val: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in env_val.split(",") if x.strip())


def _load_patient_volumes(
    root: Path, patient_ids: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    volumes, masks = {}, {}
    for pid in patient_ids:
        volumes[pid] = np.load(root / pid / "volume.npy")
        masks[pid] = np.load(root / pid / "mask.npy")
    return volumes, masks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    data_root: Path, batch_size: int, seed: int,
    hu_window: Optional[Tuple[float, float]] = None,
) -> Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]:
    all_patients = sorted(
        p.name for p in data_root.iterdir() if (p / "volume.npy").exists()
    )
    if len(all_patients) < 2:
        raise SystemExit(
            f"Need at least 2 patients at {data_root}; found {all_patients}"
        )

    # Simple deterministic split: last patient is val, rest are train.
    train_ids = all_patients[:-1]
    val_ids = all_patients[-1:]

    train_vols, train_masks = _load_patient_volumes(data_root, train_ids)
    val_vols, val_masks = _load_patient_volumes(data_root, val_ids)

    train_ct = RetinaUNetCTDataset(
        volumes=train_vols,
        masks=train_masks,
        skip_empty_slices=True,
        hu_window=hu_window,
    )
    val_ct = RetinaUNetCTDataset(
        volumes=val_vols,
        masks=val_masks,
        skip_empty_slices=True,
        hu_window=hu_window,
    )

    train_ds = RetinaUNetTorchDataset(train_ct)
    val_ds = RetinaUNetTorchDataset(val_ct)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    return train_loader, val_loader, (train_ids, val_ids)


def train_one_setting(
    seg_weight: float,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    seed: int,
    min_size: int,
    max_size: int,
    anchor_sizes,
    eval_iou: float,
    eval_score_thresh: float,
    seg_pos_weight: Optional[float] = None,
) -> List[Dict[str, float]]:
    set_seed(seed)
    model = RetinaUNet(
        num_classes=2,
        seg_num_classes=1,
        in_channels=1,
        backbone_name="resnet18",
        min_size=min_size,
        max_size=max_size,
        seg_weight=seg_weight,
        seg_pos_weight=seg_pos_weight,
        anchor_sizes=anchor_sizes,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    for epoch in range(num_epochs):
        train_stats = train_one_epoch(model, train_loader, opt, device, grad_clip=10.0)
        val_metrics = evaluate(
            model, val_loader, device,
            iou_threshold=eval_iou,
            score_threshold=eval_score_thresh,
        )
        row = {
            "seg_weight": seg_weight,
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"  [λ={seg_weight:.2f}] ep {epoch+1:>2}/{num_epochs}: "
            f"train_total={train_stats.get('loss_total', 0):.3f}  "
            f"val_ap30={val_metrics.get('ap_30', 0):.3f}  "
            f"val_f1={val_metrics['f1']:.3f}  "
            f"val_seg_iou={val_metrics['seg_iou']:.3f}  "
            f"max_score={val_metrics['max_score']:.3f}"
        )
    return history


def plot_history(
    all_histories: Dict[float, List[Dict[str, float]]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # 1) Training loss_total vs epoch
    for lam, hist in all_histories.items():
        xs = [h["epoch"] for h in hist]
        ys = [h.get("train_loss_total", 0.0) for h in hist]
        axes[0].plot(xs, ys, marker="o", label=f"λ={lam}")
    axes[0].set_title("Train loss_total")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    # 2) Val F1 vs epoch
    for lam, hist in all_histories.items():
        xs = [h["epoch"] for h in hist]
        ys = [h["val_f1"] for h in hist]
        axes[1].plot(xs, ys, marker="o", label=f"λ={lam}")
    axes[1].set_title("Val detection F1 @ IoU 0.5")
    axes[1].set_xlabel("epoch")

    # 3) Val seg IoU vs epoch
    for lam, hist in all_histories.items():
        xs = [h["epoch"] for h in hist]
        ys = [h["val_seg_iou"] for h in hist]
        axes[2].plot(xs, ys, marker="o", label=f"λ={lam}")
    axes[2].set_title("Val binary seg IoU")
    axes[2].set_xlabel("epoch")

    fig.suptitle(
        "Retina U-Net λ ablation — MSD Task04 Hippocampus (LOO val)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    num_epochs = int(os.environ.get("RUN_EPOCHS", "10"))
    batch_size = int(os.environ.get("RUN_BATCH_SIZE", "4"))
    lr = float(os.environ.get("RUN_LR", "1e-4"))
    seg_weights = _parse_lambdas(os.environ.get("RUN_LAMBDAS", "0,0.5,1.0,2.0"))
    seed = int(os.environ.get("RUN_SEED", "42"))
    min_size = int(os.environ.get("RUN_MIN_SIZE", "64"))
    max_size = int(os.environ.get("RUN_MAX_SIZE", "64"))
    eval_iou = float(os.environ.get("RUN_EVAL_IOU", "0.3"))
    eval_score_thresh = float(os.environ.get("RUN_SCORE_THRESH", "0.01"))
    anchors_env = os.environ.get("RUN_ANCHORS", "4,8,16,32,64")
    anchor_sizes = tuple((int(s),) for s in anchors_env.split(","))

    pos_w_env = os.environ.get("RUN_SEG_POS_WEIGHT", "").strip()
    seg_pos_weight = float(pos_w_env) if pos_w_env else None

    data_root = Path(os.environ.get("RUN_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    hu_env = os.environ.get("RUN_HU_WINDOW", "").strip()
    hu_window = tuple(float(x) for x in hu_env.split(",")) if hu_env else None
    if hu_window is not None and len(hu_window) != 2:
        raise SystemExit("RUN_HU_WINDOW must be 'low,high'")

    out_tag = os.environ.get("RUN_OUT_TAG", "").strip()
    out_json = DEFAULT_OUT_JSON.with_name(
        f"ablation_results{'_' + out_tag if out_tag else ''}.json"
    )
    out_png = DEFAULT_OUT_PNG.with_name(
        f"ablation_results{'_' + out_tag if out_tag else ''}.png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data root: {data_root}   HU window: {hu_window}")
    print(f"Running λ sweep over {seg_weights} for {num_epochs} epochs, batch={batch_size}")
    print(f"Config: min/max_size={min_size}/{max_size}  anchors={anchor_sizes}  "
          f"eval_iou={eval_iou}  eval_score_thresh={eval_score_thresh}  "
          f"seg_pos_weight={seg_pos_weight}")

    train_loader, val_loader, (train_ids, val_ids) = build_loaders(
        data_root, batch_size=batch_size, seed=seed, hu_window=hu_window,
    )
    print(f"Train patients: {train_ids}   Val patients: {val_ids}")
    print(f"Train slices: {len(train_loader.dataset)}   Val slices: {len(val_loader.dataset)}")

    all_histories: Dict[float, List[Dict[str, float]]] = {}
    for lam in seg_weights:
        print(f"\n== seg_weight = {lam} ==")
        all_histories[lam] = train_one_setting(
            lam, train_loader, val_loader, device, num_epochs, lr, seed,
            min_size=min_size, max_size=max_size, anchor_sizes=anchor_sizes,
            eval_iou=eval_iou, eval_score_thresh=eval_score_thresh,
            seg_pos_weight=seg_pos_weight,
        )

    print("\n=== Final val metrics ===")
    header = (
        f"{'λ':>6} | {'AP@0.3':>7} | {'AP@0.5':>7} | "
        f"{'F1':>6} | {'precision':>9} | {'recall':>6} | {'seg IoU':>7}"
    )
    print(header)
    print("-" * len(header))
    for lam, hist in all_histories.items():
        final = hist[-1]
        print(
            f"{lam:>6.2f} | {final.get('val_ap_30', 0):>7.3f} | "
            f"{final.get('val_ap_50', 0):>7.3f} | "
            f"{final['val_f1']:>6.3f} | "
            f"{final['val_precision']:>9.3f} | {final['val_recall']:>6.3f} | "
            f"{final['val_seg_iou']:>7.3f}"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": {
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "device": str(device),
                    "data_root": str(data_root),
                    "hu_window": list(hu_window) if hu_window else None,
                    "min_size": min_size,
                    "max_size": max_size,
                    "anchor_sizes": list(anchor_sizes),
                    "eval_iou": eval_iou,
                    "eval_score_thresh": eval_score_thresh,
                    "seg_pos_weight": seg_pos_weight,
                },
                "histories": {str(k): v for k, v in all_histories.items()},
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    plot_history(all_histories, out_png)


if __name__ == "__main__":
    main()
