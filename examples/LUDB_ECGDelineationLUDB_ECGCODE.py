"""
ECG-CODE training example for ECG delineation on LUDB or QTDB.

This script demonstrates how to:
1) load LUDB or QTDB with the modern BaseDataset API
2) build ECG delineation samples with masks
3) train ECG-CODE with a manual epoch loop
4) track epoch-wise loss, accuracy, and f1_micro
5) plot loss/accuracy/f1_micro over epochs

Notes
-----
ECG-CODE produces interval-level outputs shaped [B, N, 3, 3]:
- axis -2 (size 3): wave classes (P, QRS, T)
- axis -1 (size 3): (confidence, start, end)

For accuracy/f1_micro tracking in this example, we evaluate only interval-level
presence confidence:
- y_true: target confidence channel (0/1)
- y_pred: predicted confidence channel thresholded by --pred-conf-threshold
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from pyhealth.datasets import LUDBDataset, QTDBDataset, get_dataloader, split_by_patient
from pyhealth.models import ECGCODE
from pyhealth.tasks import ECGDelineationLUDB, ECGDelineationQTDB


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ECG-CODE on ECG delineation samples (LUDB or QTDB) with "
            "epoch-wise loss/accuracy/f1_micro tracking and plotting."
        )
    )

    # Dataset/task args
    parser.add_argument(
        "--dataset",
        type=str,
        default="ludb",
        choices=["ludb", "qtdb"],
        help="Dataset backend to use.",
    )
    parser.add_argument("--root", type=str, required=True, help="Dataset root path.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from PhysioNet if local files are missing.",
    )
    parser.add_argument("--dev", action="store_true", help="Enable dev mode.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional cache directory for dataset/task artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for task processing.",
    )

    # Delineation sample construction
    pulse_group = parser.add_mutually_exclusive_group()
    pulse_group.add_argument(
        "--split-by-pulse",
        dest="split_by_pulse",
        action="store_true",
        help="Use pulse-level windows.",
    )
    pulse_group.add_argument(
        "--no-split-by-pulse",
        dest="split_by_pulse",
        action="store_false",
        help="Use full-record samples.",
    )
    parser.set_defaults(split_by_pulse=True)

    parser.add_argument(
        "--pulse-window",
        type=int,
        default=250,
        help=(
            "Half-window around QRS peak in samples when pulse split is enabled "
            "(250 -> 500-sample pulse)."
        ),
    )
    parser.add_argument(
        "--keep-incomplete-pulses",
        action="store_true",
        help=(
            "If set, do NOT filter pulse samples missing P/QRS/T annotations "
            "(effective only with --split-by-pulse)."
        ),
    )

    # Model args
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier.",
    )
    parser.add_argument(
        "--interval-size",
        type=int,
        default=16,
        help="Interval size used by ECG-CODE for interval-level predictions.",
    )
    parser.add_argument(
        "--conf-tolerance",
        type=float,
        default=0.25,
        help="Confidence tolerance threshold in ECG-CODE loss.",
    )
    parser.add_argument(
        "--se-tolerance",
        type=float,
        default=0.15,
        help="Start/end tolerance threshold in ECG-CODE loss.",
    )

    # Training args
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cuda:0" or "cpu". Default: auto.',
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Metric/plot args
    parser.add_argument(
        "--pred-conf-threshold",
        type=float,
        default=0.5,
        help="Threshold on predicted interval confidence for metric computation.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help=(
            "Output path for epoch metrics plot "
            "(default: outputs/ecg_code/<dataset>_interval<interval_size>_epoch_metrics.png)."
        ),
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="",
        help="Optional custom title for the epoch metrics plot.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="DPI for the saved plot.",
    )

    args = parser.parse_args()

    if args.pulse_window <= 0:
        parser.error("--pulse-window must be positive.")
    if args.interval_size <= 0:
        parser.error("--interval-size must be positive.")
    if args.conf_tolerance < 0:
        parser.error("--conf-tolerance must be non-negative.")
    if args.se_tolerance < 0:
        parser.error("--se-tolerance must be non-negative.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if not (0.0 <= args.pred_conf_threshold <= 1.0):
        parser.error("--pred-conf-threshold must be within [0, 1].")
    if args.plot_dpi <= 0:
        parser.error("--plot-dpi must be positive.")

    return args


def build_dataset_and_task(
    args: argparse.Namespace,
) -> Tuple[
    Union[LUDBDataset, QTDBDataset],
    Union[ECGDelineationLUDB, ECGDelineationQTDB],
]:
    dataset_kwargs = {
        "root": args.root,
        "dev": args.dev,
        "num_workers": args.num_workers,
        "download": args.download,
    }
    if args.cache_dir:
        dataset_kwargs["cache_dir"] = args.cache_dir

    common_task_kwargs = {
        "split_by_pulse": args.split_by_pulse,
        "pulse_window": args.pulse_window,
        "filter_incomplete_pulses": not args.keep_incomplete_pulses,
    }

    if args.dataset == "ludb":
        base_dataset = LUDBDataset(**dataset_kwargs)
        task = ECGDelineationLUDB(**common_task_kwargs)
    else:
        base_dataset = QTDBDataset(**dataset_kwargs)
        task = ECGDelineationQTDB(**common_task_kwargs)

    return base_dataset, task


def _batch_binary_stats_from_interval_conf(
    y_true_interval: np.ndarray,
    y_prob_interval: np.ndarray,
    pred_conf_threshold: float,
) -> Dict[str, float]:
    """
    Computes TP/FP/FN and accuracy counts from interval confidence channel.

    Inputs are expected as [B, N, 3, 3], where channel index 0 is confidence.
    """
    if y_true_interval.ndim != 4 or y_true_interval.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected y_true shape [B, N, 3, 3], got {y_true_interval.shape}."
        )
    if y_prob_interval.ndim != 4 or y_prob_interval.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected y_prob shape [B, N, 3, 3], got {y_prob_interval.shape}."
        )

    y_true_conf = y_true_interval[..., 0] >= 0.5
    y_pred_conf = y_prob_interval[..., 0] >= pred_conf_threshold

    tp = float(np.logical_and(y_pred_conf, y_true_conf).sum())
    fp = float(np.logical_and(y_pred_conf, np.logical_not(y_true_conf)).sum())
    fn = float(np.logical_and(np.logical_not(y_pred_conf), y_true_conf).sum())

    correct = float((y_pred_conf == y_true_conf).sum())
    total = float(y_true_conf.size)

    return {"tp": tp, "fp": fp, "fn": fn, "correct": correct, "total": total}


def _finalize_binary_metrics(
    stats: Dict[str, float], eps: float = 1e-8
) -> Dict[str, float]:
    accuracy = stats["correct"] / (stats["total"] + eps)
    f1_micro = (2.0 * stats["tp"]) / (
        2.0 * stats["tp"] + stats["fp"] + stats["fn"] + eps
    )
    return {"accuracy": float(accuracy), "f1_micro": float(f1_micro)}


def evaluate_epoch(
    model: ECGCODE,
    dataloader,
    pred_conf_threshold: float,
) -> Dict[str, float]:
    model.eval()

    loss_all = []
    agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "correct": 0.0, "total": 0.0}

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Eval", leave=False):
            output = model(**data)
            loss_all.append(float(output["loss"].item()))

            y_true = output["y_true"].detach().cpu().numpy()
            y_prob = output["y_prob"].detach().cpu().numpy()
            batch_stats = _batch_binary_stats_from_interval_conf(
                y_true_interval=y_true,
                y_prob_interval=y_prob,
                pred_conf_threshold=pred_conf_threshold,
            )
            for k in agg:
                agg[k] += batch_stats[k]

    metrics = _finalize_binary_metrics(agg)
    metrics["loss"] = float(np.mean(loss_all)) if len(loss_all) > 0 else float("nan")
    return metrics


def plot_epoch_metrics(
    history: Dict[str, list],
    output_path: str,
    title: str,
    dpi: int = 150,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # F1 Micro
    axes[2].plot(epochs, history["train_f1_micro"], marker="o", label="Train")
    axes[2].plot(epochs, history["val_f1_micro"], marker="o", label="Val")
    axes[2].set_title("F1 Micro")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved epoch-metrics plot: {output_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1) Build base dataset + delineation task
    base_dataset, task = build_dataset_and_task(args)
    base_dataset.stats()
    if hasattr(base_dataset, "info"):
        base_dataset.info()

    # 2) Build sample dataset
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)
    print(f"Number of delineation samples: {len(sample_dataset)}")
    if len(sample_dataset) == 0:
        if args.split_by_pulse:
            raise RuntimeError(
                "No samples were generated. Try --keep-incomplete-pulses, "
                "increase --pulse-window, or use --no-split-by-pulse."
            )
        raise RuntimeError(
            "No samples were generated for full-record delineation. "
            "Check dataset root and annotations."
        )

    # 3) Split + dataloaders
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 4) Model
    model = ECGCODE(
        dataset=sample_dataset,
        signal_key="signal",
        mask_key="mask",
        width_mult=args.width_mult,
        interval_size=args.interval_size,
        conf_tolerance=args.conf_tolerance,
        se_tolerance=args.se_tolerance,
    )
    model.to(device)

    # 5) Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 6) Train loop with epoch-wise metrics
    history: Dict[str, list] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_micro": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_micro": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = []
        agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "correct": 0.0, "total": 0.0}

        for data in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            optimizer.zero_grad()
            output = model(**data)
            loss = output["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss.item()))
            y_true = output["y_true"].detach().cpu().numpy()
            y_prob = output["y_prob"].detach().cpu().numpy()

            batch_stats = _batch_binary_stats_from_interval_conf(
                y_true_interval=y_true,
                y_prob_interval=y_prob,
                pred_conf_threshold=args.pred_conf_threshold,
            )
            for k in agg:
                agg[k] += batch_stats[k]

        train_metrics = _finalize_binary_metrics(agg)
        train_metrics["loss"] = (
            float(np.mean(epoch_loss)) if len(epoch_loss) > 0 else float("nan")
        )

        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            pred_conf_threshold=args.pred_conf_threshold,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1_micro"].append(train_metrics["f1_micro"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_micro"].append(val_metrics["f1_micro"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f}, "
            f"train_acc={train_metrics['accuracy']:.6f}, "
            f"train_f1_micro={train_metrics['f1_micro']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}, "
            f"val_acc={val_metrics['accuracy']:.6f}, "
            f"val_f1_micro={val_metrics['f1_micro']:.6f}"
        )

    # 7) Final test metrics
    test_metrics = evaluate_epoch(
        model=model,
        dataloader=test_loader,
        pred_conf_threshold=args.pred_conf_threshold,
    )
    print("Final test metrics:")
    print(f"loss: {test_metrics['loss']:.6f}")
    print(f"accuracy: {test_metrics['accuracy']:.6f}")
    print(f"f1_micro: {test_metrics['f1_micro']:.6f}")

    # 8) Plot epoch curves
    plot_path = args.plot_path or (
        f"outputs/ecg_code/{args.dataset}_interval{args.interval_size}_epoch_metrics.png"
    )
    plot_title = args.plot_title or f"ECG-CODE Epoch Metrics ({args.dataset.upper()})"
    plot_epoch_metrics(
        history=history,
        output_path=plot_path,
        title=plot_title,
        dpi=args.plot_dpi,
    )


# python examples/LUDB_ECGDelineationLUDB_ECGCODE.py \
# --dataset ludb \
# --root /home/$USER/.cache/pyhealth/datasets/physionet.org/files/ludb/1.0.1/data \
#   --dev \
# --epochs 10 \
# --lr 1e-3 \
# --pred-conf-threshold 0.5 \
# --plot-path ./ludb_ecgcode_epoch_metrics.png

if __name__ == "__main__":
    main()
