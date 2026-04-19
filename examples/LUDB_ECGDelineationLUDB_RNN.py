"""
ECG RNN training example for LUDB or QTDB (pulse-split classification).

This example shows how to:
1) load LUDB or QTDB with the modern BaseDataset API
2) apply a pulse-splitting delineation task
3) train a PyHealth RNN model on pulse windows
4) track epoch-wise loss, accuracy, and f1_micro
5) plot loss/accuracy/f1_micro curves over epochs

Task setup
----------
- Input:  pulse ECG window ("signal"), shape (1, T)
- Label:  dominant wave class in the pulse mask ("label"), multiclass in {0,1,2,3}
          0=background, 1=P, 2=QRS, 3=T
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from pyhealth.datasets import LUDBDataset, QTDBDataset, get_dataloader, split_by_patient
from pyhealth.metrics import multiclass_metrics_fn
from pyhealth.models import RNN
from pyhealth.tasks import BaseTask, ECGDelineationLUDB, ECGDelineationQTDB


class PulseRNNTask(BaseTask):
    """Pulse-level ECG classification wrapper for RNN training."""

    task_name: str = "ecg_pulse_rnn_classification"
    input_schema: Dict[str, Union[str, Type]] = {"signal": "tensor"}
    output_schema: Dict[str, Union[str, Type]] = {"label": "multiclass"}

    def __init__(self, base_task: BaseTask) -> None:
        super().__init__()
        self.base_task = base_task

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        base_samples = self.base_task(patient)

        samples: List[Dict[str, Any]] = []
        for s in base_samples:
            samples.append(
                {
                    "patient_id": s["patient_id"],
                    "record_id": s["record_id"],
                    "lead": s.get("lead"),
                    "signal": s["signal"],
                    "label": int(s["label"]),
                }
            )
        return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RNN on ECG delineation pulse windows (LUDB or QTDB) "
            "with epoch-wise loss/accuracy/f1_micro tracking and plotting."
        )
    )
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
        "--pulse-window",
        type=int,
        default=250,
        help="Half-window around QRS peak in samples (250 => 500-sample pulse).",
    )
    parser.add_argument(
        "--keep-incomplete-pulses",
        action="store_true",
        help="If set, do NOT filter pulses missing P or T annotations.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for task processing.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="RNN embedding dimension.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="RNN hidden dimension.",
    )
    parser.add_argument(
        "--rnn-type",
        type=str,
        default="GRU",
        choices=["RNN", "LSTM", "GRU"],
        help="Recurrent cell type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cuda:0" or "cpu". Default: auto.',
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help=(
            "Optional output path for the epoch-metrics plot "
            "(default: output/<dataset>_rnn_epoch_metrics.png)."
        ),
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="",
        help="Optional custom title for the epoch-metrics plot.",
    )

    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.pulse_window <= 0:
        parser.error("--pulse-window must be positive.")
    return args


def build_dataset_and_task(args: argparse.Namespace):
    dataset_kwargs = {
        "root": args.root,
        "dev": args.dev,
        "num_workers": args.num_workers,
        "download": args.download,
    }
    if args.cache_dir:
        dataset_kwargs["cache_dir"] = args.cache_dir

    if args.dataset == "ludb":
        base_dataset = LUDBDataset(**dataset_kwargs)
        base_task = ECGDelineationLUDB(
            split_by_pulse=True,
            pulse_window=args.pulse_window,
            filter_incomplete_pulses=not args.keep_incomplete_pulses,
        )
    else:
        base_dataset = QTDBDataset(**dataset_kwargs)
        base_task = ECGDelineationQTDB(
            split_by_pulse=True,
            pulse_window=args.pulse_window,
            filter_incomplete_pulses=not args.keep_incomplete_pulses,
        )

    task = PulseRNNTask(base_task)
    return base_dataset, task


def _compute_multiclass_scores(
    y_true_all: np.ndarray,
    y_prob_all: np.ndarray,
) -> Dict[str, float]:
    scores = multiclass_metrics_fn(
        y_true=y_true_all,
        y_prob=y_prob_all,
        metrics=["accuracy", "f1_micro"],
    )
    return {
        "accuracy": float(scores["accuracy"]),
        "f1_micro": float(scores["f1_micro"]),
    }


def evaluate_epoch(
    model: RNN,
    dataloader,
) -> Dict[str, float]:
    model.eval()
    loss_all: List[float] = []
    y_true_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation/Test", leave=False):
            output = model(**data)
            loss_all.append(float(output["loss"].item()))
            y_true_all.append(output["y_true"].detach().cpu().numpy())
            y_prob_all.append(output["y_prob"].detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    cls_scores = _compute_multiclass_scores(y_true, y_prob)

    return {
        "loss": float(np.mean(loss_all)),
        "accuracy": cls_scores["accuracy"],
        "f1_micro": cls_scores["f1_micro"],
    }


def plot_epoch_metrics(
    history: Dict[str, List[float]],
    output_path: str,
    title: str,
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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved epoch-metrics plot to: {out}")


def main() -> None:
    args = parse_args()
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1) Base dataset + delineation task
    base_dataset, task = build_dataset_and_task(args)
    base_dataset.stats()
    if hasattr(base_dataset, "info"):
        base_dataset.info()

    # 2) Build pulse-level sample dataset
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)
    print(f"Number of pulse samples: {len(sample_dataset)}")
    if len(sample_dataset) == 0:
        raise RuntimeError(
            "No samples were generated. Try disabling pulse filtering "
            "with --keep-incomplete-pulses or using a different pulse window."
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
    model = RNN(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
    )
    model.to(device)

    # 5) Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 6) Train loop with epoch-wise metrics
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_micro": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_micro": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss: List[float] = []
        epoch_y_true: List[np.ndarray] = []
        epoch_y_prob: List[np.ndarray] = []

        for data in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            optimizer.zero_grad()
            output = model(**data)
            loss = output["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss.item()))
            epoch_y_true.append(output["y_true"].detach().cpu().numpy())
            epoch_y_prob.append(output["y_prob"].detach().cpu().numpy())

        train_y_true = np.concatenate(epoch_y_true, axis=0)
        train_y_prob = np.concatenate(epoch_y_prob, axis=0)
        train_cls_scores = _compute_multiclass_scores(train_y_true, train_y_prob)

        train_scores = {
            "loss": float(np.mean(epoch_loss)),
            "accuracy": train_cls_scores["accuracy"],
            "f1_micro": train_cls_scores["f1_micro"],
        }
        val_scores = evaluate_epoch(model, val_loader)

        history["train_loss"].append(train_scores["loss"])
        history["train_accuracy"].append(train_scores["accuracy"])
        history["train_f1_micro"].append(train_scores["f1_micro"])
        history["val_loss"].append(val_scores["loss"])
        history["val_accuracy"].append(val_scores["accuracy"])
        history["val_f1_micro"].append(val_scores["f1_micro"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_scores['loss']:.6f}, "
            f"train_acc={train_scores['accuracy']:.6f}, "
            f"train_f1_micro={train_scores['f1_micro']:.6f} | "
            f"val_loss={val_scores['loss']:.6f}, "
            f"val_acc={val_scores['accuracy']:.6f}, "
            f"val_f1_micro={val_scores['f1_micro']:.6f}"
        )

    # 7) Final test metrics
    test_scores = evaluate_epoch(model, test_loader)
    print("Final test metrics:")
    print(f"loss: {test_scores['loss']:.6f}")
    print(f"accuracy: {test_scores['accuracy']:.6f}")
    print(f"f1_micro: {test_scores['f1_micro']:.6f}")

    # 8) Plot curves
    plot_path = args.plot_path or f"output/{args.dataset}_rnn_epoch_metrics.png"
    plot_title = args.plot_title or f"RNN Epoch Metrics ({args.dataset.upper()})"
    plot_epoch_metrics(history, output_path=plot_path, title=plot_title)


# python examples/LUDB_ECGDelineationLUDB_RNN.py \
#   --dataset ludb \
#   --root /home/$USER/.cache/pyhealth/datasets/physionet.org/files/ludb/1.0.1/data \
#   --epochs 10 \
#   --lr 1e-3 \
#   --plot-path ./ludb_rnn_epoch_metrics.png
#
#
if __name__ == "__main__":
    main()
