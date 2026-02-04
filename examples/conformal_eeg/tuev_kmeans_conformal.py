"""K-means Cluster-Based Conformal Prediction (ClusterLabel) on TUEV EEG Events using ContraWR.

This script:
1) Loads the TUEV dataset and applies the EEGEventsTUEV task.
2) Splits into train/val/cal/test using split conformal protocol.
3) Trains a ContraWR model.
4) Extracts embeddings for training and calibration splits using embed=True.
5) Calibrates a ClusterLabel prediction-set predictor (K-means clustering).
6) Evaluates prediction-set coverage/miscoverage and efficiency on the test split.

Example (from repo root):
  python examples/conformal_eeg/tuev_kmeans_conformal.py --root downloads/tuev/v2.0.1/edf --n-clusters 5

Notes:
- ClusterLabel uses K-means clustering on embeddings to compute cluster-specific thresholds.
- Different K values can be tested to find optimal cluster count.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from pyhealth.calib.predictionset.cluster import ClusterLabel
from pyhealth.calib.utils import extract_embeddings
from pyhealth.datasets import TUEVDataset, get_dataloader, split_by_sample_conformal
from pyhealth.models import ContraWR
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.trainer import Trainer, get_metrics_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-means cluster-based conformal prediction (ClusterLabel) on TUEV EEG events using ContraWR."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="downloads/tuev/v2.0.1/edf",
        help="Path to TUEV edf/ folder.",
    )
    parser.add_argument("--subset", type=str, default="both", choices=["train", "eval", "both"]) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (e.g., 0.1 => 90% target coverage).")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=4,
        default=(0.6, 0.1, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "CAL", "TEST"),
        help="Split ratios for train/val/cal/test. Must sum to 1.0.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of K-means clusters for cluster-specific thresholds.",
    )
    parser.add_argument("--n-fft", type=int, default=128, help="STFT FFT size used by ContraWR.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. 'cuda:0' or 'cpu'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"TUEV root not found: {root}. "
            "Pass --root to point to your downloaded TUEV edf/ directory."
        )

    print("=" * 80)
    print("STEP 1: Load TUEV + build task dataset")
    print("=" * 80)
    dataset = TUEVDataset(root=str(root), subset=args.subset)
    sample_dataset = dataset.set_task(EEGEventsTUEV(), cache_dir="examples/conformal_eeg/cache")

    print(f"Task samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    if len(sample_dataset) == 0:
        raise RuntimeError("No samples produced. Verify TUEV root/subset/task.")

    print("\n" + "=" * 80)
    print("STEP 2: Split train/val/cal/test")
    print("=" * 80)
    train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal(
        dataset=sample_dataset, ratios=list(args.ratios), seed=args.seed
    )
    print(f"Train: {len(train_ds)}")
    print(f"Val:   {len(val_ds)}")
    print(f"Cal:   {len(cal_ds)}")
    print(f"Test:  {len(test_ds)}")

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) else None
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("\n" + "=" * 80)
    print("STEP 3: Train ContraWR")
    print("=" * 80)
    model = ContraWR(dataset=sample_dataset, n_fft=args.n_fft).to(device)
    trainer = Trainer(model=model, device=device, enable_logging=False)

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="accuracy" if val_loader is not None else None,
    )

    print("\nBase model performance on test set:")
    y_true_base, y_prob_base, _loss_base = trainer.inference(test_loader)
    base_metrics = get_metrics_fn("multiclass")(y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"])
    for metric, value in base_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("STEP 4: K-means Cluster-Based Conformal Prediction (ClusterLabel)")
    print("=" * 80)
    print(f"Target miscoverage alpha: {args.alpha} (target coverage {1 - args.alpha:.0%})")
    print(f"Number of clusters: {args.n_clusters}")

    print("Extracting embeddings for training split...")
    train_embeddings = extract_embeddings(model, train_ds, batch_size=args.batch_size, device=device)
    print(f"  train_embeddings shape: {train_embeddings.shape}")

    print("Extracting embeddings for calibration split...")
    cal_embeddings = extract_embeddings(model, cal_ds, batch_size=args.batch_size, device=device)
    print(f"  cal_embeddings shape: {cal_embeddings.shape}")

    cluster_predictor = ClusterLabel(
        model=model,
        alpha=float(args.alpha),
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )
    print("Calibrating ClusterLabel predictor (fits K-means and computes cluster-specific thresholds)...")
    cluster_predictor.calibrate(
        cal_dataset=cal_ds,
        train_embeddings=train_embeddings,
        cal_embeddings=cal_embeddings,
    )

    print("Evaluating ClusterLabel predictor on test set...")
    y_true, y_prob, _loss, extra = Trainer(model=cluster_predictor).inference(
        test_loader, additional_outputs=["y_predset"]
    )

    cluster_metrics = get_metrics_fn("multiclass")(
        y_true,
        y_prob,
        metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra["y_predset"],
    )

    predset = extra["y_predset"]
    if isinstance(predset, np.ndarray):
        predset_t = torch.tensor(predset)
    else:
        predset_t = predset
    avg_set_size = predset_t.float().sum(dim=1).mean().item()

    miscoverage = cluster_metrics["miscoverage_ps"]
    if isinstance(miscoverage, np.ndarray):
        miscoverage = float(miscoverage.item() if miscoverage.size == 1 else miscoverage.mean())
    else:
        miscoverage = float(miscoverage)

    print("\nClusterLabel Results:")
    print(f"  Accuracy: {cluster_metrics['accuracy']:.4f}")
    print(f"  Empirical miscoverage: {miscoverage:.4f}")
    print(f"  Empirical coverage: {1 - miscoverage:.4f}")
    print(f"  Average set size: {avg_set_size:.2f}")
    print(f"  Number of clusters: {args.n_clusters}")


if __name__ == "__main__":
    main()
