"""
Hallmarks of Cancer (HOC) sentence multilabel classification with :class:`~pyhealth.models.T5Classifier`.

This script supports:

1. **Synthetic demo data** (``--demo``): writes a small ``hallmarks_of_cancer.csv`` under a
   temporary directory so you can run without downloading anything.
2. **Real data**: set ``--data_root`` to a folder containing ``hallmarks_of_cancer.csv``
   (see ``examples/data_prep/export_hallmarks_of_cancer_bigbio.py``).

**Ablation (course requirement):** compares pooling strategies and learning rates under a
fixed ``t5-small`` backbone on the same train/validation splits. Metrics are reported on
the validation split (``f1_macro``, ``hamming_loss``).

First Hugging Face download of ``t5-small`` requires network access once (cached afterward).

Example::

    python examples/hallmarks_of_cancer_HallmarksOfCancerSentenceClassification_t5classifier.py --demo --epochs 2

"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import torch

from pyhealth.datasets import HallmarksOfCancerDataset, get_dataloader
from pyhealth.models import T5Classifier
from pyhealth.tasks.hallmarks_of_cancer_classification import (
    HallmarksOfCancerSentenceClassification,
)
from pyhealth.trainer import Trainer


def _write_synthetic_csv(root: Path, n_train: int = 28, n_val: int = 8) -> None:
    """Create a minimal CSV matching :class:`HallmarksOfCancerDataset` expectations."""
    rng = random.Random(0)
    label_choices = [
        "none",
        "sustaining proliferative signaling",
        "evading growth suppressors",
        "sustaining proliferative signaling##evading growth suppressors",
        "activating invasion and metastasis",
    ]
    lines = ["sentence_id,document_id,text,labels,split"]
    for i in range(n_train):
        lab = rng.choice(label_choices)
        lines.append(
            f's_tr_{i},d{i},"Synthetic oncology sentence {i} about pathways and phenotypes.",{lab},train'
        )
    for i in range(n_val):
        lab = rng.choice(label_choices)
        lines.append(
            f's_va_{i},dv{i},"Held-out validation sentence {i} for hallmark labels.",{lab},validation'
        )
    (root / "hallmarks_of_cancer.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_dataloaders(data_root: str, cache_parent: Path, batch_size: int):
    base = HallmarksOfCancerDataset(
        root=data_root,
        cache_dir=cache_parent / "hoc_cache",
        num_workers=1,
        dev=True,
    )
    train_ds = base.set_task(HallmarksOfCancerSentenceClassification(split="train"))
    val_ds = base.set_task(HallmarksOfCancerSentenceClassification(split="validation"))
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="HOC + T5Classifier ablation example")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write a tiny synthetic CSV to a temp directory and train on it.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Directory containing hallmarks_of_cancer.csv (ignored if --demo).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--pretrained",
        type=str,
        default="t5-small",
        help="Hugging Face model id (default t5-small for speed).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.demo:
        tmp = tempfile.mkdtemp(prefix="hoc_t5_demo_")
        root = Path(tmp) / "data"
        root.mkdir(parents=True)
        _write_synthetic_csv(root)
        data_root = str(root)
        cache_parent = Path(tmp)
    else:
        if not args.data_root:
            raise SystemExit("Provide --data_root or use --demo.")
        data_root = args.data_root
        cache_parent = Path(data_root).parent / "hoc_example_cache"
        cache_parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_loader, val_loader = _build_dataloaders(
        data_root, cache_parent, batch_size=args.batch_size
    )

    ablations = [
        {"pooling": "mean", "lr": 1e-4, "name": "mean_lr1e-4"},
        {"pooling": "mean", "lr": 5e-4, "name": "mean_lr5e-4"},
        {"pooling": "first", "lr": 1e-4, "name": "first_lr1e-4"},
    ]

    metrics = ["f1_macro", "hamming_loss", "loss"]

    print("=== Hallmarks of Cancer — T5Classifier ablation ===")
    print(f"data_root={data_root}, device={device}, pretrained={args.pretrained}")
    print(f"train batches ≈ {len(train_loader)}, val batches ≈ {len(val_loader)}")

    results = []
    for cfg in ablations:
        model = T5Classifier(
            dataset=train_ds,
            pretrained_model_name=args.pretrained,
            max_length=128,
            dropout=0.1,
            pooling=cfg["pooling"],
        )
        trainer = Trainer(
            model,
            device=device,
            enable_logging=False,
            metrics=metrics,
        )
        print(f"\n--- Run: {cfg['name']} (pooling={cfg['pooling']}, lr={cfg['lr']}) ---")
        trainer.train(
            train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params={"lr": cfg["lr"]},
            monitor=None,
            load_best_model_at_last=False,
        )
        final_scores = trainer.evaluate(val_loader)
        results.append((cfg["name"], final_scores))
        for k, v in final_scores.items():
            print(f"  {k}: {v:.6f}")

    print("\n=== Summary (validation) ===")
    for name, scores in results:
        f1 = scores.get("f1_macro", float("nan"))
        print(f"  {name}: f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
