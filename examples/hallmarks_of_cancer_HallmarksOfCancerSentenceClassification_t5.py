"""Hallmarks of Cancer text-to-text classification with the local T5 model.

This script supports synthetic demo data (``--demo``) and real exported HOC
data (``--data_root`` with ``hallmarks_of_cancer.csv``).

The ablation compares learning rates under the same ``t5-small`` backbone while
using the seq2seq task representation:

- source: ``hoc: <sentence>``
- target: serialized hallmark labels joined by ``" ; "``
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

from pyhealth.datasets import HallmarksOfCancerDataset, get_dataloader
from pyhealth.models import T5
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


def _evaluate_seq2seq(model: T5, dataloader) -> dict[str, float]:
    y_true = []
    y_pred = []
    for batch in dataloader:
        generated = model.generate_text(batch["source_text"])
        y_true.extend(batch["labels"])
        y_pred.extend(
            HallmarksOfCancerSentenceClassification.target_text_to_labels(text)
            for text in generated
        )

    label_space = sorted({label for labels in (y_true + y_pred) for label in labels})
    mlb = MultiLabelBinarizer(classes=label_space)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    return {
        "f1_macro": float(
            f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true_bin, y_pred_bin)),
        "exact_match": float(np.mean(np.all(y_true_bin == y_pred_bin, axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HOC + T5 seq2seq ablation example")
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
        root_path = Path(args.data_root).expanduser().resolve()
        if not root_path.is_dir():
            raise SystemExit(
                f"--data_root must be an existing directory. Not found: {root_path}\n"
                "Use a real path (e.g. ~/data/hoc) containing hallmarks_of_cancer.csv, "
                "not a documentation placeholder like /path/to/folder_with_csv."
            )
        csv_path = root_path / "hallmarks_of_cancer.csv"
        if not csv_path.is_file():
            raise SystemExit(
                f"Expected CSV at {csv_path}. Export it with "
                "examples/data_prep/export_hallmarks_of_cancer_bigbio.py or copy the file there."
            )
        data_root = str(root_path)
        # Keep caches inside the data folder so we never try to mkdir under bogus paths like /path/.
        cache_parent = root_path / ".hoc_pyhealth_cache"
        cache_parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_loader, val_loader = _build_dataloaders(
        data_root, cache_parent, batch_size=args.batch_size
    )

    ablations = [
        {"lr": 1e-4, "name": "lr1e-4"},
        {"lr": 5e-4, "name": "lr5e-4"},
        {"lr": 1e-3, "name": "lr1e-3"},
    ]

    print("=== Hallmarks of Cancer — T5 seq2seq ablation ===")
    print(f"data_root={data_root}, device={device}, pretrained={args.pretrained}")
    print(f"train batches ≈ {len(train_loader)}, val batches ≈ {len(val_loader)}")

    results = []
    for cfg in ablations:
        model = T5(
            dataset=train_ds,
            pretrained_model_name=args.pretrained,
            max_source_length=128,
            max_target_length=64,
            generation_max_length=64,
        )
        trainer = Trainer(
            model,
            device=device,
            enable_logging=False,
        )
        print(f"\n--- Run: {cfg['name']} (lr={cfg['lr']}) ---")
        trainer.train(
            train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params={"lr": cfg["lr"]},
            monitor=None,
            load_best_model_at_last=False,
        )
        final_scores = trainer.evaluate(val_loader)
        final_scores.update(_evaluate_seq2seq(model, val_loader))
        results.append((cfg["name"], final_scores))
        for k, v in final_scores.items():
            print(f"  {k}: {v:.6f}")

    print("\n=== Summary (validation) ===")
    for name, scores in results:
        f1 = scores.get("f1_macro", float("nan"))
        print(f"  {name}: f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
