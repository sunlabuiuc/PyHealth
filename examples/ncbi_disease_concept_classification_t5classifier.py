"""NCBI Disease concept classification with :class:`~pyhealth.models.T5Classifier`.

This example mirrors the local Hallmarks of Cancer T5Classifier pattern while
using the public NCBI Disease dataset added in this project.

It supports:

1. ``--demo`` mode, which copies the tiny synthetic NCBI raw files from
   ``test-resources/ncbi_disease`` into a temporary directory.
2. Real data mode via ``--data_root`` pointing to a directory containing the
   official NCBI Disease raw train/dev/test corpus files or zip archives.

The ablation compares:

- full title+abstract vs abstract-only input text
- mean vs first-token pooling

All runs reuse a shared concept-ID label space built from the combined corpus,
then train on the train split and report validation metrics on the dev split.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from pyhealth.datasets import NCBIDiseaseDataset, get_dataloader
from pyhealth.models import T5Classifier
from pyhealth.tasks import NCBIDiseaseConceptClassification
from pyhealth.trainer import Trainer


def copy_demo_root(source_root: str, target_root: str) -> None:
    source_path = Path(source_root)
    target_path = Path(target_root)
    for path in source_path.glob("NCBI*_corpus.txt"):
        shutil.copy(path, target_path / path.name)


def build_split_datasets(data_root: str, cache_parent: Path, text_source: str):
    base = NCBIDiseaseDataset(
        root=data_root,
        cache_dir=cache_parent / f"ncbi_{text_source}_cache",
        num_workers=1,
        dev=False,
    )
    all_ds = base.set_task(
        NCBIDiseaseConceptClassification(split=None, text_source=text_source)
    )

    train_indices = [i for i, sample in enumerate(all_ds) if sample["split"] == "train"]
    dev_indices = [i for i, sample in enumerate(all_ds) if sample["split"] == "dev"]

    train_ds = all_ds.subset(train_indices)
    dev_ds = all_ds.subset(dev_indices)
    train_loader = get_dataloader(train_ds, batch_size=4, shuffle=True)
    dev_loader = get_dataloader(dev_ds, batch_size=4, shuffle=False)
    return all_ds, train_ds, dev_ds, train_loader, dev_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCBI Disease concept classification + T5Classifier ablation"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Copy the tiny synthetic NCBI files into a temp directory and train "
            "on them."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Directory containing NCBI Disease corpus raw files (ignored if --demo).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--pretrained",
        type=str,
        default="t5-small",
        help="Hugging Face checkpoint for T5Classifier (default: t5-small).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="ncbi_t5classifier_demo_") as tmp:
        if args.demo:
            root = Path(tmp) / "data"
            root.mkdir(parents=True)
            copy_demo_root("test-resources/ncbi_disease", str(root))
            data_root = str(root)
            cache_parent = Path(tmp)
        else:
            if not args.data_root:
                raise SystemExit("Provide --data_root or use --demo.")
            root_path = Path(args.data_root).expanduser().resolve()
            if not root_path.is_dir():
                raise SystemExit(
                    f"--data_root must be an existing directory: {root_path}"
                )
            data_root = str(root_path)
            cache_parent = root_path / ".ncbi_pyhealth_cache"
            cache_parent.mkdir(parents=True, exist_ok=True)

        ablations = [
            {
                "text_source": "full_text",
                "pooling": "mean",
                "lr": 1e-4,
                "name": "full_text_mean_lr1e-4",
            },
            {
                "text_source": "abstract",
                "pooling": "mean",
                "lr": 1e-4,
                "name": "abstract_mean_lr1e-4",
            },
            {
                "text_source": "full_text",
                "pooling": "first",
                "lr": 1e-4,
                "name": "full_text_first_lr1e-4",
            },
        ]

        metrics = ["f1_macro", "hamming_loss"]

        print("=== NCBI Disease — T5Classifier ablation ===")
        print(f"data_root={data_root}, device={device}, pretrained={args.pretrained}")

        results = []
        for cfg in ablations:
            all_ds, train_ds, dev_ds, train_loader, dev_loader = build_split_datasets(
                data_root,
                cache_parent,
                text_source=cfg["text_source"],
            )
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

            print(
                f"\n--- Run: {cfg['name']} "
                f"(text_source={cfg['text_source']}, "
                f"pooling={cfg['pooling']}, lr={cfg['lr']}) ---"
            )
            print(
                f"  train samples={len(train_ds)}, dev samples={len(dev_ds)}, "
                f"labels={model.get_output_size()}"
            )
            trainer.train(
                train_loader,
                val_dataloader=dev_loader,
                epochs=args.epochs,
                optimizer_params={"lr": cfg["lr"]},
                monitor=None,
                load_best_model_at_last=False,
            )
            final_scores = trainer.evaluate(dev_loader)
            results.append((cfg["name"], final_scores))
            for key, value in final_scores.items():
                print(f"  {key}: {value:.6f}")

            all_ds.close()
            train_ds.close()
            dev_ds.close()

        print("\n=== Summary (dev) ===")
        for name, scores in results:
            f1 = scores.get("f1_macro", float("nan"))
            print(f"  {name}: f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
