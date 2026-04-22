"""NCBI Disease text-to-text recognition with the local T5 model.

This script supports ``--demo`` mode via the tiny synthetic corpus under
``test-resources/ncbi_disease`` and real-data mode through ``--data_root``.

The ablation compares ``full_text`` vs ``abstract`` inputs while training the
same ``t5-small`` backbone on BIO-tag target strings.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support

from pyhealth.datasets import NCBIDiseaseDataset, get_dataloader
from pyhealth.models import T5
from pyhealth.tasks import NCBIDiseaseRecognition
from pyhealth.trainer import Trainer


def copy_demo_root(source_root: str, target_root: str) -> None:
    source_path = Path(source_root)
    target_path = Path(target_root)
    for path in source_path.glob("NCBI*_corpus.txt"):
        shutil.copy(path, target_path / path.name)


def _normalize_tags(text: str, target_text: str) -> list[str]:
    tags = [tag.strip() for tag in (target_text or "").split() if tag.strip()]
    valid_tags = {"O", "B-Disease", "I-Disease"}
    tags = [tag if tag in valid_tags else "O" for tag in tags]
    expected_length = len(text.split())
    if len(tags) < expected_length:
        tags.extend(["O"] * (expected_length - len(tags)))
    return tags[:expected_length]


def _evaluate_bio(model: T5, dataloader) -> dict[str, float]:
    y_true = []
    y_pred = []
    for batch in dataloader:
        generated = model.generate_text(batch["source_text"])
        for text, target_text, predicted_text in zip(
            batch["text"], batch["target_text"], generated
        ):
            true_tags = _normalize_tags(text, target_text)
            pred_tags = _normalize_tags(text, predicted_text)
            y_true.extend(tag != "O" for tag in true_tags)
            y_pred.extend(tag != "O" for tag in pred_tags)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_split_datasets(data_root: str, cache_parent: Path, text_source: str):
    base = NCBIDiseaseDataset(
        root=data_root,
        cache_dir=cache_parent / f"ncbi_{text_source}_cache",
        num_workers=1,
        dev=False,
    )
    all_ds = base.set_task(NCBIDiseaseRecognition(text_source=text_source))

    train_indices = [i for i, sample in enumerate(all_ds) if sample["split"] == "train"]
    eval_indices = [
        i for i, sample in enumerate(all_ds) if sample["split"] in {"dev", "test"}
    ]

    train_ds = all_ds.subset(train_indices)
    dev_ds = all_ds.subset(eval_indices)
    train_loader = get_dataloader(train_ds, batch_size=4, shuffle=True)
    dev_loader = get_dataloader(dev_ds, batch_size=4, shuffle=False)
    return all_ds, train_ds, dev_ds, train_loader, dev_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCBI Disease recognition + T5 seq2seq ablation"
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
        help="Hugging Face checkpoint for the local T5 model (default: t5-small).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="ncbi_t5_demo_") as tmp:
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
                "lr": 1e-4,
                "name": "full_text_lr1e-4",
            },
            {
                "text_source": "abstract",
                "lr": 1e-4,
                "name": "abstract_lr1e-4",
            },
        ]

        print("=== NCBI Disease — T5 seq2seq ablation ===")
        print(f"data_root={data_root}, device={device}, pretrained={args.pretrained}")

        results = []
        for cfg in ablations:
            all_ds, train_ds, dev_ds, train_loader, dev_loader = build_split_datasets(
                data_root,
                cache_parent,
                text_source=cfg["text_source"],
            )
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

            print(
                f"\n--- Run: {cfg['name']} "
                f"(text_source={cfg['text_source']}, lr={cfg['lr']}) ---"
            )
            print(
                f"  train samples={len(train_ds)}, eval samples={len(dev_ds)}"
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
            final_scores.update(_evaluate_bio(model, dev_loader))
            results.append((cfg["name"], final_scores))
            for key, value in final_scores.items():
                print(f"  {key}: {value:.6f}")

            all_ds.close()
            train_ds.close()
            dev_ds.close()

        print("\n=== Summary (held-out) ===")
        for name, scores in results:
            f1 = scores.get("f1", float("nan"))
            print(f"  {name}: f1={f1:.4f}")


if __name__ == "__main__":
    main()
