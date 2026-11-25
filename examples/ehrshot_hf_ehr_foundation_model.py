"""Linear probe (or fine-tune) an hf_ehr foundation model on an EHRSHOT task.

Wornow et al., "Context Clues" (2024; https://arxiv.org/abs/2412.16178)
released gpt / llama / mamba / hyena models pretrained on Stanford EHRs with
the CLMBR vocabulary; EHRSHOT (Wornow et al., NeurIPS 2023) is the benchmark
they were evaluated on. This script:

1. loads EHRSHOT with :class:`~pyhealth.datasets.EHRShotDataset`;
2. wraps :class:`~pyhealth.tasks.BenchmarkEHRShot` in
   :class:`~pyhealth.tasks.EHRFoundationModelTask`, so each sample carries its
   raw ``(timestamp, code, value, unit)`` timeline up to the label time;
3. trains :class:`~pyhealth.models.HFEHRFoundationModel`, which tokenizes the
   timelines with the checkpoint's own vocabulary in ``forward()``, on the
   official EHRSHOT train split and reports validation and test metrics.

Access (both are gated):

- EHRSHOT: request access on Redivis (https://redivis.com/datasets/53gc-8rhx41kgt)
  under the EHRSHOT data-use agreement and download ``EHRSHOT_ASSETS``.
- Models: request access on the Hugging Face page of the checkpoint (e.g.
  https://huggingface.co/StanfordShahLab/gpt-base-1024-clmbr; CC-BY-NC-4.0,
  CITI certificate required), then run ``huggingface-cli login``.

Usage::

    python examples/ehrshot_hf_ehr_foundation_model.py \\
        --root /path/to/EHRSHOT_ASSETS --task guo_icu \\
        --model StanfordShahLab/gpt-base-1024-clmbr --mode frozen

The frozen mode is a linear probe, as in the papers' few-shot evaluation (they
fit a logistic regression on the same last-token embeddings); embeddings are
cached in memory, and on disk with ``--embedding-cache``.
"""

from __future__ import annotations

import argparse

from pyhealth.datasets import EHRShotDataset, get_dataloader
from pyhealth.models import HFEHRFoundationModel
from pyhealth.tasks import BenchmarkEHRShot, EHRFoundationModelTask
from pyhealth.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="EHRSHOT_ASSETS folder")
    parser.add_argument("--task", default="guo_icu",
                        help="An EHRSHOT task, e.g. guo_icu or new_hypertension")
    parser.add_argument("--model", default="StanfordShahLab/gpt-base-1024-clmbr")
    parser.add_argument("--mode", choices=("frozen", "finetune"), default="frozen")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Tokens per timeline (default: the checkpoint's)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default 1e-3 frozen, 1e-5 finetune")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache folder")
    parser.add_argument("--embedding-cache", default=None,
                        help="Folder for cached frozen embeddings")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Needed for the hyena checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = EHRShotDataset(root=args.root, tables=["ehrshot", "splits", args.task],
                             cache_dir=args.cache_dir)
    task = EHRFoundationModelTask(BenchmarkEHRShot(task=args.task))
    samples = dataset.set_task(task)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(samples):
        splits.setdefault(sample["split"], []).append(index)
    loaders = {
        name: get_dataloader(samples.subset(indices), batch_size=args.batch_size,
                             shuffle=name == "train")
        for name, indices in splits.items() if indices
    }

    model = HFEHRFoundationModel(
        samples, model_name=args.model, mode=args.mode, max_length=args.max_length,
        cache_dir=args.embedding_cache, trust_remote_code=args.trust_remote_code,
    )
    metrics = (["roc_auc", "pr_auc"] if model.mode == "binary"
               else ["accuracy", "f1_macro"] if model.mode == "multiclass"
               else ["roc_auc_macro", "pr_auc_macro"])
    lr = args.lr or (1e-3 if args.mode == "frozen" else 1e-5)
    trainer = Trainer(model=model, metrics=metrics)
    trainer.train(train_dataloader=loaders["train"], val_dataloader=loaders.get("val"),
                  epochs=args.epochs, optimizer_params={"lr": lr},
                  monitor=metrics[0])
    print("vocabulary coverage:", model.coverage_report())
    for name in ("val", "test"):
        if name in loaders:
            print(name, trainer.evaluate(loaders[name]))


if __name__ == "__main__":
    main()
