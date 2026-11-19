"""Linear probe (or fine-tune) CLMBR-T-base on an EHRSHOT task.

CLMBR-T-base (Wornow et al., "EHRSHOT", NeurIPS 2023;
https://arxiv.org/abs/2307.02028) is EHRSHOT's reference foundation model.
:class:`~pyhealth.models.CLMBR` reimplements its femr 0.2 architecture and
tokenizer in plain PyTorch and loads the published weights. This script:

1. loads EHRSHOT with :class:`~pyhealth.datasets.EHRShotDataset`;
2. wraps :class:`~pyhealth.tasks.BenchmarkEHRShot` in
   :class:`~pyhealth.tasks.EHRFoundationModelTask`, so each sample carries its
   raw ``(timestamp, code, value, unit)`` timeline up to the label time;
3. trains :class:`~pyhealth.models.CLMBR` on the official EHRSHOT train split
   and reports validation and test metrics.

Access (both are gated):

- EHRSHOT: request access on Redivis (https://redivis.com/datasets/53gc-8rhx41kgt)
  under the EHRSHOT data-use agreement and download ``EHRSHOT_ASSETS``.
- Models: request access on the Hugging Face page of the checkpoint (e.g.
  https://huggingface.co/StanfordShahLab/clmbr-t-base; CC-BY-NC-4.0,
  CITI certificate required), then run ``huggingface-cli login``.

Usage::

    python examples/ehrshot_clmbr_t.py \\
        --root /path/to/EHRSHOT_ASSETS --task guo_icu --mode frozen

The frozen mode is a linear probe: a linear head on the frozen last-token
CLMBR features, trained with Adam and early stopping on the validation
split (EHRSHOT's own evaluation fits a logistic regression on the same
features instead). Embeddings are cached in memory, and on disk with
``--embedding-cache``.
"""

from __future__ import annotations

import argparse

from pyhealth.datasets import EHRShotDataset, get_dataloader
from pyhealth.models import CLMBR
from pyhealth.tasks import BenchmarkEHRShot, EHRFoundationModelTask
from pyhealth.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="EHRSHOT_ASSETS folder")
    parser.add_argument("--task", default="guo_icu",
                        help="An EHRSHOT task, e.g. guo_icu or new_hypertension")
    parser.add_argument("--model", default="StanfordShahLab/clmbr-t-base")
    parser.add_argument("--mode", choices=("frozen", "finetune"), default="frozen")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Tokens per timeline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default 1e-3 frozen, 1e-5 finetune")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache folder")
    parser.add_argument("--embedding-cache", default=None,
                        help="Folder for cached frozen embeddings")
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

    model = CLMBR(samples, model_name=args.model, mode=args.mode,
                  max_length=args.max_length, cache_dir=args.embedding_cache)
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
