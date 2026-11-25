r"""PTB-XL binary classification with TaskAug: ablation over the number of stages.

Paper (TaskAug):
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag, and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms."
    Proceedings of the Conference on Health, Inference, and Learning (CHIL),
    PMLR 174:282-310, 2022. https://proceedings.mlr.press/v174/raghu22a.html

Author:
    Lisha Ranganath (GitHub: @LishaRanganath) and Meghaben D Dhalwadi

Description:
    Trains ``PTBXLResNet`` (no augmentation) and ``TaskAugResNet`` (TaskAug with
    ``K`` chained augmentation stages) on one PTB-XL diagnostic superclass (MI,
    HYP, STTC or CD) and compares them by test AUROC and AUPRC. The paper fixes
    K=2; the ablation here varies K, which the paper does not report.

    Pipeline: PTBXLDataset -> binary task -> split -> model -> Trainer.

Experimental setup:
    * ``--ablation`` trains the no-augmentation baseline and one TaskAug model
      for each value in ``--ks`` (default 1 2 3), with the same seed, data and
      optimizer, and prints one table.
    * ``--synthetic`` replaces PTB-XL by small random ECGs in which the positive
      class carries a slow oscillation. It needs no data download and runs in
      about a minute; it checks the whole pipeline, not clinical performance.
    * With real data, the official PTB-XL split is used (folds 1-8 train,
      fold 9 validation, fold 10 test); ``--train-samples 1000`` gives the
      paper's low-data setting.

Usage:
    python examples/ecg/ptbxl/ptbxl_binary_classification_taskaug_resnet.py \
        --synthetic --ablation

    python examples/ecg/ptbxl/ptbxl_binary_classification_taskaug_resnet.py \
        --root /data/ptb-xl/1.0.3 --task HYP --ablation --train-samples 1000

Findings (reference, original TaskAug code):
    Test AUROC with 1000 training ECGs, seed 42, 100 epochs, from a separate
    run of the authors' code (https://github.com/aniruddhraghu/ecg_aug). This
    PyHealth model applies the augmentation but does not learn it with the
    paper's bilevel optimization, so its numbers will differ.

        task   no aug   K=1     K=2     K=3
        MI     0.8213   0.7986  0.8040  0.8046
        HYP    0.8458   0.8523  0.8385  0.8103
        STTC   0.8761   0.8619  0.8824  0.8547
        CD     0.8201   0.8294  0.8363  0.8260

    K=2 is best or tied on MI, STTC and CD; HYP prefers K=1. TaskAug clearly
    helps only on CD. Single seed, so small differences are within noise.

Synthetic run (--synthetic --ablation, seed 42, 3 epochs, 48 train ECGs):
    Test scores on 32 random ECGs. This only checks that the pipeline runs
    end to end. With so little data the scores are close to chance, differ
    between runs and hardware, and say nothing about the value of TaskAug.

        model             roc_auc   pr_auc
        resnet (no aug)   0.5742    0.5805
        taskaug k=1       0.4766    0.5399
        taskaug k=2       0.3594    0.4389
        taskaug k=3       0.6523    0.6673
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    PTBXLDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.models import PTBXLResNet, TaskAugResNet
from pyhealth.tasks import PTBXLCD, PTBXLHYP, PTBXLMI, PTBXLSTTC
from pyhealth.trainer import Trainer

TASKS = {"MI": PTBXLMI, "HYP": PTBXLHYP, "STTC": PTBXLSTTC, "CD": PTBXLCD}


def make_synthetic_splits(seed, n_train=48, n_val=16, n_test=32, length=2496):
    """Builds small random ECG datasets with a learnable label.

    Labels alternate, so every split has both classes. Positive ECGs carry a
    slow oscillation on top of noise.

    Returns:
        tuple: The full dataset and its train, validation and test subsets.
    """
    rng = np.random.RandomState(seed)
    n = n_train + n_val + n_test
    t = np.linspace(0, 1, length, dtype=np.float32)
    wave = np.sin(2 * np.pi * 8 * t).astype(np.float32)
    samples = []
    for i in range(n):
        signal = 0.5 * rng.randn(12, length).astype(np.float32)
        if i % 2 == 1:
            signal += 0.05 * wave
        samples.append({"patient_id": f"p{i}", "signal": signal, "label": i % 2})
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="ptbxl_synthetic",
    )
    index = list(range(n))
    train = dataset.subset(index[:n_train])
    val = dataset.subset(index[n_train : n_train + n_val])
    test = dataset.subset(index[n_train + n_val :])
    return dataset, train, val, test


def load_ptbxl_splits(args):
    """Loads PTB-XL, applies the binary task and splits by the official folds."""
    kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    dataset = PTBXLDataset(
        root=str(args.root), sampling_rate=500, dev=args.dev, **kwargs
    )
    task = TASKS[args.task](dataset.scp_statements_path)
    samples = dataset.set_task(task)
    train, val, test = split_by_strat_fold(samples)
    if args.train_samples is not None and args.train_samples < len(train):
        rng = np.random.RandomState(args.seed)
        keep = rng.permutation(len(train))[: args.train_samples].tolist()
        train = train.subset(keep)
    return samples, train, val, test


def run_config(name, k, samples, train, val, test, args):
    """Trains one model (``k == 0`` is the no-augmentation baseline).

    Returns:
        dict: The model name and its test ``roc_auc`` and ``pr_auc``.
    """
    torch.manual_seed(args.seed)
    if k == 0:
        model = PTBXLResNet(dataset=samples)
    else:
        model = TaskAugResNet(dataset=samples, k=k)
    train_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test, batch_size=args.batch_size, shuffle=False)
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"], enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="roc_auc",
        optimizer_params={"lr": args.lr},
    )
    scores = trainer.evaluate(test_loader)
    return {
        "model": name,
        "roc_auc": float(scores["roc_auc"]),
        "pr_auc": float(scores["pr_auc"]),
    }


def print_table(rows):
    """Prints the results as a Markdown table."""
    print("\n| model | roc_auc | pr_auc |")
    print("|---|---|---|")
    for row in rows:
        print(f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=None,
                        help="PTB-XL version root (needed unless --synthetic)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use small random ECGs instead of PTB-XL")
    parser.add_argument("--task", choices=sorted(TASKS), default="MI",
                        help="Diagnostic superclass to detect (default: MI)")
    parser.add_argument("--ablation", action="store_true",
                        help="Compare the baseline with TaskAug for each K in --ks")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3],
                        help="Stage counts for --ablation (default: 1 2 3)")
    parser.add_argument("--model", choices=("taskaug", "resnet"), default="taskaug",
                        help="Single run: TaskAug model or the baseline")
    parser.add_argument("--k", type=int, default=2,
                        help="Single run: number of TaskAug stages (default: 2)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Default: 5 (3 with --synthetic)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Keep only this many training ECGs (low-data setting)")
    parser.add_argument("--dev", action="store_true",
                        help="Limit patients via BaseDataset.dev mode")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.synthetic and args.root is None:
        parser.error("--root is required unless --synthetic is given")
    if args.epochs is None:
        args.epochs = 3 if args.synthetic else 5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.synthetic:
        samples, train, val, test = make_synthetic_splits(args.seed)
    else:
        samples, train, val, test = load_ptbxl_splits(args)
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"task={'synthetic' if args.synthetic else args.task}")

    if args.ablation:
        configs = [("resnet (no aug)", 0)] + [(f"taskaug k={k}", k) for k in args.ks]
    elif args.model == "resnet":
        configs = [("resnet (no aug)", 0)]
    else:
        configs = [(f"taskaug k={args.k}", args.k)]

    rows = [run_config(name, k, samples, train, val, test, args)
            for name, k in configs]
    print_table(rows)


if __name__ == "__main__":
    main()