"""Reproduce the PTB-XL ECG benchmark of Nonaka & Seita (2021) in PyHealth.

Paper: *In-depth Benchmarking of Deep Neural Network Architectures for ECG
Diagnosis* (https://arxiv.org/abs/2112.13779); reference code:
https://github.com/seitalab/dnn_ecg_comparison.

This driver trains the PyHealth port ``pyhealth.models.ResNet1D`` with the
benchmark's full protocol:

- one of the five official PTB-XL fold pairs (``PTBXL_PAPER_FOLD_SPLITS``),
  or all five with ``--split all``;
- per-lead normalization with mean/std fitted on the training folds only
  (``compute_ptbxl_lead_stats`` + ``standardize="train_stats"``);
- a random 2.5 s crop per record while training
  (``training_crop_len=1250`` at 500 Hz);
- sliding-window inference at validation/test: 2.5 s windows, 1.25 s
  stride, class logits aggregated by element-wise maximum
  (``eval_window_len`` / ``eval_window_stride``);
- for the 3-way multi-class task, cross-entropy weighted
  ``[1, n_target/n_normal, n_other/n_normal]`` from the train split
  (``compute_multiclass_class_weights``);
- Adam, model selection on the best validation score
  (macro ROC-AUC for multi-label, macro F1 for multi-class), early stopping;
- test-set evaluation of the headline metric (macro one-vs-rest ROC-AUC
  for multi-label, macro F1 for multi-class) with the benchmark's
  bootstrap confidence interval: 100 resamples, bounds at the 5% and 95%
  percentiles, resamples that do not define every class discarded
  (``pyhealth.metrics.bootstrap_metrics_ci``);
- with ``--split all``, a closing summary of each metric's mean +/- std
  across the five splits, as in the paper's tables.

Requires the PTB-XL v1.0.3 corpus (https://physionet.org/content/ptb-xl/)
and ``pip install 'pyhealth[ptbxl]'``. Examples:

    python examples/ptbxl_ecg_benchmark_resnet1d.py \
        --root ~/.cache/pyhealth/datasets/ptbxl/1.0.3 \
        --split 1 --task multilabel --label-set diagnostic

    python examples/ptbxl_ecg_benchmark_resnet1d.py \
        --root ~/.cache/pyhealth/datasets/ptbxl/1.0.3 \
        --split all --task multiclass --dx AF \
        --result-json ptbxl_af_resnet18.json

Note (documented deviations from the released benchmark code):

- Normalization is per lead with train-fold statistics, following the
  paper text (the released code pooled a scaler across leads).
- Losses use mean reduction (the original used ``reduction="sum"``);
  learning rates from its grid may need a small re-tune.
- ``Trainer`` validates every epoch; the original evaluated every 5
  epochs with patience 5 (evaluations), so the default patience here is
  25 epochs.
- Multi-class weights count the 3-way labels (target precedence), while
  the original summed raw NORM/target vectors; a record with both shifts
  the two counts slightly.
- Bootstrap: the original drew resamples until 100 were usable; here
  exactly ``--boot-samples`` are drawn and unusable ones are reported as
  skipped. The original's multi-class bootstrap reused the multi-label
  positive-class check on integer labels (a latent bug); here every
  resample must contain all classes. Resampling is per test sample, as
  in the original; ``pyhealth.metrics.bootstrap_metrics_ci`` also takes
  ``groups=`` for a patient-level cluster bootstrap.
- The original ``Monitor`` defines a test-threshold Fmax (CAFA-style)
  but never calls it in its train/evaluate scripts; it is not reproduced.

Deliberately not reproduced here: the lr/batch-size grid search and
SE-ResNet and other ablation architectures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    PTBXLDataset,
    compute_ptbxl_lead_stats,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.metrics import bootstrap_metrics_ci
from pyhealth.models import ResNet1D
from pyhealth.tasks import (
    PTBXLMulticlassClassification,
    PTBXLMultilabelClassification,
    compute_multiclass_class_weights,
)
from pyhealth.tasks.ptbxl_multiclass_classification import (
    PTBXL_PAPER_FOLD_SPLITS,
)
from pyhealth.trainer import Trainer

SAMPLING_RATE = 500
ALL_FOLDS = tuple(range(1, 11))

logger = logging.getLogger("ptbxl_ecg_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PTB-XL benchmark (Nonaka & Seita 2021) with ResNet1D"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="PTB-XL v1.0.3 version directory (contains ptbxl_database.csv "
        "and scp_statements.csv)",
    )
    parser.add_argument(
        "--split",
        choices=[str(k) for k in sorted(PTBXL_PAPER_FOLD_SPLITS)] + ["all"],
        default="1",
        help="Benchmark split number ((val, test) folds "
        f"{PTBXL_PAPER_FOLD_SPLITS}) or 'all' for the five-split sweep",
    )
    parser.add_argument(
        "--task", choices=("multilabel", "multiclass"), default="multilabel"
    )
    parser.add_argument(
        "--label-set",
        choices=("all", "diagnostic", "subdiagnostic", "superdiagnostic",
                 "form", "rhythm"),
        default="diagnostic",
        help="Multi-label SCP statement set (--task multilabel)",
    )
    parser.add_argument(
        "--dx",
        choices=("AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD"),
        default="AF",
        help="Target diagnosis for the 3-way task (--task multiclass)",
    )
    parser.add_argument(
        "--depth", type=int, choices=(18, 34, 50), default=18,
        help="ResNet1D depth",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Epoch cap; early stopping usually ends earlier",
    )
    parser.add_argument(
        "--patience", type=int, default=25,
        help="Epochs without validation improvement (the original used 5 "
        "evaluations spaced 5 epochs apart)",
    )
    parser.add_argument(
        "--crop-sec", type=float, default=2.5,
        help="Random training crop length in seconds",
    )
    parser.add_argument(
        "--window-sec", type=float, default=2.5,
        help="Evaluation window length in seconds (stride is half)",
    )
    parser.add_argument(
        "--boot-samples", type=int, default=100,
        help="Bootstrap resamples for the test-set 5%%/95%% interval (the "
        "original used 100); 0 skips the bootstrap",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None,
        help="Default: cuda if available",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir", default=None,
        help="Enable Trainer logging/checkpoints under this directory",
    )
    parser.add_argument(
        "--result-json", default=None,
        help="Write per-split test metrics and bootstrap intervals to "
        "this JSON file",
    )
    return parser.parse_args()


def run_split(args: argparse.Namespace, split_number: int) -> dict:
    """Train, select and evaluate one official fold split.

    Returns:
        dict with ``split``, the test-fold metrics (``test``) and the
        bootstrap intervals for the headline metric (``bootstrap``).
    """
    val_fold, test_fold = PTBXL_PAPER_FOLD_SPLITS[split_number]
    train_folds = tuple(
        f for f in ALL_FOLDS if f not in (val_fold, test_fold)
    )
    logger.info(
        "Split %d: train folds %s, val fold %d, test fold %d",
        split_number, list(train_folds), val_fold, test_fold,
    )

    root = Path(args.root).expanduser()
    dataset = PTBXLDataset(root=str(root), sampling_rate=SAMPLING_RATE)

    # Normalization statistics come from the training folds only, so they
    # must be computed before set_task materializes the samples.
    stats_path = compute_ptbxl_lead_stats(dataset, train_folds=train_folds)

    if args.task == "multilabel":
        task = PTBXLMultilabelClassification(
            scp_statements_path=root / "scp_statements.csv",
            label_set=args.label_set,
            standardize="train_stats",
            stats_path=stats_path,
        )
        monitor = "roc_auc_macro"
        metrics = ["roc_auc_macro", "f1_macro"]
    else:
        task = PTBXLMulticlassClassification(
            target_dx=args.dx,
            standardize="train_stats",
            stats_path=stats_path,
        )
        monitor = "f1_macro"
        metrics = ["f1_macro", "roc_auc_macro_ovr"]

    samples = dataset.set_task(task)
    train_set, val_set, test_set = split_by_strat_fold(
        samples,
        train_folds=train_folds,
        val_folds=(val_fold,),
        test_folds=(test_fold,),
    )
    logger.info(
        "Samples: train=%d val=%d test=%d",
        len(train_set), len(val_set), len(test_set),
    )

    class_weight = None
    if args.task == "multiclass":
        # One pass over the training split's labels. The benchmark takes
        # the weights from the train split only.
        class_weight = compute_multiclass_class_weights(train_set)
        logger.info("Multiclass class weights: %s", class_weight)

    model = ResNet1D(
        dataset=train_set,
        depth=args.depth,
        training_crop_len=int(args.crop_sec * SAMPLING_RATE),
        eval_window_len=int(args.window_sec * SAMPLING_RATE),
        eval_window_stride=int(args.window_sec * SAMPLING_RATE) // 2,
        class_weight=class_weight,
    )

    trainer = Trainer(
        model=model,
        metrics=metrics,
        device=args.device,
        enable_logging=args.output_dir is not None,
        output_path=args.output_dir,
        exp_name=f"split{split_number}_{args.task}_resnet{args.depth}",
    )

    train_loader = get_dataloader(
        train_set, batch_size=args.batch_size, shuffle=True,
        seed=args.seed, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = get_dataloader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": args.lr},
        monitor=monitor,
        monitor_criterion="max",
        patience=args.patience,
    )

    # Trainer restores the best-validation checkpoint; evaluate it on the
    # held-out test fold with the same sliding-window inference.
    test_scores = trainer.evaluate(test_loader)
    label = args.label_set if args.task == "multilabel" else args.dx
    print(
        f"\nResNet{args.depth} | split {split_number} | "
        f"{args.task} ({label})"
    )
    for name, value in sorted(test_scores.items()):
        print(f"  {name}: {value:.4f}")

    # Benchmark reporting protocol: bootstrap interval for the headline
    # metric only, bounds at the 5% and 95% percentiles.
    bootstrap = None
    if args.boot_samples > 0:
        y_true, y_prob, _ = trainer.inference(test_loader)
        bootstrap = bootstrap_metrics_ci(
            y_true,
            y_prob,
            [monitor],
            mode=args.task,
            n_boot=args.boot_samples,
            seed=args.seed,
            alpha=0.10,
        )
        for name, ci in bootstrap.items():
            print(
                f"  {name} bootstrap mean: {ci['mean']:.4f}  "
                f"(5%: {ci['lower']:.4f}, 95%: {ci['upper']:.4f}; "
                f"{ci['n_boot']} usable, {ci['n_skipped']} skipped)"
            )

    return {"split": split_number, "test": test_scores,
            "bootstrap": bootstrap}


def print_summary(results: list[dict]) -> None:
    """Print mean +/- std per metric across the splits (paper tables)."""
    print(f"\n=== Summary across {len(results)} splits ===")
    for name in results[0]["test"]:
        values = [r["test"][name] for r in results]
        print(f"  {name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    bootstrap = results[0]["bootstrap"]
    if bootstrap:
        for name, ci in bootstrap.items():
            means = [r["bootstrap"][name]["mean"] for r in results]
            lowers = [r["bootstrap"][name]["lower"] for r in results]
            uppers = [r["bootstrap"][name]["upper"] for r in results]
            print(
                f"  bootstrap {name}: mean {np.mean(means):.4f}, "
                f"5%: {np.mean(lowers):.4f}, 95%: {np.mean(uppers):.4f}"
            )


def main() -> None:
    args = parse_args()
    if args.split == "all":
        splits = sorted(PTBXL_PAPER_FOLD_SPLITS)
    else:
        splits = [int(args.split)]

    results = []
    for split_number in splits:
        torch.manual_seed(args.seed)
        results.append(run_split(args, split_number))

    if len(results) > 1:
        print_summary(results)

    if args.result_json:
        Path(args.result_json).expanduser().write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        logger.info("Wrote per-split results to %s", args.result_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
