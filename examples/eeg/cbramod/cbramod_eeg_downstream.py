"""Shared helpers for CBraMod fine-tuning on TUH EEG downstream tasks.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC

References:
    Wang, J. et al. CBraMod: A Criss-Cross Brain Foundation Model for EEG
    Decoding. ICLR 2025. https://github.com/wjq-learning/CBraMod
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from pyhealth.datasets import (
    BaseDataset,
    SampleDataset,
    SubjectSplit,
    split_by_patient_tuh,
)
from pyhealth.models import CBraMod
from pyhealth.trainer import Trainer


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional PyHealth cache dir.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--head-lr-mult", type=float, default=5.0)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train the backbone from scratch instead of loading Hugging Face weights.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local pretrained_weights.pth path.",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Freeze the CBraMod backbone (linear probing).",
    )
    parser.add_argument(
        "--shuffle-split",
        action="store_true",
        help="Use shuffled split_by_patient_tuh instead of the sorted 80/20 CBraMod split.",
    )


def cbramod_task_kwargs() -> dict:
    return {
        "resample_rate": 200.0,
        "bandpass_filter": (0.3, 75.0),
        "notch_filter": 60.0,
        "normalization": "div_by_100",
        "compute_stft": False,
    }


def tuev_paper_subject_split(base_dataset: BaseDataset) -> SubjectSplit:
    """Sorted 80/20 train/val on the TUH train partition; eval is test."""
    train_pids: list[str] = []
    eval_pids: list[str] = []
    for patient in base_dataset.iter_patients():
        pid = patient.patient_id
        if patient.get_events("train"):
            train_pids.append(pid)
        if patient.get_events("eval"):
            eval_pids.append(pid)
    train_pids.sort(key=str)
    cutoff = int(len(train_pids) * 0.8)
    return SubjectSplit(
        (train_pids[:cutoff], train_pids[cutoff:], sorted(set(eval_pids), key=str))
    )


def tuab_paper_subject_split(base_dataset: BaseDataset) -> SubjectSplit:
    """Sorted 80/20 within normal and abnormal train subjects (CBraMod preprocessing)."""
    abnormal: list[str] = []
    normal: list[str] = []
    eval_pids: set[str] = set()
    for patient in base_dataset.iter_patients():
        pid = patient.patient_id
        if patient.get_events("eval"):
            eval_pids.add(pid)
        train_events = patient.get_events("train")
        if not train_events:
            continue
        label = train_events[0].label
        if label == "abnormal":
            abnormal.append(pid)
        else:
            normal.append(pid)
    abnormal.sort(key=str)
    normal.sort(key=str)
    n_ab = int(len(abnormal) * 0.8)
    n_no = int(len(normal) * 0.8)
    train = abnormal[:n_ab] + normal[:n_no]
    val = abnormal[n_ab:] + normal[n_no:]
    return SubjectSplit((train, val, sorted(eval_pids, key=str)))


def build_cbramod_param_groups(
    model: CBraMod,
    lr: float,
    head_lr_mult: float,
    weight_decay: float,
) -> list[dict]:
    """AdamW groups with 5x head learning rate (Table 6, CBraMod paper)."""
    return [
        {
            "params": model.backbone_parameters(),
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": model.head_parameters(),
            "lr": lr * head_lr_mult,
            "weight_decay": weight_decay,
        },
    ]


def cosine_scheduler_factory(
    eta_min: float = 1e-6,
) -> Callable[[Optimizer, int], LRScheduler]:
    """Per-iteration cosine schedule matching finetune_trainer.py."""

    def factory(optimizer: Optimizer, total_steps: int) -> LRScheduler:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    return factory


def train_cbramod(
    model: CBraMod,
    train_loader,
    val_loader,
    test_loader,
    args: argparse.Namespace,
    monitor: str,
    metrics: list[str] | None = None,
) -> dict:
    """Run CBraMod fine-tuning with the paper optimizer recipe."""
    torch.manual_seed(args.seed)
    param_groups = build_cbramod_param_groups(
        model,
        lr=args.lr,
        head_lr_mult=args.head_lr_mult,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(model=model, device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"betas": (0.9, 0.999), "eps": 1e-8},
        weight_decay=0.0,
        max_grad_norm=1.0,
        monitor=monitor,
        param_groups=param_groups,
        lr_scheduler=cosine_scheduler_factory(),
    )
    scores = trainer.evaluate(test_loader)
    if metrics:
        for key in metrics:
            print(f"{key}: {scores.get(key, float('nan')):.4f}")
    else:
        print(scores)
    return scores


def split_tuh_dataset(
    base_dataset: BaseDataset,
    task,
    paper_split_fn: Callable[[BaseDataset], SubjectSplit],
    shuffle: bool,
    seed: int,
) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    if shuffle:
        full = base_dataset.set_task(task)
        return split_by_patient_tuh(full, [0.8, 0.2], seed=seed)
    split = paper_split_fn(base_dataset)
    return base_dataset.set_task(task, split=split)


def pretrained_flag(args: argparse.Namespace) -> bool | str:
    if args.no_pretrained:
        return False
    if args.checkpoint:
        return args.checkpoint
    return True
