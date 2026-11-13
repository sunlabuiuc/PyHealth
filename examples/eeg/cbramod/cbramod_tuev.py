"""Fine-tune CBraMod on TUH EEG Events (TUEV).

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC

Paper targets (CBraMod, 5 seeds): balanced accuracy 0.6671, Cohen kappa 0.6772,
weighted F1 0.8342. Table 6 uses batch size 64, 50 epochs, AdamW lr 1e-4 with
5x head lr, weight decay 5e-2, cosine schedule to 1e-6, grad clip 1, label
smoothing 0.1. Official code defaults batch size 512; this script defaults to 64.

References:
    Wang, J. et al. CBraMod. ICLR 2025. https://arxiv.org/abs/2412.07236
    https://github.com/wjq-learning/CBraMod
    https://huggingface.co/weighting666/CBraMod
"""

import argparse

from cbramod_eeg_downstream import (
    add_common_args,
    cbramod_task_kwargs,
    pretrained_flag,
    split_tuh_dataset,
    train_cbramod,
    tuev_paper_subject_split,
)

from pyhealth.datasets import TUEVDataset, get_dataloader
from pyhealth.models import CBraMod
from pyhealth.tasks import EEGEventsTUEV


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    kwargs = {"root": args.root}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    dataset = TUEVDataset(**kwargs)
    print(dataset.stats())

    task = EEGEventsTUEV(**cbramod_task_kwargs())
    train_dataset, val_dataset, test_dataset = split_tuh_dataset(
        dataset,
        task,
        tuev_paper_subject_split,
        shuffle=args.shuffle_split,
        seed=args.seed,
    )
    print("samples train/val/test", len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = CBraMod(
        dataset=train_dataset,
        n_patches=5,
        input_scale=1.0,
        label_smoothing=0.1,
        pretrained=pretrained_flag(args),
        frozen=args.frozen,
    )
    train_cbramod(
        model,
        train_loader,
        val_loader,
        test_loader,
        args,
        monitor="cohen_kappa",
        metrics=["balanced_accuracy", "cohen_kappa", "f1_weighted"],
    )


if __name__ == "__main__":
    main()
