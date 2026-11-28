"""Fine-tune CBraMod on TUH EEG Abnormal (TUAB).

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC

Paper targets: balanced accuracy 0.8289, AUC-PR 0.9258, AUROC 0.9227.
Uses one logit with BCE (``output_schema`` overridden to binary for training).

References:
    Wang, J. et al. CBraMod. ICLR 2025. https://github.com/wjq-learning/CBraMod
"""

import argparse

from cbramod_eeg_downstream import (
    add_common_args,
    cbramod_task_kwargs,
    pretrained_flag,
    split_tuh_dataset,
    train_cbramod,
    tuab_paper_subject_split,
)

from pyhealth.datasets import TUABDataset, get_dataloader
from pyhealth.models import CBraMod
from pyhealth.tasks import EEGAbnormalTUAB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    kwargs = {"root": args.root}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    dataset = TUABDataset(**kwargs)
    print(dataset.stats())

    task = EEGAbnormalTUAB(**cbramod_task_kwargs())
    task.output_schema = {"label": "binary"}
    train_dataset, val_dataset, test_dataset = split_tuh_dataset(
        dataset,
        task,
        tuab_paper_subject_split,
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
        n_patches=10,
        input_scale=1.0,
        pretrained=pretrained_flag(args),
        frozen=args.frozen,
    )
    train_cbramod(
        model,
        train_loader,
        val_loader,
        test_loader,
        args,
        monitor="roc_auc",
        metrics=["balanced_accuracy", "pr_auc", "roc_auc"],
    )


if __name__ == "__main__":
    main()
