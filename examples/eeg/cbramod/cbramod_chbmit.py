"""Fine-tune CBraMod on CHB-MIT seizure detection.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC

Uses the CBraMod/TFM subject split (chb01-chb20 train, chb21-chb22 val,
chb23-chb24 test), matching ``preprocessing/CHB-MIT/process2.py``. The paper
text lists a slightly different hold-out (subjects 1-19 / 20-21 / 22-23).

Paper targets: balanced accuracy 0.7398, AUC-PR 0.3689, AUROC 0.8892.

References:
    Wang, J. et al. CBraMod. ICLR 2025. https://github.com/wjq-learning/CBraMod
"""

import argparse

from cbramod_eeg_downstream import (
    add_common_args,
    pretrained_flag,
    train_cbramod,
)

from pyhealth.datasets import CHBMITDataset, SampleDataset, get_dataloader
from pyhealth.models import CBraMod
from pyhealth.tasks import SeizureDetectionCHBMIT


def drop_augmented(dataset: SampleDataset) -> SampleDataset:
    keep = [i for i in range(len(dataset)) if int(dataset[i]["is_augmented"]) == 0]
    return dataset.subset(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    dataset = CHBMITDataset(root=args.root, cache_dir=args.cache_dir)
    print(dataset.stats())

    split = dataset.default_subject_split()
    task = SeizureDetectionCHBMIT(
        normalization="div_by_100",
        label_mode="cbramod",
        compute_stft=False,
        subjects=[s for part in split.subjects for s in part],
    )
    train_dataset, val_dataset, test_dataset = dataset.set_task(task, split=split)
    val_dataset = drop_augmented(val_dataset)
    test_dataset = drop_augmented(test_dataset)
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
