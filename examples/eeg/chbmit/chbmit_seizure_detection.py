"""Seizure detection on CHB-MIT with a subject-wise split.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: Loads CHB-MIT, builds 10-second windows with
    SeizureDetectionCHBMIT, and trains SparcNet. The split is the CBraMod /
    TFM-Tokenizer subject split (chb01-chb20 train, chb21-chb22 validation,
    chb23-chb24 test).

References:
    - Wang, J. et al. CBraMod: A Criss-Cross Brain Foundation Model for
      EEG Decoding. ICLR 2025. https://github.com/wjq-learning/CBraMod
    - Pradeepkumar, J. et al. Tokenizing Single-Channel EEG with
      Time-Frequency Motif Learning. ICLR 2026.
      https://github.com/Jathurshan0330/TFM-Tokenizer

The full corpus is large. Pass ``--subjects`` to try a few cases first,
for example ``--subjects 1 2 3``. ``compute_stft`` is off because SparcNet
takes a single feature. Extra (augmented) seizure windows are kept for
training but dropped from validation and test, so those metrics are on the
regular non-overlapping windows only.
"""

import argparse

from pyhealth.datasets import (
    CHBMITDataset,
    SampleDataset,
    SubjectSplit,
    get_dataloader,
)
from pyhealth.models import SparcNet
from pyhealth.tasks import SeizureDetectionCHBMIT
from pyhealth.tasks.seizure_detection_chbmit import normalize_subject
from pyhealth.trainer import Trainer


def drop_augmented(dataset: SampleDataset) -> SampleDataset:
    """Keep only the regular windows of a split part.

    Args:
        dataset: A split part returned by ``set_task``.

    Returns:
        A subset view without the ``is_augmented == 1`` windows.
    """
    keep = [i for i in range(len(dataset)) if int(dataset[i]["is_augmented"]) == 0]
    return dataset.subset(keep)


def main() -> None:
    """Train SparcNet on CHB-MIT seizure detection.

    Args:
        None. Options come from the command line: ``--root``, ``--epochs``,
        ``--batch-size``, ``--device`` and ``--subjects``.

    Returns:
        None. Prints the test-set metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/srv/local/data/physionet.org/files/chbmit/1.0.0",
        help="Directory that contains the chb01 ... chb24 folders.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject ids. With three or more, the first is train, "
        "the second is validation and the rest are test.",
    )
    args = parser.parse_args()

    dataset = CHBMITDataset(root=args.root)
    print(dataset.stats())

    if args.subjects:
        subjects = [normalize_subject(s) for s in args.subjects]
        if len(subjects) < 3:
            parser.error("--subjects needs at least one train, one val and one test id")
        split = SubjectSplit((subjects[:1], subjects[1:2], subjects[2:]))
    else:
        split = dataset.default_subject_split()

    # div_by_100 and cbramod match the CBraMod loader. STFT is omitted so
    # the sample has the one signal feature SparcNet expects. Restricting
    # the task to the split's subjects skips reading every other EDF.
    task = SeizureDetectionCHBMIT(
        normalization="div_by_100",
        label_mode="cbramod",
        compute_stft=False,
        subjects=[s for part in split.subjects for s in part],
    )
    train_dataset, val_dataset, test_dataset = dataset.set_task(task, split=split)
    val_dataset = drop_augmented(val_dataset)
    test_dataset = drop_augmented(test_dataset)
    print(
        "samples train/val/test",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = SparcNet(dataset=train_dataset)
    trainer = Trainer(model=model, device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="pr_auc",
        optimizer_params={"lr": 1e-3},
    )
    print(trainer.evaluate(test_loader))


if __name__ == "__main__":
    main()
