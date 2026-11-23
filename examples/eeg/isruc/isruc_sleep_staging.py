"""Sleep staging on ISRUC-Sleep with SparcNet.

Loads one ISRUC-Sleep subgroup with ``ISRUCDataset``, cuts each recording
into 30 second epochs on the six EEG channels with ``SleepStagingISRUC``,
splits by patient and trains ``SparcNet``.

Data: download a subgroup from https://sleeptight.isr.uc.pt/ and extract
it, then run::

    python examples/eeg/isruc/isruc_sleep_staging.py --root /path/to/ISRUC-Sleep-III

Subgroup III (10 healthy subjects) is the smallest download. The first run
writes ``isruc-pyhealth.csv`` and an ``.edf`` link for each ``.rec`` file.
"""

import argparse

from pyhealth.datasets import ISRUCDataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks import SleepStagingISRUC
from pyhealth.trainer import Trainer


def main() -> None:
    """Train SparcNet on ISRUC-Sleep sleep staging.

    Args:
        None. Options come from the command line: ``--root``, ``--annotator``,
        ``--epochs``, ``--batch-size`` and ``--device``.

    Returns:
        None. Prints the test set metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Extracted ISRUC-Sleep subgroup folder, the one holding 1, 2, ...",
    )
    parser.add_argument(
        "--annotator",
        type=int,
        default=1,
        choices=(1, 2),
        help="Which expert's scoring to train on.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = ISRUCDataset(root=args.root)
    dataset.stats()

    samples = dataset.set_task(SleepStagingISRUC(annotator=args.annotator))
    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, [0.6, 0.2, 0.2], seed=0
    )
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

    model = SparcNet(dataset=samples)
    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="cohen_kappa",
    )
    print(trainer.evaluate(test_loader))


# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    main()
