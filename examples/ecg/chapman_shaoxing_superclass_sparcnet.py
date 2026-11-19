"""Chapman-Shaoxing 7-superclass multi-label ECG classification with SparcNet.

Download the open PhysioNet release (CC BY 4.0) from
https://physionet.org/content/ecg-arrhythmia/1.0.0/ and point ``--root`` at
the version folder holding ``WFDBRecords/``. Needs ``pip install wfdb``.

Example::

    python examples/ecg/chapman_shaoxing_superclass_sparcnet.py \\
        --root /data/ecg-arrhythmia/1.0.0 --dev
"""

from __future__ import annotations

import argparse

from pyhealth.datasets import ChapmanShaoxingDataset, PatientSplit, get_dataloader
from pyhealth.models import SparcNet
from pyhealth.tasks import ChapmanShaoxingSuperclassClassification
from pyhealth.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="ecg-arrhythmia/1.0.0 folder")
    parser.add_argument("--dev", action="store_true", help="load 1,000 patients")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = ChapmanShaoxingDataset(root=args.root, dev=args.dev)
    dataset.stats()
    # One ECG per patient, so a patient split is also a record split.
    train, val, test = dataset.set_task(
        ChapmanShaoxingSuperclassClassification(),
        split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0),
    )
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = SparcNet(dataset=train)
    trainer = Trainer(model=model, metrics=["roc_auc_macro", "f1_macro"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=32, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=32),
        epochs=args.epochs,
        monitor="roc_auc_macro",
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=32)))


if __name__ == "__main__":
    main()
