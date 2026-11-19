"""ABIDE I autism vs. control from functional connectivity (logistic baseline).

Builds the BrainGB-style Pearson connectivity of the C-PAC / filt_noglobal /
cc200 ROI series and fits PyHealth's ``LogisticRegression`` on the flattened
upper triangle (19,900 features). Data: ABIDE Preprocessed (CC BY-NC-SA 3.0),
http://preprocessed-connectomes-project.org/abide/ ; ``--download`` fetches
it from the open ``fcp-indi`` S3 bucket.

Example::

    python examples/fmri/abide_autism_logistic_regression.py \\
        --root /data/abide --download --max-subjects 200

``--max-subjects`` takes the first scans in phenotypic-file order, which are
grouped by site; a small value (e.g. 12, all autism at PITT) can give a
single-class subset, and the binary label processor then fails. Use 200 or
more, or leave it out.
"""

from __future__ import annotations

import argparse

from pyhealth.datasets import ABIDEDataset, PatientSplit, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.tasks import ABIDEAutismClassification
from pyhealth.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = ABIDEDataset(
        root=args.root, download=args.download, max_subjects=args.max_subjects
    )
    dataset.stats()
    if args.max_subjects is not None and args.max_subjects < 200:
        print(
            "Warning: a small --max-subjects can hold only one class; "
            "the binary label processor needs both."
        )
    # One scan per subject, so a patient split does not leak.
    train, val, test = dataset.set_task(
        ABIDEAutismClassification(vectorize=True),
        split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0),
    )
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = LogisticRegression(dataset=train)
    trainer = Trainer(model=model, metrics=["roc_auc", "accuracy"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=32, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=32),
        epochs=args.epochs,
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=32)))


if __name__ == "__main__":
    main()
