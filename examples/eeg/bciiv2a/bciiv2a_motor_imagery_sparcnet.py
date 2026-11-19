"""BCI Competition IV 2a four-class motor imagery with SparcNet.

Uses the official cross-session protocol (session T trains, session E
tests), pooling the selected subjects. Data: BNCI Horizon 001-2014
(CC BY-ND 4.0), https://bnci-horizon-2020.eu/database/data-sets.

Example::

    python examples/eeg/bciiv2a/bciiv2a_motor_imagery_sparcnet.py \\
        --root /data/bnci/001-2014 --subjects 1 --download \\
        [--max-samples 64 --epochs 1 --dev]

``--max-samples`` caps the train and test parts (seeded random trials) for a
quick smoke run; ``--dev`` is passed to the dataset.
"""

from __future__ import annotations

import argparse

import numpy as np

from pyhealth.datasets import (
    BCICompetitionIV2aDataset,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.models import SparcNet
from pyhealth.tasks import BCIIV2aMotorImageryClassification
from pyhealth.trainer import Trainer


def cap(part, max_samples, seed=0):
    """Return a seeded random subset of at most ``max_samples`` samples."""
    if max_samples is None or len(part) <= max_samples:
        return part
    rng = np.random.default_rng(seed)
    return part.subset(sorted(rng.choice(len(part), max_samples, replace=False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="folder of A0sT/A0sE.mat")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = BCICompetitionIV2aDataset(
        root=args.root, subjects=args.subjects, download=args.download, dev=args.dev
    )
    task = BCIIV2aMotorImageryClassification(l_freq=4.0, h_freq=40.0)
    samples = dataset.set_task(task)
    # Official split: session_fold 0 = T (train), 1 = E (test). Each subject
    # appears in both parts, so a PatientSplit would be cross-subject instead.
    train, _, test = split_by_strat_fold(
        samples,
        train_folds=(0,),
        val_folds=(),
        test_folds=(1,),
        fold_field="session_fold",
    )
    train, test = cap(train, args.max_samples), cap(test, args.max_samples)
    print(f"train={len(train)} test={len(test)}")

    model = SparcNet(dataset=samples)
    trainer = Trainer(model=model, metrics=["accuracy", "cohen_kappa"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=32, shuffle=True),
        epochs=args.epochs,
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=32)))
    samples.close()


if __name__ == "__main__":
    main()
