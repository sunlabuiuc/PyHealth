"""Jichi diabetic-retinopathy Davis grading with a ResNet-18, split by patient.

Usage:
    python examples/fundus/jichi_davis_grading_resnet18.py \
        --root /data/jichi/dmr [--label-column davis_concatenated] \
        [--max-samples 64 --epochs 1 --dev]

``root`` is the extracted ``dmr/`` folder (see
:class:`pyhealth.datasets.JichiDRDataset`). The model is randomly initialised
so the script needs no network; pass ``--pretrained`` for ImageNet weights.
``--max-samples`` caps each part (seeded random images) for a quick smoke
run, and ``--dev`` loads at most 1000 patients.
"""

import argparse

import numpy as np

from pyhealth.datasets import JichiDRDataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import JichiDavisGrading
from pyhealth.trainer import Trainer


def cap(part, max_samples, seed=0):
    """Return a seeded random subset of at most ``max_samples`` samples."""
    if max_samples is None or len(part) <= max_samples:
        return part
    rng = np.random.default_rng(seed)
    return part.subset(sorted(rng.choice(len(part), max_samples, replace=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--label-column",
        default="davis_one",
        choices=["davis_one", "davis_concatenated"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    dataset = JichiDRDataset(root=args.root, cache_dir=args.cache_dir, dev=args.dev)
    dataset.stats()

    # About 3.6 images per patient: split by patient, never by image.
    parts = dataset.set_task(
        JichiDavisGrading(label_column=args.label_column),
        split=PatientSplit((0.7, 0.1, 0.2), seed=0),
    )
    train, val, test = (cap(p, args.max_samples) for p in parts)
    print(f"train/val/test samples: {len(train)}/{len(val)}/{len(test)}")

    model = TorchvisionModel(
        dataset=train,
        model_name="resnet18",
        model_config={"weights": "DEFAULT" if args.pretrained else None},
    )
    trainer = Trainer(model=model, metrics=["accuracy", "cohen_kappa"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=16, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=16, shuffle=False),
        epochs=args.epochs,
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=16, shuffle=False)))

    for part in parts:
        part.close()
