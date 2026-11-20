"""BCSS tissue-tile classification with a ResNet-18 on the official split.

Usage:
    python examples/pathology/bcss_tile_classification_resnet18.py \
        --root /data/bcss [--tile-size 224 --stride 224 --min-purity 0.9 \
        --class-codes 1 2 3 4 --image-dir rgbs_colorNormalized \
        --split official --max-samples 64 --epochs 1]

``root`` holds ``masks/`` and ``rgbs_colorNormalized/`` (see
:class:`pyhealth.datasets.BCSSDataset`). The first run tiles every ROI into
the cache. Test patients are the official held-out TCGA sites; validation
patients are carved from the training sites. The model is randomly
initialised so the script needs no network; pass ``--pretrained`` for
ImageNet weights. This recipe does not reproduce CLIMB's numbers.

``--split patient`` uses a seeded ``PatientSplit`` instead, for a subset of
ROIs that has no test-site ROI. ``--max-samples`` caps each part (seeded
random tiles) for a quick smoke run, and ``--dev`` loads at most 1000
patients.
"""

import argparse

import numpy as np

from pyhealth.datasets import BCSSDataset, PatientSplit, get_dataloader
from pyhealth.datasets.bcss import BCSS_MAIN_CLASSES
from pyhealth.models import TorchvisionModel
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
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--min-purity", type=float, default=0.9)
    parser.add_argument(
        "--class-codes",
        type=int,
        nargs="+",
        default=list(BCSS_MAIN_CLASSES),
        help="Mask codes that may label a tile (see BCSS_CLASS_CODES).",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="RGB folder under root (default: rgbs_colorNormalized, else images).",
    )
    parser.add_argument("--split", choices=["official", "patient"], default="official")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Share of training patients used for validation; must be in (0, 1).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    if args.split == "official" and not 0 < args.val_ratio < 1:
        # The script trains with a validation part, which needs val_ratio > 0.
        parser.error("--val-ratio must be in (0, 1) with --split official.")

    dataset = BCSSDataset(
        root=args.root,
        tile_size=args.tile_size,
        stride=args.stride,
        min_purity=args.min_purity,
        class_codes=args.class_codes,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        dev=args.dev,
    )
    dataset.stats()

    split = (
        dataset.official_split(val_ratio=args.val_ratio, seed=0)
        if args.split == "official"
        else PatientSplit((0.5, 0.25, 0.25), seed=0)
    )
    parts = dataset.set_task(dataset.default_task, split=split)
    train, val, test = (cap(p, args.max_samples) for p in parts)
    print(f"train/val/test tiles: {len(train)}/{len(val)}/{len(test)}")

    model = TorchvisionModel(
        dataset=train,
        model_name="resnet18",
        model_config={"weights": "DEFAULT" if args.pretrained else None},
    )
    trainer = Trainer(model=model, metrics=["accuracy", "f1_macro"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=32, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=32, shuffle=False),
        epochs=args.epochs,
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=32, shuffle=False)))

    for part in parts:
        part.close()
