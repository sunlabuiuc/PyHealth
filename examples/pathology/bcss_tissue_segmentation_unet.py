"""BCSS five-class tissue segmentation with a 2D U-Net on the official split.

Usage:
    python examples/pathology/bcss_tissue_segmentation_unet.py \
        --root /data/bcss [--tile-size 256 --min-purity 0.3 \
        --split official --max-samples 64 --epochs 1]

``root`` holds ``masks/`` and ``rgbs_colorNormalized/`` (see
:class:`pyhealth.datasets.BCSSDataset`). Tiles of every class are kept (the
dataset is built with all ``BCSS_CLASS_CODES``), and each pixel is mapped to
the BCSS baseline's five classes: other, tumor, stroma, inflammatory
infiltrate and necrosis (:class:`pyhealth.tasks.BCSSTissueSegmentation`).
Outside-ROI pixels are ignored by the loss. Test patients are the official
held-out TCGA sites. ``--split patient`` uses a seeded ``PatientSplit``
instead, for a subset of ROIs without a test-site ROI. ``--max-samples``
caps each part (seeded random tiles) and ``--dev`` loads at most 1000
patients. ``--base-channels`` and ``--depth`` shrink the U-Net for a quick
run (the defaults follow the original U-Net paper).
"""

import argparse

import numpy as np

from pyhealth.datasets import BCSSDataset, PatientSplit, get_dataloader
from pyhealth.datasets.bcss import BCSS_CLASS_CODES
from pyhealth.models import UNet2D
from pyhealth.tasks import BCSSTissueSegmentation
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
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--min-purity", type=float, default=0.3)
    parser.add_argument("--min-labelled", type=float, default=0.5)
    parser.add_argument("--image-dir", default=None)
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
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.split == "official" and not 0 < args.val_ratio < 1:
        # The script trains with a validation part, which needs val_ratio > 0.
        parser.error("--val-ratio must be in (0, 1) with --split official.")

    dataset = BCSSDataset(
        root=args.root,
        tile_size=args.tile_size,
        stride=args.stride,
        min_purity=args.min_purity,
        class_codes=BCSS_CLASS_CODES,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        dev=args.dev,
    )
    dataset.stats()

    task = BCSSTissueSegmentation(
        image_size=args.tile_size, min_labelled=args.min_labelled
    )
    split = (
        dataset.official_split(val_ratio=args.val_ratio, seed=0)
        if args.split == "official"
        else PatientSplit((0.5, 0.25, 0.25), seed=0)
    )
    parts = dataset.set_task(task, split=split)
    train, val, test = (cap(p, args.max_samples) for p in parts)
    print(f"train/val/test tiles: {len(train)}/{len(val)}/{len(test)}")

    model = UNet2D(
        train,
        in_channels=3,
        num_classes=task.num_classes,
        base_channels=args.base_channels,
        depth=args.depth,
    )
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=get_dataloader(
            train, batch_size=args.batch_size, shuffle=True
        ),
        val_dataloader=get_dataloader(val, batch_size=args.batch_size, shuffle=False),
        epochs=args.epochs,
    )
    print(
        trainer.evaluate(
            get_dataloader(test, batch_size=args.batch_size, shuffle=False)
        )
    )

    for part in parts:
        part.close()
