"""BCSS tissue-tile classification with a ResNet-18 on the official split.

Usage:
    python examples/pathology/bcss_tile_classification_resnet18.py \
        --root /data/bcss [--tile-size 224 --stride 224 --min-purity 0.9]

``root`` holds ``masks/`` and ``rgbs_colorNormalized/`` (see
:class:`pyhealth.datasets.BCSSDataset`). The first run tiles every ROI into
the cache. Test patients are the official held-out TCGA sites; validation
patients are carved from the training sites. The model is randomly
initialised so the script needs no network; pass ``--pretrained`` for
ImageNet weights. This recipe does not reproduce CLIMB's numbers.
"""

import argparse

from pyhealth.datasets import BCSSDataset, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import BCSSTileClassification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--min-purity", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    dataset = BCSSDataset(
        root=args.root,
        tile_size=args.tile_size,
        stride=args.stride,
        min_purity=args.min_purity,
        cache_dir=args.cache_dir,
    )
    dataset.stats()

    train, val, test = dataset.set_task(
        BCSSTileClassification(image_size=224),
        split=dataset.official_split(val_ratio=args.val_ratio, seed=0),
    )
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

    for part in (train, val, test):
        part.close()
