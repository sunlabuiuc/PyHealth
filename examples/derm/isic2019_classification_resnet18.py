"""Train a ResNet-18 on ISIC 2019 diagnosis (8 classes, or CLIMB's 5).

Usage:
    python examples/derm/isic2019_classification_resnet18.py \
        --root /path/to/isic2019 [--label-scheme climb5] [--epochs 1] [--dev]

``root`` holds ``ISIC_2019_Training_GroundTruth.csv``,
``ISIC_2019_Training_Metadata.csv`` and the unzipped
``ISIC_2019_Training_Input/`` folder (see
:class:`~pyhealth.datasets.ISIC2019Dataset` for the download steps). The
split is by lesion where the lesion id is known, and the "test" metrics it
prints are on a 20% lesion holdout of the training split, not on the
public ISIC 2019 test set (whose labels were released later and are loaded
with ``splits=("train", "test")``). ``--exclude-ham10000``
leaves out the 10,015 images shared with HAM10000.

The image cache holds float32 tensors, about 580 KB per image at 224 px
(about 15 GB for the training set). ``--image-size 128`` shrinks it to about
190 KB per image. ``--dev`` limits the dataset to 1,000 lesions. The model
is randomly initialised so the script needs no network; ``--pretrained``
uses torchvision's ImageNet weights instead.
"""

import argparse

from pyhealth.datasets import ISIC2019Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import ISIC2019Classification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--label-scheme", default="isic8", choices=["isic8", "climb5"])
    parser.add_argument("--exclude-ham10000", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    dataset = ISIC2019Dataset(root=args.root, cache_dir=args.cache_dir, dev=args.dev)
    dataset.stats()

    task = ISIC2019Classification(
        label_scheme=args.label_scheme,
        exclude_ham10000=args.exclude_ham10000,
        image_size=args.image_size,
        split="train",
    )
    train, val, test = dataset.set_task(
        task, split=PatientSplit((0.7, 0.1, 0.2), seed=0)
    )
    print(f"train/val/test samples: {len(train)}/{len(val)}/{len(test)}")
    train_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test, batch_size=args.batch_size, shuffle=False)

    model = TorchvisionModel(
        dataset=train,
        model_name="resnet18",
        model_config={"weights": "DEFAULT" if args.pretrained else None},
    )
    trainer = Trainer(
        model=model, metrics=["accuracy", "balanced_accuracy", "f1_macro"]
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="balanced_accuracy",
    )
    print(trainer.evaluate(test_loader))

    for part in (train, val, test):
        part.close()
