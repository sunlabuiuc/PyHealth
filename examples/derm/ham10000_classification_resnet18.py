"""Train a ResNet-18 on HAM10000 diagnosis (7 classes, or CLIMB's 5).

Usage:
    python examples/derm/ham10000_classification_resnet18.py \
        --root /path/to/isic2018_task3 [--label-scheme climb5] [--epochs 1] [--dev]

``root`` holds the unzipped ``ISIC2018_Task3_Training_Input/`` and
``ISIC2018_Task3_Training_GroundTruth/`` folders and
``ISIC2018_Task3_Training_LesionGroupings.csv`` (see
:class:`~pyhealth.datasets.HAM10000Dataset` for the download steps). The
split is by lesion, so no lesion has images in both training and test.

The image cache holds float32 tensors, about 580 KB per image at 224 px
(about 6 GB for the training set). A smaller ``--image-size`` shrinks it:
128 px is about 190 KB per image. ``--dev`` limits the dataset to 1,000
lesions. The model is randomly initialised so the script needs no network;
``--pretrained`` uses torchvision's ImageNet weights instead.
"""

import argparse

from pyhealth.datasets import HAM10000Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import HAM10000Classification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--label-scheme", default="ham7", choices=["ham7", "climb5"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    dataset = HAM10000Dataset(root=args.root, cache_dir=args.cache_dir, dev=args.dev)
    dataset.stats()

    task = HAM10000Classification(
        label_scheme=args.label_scheme, image_size=args.image_size, split="train"
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
