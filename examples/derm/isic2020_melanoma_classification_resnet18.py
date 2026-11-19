"""Train a ResNet-18 on ISIC 2020 benign/malignant classification.

Usage:
    python examples/derm/isic2020_melanoma_classification_resnet18.py \
        --root /path/to/isic2020 [--epochs 1] [--dev] [--image-size 128]

``root`` holds ``ISIC_2020_Training_GroundTruth_v2.csv``,
``ISIC_2020_Training_Duplicates.csv`` and the unzipped ``train/`` folder
(see :class:`~pyhealth.datasets.ISIC2020Dataset` for the download steps).
The split is by patient, so no patient has images in both training and
test. Only 1.8% of images are malignant, so look at ROC-AUC and PR-AUC
rather than accuracy.

The image cache holds float32 tensors, about 580 KB per image at 224 px
(about 19 GB for the full set). ``--image-size 128`` shrinks it to about
190 KB per image. ``--dev`` limits the dataset to 1,000 patients. The model
is randomly initialised so the script needs no network; ``--pretrained``
uses torchvision's ImageNet weights instead.
"""

import argparse

from pyhealth.datasets import ISIC2020Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import ISIC2020MelanomaClassification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    dataset = ISIC2020Dataset(root=args.root, cache_dir=args.cache_dir, dev=args.dev)
    dataset.stats()

    train, val, test = dataset.set_task(
        ISIC2020MelanomaClassification(image_size=args.image_size),
        split=PatientSplit((0.7, 0.1, 0.2), seed=0),
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
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc", "accuracy"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="roc_auc",
    )
    print(trainer.evaluate(test_loader))

    for part in (train, val, test):
        part.close()
