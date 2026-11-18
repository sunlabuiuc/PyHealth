"""Train a ResNet-18 on HAM10000 7-class diagnosis for 1 epoch.

Usage:
    python examples/derm/ham10000_classification_resnet18.py \
        /path/to/isic2018 [/path/to/cache] [ham7|climb5]

``root`` holds the unzipped ``ISIC2018_Task3_Training_Input/`` and
``ISIC2018_Task3_Training_GroundTruth/`` folders and
``ISIC2018_Task3_Training_LesionGroupings.csv`` (see
:class:`~pyhealth.datasets.HAM10000Dataset`). The split is by lesion, so no
lesion has images in both training and test.

The image cache holds float32 tensors, about 580 KB per image at 224 px
(about 6 GB for the training set). Pass a smaller ``image_size`` to the task
to make it smaller.
"""

import sys

from pyhealth.datasets import HAM10000Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import HAM10000Classification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    root = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
    label_scheme = sys.argv[3] if len(sys.argv) > 3 else "ham7"

    dataset = HAM10000Dataset(root=root, cache_dir=cache_dir)
    dataset.stats()

    train, val, test = dataset.set_task(
        HAM10000Classification(label_scheme=label_scheme, image_size=224),
        split=PatientSplit((0.7, 0.1, 0.2), seed=0),
    )
    train_loader = get_dataloader(train, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test, batch_size=16, shuffle=False)

    model = TorchvisionModel(
        dataset=train,
        model_name="resnet18",
        model_config={"weights": None},
    )
    trainer = Trainer(
        model=model, metrics=["accuracy", "balanced_accuracy", "f1_macro"]
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
        monitor="balanced_accuracy",
    )
    print(trainer.evaluate(test_loader))

    for part in (train, val, test):
        part.close()
