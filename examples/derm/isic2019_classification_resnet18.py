"""Train a ResNet-18 on ISIC 2019 8-class diagnosis for 1 epoch.

Usage:
    python examples/derm/isic2019_classification_resnet18.py \
        /path/to/isic2019 [/path/to/cache] [isic8|climb5]

``root`` holds ``ISIC_2019_Training_GroundTruth.csv``,
``ISIC_2019_Training_Metadata.csv`` and the unzipped
``ISIC_2019_Training_Input/`` folder (see
:class:`~pyhealth.datasets.ISIC2019Dataset`). The split is by lesion where
the lesion id is known. Pass ``exclude_ham10000=True`` to the task to leave
out the 10,015 images shared with HAM10000.

The image cache holds float32 tensors, about 580 KB per image at 224 px
(about 15 GB for the training set). Pass a smaller ``image_size`` to the task
to make it smaller.
"""

import sys

from pyhealth.datasets import ISIC2019Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import ISIC2019Classification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    root = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
    label_scheme = sys.argv[3] if len(sys.argv) > 3 else "isic8"

    dataset = ISIC2019Dataset(root=root, cache_dir=cache_dir)
    dataset.stats()

    train, val, test = dataset.set_task(
        ISIC2019Classification(label_scheme=label_scheme, image_size=224),
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
