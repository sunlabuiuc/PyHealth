"""Train a ResNet-18 on ISIC 2020 benign/malignant classification for 1 epoch.

Usage:
    python examples/derm/isic2020_melanoma_classification_resnet18.py \
        /path/to/isic2020 [/path/to/cache]

``root`` holds ``ISIC_2020_Training_GroundTruth_v2.csv``,
``ISIC_2020_Training_Duplicates.csv`` and the unzipped ``train/`` folder
(see :class:`~pyhealth.datasets.ISIC2020Dataset`). The split is by patient,
so no patient has images in both training and test. Only 1.8% of images are
malignant, so look at ROC-AUC and PR-AUC rather than accuracy.

The image cache holds float32 tensors, about 580 KB per image at 224 px
(about 19 GB for the full set). Pass ``image_size=128`` to the task, or
resize the JPEGs beforehand, to make it smaller and faster to build.
"""

import sys

from pyhealth.datasets import ISIC2020Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import ISIC2020MelanomaClassification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    root = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None

    dataset = ISIC2020Dataset(root=root, cache_dir=cache_dir)
    dataset.stats()

    train, val, test = dataset.set_task(
        ISIC2020MelanomaClassification(image_size=224),
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
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc", "accuracy"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
        monitor="roc_auc",
    )
    print(trainer.evaluate(test_loader))

    for part in (train, val, test):
        part.close()
