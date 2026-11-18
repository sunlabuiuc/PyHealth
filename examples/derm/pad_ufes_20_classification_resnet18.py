"""PAD-UFES-20 skin-lesion diagnosis with a ResNet-18, split by patient.

Usage:
    python examples/derm/pad_ufes_20_classification_resnet18.py \
        --root /data/pad_ufes_20 [--label-scheme climb5]

``root`` must hold ``metadata.csv`` and the unzipped ``images/`` folder
(see :class:`pyhealth.datasets.PADUFES20Dataset`). The model is randomly
initialised so the script needs no network; pass ``--pretrained`` to use
torchvision's ImageNet weights instead.
"""

import argparse

from pyhealth.datasets import PADUFES20Dataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import PADUFES20Classification
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--label-scheme", default="pad6", choices=["pad6", "climb5"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    dataset = PADUFES20Dataset(root=args.root, cache_dir=args.cache_dir)
    dataset.stats()

    # Split by patient: one patient can have several lesions and images.
    train, val, test = dataset.set_task(
        PADUFES20Classification(label_scheme=args.label_scheme),
        split=PatientSplit((0.7, 0.1, 0.2), seed=0),
    )
    print(f"train/val/test samples: {len(train)}/{len(val)}/{len(test)}")

    model = TorchvisionModel(
        dataset=train,
        model_name="resnet18",
        model_config={"weights": "DEFAULT" if args.pretrained else None},
    )
    trainer = Trainer(model=model, metrics=["accuracy", "f1_macro"])
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=16, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=16, shuffle=False),
        epochs=args.epochs,
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=16, shuffle=False)))

    for part in (train, val, test):
        part.close()
