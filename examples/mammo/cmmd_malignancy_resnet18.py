"""Malignant vs benign mammogram classification on CMMD with ResNet-18.

Loads CMMD with ``CMMDDataset``, builds one sample per labelled mammogram with
``CMMDMalignancyClassification`` (labels per breast side), splits by patient
with ``PatientSplit`` (processors are fitted on the training patients only)
and trains a randomly initialised ``TorchvisionModel`` ResNet-18 for one
epoch.

CMMD has no official split. A random patient split mixes the D1 and D2
cohorts; CLIMB's row-order split instead puts almost only malignant D2 cases
in the test set.

Data: open under CC BY 4.0 from
https://www.cancerimagingarchive.net/collection/cmmd/. Put the DICOMs (NBIA
Data Retriever) and ``CMMD_clinicaldata_revision.xlsx`` in one folder (the
``.xlsx`` needs ``openpyxl``; pass ``--clinical-file`` for a CSV export), then
run::

    python examples/mammo/cmmd_malignancy_resnet18.py --root /path/to/cmmd
"""

import argparse

import torch

from pyhealth.datasets import CMMDDataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import CMMDMalignancyClassification
from pyhealth.trainer import Trainer


def main() -> None:
    """Train ResNet-18 on CMMD malignancy for one epoch."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--clinical-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)

    dataset = CMMDDataset(
        root=args.root, clinical_file=args.clinical_file, cache_dir=args.cache_dir
    )
    dataset.stats()
    train_ds, val_ds, test_ds = dataset.set_task(
        CMMDMalignancyClassification(),
        split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0),
    )
    print("images train/val/test:", len(train_ds), len(val_ds), len(test_ds))

    model = TorchvisionModel(
        train_ds, model_name="resnet18", model_config={"weights": None}
    )
    trainer = Trainer(model, device=args.device, metrics=["roc_auc", "accuracy"])
    trainer.train(
        get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True),
        val_dataloader=(
            get_dataloader(val_ds, batch_size=args.batch_size) if len(val_ds) else None
        ),
        epochs=1,
    )
    print("test:", trainer.evaluate(get_dataloader(test_ds, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
