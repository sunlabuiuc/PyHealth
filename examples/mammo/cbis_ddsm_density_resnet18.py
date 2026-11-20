"""Breast density (BI-RADS 1-4) classification on CBIS-DDSM with ResNet-18.

Loads CBIS-DDSM with ``CBISDDSMDataset``, builds image-level samples with
``CBISDDSMDensityClassification`` and trains a randomly initialised
``TorchvisionModel`` ResNet-18.

The label is the ACR BI-RADS breast density of the image (4 classes).

The official train/test split is not patient-disjoint (31 patients are in
both), so the dataset is loaded with ``drop_overlap_patients=True`` and the
official split is passed to ``set_task`` as a patient split, which also fits
the processors on the training patients only. Use ``--patient-split`` for a
random 80/20 split by patient instead.

Data: open under CC BY 3.0 from
https://www.cancerimagingarchive.net/collection/cbis-ddsm/. Put the four
``*_case_description_*_set.csv`` files and the DICOMs (NBIA Data Retriever,
or ``dicom/<SeriesUID>/``) in one folder, then run::

    python examples/mammo/cbis_ddsm_density_resnet18.py --root /path/to/cbis-ddsm

For a quick smoke run on a few images add ``--max-samples 8 --epochs 1``.
"""

import argparse

import torch

from pyhealth.datasets import CBISDDSMDataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import CBISDDSMDensityClassification
from pyhealth.trainer import Trainer


def main() -> None:
    """Train ResNet-18 on CBIS-DDSM breast density."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--patient-split", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Load at most 1000 patients.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep at most this many train and test images (smoke runs).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)

    dataset = CBISDDSMDataset(
        root=args.root,
        cache_dir=args.cache_dir,
        drop_overlap_patients=True,
        dev=args.dev,
    )
    dataset.stats()
    split = (
        PatientSplit(ratios=(0.8, 0.2), seed=0)
        if args.patient_split
        else dataset.official_split()
    )
    train_ds, test_ds = dataset.set_task(CBISDDSMDensityClassification(), split=split)
    if args.max_samples is not None:
        train_ds = train_ds.subset(range(min(args.max_samples, len(train_ds))))
        test_ds = test_ds.subset(range(min(args.max_samples, len(test_ds))))
    print("images train/test:", len(train_ds), len(test_ds))

    model = TorchvisionModel(
        train_ds, model_name="resnet18", model_config={"weights": None}
    )
    trainer = Trainer(model, device=args.device, metrics=["accuracy", "f1_macro"])
    trainer.train(
        get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True),
        epochs=args.epochs,
    )
    print("test:", trainer.evaluate(get_dataloader(test_ds, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
