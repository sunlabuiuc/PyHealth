"""Molecular subtype classification of CMMD mammograms with ResNet-18.

Loads CMMD with ``CMMDDataset``, builds one sample per mammogram with
``CMMDSubtypeClassification`` (D2 malignant breasts with a subtype), splits by patient
with ``PatientSplit`` (processors are fitted on the training patients only)
and trains a randomly initialised ``TorchvisionModel`` ResNet-18.

The four classes are Luminal A, Luminal B, HER2-enriched and triple
negative; only D2 breasts with a subtype in the clinical sheet are used.

Data: open under CC BY 4.0 from
https://www.cancerimagingarchive.net/collection/cmmd/. Put the DICOMs (NBIA
Data Retriever) and ``CMMD_clinicaldata_revision.xlsx`` in one folder (the
``.xlsx`` needs ``openpyxl``; pass ``--clinical-file`` for a CSV export), then
run::

    python examples/mammo/cmmd_subtype_resnet18.py --root /path/to/cmmd

For a quick smoke run on a few images add ``--max-samples 16 --epochs 1``.
"""

import argparse

import torch

from pyhealth.datasets import CMMDDataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import CMMDSubtypeClassification
from pyhealth.trainer import Trainer


def main() -> None:
    """Train ResNet-18 on CMMD molecular subtype."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--clinical-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dev", action="store_true", help="Load at most 1000 patients.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep at most this many images per split (smoke runs).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)

    dataset = CMMDDataset(
        root=args.root,
        clinical_file=args.clinical_file,
        cache_dir=args.cache_dir,
        dev=args.dev,
    )
    dataset.stats()
    parts = dataset.set_task(
        CMMDSubtypeClassification(),
        split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0),
    )
    if args.max_samples is not None:
        parts = [p.subset(range(min(args.max_samples, len(p)))) for p in parts]
    train_ds, val_ds, test_ds = parts
    print("images train/val/test:", len(train_ds), len(val_ds), len(test_ds))

    model = TorchvisionModel(
        train_ds, model_name="resnet18", model_config={"weights": None}
    )
    trainer = Trainer(model, device=args.device, metrics=["accuracy", "f1_macro"])
    trainer.train(
        get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True),
        val_dataloader=(
            get_dataloader(val_ds, batch_size=args.batch_size) if len(val_ds) else None
        ),
        epochs=args.epochs,
    )
    print("test:", trainer.evaluate(get_dataloader(test_ds, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
