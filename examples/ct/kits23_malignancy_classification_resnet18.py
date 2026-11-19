"""KiTS23 benign vs malignant kidney tumour classification with a ResNet-18.

Loads the KiTS23 training cases with ``KiTS23Dataset``, cuts the three axial
slices through the largest tumour cross-section of each CT with
``KiTS23MalignancySliceClassification``, splits by patient (one CT each) and
trains a randomly initialised torchvision ResNet-18 for one epoch. Only 9%
of the cases are benign, so the script reports ROC AUC and PR AUC. Use
``KiTS23MalignancyClassification`` instead for a 3D model on whole volumes.

Data: open under CC BY-NC-SA 4.0 (non-commercial). Clone
https://github.com/neheller/kits23, ``pip install -e kits23`` and run
``kits23_download_data`` (about 42 GB of CT), then run::

    python examples/ct/kits23_malignancy_classification_resnet18.py \\
        --root /path/to/kits23/dataset

Cases whose ``imaging.nii.gz`` is not downloaded are skipped, so a few dozen
cases from https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging
are enough to try it. Reading each full CT makes the first ``set_task`` slow;
it is cached afterwards. For a quick smoke run add ``--max-samples 10
--epochs 1``; a small subset must still hold both benign and malignant
cases. One epoch shows the pipeline works; it is not a benchmark result.

The 3D variant, ``KiTS23MalignancyClassification``, has no example: PyHealth
has no basic 3D classifier yet (``CNN`` does not take NIfTI volumes).
"""

import argparse

from pyhealth.datasets import KiTS23Dataset, get_dataloader, split_by_patient
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import KiTS23MalignancySliceClassification
from pyhealth.trainer import Trainer

SEED = 0
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / validation / test, by patient


def main() -> None:
    """Train a ResNet-18 on KiTS23 tumour slices and print test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="The KiTS23 dataset/ folder.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep only the first N cases (quick smoke run).",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = KiTS23Dataset(root=args.root, cache_dir=args.cache_dir)
    dataset.stats()
    samples = dataset.set_task(KiTS23MalignancySliceClassification())
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples.subset(list(range(args.max_samples)))

    train_ds, val_ds, test_ds = split_by_patient(samples, SPLIT_RATIOS, seed=SEED)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise SystemExit("Too few cases for a train/val/test split; download more.")
    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=16)
    test_loader = get_dataloader(test_ds, batch_size=16)

    model = TorchvisionModel(
        dataset=samples, model_name="resnet18", model_config={"weights": None}
    )
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"], device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )
    print(trainer.evaluate(test_loader))
    samples.close()


if __name__ == "__main__":
    main()
