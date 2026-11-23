"""Four-class brain tumour MRI classification with a ResNet-18.

Loads the Bhuvaji Brain Tumor Classification (MRI) images with
``BrainTumorMRIDataset`` (byte-identical copies removed, so no test image is
also a training image), builds grayscale 224 x 224 samples with
``BrainTumorMRIClassification``, keeps the official Testing folder as the
test set (fold 10), takes fold 9 of Training for validation, and trains a
randomly initialised torchvision ResNet-18 for one epoch.

Data: open under MIT (per the Kaggle listing), no account needed. Download it
with::

    git clone --depth 1 https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet

then run::

    python examples/mri/brain_tumor_mri_classification_resnet18.py \\
        --root Brain-Tumor-Classification-DataSet

Pass ``--keep-duplicates`` for CLIMB's original split with its duplicate
images. One epoch on a CPU shows the pipeline works; it is not a benchmark
result. For a quick smoke run add ``--max-samples 32 --epochs 1``.
"""

import argparse

from pyhealth.datasets import BrainTumorMRIDataset, get_dataloader, split_by_strat_fold
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import BrainTumorMRIClassification
from pyhealth.trainer import Trainer


def main() -> None:
    """Train a ResNet-18 on the official split and print test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="Folder with Training/ and Testing/.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Load at most 1000 patients.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep at most this many images per split (smoke runs).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = BrainTumorMRIDataset(
        root=args.root,
        cache_dir=args.cache_dir,
        drop_duplicates=not args.keep_duplicates,
        dev=args.dev,
    )
    dataset.stats()
    samples = dataset.set_task(BrainTumorMRIClassification())

    # Folds 1-9 are Training (by duplicate group), fold 10 is Testing. With
    # --keep-duplicates the same image can be in Training and Testing, so the
    # patient-disjoint check would (rightly) fail and is skipped.
    train_ds, val_ds, test_ds = split_by_strat_fold(
        samples,
        train_folds=tuple(range(1, 9)),
        val_folds=(9,),
        test_folds=(10,),
        fold_field="fold",
        check_patient_disjoint=not args.keep_duplicates,
    )
    if args.max_samples is not None:
        train_ds, val_ds, test_ds = (
            part.subset(range(min(args.max_samples, len(part))))
            for part in (train_ds, val_ds, test_ds)
        )
    print("images train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size)

    model = TorchvisionModel(
        dataset=samples, model_name="resnet18", model_config={"weights": None}
    )
    trainer = Trainer(
        model=model, metrics=["accuracy", "f1_macro"], device=args.device
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )
    print(trainer.evaluate(test_loader))
    samples.close()


if __name__ == "__main__":
    main()
