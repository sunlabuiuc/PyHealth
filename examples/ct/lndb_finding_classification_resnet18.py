"""LNDb lung finding classification with a ResNet-18 on 2.5D CT patches.

Loads the LNDb training CTs with ``LNDbDataset``, cuts a 32-voxel patch
around every radiologist finding with ``LNDbFindingClassification`` (the
axial, coronal and sagittal planes through the finding, as three channels),
splits by patient (one CT each) and trains a randomly initialised
torchvision ResNet-18 for one epoch to tell non-nodules, nodules < 3 mm and
nodules >= 3 mm apart. Pass ``--binary`` for nodule >= 3 mm vs the rest.

Data: open, no account needed (the Zenodo record lists CC BY 4.0 but its
description says CC BY-NC-ND 4.0; follow the stricter terms). Download
``trainset_csv.zip`` and any of ``data0.rar`` .. ``data5.rar`` from
https://doi.org/10.5281/zenodo.8348419, extract them (``unrar x`` or
``bsdtar -xf``) into one folder, then run::

    python examples/ct/lndb_finding_classification_resnet18.py --root /path/to/lndb

Add ``--max-samples 40 --epochs 1`` for a quick smoke run on a few CTs.

Findings whose CT is not extracted are skipped, so one archive (about 40
CTs) is enough to try it. One epoch on a CPU shows the pipeline works; it
is not a benchmark result.
"""

import argparse

from pyhealth.datasets import LNDbDataset, get_dataloader, split_by_patient
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import LNDbFindingClassification
from pyhealth.trainer import Trainer

SEED = 0
SPLIT_RATIOS = (0.7, 0.1, 0.2)  # train / validation / test, by patient


def main() -> None:
    """Train a ResNet-18 on LNDb findings and print test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="Folder with the LNDb CSVs and CTs.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep only the first N findings (quick smoke run).",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = LNDbDataset(root=args.root, cache_dir=args.cache_dir)
    dataset.stats()
    task = LNDbFindingClassification(patch_mode="2.5d", binary=args.binary)
    samples = dataset.set_task(task)
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples.subset(list(range(args.max_samples)))

    train_ds, val_ds, test_ds = split_by_patient(samples, SPLIT_RATIOS, seed=SEED)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise SystemExit("Too few CTs for a train/val/test split; extract more scans.")
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32)
    test_loader = get_dataloader(test_ds, batch_size=32)

    model = TorchvisionModel(
        dataset=samples, model_name="resnet18", model_config={"weights": None}
    )
    metrics = ["roc_auc", "pr_auc"] if args.binary else ["accuracy", "f1_macro"]
    trainer = Trainer(model=model, metrics=metrics, device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )
    print(trainer.evaluate(test_loader))
    samples.close()


if __name__ == "__main__":
    main()
