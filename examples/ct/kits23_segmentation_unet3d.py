"""KiTS23 kidney / tumour / cyst segmentation with a 3D U-Net.

Loads the KiTS23 training cases with ``KiTS23Dataset``, pairs each CT with
its segmentation through ``KiTS23Segmentation`` (both resampled to
``--target-shape``), splits by patient (one CT each) and trains PyHealth's
``UNet3D`` (Dice + cross-entropy loss) on the four labels: background,
kidney, tumour, cyst. Validation and test report Dice and IoU averaged over
the foreground labels; the official KiTS23 ranking instead uses nested
evaluation classes (``kits23_compute_metrics``).

Data: open under CC BY-NC-SA 4.0 (non-commercial). Clone
https://github.com/neheller/kits23 (it holds ``kits23.json`` and every
segmentation), ``pip install -e kits23`` and run ``kits23_download_data``
(about 42 GB of CT), or fetch a few CTs from
https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging into
``dataset/case_XXXXX/imaging.nii.gz``. Then run::

    python examples/ct/kits23_segmentation_unet3d.py --root /path/to/kits23/dataset

For a quick CPU smoke run add
``--max-samples 6 --epochs 1 --target-shape 32 64 64 --base-channels 8``.
Coarse resampling of whole CTs loses small tumours and cysts; use
``target_spacing`` and patch training (``PatchDataset``) for real results.
One epoch shows the pipeline works; it is not a benchmark result.
"""

import argparse

from pyhealth.datasets import KiTS23Dataset, get_dataloader, split_by_patient
from pyhealth.models import UNet3D
from pyhealth.tasks import KiTS23Segmentation
from pyhealth.trainer import Trainer

SEED = 0
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / validation / test, by patient


def main() -> None:
    """Train a 3D U-Net on KiTS23 and print test Dice / IoU."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="The KiTS23 dataset/ folder.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep only the first N cases (quick smoke run).",
    )
    parser.add_argument(
        "--target-shape", type=int, nargs=3, default=(64, 128, 128),
        metavar=("D", "H", "W"),
    )
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = KiTS23Dataset(root=args.root, cache_dir=args.cache_dir)
    dataset.stats()
    samples = dataset.set_task(KiTS23Segmentation(target_shape=args.target_shape))
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples.subset(list(range(args.max_samples)))

    train_ds, val_ds, test_ds = split_by_patient(samples, SPLIT_RATIOS, seed=SEED)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise SystemExit("Too few cases for a train/val/test split; download more.")
    train_loader = get_dataloader(train_ds, batch_size=1, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=1)
    test_loader = get_dataloader(test_ds, batch_size=1)

    model = UNet3D(
        dataset=samples, base_channels=args.base_channels, depth=4, norm="instance"
    )
    trainer = Trainer(model=model, metrics=["dice", "iou"], device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )
    print(trainer.evaluate(test_loader))
    samples.close()


if __name__ == "__main__":
    main()
