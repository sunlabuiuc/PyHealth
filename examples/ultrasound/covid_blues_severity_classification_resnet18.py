"""Frame-level LUS severity scoring (0-3) on COVID-BLUES with ResNet-18.

Loads COVID-BLUES frames with ``COVIDBLUESDataset``, gives each frame its
video's expert severity score (Soldati et al., 2020; from ``severity.csv``)
with ``COVIDBLUESSeverityClassification``, splits by patient with
``PatientSplit`` and trains a randomly initialised ``TorchvisionModel``
ResNet-18. Reports accuracy and macro F1 over the four scores, per frame.

Data: CC BY-NC-ND 4.0 (non-commercial; do not share extracted frames).
Clone https://github.com/NinaWie/COVID-BLUES and extract frames, e.g.::

    for v in COVID-BLUES/lus_videos/*.mp4; do
      s=$(basename "$v" .mp4); mkdir -p frames/$s
      ffmpeg -i "$v" frames/$s/%06d.png
    done
    python examples/ultrasound/covid_blues_severity_classification_resnet18.py \\
        --root COVID-BLUES --frames-dir frames --frame-step 5

Pass ``--decode`` instead of ``--frames-dir`` to decode the videos with the
optional ``av`` or ``opencv-python`` package. For a quick smoke run add
``--max-samples 500 --epochs 1``.
"""

import argparse

import torch

from pyhealth.datasets import COVIDBLUESDataset, PatientSplit, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import COVIDBLUESSeverityClassification
from pyhealth.trainer import Trainer


def main() -> None:
    """Train ResNet-18 on COVID-BLUES severity scores and print test metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Keep only the first N frames of each split (quick smoke run).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)

    dataset = COVIDBLUESDataset(
        root=args.root,
        frames_dir=args.frames_dir,
        decode_videos=args.decode,
        frame_step=args.frame_step,
        cache_dir=args.cache_dir,
    )
    dataset.stats()
    train_ds, test_ds = dataset.set_task(
        COVIDBLUESSeverityClassification(),
        split=PatientSplit(ratios=(0.75, 0.25), seed=0),
    )
    if args.max_samples is not None:
        train_ds, test_ds = (
            ds.subset(list(range(min(args.max_samples, len(ds)))))
            for ds in (train_ds, test_ds)
        )
    print("frames train/test:", len(train_ds), len(test_ds))

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
