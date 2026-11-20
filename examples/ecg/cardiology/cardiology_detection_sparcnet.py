"""Arrhythmia detection on the PhysioNet/CinC Challenge 2020 ECGs with SparcNet.

Pipeline (CardiologyDataset -> CardiologyDetection -> SparcNet):

1. Load the Challenge 2020 training set (or some of its six sources).
2. Cut 10-second windows at 500 Hz, labeled 1 when the recording has a
   diagnosis of the chosen category (default ``AR``, arrhythmias).
3. Split by patient 70/10/20; each recording is its own patient.
4. Train SparcNet and report the test metrics.

This is the PyHealth 2.0 version of the 1.x example
``examples/old/cardiology_detection_isAR_SparcNet.py``.

Usage:
    # Quick check on synthetic ECGs (no download, runs on CPU):
    python examples/ecg/cardiology/cardiology_detection_sparcnet.py --synthetic

    # Real data (open access, 4.5 GB zipped):
    #   wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/
    python examples/ecg/cardiology/cardiology_detection_sparcnet.py \\
        --root physionet.org/files/challenge-2020/1.0.2/training \\
        --sources cpsc_2018 georgia --category AR --epochs 5

Note:
    Results on ``--synthetic`` data only show that the pipeline runs.
"""

import argparse
import os
import tempfile

import numpy as np
from scipy.io import savemat

from pyhealth.datasets import CardiologyDataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks import CardiologyDetection
from pyhealth.trainer import Trainer

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def build_synthetic_challenge(root: str, n_records: int = 40, seed: int = 0) -> None:
    """Writes a small Challenge-like folder: a 500 Hz and a 1000 Hz source.

    Half of the recordings carry an arrhythmia code (atrial fibrillation,
    164889003) and an added fast rhythm, so the label is learnable.
    """
    rng = np.random.default_rng(seed)
    for source, fs in (("cpsc_2018", 500), ("ptb", 1000)):
        folder = os.path.join(root, source, "g1")
        os.makedirs(folder)
        for i in range(n_records // 2):
            name = f"{source[0].upper()}{i + 1:04d}"
            n = fs * 15
            t = np.arange(n) / fs
            arrhythmia = i % 2 == 0
            signal = 100 * rng.normal(size=(12, n)) + 500 * np.sin(2 * np.pi * t)
            if arrhythmia:
                signal += 400 * np.sin(2 * np.pi * 7 * t)
            savemat(
                os.path.join(folder, f"{name}.mat"), {"val": signal.astype(np.int16)}
            )
            dx = "164889003" if arrhythmia else "426783006"
            header = [f"{name} 12 {fs} {n}"]
            header += [
                f"{name}.mat 16x1+24 1000.0(0)/mV 16 0 0 0 0 {lead}" for lead in LEADS
            ]
            header += [
                "# Age: 60",
                "# Sex: Female",
                f"# Dx: {dx}",
                "# Rx: Unknown",
                "# Hx: Unknown",
                "# Sx: Unknown",
            ]
            with open(os.path.join(folder, f"{name}.hea"), "w") as handle:
                handle.write("\n".join(header) + "\n")


def main() -> None:
    """Parses arguments and runs the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="the Challenge's training/ folder")
    parser.add_argument(
        "--synthetic", action="store_true", help="tiny fake dataset for a smoke test"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="source folders to load (default: all found)",
    )
    parser.add_argument(
        "--category", default="AR", choices=["AR", "BBBFB", "AD", "CD", "WA"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dev", action="store_true", help="small subset of patients")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.path.expanduser("~"), ".cache", "pyhealth", "cardiology_example"
        ),
        help="processed-data cache, kept between runs "
        "(ignored with --synthetic; default: %(default)s)",
    )
    args = parser.parse_args()
    if not args.root and not args.synthetic:
        parser.error(
            "pass --root /path/to/challenge-2020/1.0.2/training (or --synthetic)"
        )

    with tempfile.TemporaryDirectory() as scratch:
        if args.synthetic:
            args.root = os.path.join(scratch, "training")
            build_synthetic_challenge(args.root)
            args.epochs = min(args.epochs, 3)
        dataset = CardiologyDataset(
            root=args.root,
            sources=args.sources,
            dev=args.dev,
            cache_dir=(
                os.path.join(scratch, "cache") if args.synthetic else args.cache_dir
            ),
        )
        dataset.stats()
        samples = dataset.set_task(CardiologyDetection(args.category))
        train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=0)
        print(f"windows: train={len(train)} val={len(val)} test={len(test)}")

        model = SparcNet(dataset=samples)
        trainer = Trainer(
            model=model, device=args.device, metrics=["pr_auc", "roc_auc", "f1"]
        )
        trainer.train(
            train_dataloader=get_dataloader(
                train, batch_size=args.batch_size, shuffle=True
            ),
            val_dataloader=get_dataloader(val, batch_size=args.batch_size),
            epochs=args.epochs,
            monitor="pr_auc",
        )
        print(
            "test:", trainer.evaluate(get_dataloader(test, batch_size=args.batch_size))
        )


if __name__ == "__main__":
    main()
