"""DREAMT sleep staging with SparcNet on raw 30-second windows.

``SleepStagingDREAMT`` emits ``signal`` tensors of shape ``(n_channels, 1920)``
at 64 Hz. This example keeps that **raw multichannel waveform** and trains
:class:`~pyhealth.models.SparcNet`, which applies 1-D convolutions over time.

For a lighter baseline that pools each epoch to summary statistics and uses
:class:`~pyhealth.models.RNN`, see ``dreamt_sleep_staging_rnn.py``.

Usage (synthetic demo, no PhysioNet download)::

    python dreamt_sleep_staging_sparcnet.py --demo

Usage (local DREAMT checkout)::

    python dreamt_sleep_staging_sparcnet.py --root /path/to/dreamt/version
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Dict, List

import numpy as np

from dreamt_sleep_staging_demo_utils import generate_demo_samples

from pyhealth.datasets import DREAMTDataset, create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks.sleep_staging_dreamt import SleepStagingDREAMT
from pyhealth.trainer import Trainer

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_ROOT = os.path.expanduser("~/.pyhealth/dreamt")


def _resolve_root(root_arg: str | None) -> str:
    candidates = (
        [root_arg]
        if root_arg
        else [
            DEFAULT_ROOT,
            os.path.expanduser("~/data/dreamt"),
            os.path.expanduser("~/dreamt"),
        ]
    )
    for path in candidates:
        if path and os.path.isdir(path):
            info = os.path.join(path, "participant_info.csv")
            if os.path.isfile(info):
                return path
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isdir(subpath) and os.path.isfile(
                    os.path.join(subpath, "participant_info.csv")
                ):
                    return subpath
    raise SystemExit(
        "Could not find DREAMT root with participant_info.csv. "
        "Download from https://physionet.org/content/dreamt/ "
        "or pass --root."
    )


def build_raw_signal_dataset(samples: List[Dict[str, Any]]) -> Any:
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="dreamt_sparcnet",
        task_name="sleep_staging",
    )


def train_and_evaluate(
    dataset: Any,
    *,
    device: str,
    epochs: int,
) -> Dict[str, Any]:
    model = SparcNet(dataset=dataset)
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_macro", "cohen_kappa"],
        device=device,
        enable_logging=False,
    )
    train_ds, val_ds, test_ds = split_by_patient(
        dataset, [0.7, 0.1, 0.2], seed=42,
    )
    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=16, shuffle=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
    )
    return trainer.evaluate(test_loader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DREAMT sleep staging with SparcNet (raw signal)",
    )
    parser.add_argument("--root", default=None, help="DREAMT version directory")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data (no dataset on disk)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.demo:
        print("Demo mode: synthetic patients ...")
        raw = generate_demo_samples(
            n_classes=5,
            n_patients=8,
            epochs_per_patient=12,
            seed=7,
        )
    else:
        root = _resolve_root(args.root)
        print(f"Loading DREAMT from {root} ...")
        ds = DREAMTDataset(root=root)
        task = SleepStagingDREAMT(n_classes=5)
        sample_ds = ds.set_task(task)
        raw = [sample_ds[i] for i in range(len(sample_ds))]

    dataset = build_raw_signal_dataset(raw)
    print(f"Samples: {len(raw)}  |  schema signal shape (first): "
          f"{raw[0]['signal'].shape}")
    results = train_and_evaluate(
        dataset,
        device=args.device,
        epochs=args.epochs,
    )
    print("Test metrics:", results)


if __name__ == "__main__":
    main()
