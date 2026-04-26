"""ibi_sleep_staging_ibi_watchsleepnet.py — WatchSleepNet two-phase transfer learning.

Demonstrates pre-training on clinical IBI data (SHHS/MESA) then fine-tuning on
wearable IBI data (DREAMT), following the WatchSleepNet paper methodology.

Quick start with synthetic data:

    python examples/ibi_sleep_staging_ibi_watchsleepnet.py --synthetic

With real preprocessed data:

    python examples/ibi_sleep_staging_ibi_watchsleepnet.py \\
        --clinical_root ~/watchsleepnet_data/shhs \\
        --wearable_root ~/watchsleepnet_data/dreamt

To produce the NPZ directories from raw recordings, run the preprocessing scripts:

    python examples/preprocess_shhs_to_ibi.py \\
        --src_dir /data/shhs/polysomnography \\
        --dst_dir ~/watchsleepnet_data/shhs \\
        --harmonized_csv /data/shhs/shhs-harmonized-dataset.csv
    python examples/preprocess_mesa_to_ibi.py \\
        --src_dir /data/mesa/polysomnography \\
        --dst_dir ~/watchsleepnet_data/mesa \\
        --harmonized_csv /data/mesa/mesa-sleep-harmonized-dataset.csv
    python examples/preprocess_dreamt_to_ibi.py \\
        --src_dir /data/dreamt/raw \\
        --dst_dir ~/watchsleepnet_data/dreamt \\
        --participant_info /data/dreamt/participant_info.csv
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch

from pyhealth.datasets import IBISleepDataset, get_dataloader
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingIBI
from pyhealth.trainer import Trainer


def _make_synthetic_data(
    root: str,
    n_subjects: int,
    epochs_per_subject: int,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    n = epochs_per_subject * 750
    for i in range(n_subjects):
        np.savez(
            os.path.join(root, f"S{i:04d}.npz"),
            data=rng.random(n).astype(np.float32) * 0.5 + 0.6,
            stages=rng.integers(0, 5, size=n).astype(np.int32),
            fs=np.int64(25),
            ahi=np.float32(rng.uniform(0.0, 30.0)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WatchSleepNet two-phase transfer learning"
    )
    parser.add_argument(
        "--clinical_root",
        help="Directory of preprocessed clinical NPZ files (SHHS/MESA)",
    )
    parser.add_argument(
        "--wearable_root",
        help="Directory of preprocessed wearable NPZ files (DREAMT)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data and run on it",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=30,
        help="Epochs for phase 1 clinical pre-training",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=30,
        help="Epochs for phase 2 wearable fine-tuning",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.synthetic:
        _tmpdir = tempfile.mkdtemp(prefix="watchsleepnet_")
        clinical_root = os.path.join(_tmpdir, "clinical")
        wearable_root = os.path.join(_tmpdir, "wearable")
        print(f"Generating synthetic data in {_tmpdir}")
        _make_synthetic_data(clinical_root, n_subjects=40, epochs_per_subject=20)
        _make_synthetic_data(
            wearable_root, n_subjects=20, epochs_per_subject=15, seed=1
        )
    else:
        if not args.clinical_root or not args.wearable_root:
            parser.error(
                "--clinical_root and --wearable_root are required (or use --synthetic)"
            )
        clinical_root = os.path.expanduser(args.clinical_root)
        wearable_root = os.path.expanduser(args.wearable_root)
        for path, name in [
            (clinical_root, "--clinical_root"),
            (wearable_root, "--wearable_root"),
        ]:
            if not os.path.isdir(path):
                parser.error(f"{name}: directory not found: {path}")

    # step 1: load datasets
    clinical_ds = IBISleepDataset(root=clinical_root, source="shhs")
    wearable_ds = IBISleepDataset(root=wearable_root, source="dreamt")

    # step 2: set task
    clinical_samples = clinical_ds.set_task(SleepStagingIBI(num_classes=5))
    wearable_samples = wearable_ds.set_task(SleepStagingIBI(num_classes=3))

    # step 3: define model and dataloaders for phase 1
    model = WatchSleepNet(num_classes=5)

    train_clin, val_clin, _ = split_by_patient(clinical_samples, [0.7, 0.15, 0.15])
    train_loader = get_dataloader(train_clin, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_clin, batch_size=32, shuffle=False)

    # step 4: phase 1 — pre-train on clinical data
    trainer = Trainer(model=model, device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.pretrain_epochs,
        optimizer_params={"lr": 1e-3},
        monitor="accuracy",
    )

    # step 5: phase 2 — replace head and fine-tune on wearable data
    model_ft = WatchSleepNet(num_classes=3)
    backbone_state = {
        k: v for k, v in model.state_dict().items() if not k.startswith("fc.")
    }
    model_ft.load_state_dict(backbone_state, strict=False)

    train_wear, val_wear, test_wear = split_by_patient(
        wearable_samples, [0.6, 0.2, 0.2]
    )
    train_loader = get_dataloader(train_wear, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_wear, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_wear, batch_size=32, shuffle=False)

    trainer_ft = Trainer(model=model_ft, device=args.device)
    trainer_ft.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.finetune_epochs,
        optimizer_params={"lr": 1e-4},
        monitor="accuracy",
    )

    # step 6: evaluate
    print(trainer_ft.evaluate(test_loader))
