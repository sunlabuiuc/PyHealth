"""
DREAMT Sleep Staging — WatchSleepNet Task Ablations
====================================================
Paper: Wang et al., "WatchSleepNet: A Novel Model and Pretraining Approach for
Advancing Sleep Staging with Smartwatches", 2025.
https://doi.org/10.48550/arXiv.2501.17268

Dataset: DREAMT (PhysioNet) — https://physionet.org/content/dreamt/

Implements the DREAMTSleepClassification task and three novel ablation studies not
present in the original paper:

    Ablation 1 — Label granularity : 3-class (Wake/NREM/REM) vs 4-class
                                     (Wake/N1/N2/N3/REM)
    Ablation 2 — Accelerometer     : IBI-only vs IBI + ACC_X/Y/Z
    Ablation 3 — Epoch duration    : 15 s / 30 s (paper default) / 60 s

Quick start (no download required):
    python examples/dreamt_sleep_staging_rnn.py --demo

Real data:
    python examples/dreamt_sleep_staging_rnn.py --root /path/to/dreamt/2.1.0
"""

import argparse
import collections
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _build_demo_root(n_patients: int = 6, n_rows: int = 3840) -> str:
    """Create a minimal synthetic DREAMT directory tree for demo mode."""
    tmp = tempfile.mkdtemp(prefix="dreamt_demo_")
    root = Path(tmp)
    (root / "data_64Hz").mkdir()
    (root / "data_100Hz").mkdir()

    rng = np.random.default_rng(0)
    stage_cycle = (
        ["W"] * 640 + ["N1"] * 640 + ["N2"] * 640 + ["N3"] * 640 + ["R"] * 640
    ) * 2

    rows = []
    for i in range(1, n_patients + 1):
        sid = f"S{i:03d}"
        ibi = np.zeros(n_rows)
        beat_idx = np.arange(0, n_rows, 51)
        ibi[beat_idx] = rng.uniform(0.7, 1.1, len(beat_idx))

        df = pd.DataFrame({
            "TIMESTAMP":   np.arange(n_rows) / 64.0,
            "BVP":         rng.standard_normal(n_rows),
            "HR":          rng.integers(50, 90, n_rows).astype(float),
            "EDA":         rng.uniform(0.0, 1.0, n_rows),
            "TEMP":        rng.uniform(33.0, 37.0, n_rows),
            "ACC_X":       rng.standard_normal(n_rows),
            "ACC_Y":       rng.standard_normal(n_rows),
            "ACC_Z":       rng.standard_normal(n_rows),
            "IBI":         ibi,
            "Sleep_Stage": stage_cycle[:n_rows],
        })
        df.to_csv(root / "data_64Hz" / f"{sid}_whole_df.csv", index=False)
        pd.DataFrame({"a": [1]}).to_csv(
            root / "data_100Hz" / f"{sid}_PSG_df.csv", index=False
        )
        rows.append({
            "SID": sid, "AGE": int(rng.integers(25, 65)),
            "GENDER": rng.choice(["M", "F"]), "BMI": int(rng.integers(18, 40)),
            "OAHI": int(rng.integers(0, 30)), "AHI": int(rng.integers(0, 30)),
            "Mean_SaO2": f"{int(rng.integers(90, 99))}%",
            "Arousal Index": int(rng.integers(5, 30)),
            "MEDICAL_HISTORY": "None", "Sleep_Disorders": "None",
        })

    pd.DataFrame(rows).to_csv(root / "participant_info.csv", index=False)
    print(f"[demo] Synthetic DREAMT root: {root}")
    return tmp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_samples(ds):
    return [ds[i] for i in range(len(ds))]


def summarise(ds, name: str) -> None:
    all_s = _all_samples(ds)
    n = len(all_s)
    counts = dict(sorted(
        collections.Counter(s["label"].item() for s in all_s).items()
    ))
    print(f"  [{name}]")
    print(f"    Total epochs : {n}")
    print(f"    Label dist   : {counts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(root: str) -> None:
    from pyhealth.datasets import DREAMTDataset
    from pyhealth.tasks import DREAMTSleepClassification

    # Step 1 — Load dataset
    print("\n=== Step 1 — Load DREAMTDataset ===")
    dreamt = DREAMTDataset(root=root)
    dreamt.stats()
    print(f"Patients loaded: {len(dreamt.unique_patient_ids)}")

    # Ablation 1 — Label granularity
    print("\n=== Ablation 1 — Label Granularity ===")
    ds_3cls = dreamt.set_task(DREAMTSleepClassification(num_classes=3))
    ds_4cls = dreamt.set_task(DREAMTSleepClassification(num_classes=4))
    print("Label granularity comparison:")
    summarise(ds_3cls, "3-class  Wake=0 / NREM=1 / REM=2")
    summarise(ds_4cls, "4-class  Wake=0 / N1=1 / N2=2 / N3=3 / REM=4")
    print(
        "\nObservation: both datasets share the same epoch count; "
        "4-class spreads NREM epochs across three labels."
    )

    # Ablation 2 — Accelerometer augmentation
    print("\n=== Ablation 2 — Accelerometer Augmentation ===")
    ds_ibi_only = dreamt.set_task(DREAMTSleepClassification(num_classes=3, use_accelerometer=False))
    ds_ibi_acc  = dreamt.set_task(DREAMTSleepClassification(num_classes=3, use_accelerometer=True))
    print("Accelerometer augmentation comparison:")
    summarise(ds_ibi_only, "IBI-only        input keys: ibi_sequence")
    summarise(ds_ibi_acc,  "IBI + ACC       input keys: ibi_sequence, accelerometer")
    acc_samples = _all_samples(ds_ibi_acc)
    if acc_samples:
        print(f"\nACC tensor shape per epoch: {acc_samples[0]['accelerometer'].shape}  (rows x 3 axes)")
    print(
        "\nTo train with ACC: replace feature_keys=['ibi_sequence'] with "
        "['ibi_sequence', 'accelerometer'] and compare Wake F1."
    )

    # Ablation 3 — Epoch duration
    print("\n=== Ablation 3 — Epoch Duration ===")
    print(f"{'Epoch (s)':<10} {'Total epochs':<15} {'Avg IBI vals/epoch':<20}")
    print("-" * 45)
    for epoch_secs in (15, 30, 60):
        ds_ep = dreamt.set_task(DREAMTSleepClassification(epoch_seconds=epoch_secs, num_classes=3))
        ep_samples = _all_samples(ds_ep)
        n = len(ep_samples)
        avg_ibi = (
            np.mean([len(s["ibi_sequence"]) for s in ep_samples])
            if ep_samples else 0.0
        )
        marker = " <- paper default" if epoch_secs == 30 else ""
        print(f"{epoch_secs:<10} {n:<15} {avg_ibi:<20.1f}{marker}")
    print(
        "\nObservation: halving epoch duration doubles epoch count "
        "but halves the average IBI count per window."
    )

    # Step 2 — Train RNN
    print("\n=== Step 2 — Train RNN on 3-class task ===")
    from pyhealth.datasets import get_dataloader, split_by_patient
    from pyhealth.models import RNN
    from pyhealth.trainer import Trainer
    from sklearn.metrics import cohen_kappa_score, f1_score
    import torch

    train_ds, val_ds, test_ds = split_by_patient(ds_3cls, [0.7, 0.15, 0.15])
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=32, shuffle=False)
    print(f"Split -- train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)} epochs")

    model = RNN(dataset=ds_3cls)
    trainer = Trainer(model=model, device="cpu")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,
        monitor="accuracy",
    )

    results = trainer.evaluate(test_loader)
    print(f"\nTest results: {results}")

    # Per-class metrics
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            output = model(**batch)
            preds  = output["y_prob"].argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])
    kappa = cohen_kappa_score(all_labels, all_preds)
    acc   = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\nAccuracy     : {acc:.4f}")
    print(f"Wake F1      : {per_class_f1[0]:.4f}")
    print(f"NREM F1      : {per_class_f1[1]:.4f}")
    print(f"REM F1       : {per_class_f1[2]:.4f}")
    print(f"Cohen Kappa  : {kappa:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DREAMT sleep staging ablations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Run with synthetic data")
    group.add_argument("--root", type=str, help="Path to local DREAMT 2.1.0 directory")
    args = parser.parse_args()

    tmpdir = None
    if args.demo:
        tmpdir = _build_demo_root()
        root = tmpdir
    else:
        root = args.root

    try:
        main(root)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir)
            print(f"\n[demo] Cleaned up {tmpdir}")
