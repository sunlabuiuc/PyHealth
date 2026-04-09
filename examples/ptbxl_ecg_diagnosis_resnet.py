"""
PTB-XL ECG Diagnosis — Ablation Study
======================================

Reproduces part of the benchmark from:
    Nonaka, K., & Seita, D. (2021). In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis.
    Proceedings of Machine Learning Research, 149, 414-424.
    https://proceedings.mlr.press/v149/nonaka21a.html

This script demonstrates the PTBXLDataset + PTBXLDiagnosis pipeline and runs
a real ablation study using PyHealth's MLP model on synthetic data, so it
runs without downloading the real PTB-XL dataset.

Ablation dimensions explored
-----------------------------
1. Task type        — multilabel vs. multiclass (label definition)
2. Hidden dimension — MLP hidden_dim in {32, 64, 128}
3. Number of layers — MLP n_layers in {1, 2, 3}
4. Sampling rate    — 100 Hz vs. 500 Hz metadata parsing

Usage
-----
    python examples/ptbxl_ecg_diagnosis_resnet.py

Requirements
------------
    pip install pandas numpy torch pyhealth

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets.ptbxl import PTBXLDataset
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.trainer import Trainer
from pyhealth.tasks.ptbxl_diagnosis import (
    PTBXLDiagnosis,
    PTBXLMulticlassDiagnosis,
    _scp_to_superclasses,
    SUPERCLASSES,
    SCP_TO_SUPER,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Synthetic ECG profiles (mirrors real PTB-XL SCP distributions) ───────────
SYNTHETIC_SCP_PROFILES = [
    "{'NORM': 100.0}",
    "{'IMI': 80.0, 'CLBBB': 20.0}",
    "{'STD_': 90.0}",
    "{'LVH': 70.0, 'HVOLT': 30.0}",
    "{'NORM': 60.0, 'IMI': 40.0}",
    "{'CRBBB': 100.0}",
    "{'ISCA': 85.0}",
    "{'RVH': 55.0}",
    "{'NORM': 100.0}",
    "{'AMI': 75.0, 'STD_': 25.0}",
    "{'NORM': 100.0}",
    "{'ILMI': 70.0}",
    "{'LNGQT': 90.0}",
    "{'LAFB': 80.0}",
    "{'NORM': 100.0}",
    "{'HYP': 60.0, 'LVH': 40.0}",
    "{'ISCI': 85.0}",
    "{'RBBB': 100.0}",
    "{'NORM': 100.0}",
    "{'AMI': 90.0}",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_db(root: Path, sampling_rate: int = 100) -> None:
    """Write a synthetic ptbxl_database.csv for demonstration."""
    rows = []
    for i, scp in enumerate(SYNTHETIC_SCP_PROFILES):
        rows.append({
            "ecg_id": i + 1,
            "patient_id": i + 1,
            "filename_lr": f"records100/00000/{i + 1:05d}_lr",
            "filename_hr": f"records500/00000/{i + 1:05d}_hr",
            "scp_codes": scp,
        })
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)


class _FakeEvent:
    def __init__(self, record_id, signal_file, scp_codes):
        self.record_id = record_id
        self.signal_file = signal_file
        self.scp_codes = scp_codes

    def get(self, key, default=None):
        return getattr(self, key, default)


class _FakePatient:
    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type="ptbxl"):
        return self._events


def build_fake_patients(db_path: Path) -> List[_FakePatient]:
    df = pd.read_csv(db_path)
    patients = []
    for _, row in df.iterrows():
        event = _FakeEvent(
            record_id=int(row["ecg_id"]),
            signal_file=str(row["filename_lr"]),
            scp_codes=str(row["scp_codes"]),
        )
        patients.append(
            _FakePatient(patient_id=str(int(row["patient_id"])), events=[event])
        )
    return patients


def samples_to_feature_vectors(
    samples: List[Dict], label_key: str, superclasses: List[str]
) -> List[Dict]:
    """
    Convert PTB-XL task samples into feature-vector samples for MLP.

    Since we don't have real signal files, we simulate a 12-lead ECG feature
    vector using the superclass one-hot encoding as a proxy feature.
    In a real pipeline this would be replaced by wfdb.rdsamp() signal loading.
    """
    feature_samples = []
    for s in samples:
        # Simulate a 5-dim feature vector (one-hot over superclasses)
        # In real usage: load signal with wfdb, extract features
        feat = [1.0 if sc in s.get("labels", [s.get("label", "")]) else 0.0
                for sc in superclasses]
        # Add small noise to make it non-trivial
        feat = [f + float(np.random.normal(0, 0.1)) for f in feat]

        if label_key == "label":
            label = s["label"]
        else:
            # For multilabel, use first label as proxy for binary demo
            label = s["labels"][0] if s["labels"] else "NORM"

        feature_samples.append({
            "patient_id": s["patient_id"],
            "visit_id": str(s["record_id"]),
            "ecg_features": feat,
            "label": label,
        })
    return feature_samples


def train_one_epoch(model, loader) -> float:
    """Run one training epoch using PyHealth Trainer, return mean loss."""
    trainer = Trainer(
        model=model,
        enable_logging=False,   # suppress file output during ablation
    )
    trainer.train(
        train_dataloader=loader,
        epochs=1,
        optimizer_params={"lr": 1e-3},
    )
    # evaluate loss on same loader
    scores = trainer.evaluate(loader)
    return scores["loss"]


def eval_accuracy(model, loader) -> float:
    """Evaluate accuracy using PyHealth Trainer."""
    trainer = Trainer(model=model, enable_logging=False)
    y_true_all, y_prob_all, _ = trainer.inference(loader)
    preds = y_prob_all.argmax(axis=-1)
    true  = y_true_all.argmax(axis=-1)
    return float((preds == true).mean())


# ── Main ablation study ──────────────────────────────────────────────────────

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_synthetic_db(root)
        patients = build_fake_patients(root / "ptbxl_database.csv")

        # ── Ablation 1: Sampling rate — metadata parsing ─────────────────────
        print("=" * 60)
        print("Ablation 1: Sampling rate — metadata parsing")
        print("=" * 60)
        for sr in (100, 500):
            make_synthetic_db(root)
            ds = PTBXLDataset.__new__(PTBXLDataset)
            ds.sampling_rate = sr
            ds.root = str(root)
            ds.prepare_metadata(str(root))
            df = pd.read_csv(root / "ptbxl-metadata-pyhealth.csv")
            rate_str = "records100" if sr == 100 else "records500"
            ok = df["signal_file"].str.contains(rate_str).all()
            print(f"  {sr} Hz → {len(df)} records, correct paths: {ok}")
            (root / "ptbxl-metadata-pyhealth.csv").unlink()

        # ── Ablation 2: Task type — multilabel vs. multiclass ────────────────
        print()
        print("=" * 60)
        print("Ablation 2: Task type — label definition")
        print("=" * 60)

        ml_task = PTBXLDiagnosis()
        mc_task = PTBXLMulticlassDiagnosis()

        ml_samples = [s for p in patients for s in ml_task(p)]
        mc_samples = [s for p in patients for s in mc_task(p)]

        print(f"  Multilabel samples : {len(ml_samples)}")
        print(f"  Multiclass samples : {len(mc_samples)}")

        # ── Ablation 3: MLP hidden_dim — model performance comparison ────────
        print()
        print("=" * 60)
        print("Ablation 3: MLP hidden_dim ∈ {32, 64, 128} on multiclass task")
        print("=" * 60)
        print(f"  {'hidden_dim':<12} {'n_layers':<10} {'train_loss':<12} {'accuracy'}")
        print(f"  {'-'*50}")

        feature_samples = samples_to_feature_vectors(
            mc_samples, "label", SUPERCLASSES
        )

        input_schema = {"ecg_features": "sequence"}
        output_schema = {"label": "multiclass"}

        for hidden_dim in (32, 64, 128):
            for n_layers in (1, 2):
                sample_ds = create_sample_dataset(
                    samples=feature_samples,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    dataset_name="ptbxl_synthetic",
                )
                loader = get_dataloader(sample_ds, batch_size=4, shuffle=True)
                model = MLP(
                    dataset=sample_ds,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                )
                loss = train_one_epoch(model, loader)
                acc = eval_accuracy(model, loader)
                print(f"  {hidden_dim:<12} {n_layers:<10} {loss:<12.4f} {acc:.4f}")
                sample_ds.close()

        # ── Ablation 4: SCP code coverage ────────────────────────────────────
        print()
        print("=" * 60)
        print("Ablation 4: SCP → superclass mapping coverage")
        print("=" * 60)
        print(f"  Total mapped SCP codes : {len(SCP_TO_SUPER)}")
        print(f"  Superclasses covered   : {sorted(set(SCP_TO_SUPER.values()))}")

        print()
        print("Ablation study complete.")
        print()
        print("Next steps with real PTB-XL data:")
        print("  1. Download from https://physionet.org/content/ptb-xl/1.0.3/")
        print("  2. dataset = PTBXLDataset(root='/path/to/ptb-xl')")
        print("  3. samples = dataset.set_task(PTBXLDiagnosis())")
        print("  4. Replace ecg_features with wfdb.rdsamp() signal loading")
        print("  5. Evaluate with ROC-AUC (multilabel) or F1 (multiclass)")
        print("     as in Nonaka & Seita (2021).")


if __name__ == "__main__":
    main()
