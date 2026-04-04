"""
PTB-XL ECG Diagnosis — Ablation Study
======================================

Reproduces part of the benchmark from:
    Nonaka, K., & Seita, D. (2021). In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis.
    Proceedings of Machine Learning Research, 149, 414-424.
    https://proceedings.mlr.press/v149/nonaka21a.html

This script demonstrates the PTBXLDataset + PTBXLDiagnosis pipeline using
**synthetic data** so it runs without downloading the real PTB-XL dataset.

Ablation dimensions explored
-----------------------------
1. Task type        — multilabel (ROC-AUC) vs. multiclass (F1)
2. Sampling rate    — 100 Hz vs. 500 Hz metadata parsing
3. Label filtering  — all superclasses vs. NORM-only subset

Usage
-----
    python examples/ptbxl_ecg_diagnosis_resnet.py

Requirements
------------
    pip install pandas numpy scikit-learn

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import ast
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── PTBXLDataset helpers ────────────────────────────────────────────────────
from pyhealth.datasets.ptbxl import PTBXLDataset
from pyhealth.tasks.ptbxl_diagnosis import (
    PTBXLDiagnosis,
    PTBXLMulticlassDiagnosis,
    _scp_to_superclasses,
    SUPERCLASSES,
    SCP_TO_SUPER,
)

# ── Synthetic data generation ───────────────────────────────────────────────

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
]


def make_synthetic_db(root: Path, sampling_rate: int = 100) -> None:
    """Write a synthetic ptbxl_database.csv for demonstration purposes."""
    rows = []
    for i, scp in enumerate(SYNTHETIC_SCP_PROFILES):
        rows.append(
            {
                "ecg_id": i + 1,
                "patient_id": i + 1,
                "filename_lr": f"records100/00000/{i + 1:05d}_lr",
                "filename_hr": f"records500/00000/{i + 1:05d}_hr",
                "scp_codes": scp,
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)


# ── Fake patient / event stubs (mirrors test helpers) ───────────────────────

class _FakeEvent:
    def __init__(self, record_id, signal_file, scp_codes, sampling_rate=100):
        self.record_id = record_id
        self.signal_file = signal_file
        self.scp_codes = scp_codes
        self.sampling_rate = sampling_rate

    def get(self, key, default=None):
        return getattr(self, key, default)


class _FakePatient:
    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type="ptbxl"):
        return self._events


def build_fake_patients(db_path: Path) -> List[_FakePatient]:
    """Build fake patient objects from the synthetic CSV."""
    df = pd.read_csv(db_path)
    patients = []
    for _, row in df.iterrows():
        event = _FakeEvent(
            record_id=int(row["ecg_id"]),
            signal_file=str(row["filename_lr"]),
            scp_codes=str(row["scp_codes"]),
        )
        patients.append(_FakePatient(patient_id=str(int(row["patient_id"])), events=[event]))
    return patients


# ── Ablation helpers ─────────────────────────────────────────────────────────

def run_multilabel_task(patients: List[_FakePatient]) -> List[Dict]:
    task = PTBXLDiagnosis()
    samples = []
    for p in patients:
        samples.extend(task(p))
    return samples


def run_multiclass_task(patients: List[_FakePatient]) -> List[Dict]:
    task = PTBXLMulticlassDiagnosis()
    samples = []
    for p in patients:
        samples.extend(task(p))
    return samples


def label_distribution(samples: List[Dict], key: str = "labels") -> Dict:
    """Count label occurrences across samples."""
    counts: Dict[str, int] = {}
    for s in samples:
        val = s[key]
        if isinstance(val, list):
            for v in val:
                counts[v] = counts.get(v, 0) + 1
        else:
            counts[val] = counts.get(val, 0) + 1
    return dict(sorted(counts.items()))


# ── Main ablation study ──────────────────────────────────────────────────────

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # ── Ablation 1: Metadata parsing at 100 Hz vs 500 Hz ────────────────
        print("=" * 60)
        print("Ablation 1: Sampling rate — metadata parsing")
        print("=" * 60)
        for sr in (100, 500):
            make_synthetic_db(root, sampling_rate=sr)
            ds = PTBXLDataset.__new__(PTBXLDataset)
            ds.sampling_rate = sr
            ds.root = str(root)
            ds.prepare_metadata(str(root))
            df = pd.read_csv(root / "ptbxl-metadata-pyhealth.csv")
            col = "filename_lr" if sr == 100 else "filename_hr"
            rate_str = "records100" if sr == 100 else "records500"
            all_correct = df["signal_file"].str.contains(rate_str).all()
            print(f"  {sr} Hz → {len(df)} records, paths contain '{rate_str}': {all_correct}")
            # Clean up for next iteration
            (root / "ptbxl-metadata-pyhealth.csv").unlink()

        # ── Build patients for task ablations ────────────────────────────────
        make_synthetic_db(root)
        patients = build_fake_patients(root / "ptbxl_database.csv")

        # ── Ablation 2: Task type — multilabel vs. multiclass ────────────────
        print()
        print("=" * 60)
        print("Ablation 2: Task type — multilabel vs. multiclass")
        print("=" * 60)

        ml_samples = run_multilabel_task(patients)
        mc_samples = run_multiclass_task(patients)

        print(f"  Multilabel samples : {len(ml_samples)}")
        print(f"  Multiclass samples : {len(mc_samples)}")
        print()
        print("  Multilabel label distribution:")
        for label, cnt in label_distribution(ml_samples, "labels").items():
            print(f"    {label}: {cnt}")
        print()
        print("  Multiclass label distribution:")
        for label, cnt in label_distribution(mc_samples, "label").items():
            print(f"    {label}: {cnt}")

        # ── Ablation 3: Label filtering — all classes vs. NORM-only ──────────
        print()
        print("=" * 60)
        print("Ablation 3: Label filtering — all superclasses vs. NORM-only")
        print("=" * 60)

        norm_only = [s for s in ml_samples if s["labels"] == ["NORM"]]
        multi_label = [s for s in ml_samples if len(s["labels"]) > 1]
        print(f"  NORM-only samples  : {len(norm_only)}")
        print(f"  Multi-label samples: {len(multi_label)}")
        print(f"  Total samples      : {len(ml_samples)}")

        # ── Ablation 4: SCP code coverage ────────────────────────────────────
        print()
        print("=" * 60)
        print("Ablation 4: SCP → superclass mapping coverage")
        print("=" * 60)
        all_codes = set(SCP_TO_SUPER.keys())
        print(f"  Total mapped SCP codes : {len(all_codes)}")
        print(f"  Superclasses covered   : {sorted(set(SCP_TO_SUPER.values()))}")

        print()
        print("Ablation study complete.")
        print()
        print("Next steps with real PTB-XL data:")
        print("  1. Download from https://physionet.org/content/ptb-xl/1.0.3/")
        print("  2. dataset = PTBXLDataset(root='/path/to/ptb-xl')")
        print("  3. samples = dataset.set_task(PTBXLDiagnosis())")
        print("  4. Train a ResNet/CNN model and evaluate with ROC-AUC (multilabel)")
        print("     or F1 score (multiclass) as in Nonaka & Seita (2021).")


if __name__ == "__main__":
    main()
