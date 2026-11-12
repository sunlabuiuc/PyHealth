"""Synthetic unit tests for TCGARNASeqDataset.

Uses three synthetic patients written to a temp dir. No real TCGA data, no
Hugging Face weights, no network.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import TCGARNASeqDataset


def make_csv(root: Path, n: int = 3) -> Path:
    """Writes a minimal tcga_rna_seq.csv in the shape BaseDataset expects."""
    rows = []
    for i in range(n):
        rows.append({
            "patient_id": f"TCGA-X-{i:04d}",
            "cohort": ["TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC"][i],
            "survival_time": 100.0 + 50 * i,
            "event": 1 if i % 2 == 0 else 0,
            "embedding": json.dumps(np.random.randn(256).astype(np.float32).tolist()),
        })
    csv_path = root / "tcga_rna_seq.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_dataset_loads_and_exposes_patients():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_csv(root)
        ds = TCGARNASeqDataset(root=str(root), cache_dir=str(root / "cache"))
        assert ds.dataset_name == "tcga_rna_seq"
        assert len(ds.unique_patient_ids) == 3
        p = ds.get_patient("TCGA-X-0000")
        events = p.get_events(event_type="tcga_rna_seq")
        assert len(events) == 1
        assert float(events[0]["survival_time"]) == 100.0
        assert int(float(events[0]["event"])) == 1
        assert str(events[0]["cohort"]) == "TCGA-BRCA"
        emb = json.loads(events[0]["embedding"])
        assert len(emb) == 256


def test_missing_csv_and_artifacts_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # No CSV, no artifacts/ -> FileNotFoundError from prepare
        try:
            TCGARNASeqDataset(root=str(root), cache_dir=str(root / "cache"))
        except FileNotFoundError:
            return
        raise AssertionError("Expected FileNotFoundError")