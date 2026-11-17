import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets import TCGARNASeqDataset
from pyhealth.datasets import tcga_rna_seq as dsmod


def _write_csv(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    cohorts = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC"]
    rows = []
    for i, cohort in enumerate(cohorts):
        rows.append({
            "patient_id": f"TCGA-{cohort[5:]}-{i:04d}",
            "cohort": cohort,
            "duration": 100.0 + 100 * i,
            "event": 1 if i % 2 == 0 else 0,
            "embedding": json.dumps(
                rng.standard_normal(256).astype(np.float32).tolist()
            ),
        })
    csv_path = root / "tcga_rna_seq.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_dataset_loads_and_exposes_patients():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_csv(tmp)
        ds = TCGARNASeqDataset(
            root=str(tmp), cache_dir=str(tmp / "cache")
        )
        assert ds.dataset_name == "tcga_rna_seq"
        assert len(ds.unique_patient_ids) == 3

        p = ds.get_patient("TCGA-BRCA-0000")
        events = p.get_events(event_type="tcga_rna_seq")
        assert len(events) == 1
        assert float(events[0]["duration"]) == 100.0
        assert int(float(events[0]["event"])) == 1
        assert str(events[0]["cohort"]) == "TCGA-BRCA"
        assert len(json.loads(events[0]["embedding"])) == 256


def test_dataset_raises_on_missing_csv():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="tcga_rna_seq.csv"):
            TCGARNASeqDataset(
                root=tmp, cache_dir=str(Path(tmp) / "cache")
            )


def test_from_huggingface_downloads_and_loads():
    """Mock hf_hub_download so the test never touches the network."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_hf_dir = tmp / "fake_hf_cache"
        fake_hf_dir.mkdir()
        _write_csv(fake_hf_dir)

        with mock.patch(
            "huggingface_hub.hf_hub_download",
            return_value=str(fake_hf_dir / "tcga_rna_seq.csv"),
        ) as mocked:
            ds = TCGARNASeqDataset.from_huggingface(
                repo_id="cstein2/TCGABulkBertRNAEmbeddings",
                cache_dir=str(tmp / "cache"),
            )

        mocked.assert_called_once_with(
            repo_id="cstein2/TCGABulkBertRNAEmbeddings",
            filename="tcga_rna_seq.csv",
            repo_type="dataset",
            revision="c1cca7f308c5d92b3337c6c36bdb3e857ba7991e",
        )
        assert len(ds.unique_patient_ids) == 3