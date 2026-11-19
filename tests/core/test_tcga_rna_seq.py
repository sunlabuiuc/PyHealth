import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from pyhealth.datasets import TCGARNASeqDataset


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

@pytest.mark.filterwarnings("ignore:The given buffer is not writable")
def test_dataset_to_model_pipeline():
    """Load synthetic CSV, apply task, build model, run a forward pass."""
    from pyhealth.datasets import get_dataloader
    from pyhealth.models import BulkRNABert
    from pyhealth.tasks import TCGASurvivalPrediction

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        _write_csv(tmp)

        ds = TCGARNASeqDataset(root=str(tmp), cache_dir=str(tmp / "cache"))
        samples = ds.set_task(TCGASurvivalPrediction())
        assert len(samples) == 3

        model = BulkRNABert(dataset=samples)
        loader = get_dataloader(samples, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        out = model(**batch)
        assert out["logit"].shape == (2, 1)
        assert "loss" in out
        assert torch.isfinite(out["loss"])

def _load_example_module():
    """Import the example script by path (examples/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "tcga_example",
        Path(__file__).parents[2] / "examples"
        / "tcga_rna_seq_tcga_survival_prediction_bulk_rna_bert.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_per_cohort_c_index_aggregation():
    """Hand-computed case for the ablation's per-cohort metric helper."""
    mod = _load_example_module()

    # Two cohorts, 4 patients each. Cohort 0 ranks perfectly (C=1.0),
    # cohort 1 ranks reversed (C=0.0). Overall = 0.5.
    duration = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    risk = np.array([4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    event = np.ones(8)
    cohort = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    result = mod.per_cohort_c_index(duration, risk, event, cohort)

    assert result["per_cohort"][0] == 1.0
    assert result["per_cohort"][1] == 0.0
    assert abs(result["c_index_macro"] - 0.5) < 1e-9
    assert abs(result["c_index_weighted"] - 0.5) < 1e-9  # equal cohort sizes

def test_per_cohort_c_index_excludes_small_cohorts():
    """Cohorts with fewer than 2 samples are excluded from aggregates."""
    mod = _load_example_module()

    duration = np.array([1.0, 2.0, 3.0])
    risk = np.array([3.0, 2.0, 1.0])
    event = np.ones(3)
    cohort = np.array([0, 0, 1])

    result = mod.per_cohort_c_index(duration, risk, event, cohort)
    assert result["per_cohort"][0] == 1.0
    assert np.isnan(result["per_cohort"][1])
    assert result["c_index_macro"] == 1.0
