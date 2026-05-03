"""Tests for TCGARNASeqDataset and TCGA RNA-seq tasks.

Uses synthetic data only — no real TCGA downloads required.
All tests complete in milliseconds.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

N_PATIENTS = 4
N_GENES = 20
GENE_NAMES = [f"GENE{i}" for i in range(N_GENES)]
COHORTS = ["BRCA", "LUAD", "BRCA", "BLCA"]


def _make_synthetic_rnaseq(root: str) -> str:
    np.random.seed(42)
    expr = np.random.exponential(scale=10.0, size=(N_PATIENTS, N_GENES))
    df = pd.DataFrame(expr, columns=GENE_NAMES)
    df.insert(0, "patient_id", [f"TCGA-{i:02d}" for i in range(N_PATIENTS)])
    df.insert(1, "cohort", COHORTS)
    path = os.path.join(root, "rna_seq.csv")
    df.to_csv(path, index=False)
    return path


def _make_synthetic_clinical(root: str) -> str:
    df = pd.DataFrame({
        "patient_id": [f"TCGA-{i:02d}" for i in range(N_PATIENTS)],
        "cohort": COHORTS,
        "vital_status": ["dead", "alive", "dead", "alive"],
        "days_to_death": [365.0, None, 180.0, None],
        "days_to_last_follow_up": [None, 700.0, None, 500.0],
    })
    path = os.path.join(root, "clinical.csv")
    df.to_csv(path, index=False)
    return path


class TestTCGARNASeqPreprocessing:

    def test_log_transform_and_binning(self, tmp_path):
        root = str(tmp_path)
        _make_synthetic_rnaseq(root)
        _make_synthetic_clinical(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
        rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
        clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")
        TCGARNASeqDataset._prepare_metadata(root, 64, None, rnaseq_out, clinical_out)
        assert os.path.exists(rnaseq_out)
        df = pd.read_csv(rnaseq_out)
        gene_cols = [c for c in df.columns if c not in ("patient_id", "cohort")]
        values = df[gene_cols].values
        assert values.min() >= 0
        assert values.max() < 64

    def test_gene_file_written(self, tmp_path):
        root = str(tmp_path)
        _make_synthetic_rnaseq(root)
        _make_synthetic_clinical(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
        rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
        clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")
        TCGARNASeqDataset._prepare_metadata(root, 64, None, rnaseq_out, clinical_out)
        gene_file = os.path.join(root, "tcga_rnaseq_genes.txt")
        assert os.path.exists(gene_file)
        with open(gene_file) as f:
            genes = [l.strip() for l in f if l.strip()]
        assert len(genes) == N_GENES

    def test_placeholder_created_when_no_raw(self, tmp_path):
        root = str(tmp_path)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
        rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
        clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")
        TCGARNASeqDataset._prepare_metadata(root, 64, None, rnaseq_out, clinical_out)
        assert os.path.exists(rnaseq_out)
        assert os.path.exists(clinical_out)

    def test_n_genes_filtering(self, tmp_path):
        root = str(tmp_path)
        _make_synthetic_rnaseq(root)
        _make_synthetic_clinical(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
        rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
        clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")
        TCGARNASeqDataset._prepare_metadata(root, 64, 10, rnaseq_out, clinical_out)
        df = pd.read_csv(rnaseq_out)
        gene_cols = [c for c in df.columns if c not in ("patient_id", "cohort")]
        assert len(gene_cols) == 10


class TestTCGARNASeqDatasetIntegration:

    def test_instantiate_and_runtime_config(self, tmp_path):
        root = str(tmp_path)
        _make_synthetic_rnaseq(root)
        _make_synthetic_clinical(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset

        cache = os.path.join(root, "ds_cache")
        ds = TCGARNASeqDataset(
            root=root,
            n_bins=32,
            n_genes=8,
            cache_dir=cache,
            num_workers=1,
            dev=False,
        )
        assert os.path.isfile(os.path.join(root, "tcga_rnaseq_pyhealth_config.yaml"))
        assert len(ds.gene_names) == 8
        assert len(ds.unique_patient_ids) == N_PATIENTS

    def test_get_patient_rnaseq_event_token_order(self, tmp_path):
        root = str(tmp_path)
        _make_synthetic_rnaseq(root)
        _make_synthetic_clinical(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
        from pyhealth.tasks.tcga_rnaseq_tasks import _extract_token_ids

        cache = os.path.join(root, "cache_evt")
        ds = TCGARNASeqDataset(
            root=root,
            n_bins=64,
            n_genes=6,
            cache_dir=cache,
            num_workers=1,
            dev=False,
        )
        pid = ds.unique_patient_ids[0]
        patient = ds.get_patient(pid)
        rnaseq_events = patient.get_events(event_type="rnaseq")
        assert len(rnaseq_events) >= 1
        toks = _extract_token_ids(rnaseq_events[0])
        assert len(toks) == 6
        assert all(0 <= t < 64 for t in toks)

    def test_set_task_default_cancer_type_smoke(self, tmp_path):
        root = str(tmp_path)
        _make_synthetic_rnaseq(root)
        _make_synthetic_clinical(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset

        cache = os.path.join(root, "cache_task")
        ds = TCGARNASeqDataset(
            root=root,
            n_bins=64,
            n_genes=5,
            cache_dir=cache,
            num_workers=1,
            dev=False,
        )
        samples = ds.set_task(num_workers=1)
        assert len(samples) >= 1
        row = samples[0]
        assert "token_ids" in row
        assert "cancer_type" in row


class _FakeEvent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakePatient:
    def __init__(self, patient_id, rnaseq_events, clinical_events):
        self.patient_id = patient_id
        self._events = {"rnaseq": rnaseq_events, "clinical": clinical_events}

    def get_events(self, event_type):
        return self._events.get(event_type, [])


def _make_rnaseq_event(cohort="BRCA", n_genes=10, n_bins=64):
    kwargs = {"cohort": cohort}
    for i in range(n_genes):
        kwargs[f"GENE{i}"] = np.random.randint(0, n_bins)
    return _FakeEvent(**kwargs)


class TestTCGACancerTypeTask:

    def test_returns_sample_with_token_ids(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGACancerTypeTask
        task = TCGACancerTypeTask()
        event = _make_rnaseq_event("BRCA")
        patient = _FakePatient("P1", [event], [])
        samples = task(patient)
        assert len(samples) == 1
        assert "token_ids" in samples[0]
        assert samples[0]["cancer_type"] == "BRCA"

    def test_cohort_filter_excludes_other_cohorts(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGACancerTypeTask
        task = TCGACancerTypeTask(cohorts=["LUAD"])
        event = _make_rnaseq_event("BRCA")
        patient = _FakePatient("P1", [event], [])
        assert task(patient) == []

    def test_cohort_filter_includes_matching_cohort(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGACancerTypeTask
        task = TCGACancerTypeTask(cohorts=["BRCA", "LUAD"])
        event = _make_rnaseq_event("LUAD")
        patient = _FakePatient("P1", [event], [])
        assert len(task(patient)) == 1

    def test_missing_rnaseq_returns_empty(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGACancerTypeTask
        task = TCGACancerTypeTask()
        assert task(_FakePatient("P1", [], [])) == []

    def test_missing_cohort_returns_empty(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGACancerTypeTask
        task = TCGACancerTypeTask()
        event = _FakeEvent(cohort=None, GENE0=5, GENE1=10)
        assert task(_FakePatient("P1", [event], [])) == []


class TestTCGASurvivalTask:

    def _make_clinical(self, vital="dead", days_death=365.0, days_follow=None):
        return _FakeEvent(
            vital_status=vital,
            days_to_death=days_death,
            days_to_last_follow_up=days_follow,
        )

    def test_deceased_patient_returns_sample(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGASurvivalTask
        task = TCGASurvivalTask()
        patient = _FakePatient(
            "P1", [_make_rnaseq_event("BRCA")], [self._make_clinical("dead", 365.0)]
        )
        samples = task(patient)
        assert len(samples) == 1
        assert samples[0]["event"] == 1
        assert samples[0]["survival_time"] == pytest.approx(365.0)

    def test_censored_patient_returns_sample(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGASurvivalTask
        task = TCGASurvivalTask()
        patient = _FakePatient(
            "P1",
            [_make_rnaseq_event("LUAD")],
            [self._make_clinical("alive", None, 700.0)],
        )
        samples = task(patient)
        assert len(samples) == 1
        assert samples[0]["event"] == 0
        assert samples[0]["survival_time"] == pytest.approx(700.0)

    def test_missing_clinical_returns_empty(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGASurvivalTask
        task = TCGASurvivalTask()
        assert task(_FakePatient("P1", [_make_rnaseq_event()], [])) == []

    def test_unknown_vital_status_returns_empty(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGASurvivalTask
        task = TCGASurvivalTask()
        patient = _FakePatient(
            "P1",
            [_make_rnaseq_event()],
            [self._make_clinical("unknown", None, None)],
        )
        assert task(patient) == []

    def test_cohort_filter(self):
        from pyhealth.tasks.tcga_rnaseq_tasks import TCGASurvivalTask
        task = TCGASurvivalTask(cohorts=["BLCA"])
        patient = _FakePatient(
            "P1", [_make_rnaseq_event("BRCA")], [self._make_clinical("dead", 200.0)]
        )
        assert task(patient) == []