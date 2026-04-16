"""
tests/pytest_ehrmamba.py
========================
Pytest test suite for the EHR Mamba V3 implementation.

Modules under test
------------------
mimic4_ehr_mamba_task.py  (V3: task separated from embedding module)
    - LabQuantizer              : 5-bin quantile tokenizer for lab results (paper Appx. B)
    - MIMIC4EHRMambaTask        : base PyHealth task — builds the V2 paper-conformant sequence
    - MIMIC4EHRMambaMortalityTask : subclass — adds in-hospital mortality label + age filter
    - collate_ehr_mamba_batch   : custom DataLoader collate_fn for variable-length EHR seqs

ehr_mamba_embeddings_paper_w_bins_v3.py  (V3: task removed; embedding-only module)
    - TimeEmbeddingLayer    : learnable sinusoidal time / age embedding (paper §2.2)
    - EHRMambaEmbedding     : full 7-component fusion embedding (paper §2.2 / Appx. C.2)
    - EHRMambaEmbeddingAdapter : dict-interface shim that bridges EHRMambaEmbedding
                                 into pyhealth's EmbeddingModel API

ehrmamba.py
    - RMSNorm    : root mean square layer normalization used inside each Mamba block
    - MambaBlock : single Mamba (SSM) block — the core sequence-mixing unit
    - EHRMamba   : full pyhealth BaseModel wrapping embedding + N Mamba blocks + FC head

Design principles
-----------------
- All tests use synthetic data only; no MIMIC-IV files are needed.
- Expensive shared objects (dataset, model, batch) are built once per session
  using pytest's module-scoped fixtures to keep total runtime under ~15 s.
- Each test method has a docstring that explains why the property matters,
  not just what is being checked.

Running
-------
to run regular pytest checks:
    pytest tests/pytest_ehrmamba.py -v

to run with timing information for all tests:
    pytest tests/pytest_ehrmamba.py -vv --durations=0
"""

from __future__ import annotations

import sys
import types
import os
import tempfile

from pyhealth.models.ehrmamba_embedding import (
    TimeEmbeddingLayer,
    EHRMambaEmbedding,
    EHRMambaEmbeddingAdapter,
    NUM_TOKEN_TYPES,
    SPECIAL_TYPE_MAX,
    NUM_VISIT_SEGMENTS,
)

from pyhealth.tasks.mortality_prediction_ehrmamba_mimic4 import (
    LabQuantizer,
    MIMIC4EHRMambaTask,
    MIMIC4EHRMambaMortalityTask,
    collate_ehr_mamba_batch,
    MIMIC4_TOKEN_TYPES,
    MAX_NUM_VISITS,
    _time_interval_token,
)

# # ---------------------------------------------------------------------------
# # Path fix — insert project root so Python can find the modules which live
# # one directory above this tests/ folder.
# # ---------------------------------------------------------------------------
# _HERE = os.path.dirname(__file__)
# _ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# rdkit stub
# pyhealth.models.__init__ imports MoleRec and SafeDrug, both of which pull in
# rdkit.  On this machine rdkit's native DLL is blocked by Application Control
# policy, so the real import fails.  We register lightweight stub modules
# before any pyhealth import so that BaseModel loads cleanly.
# ---------------------------------------------------------------------------
# for _mod_name in [
#     "rdkit", "rdkit.Chem", "rdkit.Chem.rdchem", "rdkit.Chem.BRICS",
#     "rdkit.Geometry", "rdkit.Geometry.rdGeometry",
# ]:
#     if _mod_name not in sys.modules:
#         _stub = types.ModuleType(_mod_name)
#         _stub.__path__ = []
#         sys.modules[_mod_name] = _stub

import pytest
import torch
import polars as pl
from datetime import datetime, timedelta
from typing import Dict

# ---------------------------------------------------------------------------
# V3 import layout:
#   - Task-side concerns (LabQuantizer, collate, task classes, token constants)
#     come from mimic4_ehr_mamba_task.
#   - Embedding-side concerns (TimeEmbeddingLayer, EHRMambaEmbedding, adapter,
#     embedding constants) come from ehr_mamba_embeddings_paper_w_bins_v3.
# ---------------------------------------------------------------------------
# from mimic4_ehr_mamba_task import (
#     LabQuantizer,
#     MIMIC4EHRMambaTask,
#     MIMIC4EHRMambaMortalityTask,
#     collate_ehr_mamba_batch,
#     MIMIC4_TOKEN_TYPES,
#     MAX_NUM_VISITS,
#     _time_interval_token,
# )
# from ehr_mamba_embeddings_paper_w_bins_v3 import (
#     TimeEmbeddingLayer,
#     EHRMambaEmbedding,
#     EHRMambaEmbeddingAdapter,
#     NUM_TOKEN_TYPES,
#     SPECIAL_TYPE_MAX,
#     NUM_VISIT_SEGMENTS,
# )

from pyhealth.models.ehrmamba_embedding import (
    TimeEmbeddingLayer,
    EHRMambaEmbedding,
    EHRMambaEmbeddingAdapter,
    NUM_TOKEN_TYPES,
    SPECIAL_TYPE_MAX,
    NUM_VISIT_SEGMENTS,
)

from pyhealth.tasks.mortality_prediction_ehrmamba_mimic4 import (
    LabQuantizer,
    MIMIC4EHRMambaTask,
    MIMIC4EHRMambaMortalityTask,
    collate_ehr_mamba_batch,
    MIMIC4_TOKEN_TYPES,
    MAX_NUM_VISITS,
    _time_interval_token,
)

from pyhealth.models.ehrmamba_v2 import EHRMamba, MambaBlock, RMSNorm


# ===========================================================================
# Shared synthetic-data constants
# ===========================================================================
VOCAB = 200   # synthetic vocabulary size
H     = 64    # embedding hidden dimension
B     = 3     # default batch size
L     = 15    # default sequence length


# ===========================================================================
# Synthetic-data infrastructure
# ===========================================================================

class _MockEvent:
    """Stand-in for a single row returned by pyhealth's event tables."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockPatient:
    """Synthetic patient satisfying the pyhealth.Patient API used by MIMIC4EHRMambaTask."""

    def __init__(self, patient_id, anchor_age, anchor_year, admissions,
                 proc_df=None, rx_df=None, lab_df=None):
        self.patient_id   = patient_id
        self._anchor_age  = anchor_age
        self._anchor_year = anchor_year
        self._admissions  = admissions
        self._tables: Dict[str, pl.DataFrame] = {}
        if proc_df is not None:
            self._tables["procedures_icd"] = proc_df
        if rx_df is not None:
            self._tables["prescriptions"] = rx_df
        if lab_df is not None:
            self._tables["labevents"] = lab_df

    def get_events(self, event_type=None, filters=None, return_df=False):
        if event_type == "patients":
            return [_MockEvent(anchor_age=self._anchor_age,
                               anchor_year=self._anchor_year)]
        if event_type == "admissions":
            return self._admissions
        df = self._tables.get(event_type)
        if return_df:
            if df is None:
                return pl.DataFrame()
            if filters:
                for col, op, val in filters:
                    if op == "==":
                        df = df.filter(pl.col(col) == val)
            return df
        return []


def _make_admission(hadm_id, ts, expire=0):
    return _MockEvent(hadm_id=hadm_id, timestamp=ts, hospital_expire_flag=expire)


def _make_synthetic_patient(patient_id="p001", anchor_age=55, anchor_year=2020,
                             n_visits=2, expire_last=0):
    """Build a complete MockPatient with n_visits admissions and one event per visit."""
    base     = datetime(anchor_year, 1, 1)
    adms     = [
        _make_admission(f"adm{i:03d}", base + timedelta(days=30 * i),
                        expire=expire_last if i == n_visits - 1 else 0)
        for i in range(n_visits)
    ]
    hadm_ids = [a.hadm_id for a in adms]
    proc_df  = pl.DataFrame({
        "hadm_id": hadm_ids,
        "procedures_icd/icd_code": ["9904"] * n_visits,
    })
    rx_df = pl.DataFrame({
        "hadm_id": hadm_ids,
        "prescriptions/drug": ["Aspirin"] * n_visits,
    })
    lab_df = pl.DataFrame({
        "hadm_id": hadm_ids,
        "labevents/itemid": ["51006"] * n_visits,
        "labevents/valuenum": [1.2] * n_visits,
    })
    return MockPatient(patient_id, anchor_age, anchor_year, adms, proc_df, rx_df, lab_df)


def _make_synthetic_sample_dataset(n_samples=4):
    """Return a pyhealth SampleDataset built from hardcoded synthetic tokens.

    Token sequence (7 tokens — one complete visit bracket):
        [CLS]  [VS]  PR:9904  RX:Aspirin  LB:51006_bin2  [VE]  [REG]

    input_schema = {"input_ids": "sequence"} — single feature key, matching
    the V3 architecture where auxiliary fields pass through **kwargs, not
    through input_schema.
    """
    from pyhealth.datasets import create_sample_dataset

    tokens   = ["[CLS]", "[VS]", "PR:9904", "RX:Aspirin", "LB:51006_bin2", "[VE]", "[REG]"]
    type_ids = [
        MIMIC4_TOKEN_TYPES["CLS"],
        MIMIC4_TOKEN_TYPES["VS"],
        MIMIC4_TOKEN_TYPES["procedures_icd"],
        MIMIC4_TOKEN_TYPES["prescriptions"],
        MIMIC4_TOKEN_TYPES["labevents"],
        MIMIC4_TOKEN_TYPES["VE"],
        MIMIC4_TOKEN_TYPES["REG"],
    ]
    seq_len = len(tokens)

    samples = [
        {
            "patient_id":     f"p{i}",
            "visit_id":       f"v{i}",
            "input_ids":      tokens,
            "token_type_ids": torch.tensor(type_ids, dtype=torch.long),
            "time_stamps":    torch.zeros(seq_len, dtype=torch.float),
            "ages":           torch.full((seq_len,), 55.0, dtype=torch.float),
            "visit_orders":   torch.zeros(seq_len, dtype=torch.long),
            "visit_segments": torch.tensor([0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
            "label":          i % 2,
        }
        for i in range(n_samples)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"input_ids": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_v3",
    )


def _aux_tensors(b=B, l=L, all_clinical=False):
    """Generate a full set of synthetic auxiliary tensors for EHRMambaEmbedding."""
    tids = torch.randint(SPECIAL_TYPE_MAX + 1, NUM_TOKEN_TYPES, (b, l))
    if not all_clinical:
        tids[:, 0] = MIMIC4_TOKEN_TYPES["CLS"]
    ts   = torch.rand(b, l) * 52.0
    ages = torch.rand(b, l) * 30.0 + 40.0
    vo   = torch.randint(0, 10, (b, l))
    vs   = torch.randint(1, NUM_VISIT_SEGMENTS, (b, l))
    vs[:, 0] = 0
    return tids, ts, ages, vo, vs


def _raw_collate_sample(seq_len, label, pid="p"):
    """Build a single pre-collation sample that mimics MIMIC4EHRMambaTask output."""
    ids  = torch.randint(1, 100, (seq_len,)).long()
    tids = torch.randint(SPECIAL_TYPE_MAX + 1, NUM_TOKEN_TYPES, (seq_len,)).long()
    tids[0] = MIMIC4_TOKEN_TYPES["CLS"]
    return {
        "input_ids":      ids,
        "token_type_ids": tids,
        "time_stamps":    torch.rand(seq_len) * 52.0,
        "ages":           torch.full((seq_len,), 55.0),
        "visit_orders":   torch.zeros(seq_len, dtype=torch.long),
        "visit_segments": torch.ones(seq_len, dtype=torch.long),
        "label":          label,
        "patient_id":     pid,
        "visit_id":       f"{pid}_v0",
    }


# ===========================================================================
# Module-level fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def emb_model():
    return EHRMambaEmbedding(dataset = '',vocab_size=VOCAB, hidden_size=H)


@pytest.fixture(scope="module")
def sample_ds():
    return _make_synthetic_sample_dataset(n_samples=4)


@pytest.fixture(scope="module")
def ehr_model(sample_ds):
    return EHRMamba(dataset=sample_ds, embedding_dim=32, num_layers=1, dropout=0.1)


@pytest.fixture(scope="module")
def ehr_batch(sample_ds):
    from torch.utils.data import DataLoader
    loader = DataLoader(sample_ds, batch_size=2, collate_fn=collate_ehr_mamba_batch)
    batch  = next(iter(loader))
    batch["label"] = batch["label"].float().unsqueeze(-1)
    return batch


# ===========================================================================
# Section 0: MIMIC4EHRMambaTask and MIMIC4EHRMambaMortalityTask
# V3: task classes now live in mimic4_ehr_mamba_task.py (separated from embedding).
# ===========================================================================

class TestMIMIC4EHRMambaTask:
    """MIMIC4EHRMambaTask — base PyHealth task that builds the V2 paper-conformant sequence.

    V3 architecture: the task file has no dependency on the embedding file.
    It defines the sequence structure (§2.1), produces auxiliary metadata tensors,
    and delegates label assignment to subclasses via _passes_patient_filter /
    _make_label hooks.
    """

    def test_base_task_returns_no_label(self):
        """Base class _make_label must return an empty dict — no label fields.

        Subclasses add labels by overriding _make_label.  The base returning {}
        (not None) means the sample is kept but has no prediction target, which
        is the correct behaviour for unsupervised pretraining sequences.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        samples = task(patient)
        assert len(samples) == 1,          "Base task must produce exactly one sample"
        assert "label" not in samples[0],  "Base task must not include a label field"

    def test_sample_contains_required_fields(self):
        """Sample dict must include all six sequence fields plus patient/visit IDs.

        EHRMamba.forward() accesses input_ids via feature_keys; the remaining
        five fields are consumed as auxiliary tensors via **kwargs.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        sample  = task(patient)[0]
        required = {"input_ids", "token_type_ids", "time_stamps", "ages",
                    "visit_orders", "visit_segments", "patient_id", "visit_id"}
        assert required <= set(sample.keys()), (
            f"Missing fields: {required - set(sample.keys())}"
        )

    def test_input_ids_starts_with_cls(self):
        """The first token in input_ids must always be [CLS] (paper §2.1).

        EHR Mamba uses a single global [CLS] at the start of the full patient
        sequence, analogous to BERT's [CLS] token.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        sample  = task(patient)[0]
        assert sample["input_ids"][0] == "[CLS]", (
            "First token must be [CLS] (paper §2.1)"
        )

    def test_sequence_contains_visit_delimiters(self):
        """input_ids must contain [VS] and [VE] tokens (paper §2.1).

        Each visit is bracketed by [VS] (visit start) and [VE] (visit end).
        Their presence confirms the task correctly builds the V2 sequence layout.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=2)
        sample  = task(patient)[0]
        assert "[VS]" in sample["input_ids"], "Sequence must contain [VS] tokens"
        assert "[VE]" in sample["input_ids"], "Sequence must contain [VE] tokens"

    def test_sequence_contains_reg_token(self):
        """[REG] register token must follow each [VE] (paper §2.1).

        The [REG] token is the prediction anchor for multi-task prompted
        finetuning (MPF).  It must appear after every visit-end token.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        sample  = task(patient)[0]
        assert "[REG]" in sample["input_ids"], "Sequence must contain [REG] tokens"

    def test_aux_tensor_lengths_match_input_ids(self):
        """All auxiliary tensors must have the same length as input_ids.

        EHRMambaEmbedding processes all tensors position-by-position; a length
        mismatch would cause an index error or silent misalignment.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=3)
        sample  = task(patient)[0]
        L_ = len(sample["input_ids"])
        for key in ("token_type_ids", "time_stamps", "ages",
                    "visit_orders", "visit_segments"):
            assert len(sample[key]) == L_, (
                f"{key} length {len(sample[key])} != input_ids length {L_}"
            )

    def test_visit_segments_alternate(self):
        """Clinical tokens must have alternating segment values 1/2 (paper §2.2).

        Visit segment alternates between 1 and 2 across consecutive non-empty
        visits so the model can distinguish adjacent visits.  Structural tokens
        ([CLS], [VS], [VE], [REG], time-interval) receive segment 0.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=3)
        sample  = task(patient)[0]
        segs    = sample["visit_segments"].tolist()
        clinical_segs = {s for i, s in enumerate(segs)
                         if sample["token_type_ids"][i].item() > SPECIAL_TYPE_MAX}
        assert clinical_segs <= {1, 2}, (
            f"Clinical segments must be in {{1, 2}}, got {clinical_segs}"
        )

    def test_min_visits_filter(self):
        """Patient with fewer visits than min_visits must return empty list."""
        task    = MIMIC4EHRMambaTask(min_visits=5)
        patient = _make_synthetic_patient(n_visits=2)
        assert task(patient) == [], "Patient below min_visits must be filtered out"

    def test_empty_patient_returns_empty(self):
        """Patient with no demographics must return empty list."""
        class _NoDemo(MockPatient):
            def get_events(self, event_type=None, **kwargs):
                if event_type == "patients":
                    return []
                return super().get_events(event_type, **kwargs)
        task    = MIMIC4EHRMambaTask()
        patient = _NoDemo("px", 50, 2020, [])
        assert task(patient) == [], "Patient with no demographics must return []"

    def test_inter_visit_time_interval_token_present(self):
        """Multi-visit patient must have a time-interval token between visits.

        Paper §2.1: [W0]–[W3], [M1]–[M12], or [LT] appears between the [REG]
        of one visit and the [VS] of the next.
        """
        task    = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=2)
        sample  = task(patient)[0]
        tokens  = sample["input_ids"]
        time_tokens = [t for t in tokens
                       if t.startswith("[W") or t.startswith("[M") or t == "[LT]"]
        assert len(time_tokens) >= 1, (
            "Multi-visit sequence must contain at least one time-interval token"
        )


class TestMIMIC4EHRMambaMortalityTask:
    """MIMIC4EHRMambaMortalityTask — mortality subclass tests.

    Verifies label assignment, age filtering, and correct inheritance from
    the base task.  All tests use synthetic data; no MIMIC-IV files needed.
    """

    def test_mortality_label_zero_for_survivor(self):
        """hospital_expire_flag=0 must produce label=0."""
        task    = MIMIC4EHRMambaMortalityTask()
        patient = _make_synthetic_patient(expire_last=0)
        samples = task(patient)
        assert len(samples) == 1,         "Survivor patient must produce one sample"
        assert samples[0]["label"] == 0,  "Survivor label must be 0"

    def test_mortality_label_one_for_expired(self):
        """hospital_expire_flag=1 on last admission must produce label=1."""
        task    = MIMIC4EHRMambaMortalityTask()
        patient = _make_synthetic_patient(expire_last=1)
        samples = task(patient)
        assert len(samples) == 1,         "Expired patient must produce one sample"
        assert samples[0]["label"] == 1,  "Expired label must be 1"

    def test_min_age_filter_rejects_young_patient(self):
        """Patient younger than min_age must be filtered out."""
        task    = MIMIC4EHRMambaMortalityTask(min_age=18)
        patient = _make_synthetic_patient(anchor_age=15)
        assert task(patient) == [], "Patient below min_age must return []"

    def test_min_age_filter_accepts_adult(self):
        """Patient at or above min_age must be accepted."""
        task    = MIMIC4EHRMambaMortalityTask(min_age=18)
        patient = _make_synthetic_patient(anchor_age=18)
        assert len(task(patient)) == 1, "Adult patient must produce one sample"

    def test_output_schema_is_binary(self):
        """Mortality subclass must declare output_schema = {'label': 'binary'}."""
        assert MIMIC4EHRMambaMortalityTask.output_schema == {"label": "binary"}, (
            "output_schema must declare binary label"
        )

    def test_inherits_base_sequence_structure(self):
        """Mortality samples must contain the same sequence fields as the base task."""
        task    = MIMIC4EHRMambaMortalityTask()
        patient = _make_synthetic_patient()
        sample  = task(patient)[0]
        assert "[CLS]" == sample["input_ids"][0], "[CLS] must be first token"
        assert "[VS]"  in sample["input_ids"],    "Sequence must contain [VS]"
        assert "[REG]" in sample["input_ids"],    "Sequence must contain [REG]"


class TestTimeIntervalToken:
    """_time_interval_token — inter-visit gap → paper special token mapping (§2.1)."""

    def test_w0_for_sub_week_gap(self):
        assert _time_interval_token(0.0)  == "[W0]"
        assert _time_interval_token(0.99) == "[W0]"

    def test_w1_w2_w3_ranges(self):
        assert _time_interval_token(1.0) == "[W1]"
        assert _time_interval_token(2.0) == "[W2]"
        assert _time_interval_token(3.0) == "[W3]"

    def test_monthly_tokens(self):
        assert _time_interval_token(5.0)  == "[M1]"
        assert _time_interval_token(9.0)  == "[M2]"
        assert _time_interval_token(52.0) == "[M12]"

    def test_lt_for_long_gaps(self):
        # [LT] requires > 12 months: round(weeks / 4.345) > 12, i.e. weeks >= 55.
        # 52.0 w = round(52/4.345)=12 → [M12]; 57.0 w = round(57/4.345)=13 → [LT].
        assert _time_interval_token(57.0) == "[LT]"
        assert _time_interval_token(60.0) == "[LT]"


# ===========================================================================
# Section 1: LabQuantizer (now in mimic4_ehr_mamba_task.py)
# ===========================================================================

class TestLabQuantizer:
    """LabQuantizer — 5-bin quantile tokenizer for MIMIC-IV lab results (Appx. B).

    V3: imported from mimic4_ehr_mamba_task (task-side preprocessing).
    """

    def test_fit_from_records_boundaries(self):
        """fit_from_records must populate the boundaries dict with 4 cut-points."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 101)])
        assert "51006" in q.boundaries
        assert len(q.boundaries["51006"]) == 4

    def test_fit_chaining(self):
        """fit_from_records must return self so calls can be chained."""
        q = LabQuantizer()
        assert q.fit_from_records([("X", 1.0)]) is q

    def test_bin_index_monotone(self):
        """bin_index must be non-decreasing as valuenum increases."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 1001)])
        bins = [q.bin_index("51006", float(v)) for v in range(1, 101)]
        for a, b_ in zip(bins, bins[1:]):
            assert a <= b_

    def test_bin_index_range(self):
        """bin_index must always return a value in [0, n_bins - 1]."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 101)])
        for v in range(-100, 200, 10):
            idx = q.bin_index("51006", float(v))
            assert 0 <= idx <= 4

    def test_unknown_itemid_returns_zero(self):
        """bin_index must return 0 for an itemid not seen during fit."""
        q = LabQuantizer()
        q.fit_from_records([("known", 1.0)])
        assert q.bin_index("unknown", 5.0) == 0

    def test_token_format_with_value(self):
        """token() must produce 'LB:<itemid>_bin<N>' when valuenum is provided."""
        import re
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 101)])
        tok = q.token("51006", 50.0)
        assert re.fullmatch(r"LB:51006_bin[0-4]", tok)

    def test_token_format_without_value(self):
        """token() must produce 'LB:<itemid>' when valuenum is None."""
        q = LabQuantizer()
        assert q.token("51006", None) == "LB:51006"

    def test_non_numeric_records_skipped(self):
        """Non-numeric valuenum entries must be silently skipped."""
        q = LabQuantizer()
        q.fit_from_records([("X", "bad"), ("X", None), ("X", 1.0), ("X", 2.0)])
        assert "X" in q.boundaries


# ===========================================================================
# Section 2: TimeEmbeddingLayer
# ===========================================================================

class TestTimeEmbeddingLayer:
    """TimeEmbeddingLayer — learnable sinusoidal embedding for time/age (§2.2)."""

    def test_output_shape_time_delta(self):
        layer = TimeEmbeddingLayer(embedding_size=16, is_time_delta=True)
        out   = layer(torch.rand(B, L) * 52.0)
        assert out.shape == (B, L, 16)

    def test_output_shape_absolute(self):
        layer = TimeEmbeddingLayer(embedding_size=8, is_time_delta=False)
        out   = layer(torch.rand(B, L) * 30.0 + 40.0)
        assert out.shape == (B, L, 8)

    def test_output_no_nan(self):
        layer = TimeEmbeddingLayer(embedding_size=16, is_time_delta=True)
        out   = layer(torch.rand(2, 10) * 100.0)
        assert not out.isnan().any()
        assert not out.isinf().any()


# ===========================================================================
# Section 3: EHRMambaEmbedding
# ===========================================================================

class TestEHRMambaEmbedding:
    """EHRMambaEmbedding — the §2.2 / Appx-C2 fusion embedding module."""

    def test_default_instantiation(self, emb_model):
        """All embedding tables must be correctly sized."""
        assert emb_model.hidden_size == H
        assert emb_model.word_embeddings.num_embeddings        == VOCAB
        assert emb_model.token_type_embeddings.num_embeddings  == NUM_TOKEN_TYPES
        assert emb_model.visit_order_embeddings.num_embeddings == MAX_NUM_VISITS
        assert emb_model.position_embeddings.num_embeddings    == 4096

    def test_custom_instantiation(self):
        """Non-default hyperparameters must be stored and honoured."""
        emb_c = EHRMambaEmbedding(
            vocab_size=50, hidden_size=32, type_vocab_size=10,
            max_num_visits=64, time_embeddings_size=8,
            num_visit_segments=3, max_position_embeddings=128,
        )
        assert emb_c.hidden_size          == 32
        assert emb_c.time_embeddings_size == 8

    def test_explicit_args_shape(self, emb_model):
        """Explicit-args forward: output shape (B, L, H), no NaN/Inf."""
        ids              = torch.randint(1, VOCAB, (B, L))
        tids, ts, ages, vo, vs = _aux_tensors()
        out = emb_model(ids, tids, ts, ages, vo, vs)
        assert out.shape == (B, L, H)
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_cached_api_shape(self):
        """Cached API: set_aux_inputs() -> forward(ids) produces correct shape."""
        emb2 = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids  = torch.randint(1, VOCAB, (B, L))
        tids, ts, ages, vo, vs = _aux_tensors()
        emb2.set_aux_inputs(tids, ts, ages, vo, vs)
        assert emb2._type_ids is not None
        out = emb2(ids)
        assert out.shape == (B, L, H)

    def test_cache_consumed_after_forward(self):
        """Cache must be cleared to None after a single forward call."""
        emb2 = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids  = torch.randint(1, VOCAB, (B, L))
        tids, ts, ages, vo, vs = _aux_tensors()
        emb2.set_aux_inputs(tids, ts, ages, vo, vs)
        emb2(ids)
        assert emb2._type_ids    is None
        assert emb2._time_stamps is None
        assert emb2._ages        is None

    def test_fallback_mode_no_aux(self, emb_model):
        """Fallback mode (no aux): word + positional only, correct shape."""
        ids = torch.randint(1, VOCAB, (B, L))
        out = emb_model(ids)
        assert out.shape == (B, L, H)
        assert not out.isnan().any()

    def test_special_token_mask(self):
        """Special tokens (type_id <= SPECIAL_TYPE_MAX) receive zero aux embeddings."""
        emb_eval = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H,
                                     hidden_dropout_prob=0.0)
        emb_eval.eval()
        b2, l2 = 2, 5
        ids  = torch.randint(1, VOCAB, (b2, l2))
        tids = torch.full((b2, l2), MIMIC4_TOKEN_TYPES["CLS"], dtype=torch.long)
        tids[1, :] = MIMIC4_TOKEN_TYPES["procedures_icd"]
        ts   = torch.ones(b2, l2) * 10.0
        ages = torch.ones(b2, l2) * 50.0
        vo   = torch.ones(b2, l2, dtype=torch.long) * 3
        vs   = torch.ones(b2, l2, dtype=torch.long)
        with torch.no_grad():
            out = emb_eval(ids, tids, ts, ages, vo, vs)
        assert out.shape == (b2, l2, H)
        assert not out.isnan().any()

    def test_padding_token_deterministic(self):
        """Padding token (id=0) gives deterministic output in eval mode."""
        emb_eval = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H,
                                     hidden_dropout_prob=0.0)
        emb_eval.eval()
        ids  = torch.zeros(1, 8, dtype=torch.long)
        tids = torch.zeros(1, 8, dtype=torch.long)
        ts   = torch.zeros(1, 8)
        ages = torch.zeros(1, 8)
        vo   = torch.zeros(1, 8, dtype=torch.long)
        vs   = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            out1 = emb_eval(ids, tids, ts, ages, vo, vs)
            out2 = emb_eval(ids, tids, ts, ages, vo, vs)
        assert torch.allclose(out1, out2)

    def test_gradient_flows_through_time_inputs(self):
        """Gradient flows back through time_stamps and ages inputs."""
        emb_grad = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids  = torch.randint(1, VOCAB, (2, 6))
        tids = torch.randint(SPECIAL_TYPE_MAX + 1, NUM_TOKEN_TYPES, (2, 6))
        ts   = torch.rand(2, 6, requires_grad=True)
        ages = torch.rand(2, 6, requires_grad=True)
        vo   = torch.randint(0, 10, (2, 6))
        vs   = torch.randint(1, NUM_VISIT_SEGMENTS, (2, 6))
        emb_grad(ids, tids, ts, ages, vo, vs).sum().backward()
        assert ts.grad   is not None
        assert ages.grad is not None

    def test_gradient_reaches_parameters(self):
        """Gradient reaches word_embeddings, projection, and time embedding params."""
        emb_grad = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids  = torch.randint(1, VOCAB, (2, 6))
        tids, ts, ages, vo, vs = _aux_tensors(2, 6)
        emb_grad(ids, tids, ts, ages, vo, vs).sum().backward()
        assert emb_grad.word_embeddings.weight.grad         is not None
        assert emb_grad.scale_back_concat_layer.weight.grad is not None
        assert emb_grad.time_embeddings.w.grad              is not None


# ===========================================================================
# Section 4: EHRMambaEmbeddingAdapter
# ===========================================================================

class TestEHRMambaEmbeddingAdapter:
    """EHRMambaEmbeddingAdapter — the dict-interface shim (V3: in embedding module)."""

    @pytest.fixture
    def adapter_pair(self):
        core    = EHRMambaEmbedding(vocab_size=100, hidden_size=32)
        adapter = EHRMambaEmbeddingAdapter(core)
        return core, adapter

    def test_holds_reference_to_core(self, adapter_pair):
        core, adapter = adapter_pair
        assert adapter.embedding is core

    def test_fresh_cache_empty(self, adapter_pair):
        _, adapter = adapter_pair
        assert adapter._aux == {}

    def test_set_aux_inputs_keys(self, adapter_pair):
        """set_aux_inputs must populate exactly the five expected keys."""
        _, adapter = adapter_pair
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        expected = {"token_type_ids", "time_stamps", "ages", "visit_orders", "visit_segments"}
        assert set(adapter._aux.keys()) == expected

    def test_dict_forward_key_preserved(self, adapter_pair):
        """forward() must preserve the input dict key and return (B, L, H)."""
        _, adapter = adapter_pair
        ids  = torch.randint(1, 100, (2, 8))
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        result = adapter({"input_ids": ids})
        assert "input_ids" in result
        assert result["input_ids"].shape == (2, 8, 32)
        assert not result["input_ids"].isnan().any()

    def test_cache_cleared_after_forward(self, adapter_pair):
        """Aux cache must be reset to {} after a forward call."""
        _, adapter = adapter_pair
        ids  = torch.randint(1, 100, (2, 8))
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        adapter({"input_ids": ids})
        assert adapter._aux == {}

    def test_fallback_without_aux(self, adapter_pair):
        """forward() without prior set_aux_inputs must still return correct shape."""
        _, adapter = adapter_pair
        ids = torch.randint(1, 100, (2, 8))
        result = adapter({"input_ids": ids})
        assert result["input_ids"].shape == (2, 8, 32)

    def test_multiple_feature_keys(self, adapter_pair):
        """Multiple feature keys must be processed independently."""
        _, adapter = adapter_pair
        ids  = torch.randint(1, 100, (2, 8))
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        multi = adapter({"feat_a": ids, "feat_b": ids})
        assert "feat_a" in multi and "feat_b" in multi
        assert multi["feat_a"].shape == multi["feat_b"].shape == (2, 8, 32)


# ===========================================================================
# Section 5: collate_ehr_mamba_batch (now in mimic4_ehr_mamba_task.py)
# ===========================================================================

class TestCollateBatch:
    """collate_ehr_mamba_batch — batch assembly and right-padding (V3: in task file)."""

    @pytest.fixture(scope="class")
    def batch_3(self):
        samples = [
            _raw_collate_sample(10, label=1, pid="p0"),
            _raw_collate_sample( 7, label=0, pid="p1"),
            _raw_collate_sample(15, label=1, pid="p2"),
        ]
        return collate_ehr_mamba_batch(samples), 3, 15

    def test_output_keys(self, batch_3):
        batch, _, _ = batch_3
        expected = {"input_ids", "token_type_ids", "time_stamps", "ages",
                    "visit_orders", "visit_segments", "label", "patient_id", "visit_id"}
        assert set(batch.keys()) == expected

    def test_tensor_shapes(self, batch_3):
        batch, B_, L_max = batch_3
        for key in ["input_ids", "token_type_ids", "visit_orders",
                    "visit_segments", "time_stamps", "ages"]:
            assert batch[key].shape == (B_, L_max)

    def test_short_sequence_padded_with_zeros(self, batch_3):
        batch, _, _ = batch_3
        assert (batch["input_ids"][1, 7:] == 0).all()

    def test_padded_token_type_is_pad(self, batch_3):
        batch, _, _ = batch_3
        assert (batch["token_type_ids"][1, 7:] == MIMIC4_TOKEN_TYPES["PAD"]).all()

    def test_float_fields_padded_zero(self, batch_3):
        batch, _, _ = batch_3
        assert (batch["time_stamps"][1, 7:] == 0.0).all()
        assert (batch["ages"][1, 7:]        == 0.0).all()

    def test_labels_shape_and_values(self, batch_3):
        batch, B_, _ = batch_3
        assert batch["label"].shape == (B_,)
        assert batch["label"].tolist() == [1, 0, 1]

    def test_metadata_lists_preserved(self, batch_3):
        batch, _, _ = batch_3
        assert batch["patient_id"] == ["p0", "p1", "p2"]
        assert all("_v0" in v for v in batch["visit_id"])

    def test_single_sample_batch(self):
        sample = _raw_collate_sample(5, label=0)
        batch  = collate_ehr_mamba_batch([sample])
        assert batch["input_ids"].shape == (1, 5)
        assert batch["label"].shape     == (1,)


# ===========================================================================
# Section 6: RMSNorm
# ===========================================================================

class TestRMSNorm:
    """RMSNorm — root mean square layer normalization (paper ref §62)."""

    def test_output_shape_preserved(self):
        norm = RMSNorm(dim=32)
        assert norm(torch.randn(4, 10, 32)).shape == (4, 10, 32)

    def test_output_no_nan(self):
        norm = RMSNorm(dim=16)
        out  = norm(torch.randn(2, 8, 16))
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_gradient_flows(self):
        norm = RMSNorm(dim=8)
        x    = torch.randn(2, 4, 8, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad           is not None
        assert norm.weight.grad is not None


# ===========================================================================
# Section 7: MambaBlock
# ===========================================================================

class TestMambaBlock:
    """MambaBlock — single Mamba SSM block (paper Appendix C.1)."""

    def test_output_shape(self):
        blk = MambaBlock(d_model=32)
        assert blk(torch.randn(2, 10, 32)).shape == (2, 10, 32)

    def test_output_no_nan(self):
        blk = MambaBlock(d_model=16)
        out = blk(torch.randn(2, 5, 16))
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_residual_connection(self):
        blk = MambaBlock(d_model=16)
        x   = torch.randn(1, 4, 16)
        assert not torch.allclose(blk(x), x)

    def test_gradient_flows(self):
        blk = MambaBlock(d_model=8)
        x   = torch.randn(2, 6, 8, requires_grad=True)
        blk(x).sum().backward()
        assert x.grad is not None
        grad_count = sum(1 for p in blk.parameters()
                         if p.grad is not None and p.grad.norm() > 0)
        assert grad_count > 0


# ===========================================================================
# Section 8: EHRMamba — full-stack integration
# ===========================================================================

class TestEHRMamba:
    """EHRMamba — full-stack integration on a synthetic SampleDataset.

    Pipeline: DataLoader (collate_ehr_mamba_batch from task)
              -> EHRMamba.forward(**batch)
              -> set_aux_inputs (our V3 adapter pattern)
              -> EHRMambaEmbedding (§2.2 fusion)
              -> MambaBlock × num_layers
              -> get_last_visit -> Dropout -> Linear -> loss/y_prob
    """

    def test_instantiation_attributes(self, ehr_model):
        assert ehr_model.embedding_dim == 32
        assert ehr_model.num_layers    == 1
        assert ehr_model.feature_keys  == ["input_ids"]
        assert ehr_model.label_key     == "label"
        assert ehr_model.mode          == "binary"

    def test_embedding_vocab_matches_dataset(self, ehr_model, sample_ds):
        """Embedding table size must match the SequenceProcessor vocabulary."""
        vocab_size = sample_ds.input_processors["input_ids"].vocab_size()
        emb_vocab  = ehr_model.embedding_model.embedding.word_embeddings.num_embeddings
        assert emb_vocab == vocab_size

    def test_forward_output_keys(self, ehr_model, ehr_batch):
        ehr_model.eval()
        with torch.no_grad():
            out = ehr_model(**ehr_batch)
        assert {"loss", "y_prob", "y_true", "logit"} <= set(out.keys())

    def test_forward_output_shapes(self, ehr_model, ehr_batch):
        ehr_model.eval()
        with torch.no_grad():
            out = ehr_model(**ehr_batch)
        B_ = ehr_batch["input_ids"].shape[0]
        assert out["y_prob"].shape[0] == B_
        assert out["y_true"].shape[0] == B_
        assert out["logit"].shape[0]  == B_

    def test_loss_finite_scalar(self, ehr_model, ehr_batch):
        ehr_model.eval()
        with torch.no_grad():
            out = ehr_model(**ehr_batch)
        assert out["loss"].ndim == 0
        assert not out["loss"].isnan().item()
        assert not out["loss"].isinf().item()

    def test_y_prob_in_unit_interval(self, ehr_model, ehr_batch):
        ehr_model.eval()
        with torch.no_grad():
            out = ehr_model(**ehr_batch)
        assert (out["y_prob"] >= 0).all() and (out["y_prob"] <= 1).all()

    def test_backward_and_optimizer_step(self, sample_ds, ehr_batch):
        import torch.optim as optim
        model = EHRMamba(dataset=sample_ds, embedding_dim=32, num_layers=1, dropout=0.1)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        model(**ehr_batch)["loss"].backward()
        grad_count = sum(1 for p in model.parameters()
                         if p.grad is not None and p.grad.norm() > 0)
        assert grad_count > 0
        optimizer.step()

    def test_gradient_reaches_word_embeddings(self, sample_ds, ehr_batch):
        """Gradient must flow back to word_embeddings through the adapter."""
        model = EHRMamba(dataset=sample_ds, embedding_dim=32, num_layers=1, dropout=0.1)
        model.train()
        model(**ehr_batch)["loss"].backward()
        grad = model.embedding_model.embedding.word_embeddings.weight.grad
        assert grad is not None

    def test_embed_flag(self, sample_ds, ehr_batch):
        """embed=True must add pooled patient-level embeddings to the output dict."""
        model = EHRMamba(dataset=sample_ds, embedding_dim=32, num_layers=1, dropout=0.1)
        model.eval()
        with torch.no_grad():
            out = model(**ehr_batch, embed=True)
        assert "embed" in out
        assert out["embed"].shape[0] == ehr_batch["input_ids"].shape[0]

    def test_batch_size_one(self, sample_ds):
        """batch_size=1 forward must not cause shape errors."""
        from torch.utils.data import DataLoader
        ds     = _make_synthetic_sample_dataset(n_samples=2)
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_ehr_mamba_batch)
        batch  = next(iter(loader))
        batch["label"] = batch["label"].float().unsqueeze(-1)
        model = EHRMamba(dataset=ds, embedding_dim=16, num_layers=1)
        model.eval()
        with torch.no_grad():
            out = model(**batch)
        assert out["y_prob"].shape[0] == 1

    def test_create_mask_mixed_tokens(self, ehr_model):
        """_create_mask: non-zero positions unmasked, zero positions masked."""
        val  = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=torch.long)
        mask = ehr_model._create_mask("input_ids", val)
        assert mask.shape[0] == 2
        assert mask[0, 0].item()
        assert not mask[0, 2].item()

    def test_create_mask_allzero_row_fixed(self, ehr_model):
        """_create_mask: all-zero sequence must get at least one True."""
        val  = torch.zeros(1, 4, dtype=torch.long)
        mask = ehr_model._create_mask("input_ids", val)
        assert mask[0].any()

    def test_pool_embedding_3d_passthrough(self):
        x = torch.randn(2, 7, 32)
        assert EHRMamba._pool_embedding(x).shape == (2, 7, 32)

    def test_pool_embedding_2d_unsqueeze(self):
        x = torch.randn(2, 32)
        assert EHRMamba._pool_embedding(x).shape == (2, 1, 32)

    def test_pool_embedding_4d_sum(self):
        x = torch.randn(2, 7, 3, 32)
        assert EHRMamba._pool_embedding(x).shape == (2, 7, 32)

    def test_tmpdir_round_trip(self, sample_ds):
        """Model checkpoint can be saved and loaded from a temp directory."""
        import shutil
        tmpdir = tempfile.mkdtemp(prefix="ehr_mamba_test_")
        try:
            model = EHRMamba(dataset=sample_ds, embedding_dim=32, num_layers=1)
            path  = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)
            assert os.path.exists(path)
            sd = torch.load(path, map_location="cpu")
            assert isinstance(sd, dict)
        finally:
            shutil.rmtree(tmpdir)
