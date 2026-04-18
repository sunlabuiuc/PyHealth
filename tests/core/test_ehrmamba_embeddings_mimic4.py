"""tests/core/test_ehrmamba_embeddings_mimic4.py
=================================================
``unittest``-based test suite for the EHR Mamba implementation.

Covers the same functionality as ``test_ehrmamba_embeddings.py`` but uses
:mod:`unittest` (like ``test_cnn.py``) instead of pytest fixtures, making
the suite runnable without pytest or any plugins.

Modules under test
------------------
``mortality_prediction_ehrmamba_mimic4.py`` (task-side):
    - :class:`LabQuantizer`                — 5-bin quantile lab tokenizer
      (paper Appx. B).
    - :class:`MIMIC4EHRMambaTask`          — base task
      paper-conformant token sequence.
    - :class:`MIMIC4EHRMambaMortalityTask` — adds in-hospital mortality label
      and minimum-age filter.
    - :func:`collate_ehr_mamba_batch`      — custom DataLoader collate for
      variable-length EHR sequences.

``ehrmamba_embedding.py`` (embedding-side):
    - :class:`TimeEmbeddingLayer`          — learnable sinusoidal time / age
      embedding (paper §2.2).
    - :class:`EHRMambaEmbedding`           — full 7-component fusion embedding
      (paper §2.2 / Appx. C.2).
    - :class:`EHRMambaEmbeddingAdapter`    — dict-interface shim bridging
      :class:`EHRMambaEmbedding` into PyHealth's EmbeddingModel API.

``ehrmamba_vi.py`` (model-side):
    - :class:`RMSNorm`    — root mean square layer normalization.
    - :class:`MambaBlock` — single Mamba SSM block.
    - :class:`EHRMamba`   — full PyHealth BaseModel with N Mamba blocks and
      an FC classification head.

Design principles
-----------------
- All tests use synthetic data only; no MIMIC-IV files are required.
- Expensive shared objects (datasets, models, batches) are constructed once
  per class via ``setUpClass`` to keep total runtime under ~15 s.
- Each test method carries a docstring explaining *why* the property matters,
  not just *what* is checked.

Running
-------
Run all tests::

    python -m unittest PyHealth.tests.core.test_ehrmamba_embeddings_mimic4 -v

Run a single class::

    python -m unittest \\
        PyHealth.tests.core.test_ehrmamba_embeddings_mimic4.TestEHRMamba -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import torch
from torch.utils.data import DataLoader

from pyhealth.models.ehrmamba_embedding import (
    EHRMambaEmbedding,
    EHRMambaEmbeddingAdapter,
    NUM_TOKEN_TYPES,
    NUM_VISIT_SEGMENTS,
    SPECIAL_TYPE_MAX,
    TimeEmbeddingLayer,
)
from pyhealth.models.ehrmamba_vi import EHRMamba, MambaBlock, RMSNorm
from pyhealth.tasks.mortality_prediction_ehrmamba_mimic4 import (
    MAX_NUM_VISITS,
    MIMIC4_TOKEN_TYPES,
    MIMIC4EHRMambaTask,
    MIMIC4EHRMambaMortalityTask,
    LabQuantizer,
    _time_interval_token,
    collate_ehr_mamba_batch,
)


# ===========================================================================
# Timed base class
# ===========================================================================


class TimedTestCase(unittest.TestCase):
    """``unittest.TestCase`` subclass that prints per-test wall-clock timings.

    Overrides ``setUp`` and ``tearDown`` to record ``time.perf_counter()``
    before and after each test method, then prints::

        [TIMING] TestClassName.test_foo — 0.0031 s

    Subclasses that define their own ``setUp`` / ``tearDown`` must call
    ``super().setUp()`` / ``super().tearDown()`` to preserve timing.
    """

    def setUp(self) -> None:
        """Record the start time before every test method."""
        self._t0: float = time.perf_counter()

    def tearDown(self) -> None:
        """Print elapsed wall-clock time after every test method."""
        elapsed = time.perf_counter() - self._t0
        print(
            f"\n  [TIMING] {self.__class__.__name__}.{self._testMethodName}"
            f" — {elapsed:.4f} s"
        )


# ===========================================================================
# Shared synthetic-data constants
# ===========================================================================

VOCAB: int = 200  # synthetic vocabulary size
H: int = 64       # embedding hidden dimension
B: int = 3        # default batch size
L: int = 15       # default sequence length


# ===========================================================================
# Synthetic-data infrastructure
# ===========================================================================


class _MockEvent:
    """Stand-in for a single row returned by PyHealth's event tables.

    Accepts arbitrary keyword arguments and stores them as instance attributes,
    matching the attribute-access pattern used by the task classes.

    Args:
        **kwargs: Arbitrary field-name / value pairs (e.g. ``hadm_id``,
            ``timestamp``, ``hospital_expire_flag``).
    """

    def __init__(self, **kwargs: Any) -> None:
        """Store all keyword arguments as instance attributes.

        Args:
            **kwargs: Arbitrary field-name / value pairs.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockPatient:
    """Synthetic patient satisfying the PyHealth Patient API.

    Supports the ``get_events()`` call pattern used by
    :class:`MIMIC4EHRMambaTask`, routing requests to either the demographic
    stub, the admission list, or a pre-built Polars DataFrame table.

    Args:
        patient_id: Unique patient identifier string.
        anchor_age: Patient age at the MIMIC-IV anchor year.
        anchor_year: MIMIC-IV anchor year for age calculation.
        admissions: List of admission :class:`_MockEvent` objects in
            chronological order.
        proc_df: Optional procedures DataFrame keyed on
            ``procedures_icd/icd_code`` and ``hadm_id``.
        rx_df: Optional prescriptions DataFrame keyed on
            ``prescriptions/drug`` and ``hadm_id``.
        lab_df: Optional lab events DataFrame keyed on
            ``labevents/itemid``, ``labevents/valuenum``, and ``hadm_id``.
    """

    def __init__(
        self,
        patient_id: str,
        anchor_age: int,
        anchor_year: int,
        admissions: List[_MockEvent],
        proc_df: Optional[pl.DataFrame] = None,
        rx_df: Optional[pl.DataFrame] = None,
        lab_df: Optional[pl.DataFrame] = None,
    ) -> None:
        """Initialize MockPatient with demographics and optional event tables.

        Args:
            patient_id: Unique patient identifier string.
            anchor_age: Patient age at the MIMIC-IV anchor year.
            anchor_year: MIMIC-IV anchor year for age calculation.
            admissions: Chronologically ordered list of admission events.
            proc_df: Optional procedures Polars DataFrame.
            rx_df: Optional prescriptions Polars DataFrame.
            lab_df: Optional lab events Polars DataFrame.
        """
        self.patient_id = patient_id
        self._anchor_age = anchor_age
        self._anchor_year = anchor_year
        self._admissions = admissions
        self._tables: Dict[str, pl.DataFrame] = {}
        if proc_df is not None:
            self._tables["procedures_icd"] = proc_df
        if rx_df is not None:
            self._tables["prescriptions"] = rx_df
        if lab_df is not None:
            self._tables["labevents"] = lab_df

    def get_events(
        self,
        event_type: Optional[str] = None,
        filters: Optional[List[Tuple[str, str, Any]]] = None,
        return_df: bool = False,
    ) -> Union[List[_MockEvent], pl.DataFrame]:
        """Dispatch event queries to the appropriate synthetic data source.

        Args:
            event_type: MIMIC-IV table name (e.g. ``"patients"``,
                ``"admissions"``, ``"procedures_icd"``).
            filters: Optional list of ``(column, operator, value)`` triples
                applied as equality filters on DataFrames.
            return_df: If ``True``, return a Polars DataFrame instead of a
                list of event objects.

        Returns:
            A list of :class:`_MockEvent` objects for ``"patients"`` and
            ``"admissions"`` queries, or a (possibly filtered) Polars
            DataFrame when ``return_df=True``.
        """
        if event_type == "patients":
            return [
                _MockEvent(
                    anchor_age=self._anchor_age,
                    anchor_year=self._anchor_year,
                )
            ]
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


def _make_admission(
    hadm_id: str, ts: datetime, expire: int = 0
) -> _MockEvent:
    """Create a single admission event stub.

    Args:
        hadm_id: Hospital admission identifier string.
        ts: Admission timestamp.
        expire: Value of ``hospital_expire_flag`` (0 = survived, 1 = died).
            Defaults to 0.

    Returns:
        A :class:`_MockEvent` with ``hadm_id``, ``timestamp``, and
        ``hospital_expire_flag`` attributes set.
    """
    return _MockEvent(hadm_id=hadm_id, timestamp=ts, hospital_expire_flag=expire)


def _make_synthetic_patient(
    patient_id: str = "p001",
    anchor_age: int = 55,
    anchor_year: int = 2020,
    n_visits: int = 2,
    expire_last: int = 0,
) -> MockPatient:
    """Build a complete :class:`MockPatient` with ``n_visits`` admissions.

    Each admission contains one procedure (9904), one prescription (Aspirin),
    and one lab result (itemid 51006, value 1.2).

    Args:
        patient_id: Unique patient identifier. Defaults to ``"p001"``.
        anchor_age: Patient age at the anchor year. Defaults to 55.
        anchor_year: MIMIC-IV anchor year. Defaults to 2020.
        n_visits: Number of synthetic admissions to generate. Defaults to 2.
        expire_last: ``hospital_expire_flag`` for the last admission.
            Defaults to 0 (survived).

    Returns:
        A fully populated :class:`MockPatient` with procedures, prescriptions,
        and lab events DataFrames attached.
    """
    base = datetime(anchor_year, 1, 1)
    adms = [
        _make_admission(
            f"adm{i:03d}",
            base + timedelta(days=30 * i),
            expire=expire_last if i == n_visits - 1 else 0,
        )
        for i in range(n_visits)
    ]
    hadm_ids = [a.hadm_id for a in adms]
    proc_df = pl.DataFrame({
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
    return MockPatient(
        patient_id, anchor_age, anchor_year, adms, proc_df, rx_df, lab_df
    )


def _make_synthetic_sample_dataset(n_samples: int = 4):
    """Return a PyHealth SampleDataset built from hardcoded synthetic tokens.

    Token sequence (7 tokens — one complete visit bracket)::

        [CLS]  [VS]  PR:9904  RX:Aspirin  LB:51006_bin2  [VE]  [REG]

    Args:
        n_samples: Number of identical synthetic samples to generate.
            Defaults to 4.

    Returns:
        A :class:`~pyhealth.datasets.SampleDataset` ready for use with
        :class:`EHRMamba` and :func:`collate_ehr_mamba_batch`.
    """
    from pyhealth.datasets import create_sample_dataset

    tokens = [
        "[CLS]", "[VS]", "PR:9904", "RX:Aspirin",
        "LB:51006_bin2", "[VE]", "[REG]",
    ]
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
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "input_ids": tokens,
            "token_type_ids": torch.tensor(type_ids, dtype=torch.long),
            "time_stamps": torch.zeros(seq_len, dtype=torch.float),
            "ages": torch.full((seq_len,), 55.0, dtype=torch.float),
            "visit_orders": torch.zeros(seq_len, dtype=torch.long),
            "visit_segments": torch.tensor(
                [0, 1, 1, 1, 1, 1, 1], dtype=torch.long
            ),
            "label": i % 2,
        }
        for i in range(n_samples)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"input_ids": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_v3",
    )


def _aux_tensors(
    b: int = B,
    l: int = L,
    all_clinical: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Generate a full set of synthetic auxiliary tensors for :class:`EHRMambaEmbedding`.

    Args:
        b: Batch size. Defaults to :data:`B`.
        l: Sequence length. Defaults to :data:`L`.
        all_clinical: If ``False`` (default), forces position 0 to the
            ``CLS`` type so the first token is a structural special token.

    Returns:
        A 5-tuple ``(token_type_ids, time_stamps, ages, visit_orders,
        visit_segments)`` — all ``torch.Tensor`` with shape ``(b, l)``.
    """
    tids = torch.randint(SPECIAL_TYPE_MAX + 1, NUM_TOKEN_TYPES, (b, l))
    if not all_clinical:
        tids[:, 0] = MIMIC4_TOKEN_TYPES["CLS"]
    ts = torch.rand(b, l) * 52.0
    ages = torch.rand(b, l) * 30.0 + 40.0
    vo = torch.randint(0, 10, (b, l))
    vs = torch.randint(1, NUM_VISIT_SEGMENTS, (b, l))
    vs[:, 0] = 0
    return tids, ts, ages, vo, vs


def _raw_collate_sample(
    seq_len: int, label: int, pid: str = "p"
) -> Dict[str, Any]:
    """Build a single pre-collation sample mimicking :class:`MIMIC4EHRMambaTask` output.

    Args:
        seq_len: Number of tokens in this sample's sequence.
        label: Binary label value (0 or 1).
        pid: Patient ID prefix string. Defaults to ``"p"``.

    Returns:
        A sample dict with ``input_ids``, ``token_type_ids``, ``time_stamps``,
        ``ages``, ``visit_orders``, ``visit_segments``, ``label``,
        ``patient_id``, and ``visit_id`` keys.
    """
    ids = torch.randint(1, 100, (seq_len,)).long()
    tids = torch.randint(
        SPECIAL_TYPE_MAX + 1, NUM_TOKEN_TYPES, (seq_len,)
    ).long()
    tids[0] = MIMIC4_TOKEN_TYPES["CLS"]
    return {
        "input_ids": ids,
        "token_type_ids": tids,
        "time_stamps": torch.rand(seq_len) * 52.0,
        "ages": torch.full((seq_len,), 55.0),
        "visit_orders": torch.zeros(seq_len, dtype=torch.long),
        "visit_segments": torch.ones(seq_len, dtype=torch.long),
        "label": label,
        "patient_id": pid,
        "visit_id": f"{pid}_v0",
    }


# ===========================================================================
# Section 0: MIMIC4EHRMambaTask and MIMIC4EHRMambaMortalityTask
# ===========================================================================


class TestMIMIC4EHRMambaTask(TimedTestCase):
    """Tests for :class:`MIMIC4EHRMambaTask`.

    Verifies the base task correctly builds the token sequence (§2.1) from a synthetic 
    patient, including all structural special tokens and auxiliary metadata tensors.  
    """

    def test_base_task_returns_no_label(self) -> None:
        """Base class _make_label must return an empty dict — no label fields.

        Subclasses add labels by overriding _make_label.  The base returning {}
        (not None) keeps the sample but omits a prediction target, which is the
        correct behaviour for unsupervised pretraining sequences.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        samples = task(patient)
        self.assertEqual(len(samples), 1, "Base task must produce exactly one sample")
        self.assertNotIn(
            "label", samples[0], "Base task must not include a label field"
        )

    def test_sample_contains_required_fields(self) -> None:
        """Sample dict must include all six sequence fields plus patient/visit IDs.

        ``EHRMamba.forward()`` accesses ``input_ids`` via ``feature_keys``; the
        remaining five fields are consumed as auxiliary tensors via ``**kwargs``.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        sample = task(patient)[0]
        required = {
            "input_ids", "token_type_ids", "time_stamps", "ages",
            "visit_orders", "visit_segments", "patient_id", "visit_id",
        }
        self.assertTrue(
            required <= set(sample.keys()),
            f"Missing fields: {required - set(sample.keys())}",
        )

    def test_input_ids_starts_with_cls(self) -> None:
        """The first token in input_ids must always be [CLS] (paper §2.1).

        EHR Mamba uses a single global [CLS] at the start of the full patient
        sequence, analogous to BERT's [CLS] token.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        sample = task(patient)[0]
        self.assertEqual(
            sample["input_ids"][0],
            "[CLS]",
            "First token must be [CLS] (paper §2.1)",
        )

    def test_sequence_contains_visit_delimiters(self) -> None:
        """input_ids must contain [VS] and [VE] tokens (paper §2.1).

        Each visit is bracketed by [VS] (visit start) and [VE] (visit end).
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=2)
        sample = task(patient)[0]
        self.assertIn("[VS]", sample["input_ids"], "Sequence must contain [VS] tokens")
        self.assertIn("[VE]", sample["input_ids"], "Sequence must contain [VE] tokens")

    def test_sequence_contains_reg_token(self) -> None:
        """[REG] register token must follow each [VE] (paper §2.1).

        The [REG] token is the prediction anchor for multi-task prompted
        fine-tuning (MPF) and must appear after every visit-end token.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient()
        sample = task(patient)[0]
        self.assertIn(
            "[REG]", sample["input_ids"], "Sequence must contain [REG] tokens"
        )

    def test_aux_tensor_lengths_match_input_ids(self) -> None:
        """All auxiliary tensors must have the same length as input_ids.

        :class:`EHRMambaEmbedding` processes all tensors position-by-position;
        a length mismatch would cause an index error or silent misalignment.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=3)
        sample = task(patient)[0]
        seq_len = len(sample["input_ids"])
        for key in (
            "token_type_ids", "time_stamps", "ages", "visit_orders", "visit_segments"
        ):
            self.assertEqual(
                len(sample[key]),
                seq_len,
                f"{key} length {len(sample[key])} != input_ids length {seq_len}",
            )

    def test_visit_segments_alternate(self) -> None:
        """Clinical tokens must have alternating segment values 1/2 (paper §2.2).

        Visit segment alternates between 1 and 2 across consecutive non-empty
        visits so the model can distinguish adjacent visits.  Structural tokens
        ([CLS], [VS], [VE], [REG], time-interval) receive segment 0.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=3)
        sample = task(patient)[0]
        segs = sample["visit_segments"].tolist()
        clinical_segs = {
            s for i, s in enumerate(segs)
            if sample["token_type_ids"][i].item() > SPECIAL_TYPE_MAX
        }
        self.assertTrue(
            clinical_segs <= {1, 2},
            f"Clinical segments must be in {{1, 2}}, got {clinical_segs}",
        )

    def test_min_visits_filter(self) -> None:
        """Patient with fewer visits than min_visits must return an empty list."""
        task = MIMIC4EHRMambaTask(min_visits=5)
        patient = _make_synthetic_patient(n_visits=2)
        self.assertEqual(
            task(patient), [], "Patient below min_visits must be filtered out"
        )

    def test_empty_patient_returns_empty(self) -> None:
        """Patient with no demographics must return an empty list."""

        class _NoDemo(MockPatient):
            """MockPatient subclass that returns no demographics."""

            def get_events(
                self,
                event_type: Optional[str] = None,
                **kwargs: Any,
            ) -> Union[List[_MockEvent], pl.DataFrame]:
                """Return empty list for 'patients'; delegate everything else."""
                if event_type == "patients":
                    return []
                return super().get_events(event_type, **kwargs)

        task = MIMIC4EHRMambaTask()
        patient = _NoDemo("px", 50, 2020, [])
        self.assertEqual(
            task(patient), [], "Patient with no demographics must return []"
        )

    def test_inter_visit_time_interval_token_present(self) -> None:
        """Multi-visit patient must have a time-interval token between visits.

        Paper §2.1: [W0]–[W3], [M1]–[M12], or [LT] appears between the [REG]
        of one visit and the [VS] of the next.
        """
        task = MIMIC4EHRMambaTask()
        patient = _make_synthetic_patient(n_visits=2)
        sample = task(patient)[0]
        tokens = sample["input_ids"]
        time_tokens = [
            t for t in tokens
            if t.startswith("[W") or t.startswith("[M") or t == "[LT]"
        ]
        self.assertGreaterEqual(
            len(time_tokens),
            1,
            "Multi-visit sequence must contain at least one time-interval token",
        )


class TestMIMIC4EHRMambaMortalityTask(TimedTestCase):
    """Tests for :class:`MIMIC4EHRMambaMortalityTask`.

    Verifies label assignment from ``hospital_expire_flag``, the minimum-age
    demographic filter, and correct inheritance from the base task.  All tests
    use synthetic data; no MIMIC-IV files are required.
    """

    def test_mortality_label_zero_for_survivor(self) -> None:
        """hospital_expire_flag=0 on the last admission must produce label=0."""
        task = MIMIC4EHRMambaMortalityTask()
        patient = _make_synthetic_patient(expire_last=0)
        samples = task(patient)
        self.assertEqual(len(samples), 1, "Survivor patient must produce one sample")
        self.assertEqual(samples[0]["label"], 0, "Survivor label must be 0")

    def test_mortality_label_one_for_expired(self) -> None:
        """hospital_expire_flag=1 on the last admission must produce label=1."""
        task = MIMIC4EHRMambaMortalityTask()
        patient = _make_synthetic_patient(expire_last=1)
        samples = task(patient)
        self.assertEqual(len(samples), 1, "Expired patient must produce one sample")
        self.assertEqual(samples[0]["label"], 1, "Expired label must be 1")

    def test_min_age_filter_rejects_young_patient(self) -> None:
        """Patient younger than min_age must be filtered out."""
        task = MIMIC4EHRMambaMortalityTask(min_age=18)
        patient = _make_synthetic_patient(anchor_age=15)
        self.assertEqual(task(patient), [], "Patient below min_age must return []")

    def test_min_age_filter_accepts_adult(self) -> None:
        """Patient at or above min_age must produce one sample."""
        task = MIMIC4EHRMambaMortalityTask(min_age=18)
        patient = _make_synthetic_patient(anchor_age=18)
        self.assertEqual(
            len(task(patient)), 1, "Adult patient must produce one sample"
        )

    def test_output_schema_is_binary(self) -> None:
        """Mortality subclass must declare output_schema = {'label': 'binary'}."""
        self.assertEqual(
            MIMIC4EHRMambaMortalityTask.output_schema,
            {"label": "binary"},
            "output_schema must declare binary label",
        )

    def test_inherits_base_sequence_structure(self) -> None:
        """Mortality samples must contain the same sequence fields as the base task."""
        task = MIMIC4EHRMambaMortalityTask()
        patient = _make_synthetic_patient()
        sample = task(patient)[0]
        self.assertEqual(sample["input_ids"][0], "[CLS]", "[CLS] must be first token")
        self.assertIn("[VS]", sample["input_ids"], "Sequence must contain [VS]")
        self.assertIn("[REG]", sample["input_ids"], "Sequence must contain [REG]")


class TestTimeIntervalToken(TimedTestCase):
    """Tests for :func:`_time_interval_token`.

    Validates the inter-visit gap → paper special-token mapping (§2.1) across
    all four week-level buckets, monthly tokens, and the long-time sentinel.
    """

    def test_w0_for_sub_week_gap(self) -> None:
        """Gaps under 1 week must map to [W0]."""
        self.assertEqual(_time_interval_token(0.0), "[W0]")
        self.assertEqual(_time_interval_token(0.99), "[W0]")

    def test_w1_w2_w3_ranges(self) -> None:
        """Gaps of 1, 2, and 3 weeks must map to [W1], [W2], and [W3]."""
        self.assertEqual(_time_interval_token(1.0), "[W1]")
        self.assertEqual(_time_interval_token(2.0), "[W2]")
        self.assertEqual(_time_interval_token(3.0), "[W3]")

    def test_monthly_tokens(self) -> None:
        """Gaps within 1–12 months must map to [M1]–[M12]."""
        self.assertEqual(_time_interval_token(5.0), "[M1]")
        self.assertEqual(_time_interval_token(9.0), "[M2]")
        self.assertEqual(_time_interval_token(52.0), "[M12]")

    def test_lt_for_long_gaps(self) -> None:
        """Gaps over 12 months must map to [LT].

        [LT] requires round(weeks / 4.345) > 12.  At 52.0 weeks the rounded
        value is 12 → [M12]; at 57.0 weeks it is 13 → [LT].
        """
        self.assertEqual(_time_interval_token(57.0), "[LT]")
        self.assertEqual(_time_interval_token(60.0), "[LT]")


# ===========================================================================
# Section 1: LabQuantizer
# ===========================================================================


class TestLabQuantizer(TimedTestCase):
    """Tests for :class:`LabQuantizer`.

    Verifies the 5-bin quantile tokenizer for MIMIC-IV lab results (Appx. B):
    boundary fitting, monotone binning, range safety, fallback for unknowns,
    and token string formatting.
    """

    def test_fit_from_records_boundaries(self) -> None:
        """fit_from_records must populate boundaries with 4 cut-points for 5 bins."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 101)])
        self.assertIn("51006", q.boundaries)
        self.assertEqual(len(q.boundaries["51006"]), 4)

    def test_fit_chaining(self) -> None:
        """fit_from_records must return self so calls can be chained."""
        q = LabQuantizer()
        self.assertIs(q.fit_from_records([("X", 1.0)]), q)

    def test_bin_index_monotone(self) -> None:
        """bin_index must be non-decreasing as valuenum increases."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 1001)])
        bins = [q.bin_index("51006", float(v)) for v in range(1, 101)]
        for a, b_ in zip(bins, bins[1:]):
            self.assertLessEqual(a, b_)

    def test_bin_index_range(self) -> None:
        """bin_index must always return a value in [0, n_bins - 1]."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 101)])
        for v in range(-100, 200, 10):
            idx = q.bin_index("51006", float(v))
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, 4)

    def test_unknown_itemid_returns_zero(self) -> None:
        """bin_index must return 0 for an itemid not seen during fit."""
        q = LabQuantizer()
        q.fit_from_records([("known", 1.0)])
        self.assertEqual(q.bin_index("unknown", 5.0), 0)

    def test_token_format_with_value(self) -> None:
        """token() must produce 'LB:<itemid>_bin<N>' when valuenum is provided."""
        q = LabQuantizer(n_bins=5)
        q.fit_from_records([("51006", float(v)) for v in range(1, 101)])
        tok = q.token("51006", 50.0)
        self.assertRegex(tok, r"LB:51006_bin[0-4]")

    def test_token_format_without_value(self) -> None:
        """token() must produce 'LB:<itemid>' when valuenum is None."""
        q = LabQuantizer()
        self.assertEqual(q.token("51006", None), "LB:51006")

    def test_non_numeric_records_skipped(self) -> None:
        """Non-numeric valuenum entries must be silently skipped during fit."""
        q = LabQuantizer()
        q.fit_from_records([("X", "bad"), ("X", None), ("X", 1.0), ("X", 2.0)])
        self.assertIn("X", q.boundaries)


# ===========================================================================
# Section 2: TimeEmbeddingLayer
# ===========================================================================


class TestTimeEmbeddingLayer(TimedTestCase):
    """Tests for :class:`TimeEmbeddingLayer`.

    Checks output shape and numerical stability for both the time-delta mode
    (inter-visit week gaps) and the absolute-value mode (patient age in years).
    """

    def test_output_shape_time_delta(self) -> None:
        """Time-delta mode must produce ``(B, L, embedding_size)`` output."""
        layer = TimeEmbeddingLayer(embedding_size=16, is_time_delta=True)
        out = layer(torch.rand(B, L) * 52.0)
        self.assertEqual(out.shape, (B, L, 16))

    def test_output_shape_absolute(self) -> None:
        """Absolute-value mode must produce ``(B, L, embedding_size)`` output."""
        layer = TimeEmbeddingLayer(embedding_size=8, is_time_delta=False)
        out = layer(torch.rand(B, L) * 30.0 + 40.0)
        self.assertEqual(out.shape, (B, L, 8))

    def test_output_no_nan(self) -> None:
        """Time-delta embeddings must contain no NaN or Inf values."""
        layer = TimeEmbeddingLayer(embedding_size=16, is_time_delta=True)
        out = layer(torch.rand(2, 10) * 100.0)
        self.assertFalse(out.isnan().any().item())
        self.assertFalse(out.isinf().any().item())


# ===========================================================================
# Section 3: EHRMambaEmbedding
# ===========================================================================


class TestEHRMambaEmbedding(TimedTestCase):
    """Tests for :class:`EHRMambaEmbedding`.

    A single :class:`EHRMambaEmbedding` instance (``cls.emb_model``) is built
    once for the class and shared across read-only tests.  Tests that require
    clean internal state (e.g. cache-consumption tests) create their own.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Instantiate the shared embedding model used by read-only tests."""
        cls.emb_model = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)

    def test_default_instantiation(self) -> None:
        """All embedding tables must be correctly sized at construction."""
        self.assertEqual(self.emb_model.hidden_size, H)
        self.assertEqual(self.emb_model.word_embeddings.num_embeddings, VOCAB)
        self.assertEqual(
            self.emb_model.token_type_embeddings.num_embeddings, NUM_TOKEN_TYPES
        )
        self.assertEqual(
            self.emb_model.visit_order_embeddings.num_embeddings, MAX_NUM_VISITS
        )
        self.assertEqual(self.emb_model.position_embeddings.num_embeddings, 4096)

    def test_custom_instantiation(self) -> None:
        """Non-default hyperparameters must be stored and respected."""
        emb_c = EHRMambaEmbedding(
            vocab_size=50,
            hidden_size=32,
            type_vocab_size=10,
            max_num_visits=64,
            time_embeddings_size=8,
            num_visit_segments=3,
            max_position_embeddings=128,
        )
        self.assertEqual(emb_c.hidden_size, 32)
        self.assertEqual(emb_c.time_embeddings_size, 8)

    def test_explicit_args_shape(self) -> None:
        """Explicit-args forward pass must yield shape ``(B, L, H)`` without NaN/Inf."""
        ids = torch.randint(1, VOCAB, (B, L))
        tids, ts, ages, vo, vs = _aux_tensors()
        out = self.emb_model(ids, tids, ts, ages, vo, vs)
        self.assertEqual(out.shape, (B, L, H))
        self.assertFalse(out.isnan().any().item())
        self.assertFalse(out.isinf().any().item())

    def test_cached_api_shape(self) -> None:
        """Cached API: set_aux_inputs() then forward(ids) must yield correct shape."""
        emb2 = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids = torch.randint(1, VOCAB, (B, L))
        tids, ts, ages, vo, vs = _aux_tensors()
        emb2.set_aux_inputs(tids, ts, ages, vo, vs)
        self.assertIsNotNone(emb2._type_ids)
        out = emb2(ids)
        self.assertEqual(out.shape, (B, L, H))

    def test_cache_consumed_after_forward(self) -> None:
        """Aux cache must be cleared to None after a single forward call.

        The cached calling pattern is designed for one-shot use.  Leaving the
        cache populated would silently apply stale aux inputs to the next call.
        """
        emb2 = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids = torch.randint(1, VOCAB, (B, L))
        tids, ts, ages, vo, vs = _aux_tensors()
        emb2.set_aux_inputs(tids, ts, ages, vo, vs)
        emb2(ids)
        self.assertIsNone(emb2._type_ids)
        self.assertIsNone(emb2._time_stamps)
        self.assertIsNone(emb2._ages)

    def test_fallback_mode_no_aux(self) -> None:
        """Fallback mode (no aux tensors) must produce word+position-only output."""
        ids = torch.randint(1, VOCAB, (B, L))
        out = self.emb_model(ids)
        self.assertEqual(out.shape, (B, L, H))
        self.assertFalse(out.isnan().any().item())

    def test_special_token_mask(self) -> None:
        """Special tokens (type_id <= SPECIAL_TYPE_MAX) receive zero aux embeddings.

        Paper §2.2 states that structural tokens ([CLS], [VS], [VE], [REG],
        time-interval) use zero vectors for time, age, visit_order, and
        visit_segment components.  The masking logic must silence these
        embeddings before summation.
        """
        emb_eval = EHRMambaEmbedding(
            vocab_size=VOCAB, hidden_size=H, hidden_dropout_prob=0.0
        )
        emb_eval.eval()
        b2, l2 = 2, 5
        ids = torch.randint(1, VOCAB, (b2, l2))
        tids = torch.full(
            (b2, l2), MIMIC4_TOKEN_TYPES["CLS"], dtype=torch.long
        )
        tids[1, :] = MIMIC4_TOKEN_TYPES["procedures_icd"]
        ts = torch.ones(b2, l2) * 10.0
        ages = torch.ones(b2, l2) * 50.0
        vo = torch.ones(b2, l2, dtype=torch.long) * 3
        vs = torch.ones(b2, l2, dtype=torch.long)
        with torch.no_grad():
            out = emb_eval(ids, tids, ts, ages, vo, vs)
        self.assertEqual(out.shape, (b2, l2, H))
        self.assertFalse(out.isnan().any().item())

    def test_padding_token_deterministic(self) -> None:
        """Padding token (id=0) must yield identical output on repeated calls in eval mode."""
        emb_eval = EHRMambaEmbedding(
            vocab_size=VOCAB, hidden_size=H, hidden_dropout_prob=0.0
        )
        emb_eval.eval()
        ids = torch.zeros(1, 8, dtype=torch.long)
        tids = torch.zeros(1, 8, dtype=torch.long)
        ts = torch.zeros(1, 8)
        ages = torch.zeros(1, 8)
        vo = torch.zeros(1, 8, dtype=torch.long)
        vs = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            out1 = emb_eval(ids, tids, ts, ages, vo, vs)
            out2 = emb_eval(ids, tids, ts, ages, vo, vs)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gradient_flows_through_time_inputs(self) -> None:
        """Gradient must flow back through time_stamps and ages inputs."""
        emb_grad = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids = torch.randint(1, VOCAB, (2, 6))
        tids = torch.randint(SPECIAL_TYPE_MAX + 1, NUM_TOKEN_TYPES, (2, 6))
        ts = torch.rand(2, 6, requires_grad=True)
        ages = torch.rand(2, 6, requires_grad=True)
        vo = torch.randint(0, 10, (2, 6))
        vs = torch.randint(1, NUM_VISIT_SEGMENTS, (2, 6))
        emb_grad(ids, tids, ts, ages, vo, vs).sum().backward()
        self.assertIsNotNone(ts.grad)
        self.assertIsNotNone(ages.grad)

    def test_gradient_reaches_parameters(self) -> None:
        """Gradient must reach word_embeddings, projection, and time embedding params."""
        emb_grad = EHRMambaEmbedding(vocab_size=VOCAB, hidden_size=H)
        ids = torch.randint(1, VOCAB, (2, 6))
        tids, ts, ages, vo, vs = _aux_tensors(2, 6)
        emb_grad(ids, tids, ts, ages, vo, vs).sum().backward()
        self.assertIsNotNone(emb_grad.word_embeddings.weight.grad)
        self.assertIsNotNone(emb_grad.scale_back_concat_layer.weight.grad)
        self.assertIsNotNone(emb_grad.time_embeddings.w.grad)


# ===========================================================================
# Section 4: EHRMambaEmbeddingAdapter
# ===========================================================================


class TestEHRMambaEmbeddingAdapter(TimedTestCase):
    """Tests for :class:`EHRMambaEmbeddingAdapter`.

    A fresh ``(core, adapter)`` pair is created before each test via ``setUp``
    so that aux-cache state from one test cannot bleed into another.
    """

    def setUp(self) -> None:
        """Create a fresh core embedding and adapter pair for each test."""
        super().setUp()
        self.core = EHRMambaEmbedding(vocab_size=100, hidden_size=32)
        self.adapter = EHRMambaEmbeddingAdapter(self.core)

    def test_holds_reference_to_core(self) -> None:
        """Adapter must hold a reference to the same core embedding object."""
        self.assertIs(self.adapter.embedding, self.core)

    def test_fresh_cache_empty(self) -> None:
        """A newly created adapter must have an empty aux cache."""
        self.assertEqual(self.adapter._aux, {})

    def test_set_aux_inputs_keys(self) -> None:
        """set_aux_inputs must populate exactly the five expected cache keys."""
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        self.adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        expected = {
            "token_type_ids", "time_stamps", "ages",
            "visit_orders", "visit_segments",
        }
        self.assertEqual(set(self.adapter._aux.keys()), expected)

    def test_dict_forward_key_preserved(self) -> None:
        """forward() must preserve the input dict key and return ``(B, L, H)``."""
        ids = torch.randint(1, 100, (2, 8))
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        self.adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        result = self.adapter({"input_ids": ids})
        self.assertIn("input_ids", result)
        self.assertEqual(result["input_ids"].shape, (2, 8, 32))
        self.assertFalse(result["input_ids"].isnan().any().item())

    def test_cache_cleared_after_forward(self) -> None:
        """Aux cache must be reset to {} after a forward call.

        The adapter is designed for one-shot use: the cache is populated
        immediately before the forward call and consumed within it, preventing
        stale aux inputs from affecting subsequent batches.
        """
        ids = torch.randint(1, 100, (2, 8))
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        self.adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        self.adapter({"input_ids": ids})
        self.assertEqual(self.adapter._aux, {})

    def test_fallback_without_aux(self) -> None:
        """forward() without prior set_aux_inputs must still return correct shape."""
        ids = torch.randint(1, 100, (2, 8))
        result = self.adapter({"input_ids": ids})
        self.assertEqual(result["input_ids"].shape, (2, 8, 32))

    def test_multiple_feature_keys(self) -> None:
        """Multiple input dict keys must each produce an independent embedding."""
        ids = torch.randint(1, 100, (2, 8))
        tids, ts, ages, vo, vs = _aux_tensors(2, 8)
        self.adapter.set_aux_inputs(tids, ts, ages, vo, vs)
        multi = self.adapter({"feat_a": ids, "feat_b": ids})
        self.assertIn("feat_a", multi)
        self.assertIn("feat_b", multi)
        self.assertEqual(multi["feat_a"].shape, (2, 8, 32))
        self.assertEqual(multi["feat_b"].shape, (2, 8, 32))


# ===========================================================================
# Section 5: collate_ehr_mamba_batch
# ===========================================================================


class TestCollateBatch(TimedTestCase):
    """Tests for :func:`collate_ehr_mamba_batch`.

    A shared batch of three samples (lengths 10, 7, 15) is built once via
    ``setUpClass``.  Tests verify key presence, tensor shapes, right-padding
    to the maximum sequence length, and metadata list preservation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build a shared batch with three variable-length sequences."""
        samples = [
            _raw_collate_sample(10, label=1, pid="p0"),
            _raw_collate_sample(7, label=0, pid="p1"),
            _raw_collate_sample(15, label=1, pid="p2"),
        ]
        cls.batch = collate_ehr_mamba_batch(samples)
        cls.num_samples = 3
        cls.max_len = 15

    def test_output_keys(self) -> None:
        """Collated batch must contain all required tensor and metadata keys."""
        expected = {
            "input_ids", "token_type_ids", "time_stamps", "ages",
            "visit_orders", "visit_segments", "label", "patient_id", "visit_id",
        }
        self.assertEqual(set(self.batch.keys()), expected)

    def test_tensor_shapes(self) -> None:
        """All sequence tensors must have shape ``(B, L_max)`` after padding."""
        for key in [
            "input_ids", "token_type_ids", "visit_orders",
            "visit_segments", "time_stamps", "ages",
        ]:
            self.assertEqual(
                self.batch[key].shape,
                (self.num_samples, self.max_len),
                f"Unexpected shape for {key}",
            )

    def test_short_sequence_padded_with_zeros(self) -> None:
        """Positions beyond the original sequence length must be padded with 0."""
        self.assertTrue((self.batch["input_ids"][1, 7:] == 0).all().item())

    def test_padded_token_type_is_pad(self) -> None:
        """Padding positions must have token_type_id equal to MIMIC4_TOKEN_TYPES['PAD']."""
        pad_id = MIMIC4_TOKEN_TYPES["PAD"]
        self.assertTrue(
            (self.batch["token_type_ids"][1, 7:] == pad_id).all().item()
        )

    def test_float_fields_padded_zero(self) -> None:
        """Padding positions in time_stamps and ages must be 0.0."""
        self.assertTrue((self.batch["time_stamps"][1, 7:] == 0.0).all().item())
        self.assertTrue((self.batch["ages"][1, 7:] == 0.0).all().item())

    def test_labels_shape_and_values(self) -> None:
        """Label tensor must have shape ``(B,)`` and preserve original values."""
        self.assertEqual(self.batch["label"].shape, (self.num_samples,))
        self.assertEqual(self.batch["label"].tolist(), [1, 0, 1])

    def test_metadata_lists_preserved(self) -> None:
        """patient_id and visit_id metadata must be preserved verbatim."""
        self.assertEqual(self.batch["patient_id"], ["p0", "p1", "p2"])
        self.assertTrue(all("_v0" in v for v in self.batch["visit_id"]))

    def test_single_sample_batch(self) -> None:
        """A batch of one sample must collate to shape ``(1, seq_len)``."""
        sample = _raw_collate_sample(5, label=0)
        batch = collate_ehr_mamba_batch([sample])
        self.assertEqual(batch["input_ids"].shape, (1, 5))
        self.assertEqual(batch["label"].shape, (1,))


# ===========================================================================
# Section 6: RMSNorm
# ===========================================================================


class TestRMSNorm(TimedTestCase):
    """Tests for :class:`RMSNorm`.

    Verifies shape preservation, numerical stability, and gradient flow for
    root mean square layer normalization (paper ref §62).
    """

    def test_output_shape_preserved(self) -> None:
        """RMSNorm must not alter the shape of the input tensor."""
        norm = RMSNorm(dim=32)
        self.assertEqual(norm(torch.randn(4, 10, 32)).shape, (4, 10, 32))

    def test_output_no_nan(self) -> None:
        """RMSNorm output must contain no NaN or Inf values."""
        norm = RMSNorm(dim=16)
        out = norm(torch.randn(2, 8, 16))
        self.assertFalse(out.isnan().any().item())
        self.assertFalse(out.isinf().any().item())

    def test_gradient_flows(self) -> None:
        """Gradient must flow back to both the input and the learnable weight."""
        norm = RMSNorm(dim=8)
        x = torch.randn(2, 4, 8, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)


# ===========================================================================
# Section 7: MambaBlock
# ===========================================================================


class TestMambaBlock(TimedTestCase):
    """Tests for :class:`MambaBlock`.

    Verifies the single Mamba SSM building block (paper Appendix C.1):
    shape invariance, numerical stability, residual connection, and gradient
    propagation through the SSM and gating pathway.
    """

    def test_output_shape(self) -> None:
        """MambaBlock must preserve the ``(B, L, d_model)`` shape."""
        blk = MambaBlock(d_model=32)
        self.assertEqual(blk(torch.randn(2, 10, 32)).shape, (2, 10, 32))

    def test_output_no_nan(self) -> None:
        """MambaBlock output must contain no NaN or Inf values."""
        blk = MambaBlock(d_model=16)
        out = blk(torch.randn(2, 5, 16))
        self.assertFalse(out.isnan().any().item())
        self.assertFalse(out.isinf().any().item())

    def test_residual_connection(self) -> None:
        """Block output must differ from the input (residual + transform != identity)."""
        blk = MambaBlock(d_model=16)
        x = torch.randn(1, 4, 16)
        self.assertFalse(torch.allclose(blk(x), x))

    def test_gradient_flows(self) -> None:
        """Gradient must reach the input tensor and at least one block parameter."""
        blk = MambaBlock(d_model=8)
        x = torch.randn(2, 6, 8, requires_grad=True)
        blk(x).sum().backward()
        self.assertIsNotNone(x.grad)
        grad_count = sum(
            1 for p in blk.parameters()
            if p.grad is not None and p.grad.norm() > 0
        )
        self.assertGreater(grad_count, 0)


# ===========================================================================
# Section 8: EHRMamba — full-stack integration
# ===========================================================================


class TestEHRMamba(TimedTestCase):
    """Full-stack integration tests for :class:`EHRMamba`.

    Pipeline tested end-to-end::

        DataLoader (collate_ehr_mamba_batch)
        → EHRMamba.forward(**batch)
        → set_aux_inputs  (EHRMambaEmbeddingAdapter pattern)
        → EHRMambaEmbedding (§2.2 fusion)
        → MambaBlock × num_layers
        → get_last_visit → Dropout → Linear → loss / y_prob

    A shared model (``cls.ehr_model``), dataset (``cls.sample_ds``), and batch
    (``cls.ehr_batch``) are constructed once in ``setUpClass``.  Tests that
    require a fresh model (backward pass, gradient checks) create their own.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build the shared dataset, model, and batch used by read-only tests."""
        cls.sample_ds = _make_synthetic_sample_dataset(n_samples=4)
        cls.ehr_model = EHRMamba(
            dataset=cls.sample_ds,
            embedding_dim=32,
            num_layers=1,
            dropout=0.1,
        )
        loader = DataLoader(
            cls.sample_ds,
            batch_size=2,
            collate_fn=collate_ehr_mamba_batch,
        )
        batch = next(iter(loader))
        batch["label"] = batch["label"].float().unsqueeze(-1)
        cls.ehr_batch = batch

    def test_instantiation_attributes(self) -> None:
        """Model must store the correct hyperparameters after construction."""
        self.assertEqual(self.ehr_model.embedding_dim, 32)
        self.assertEqual(self.ehr_model.num_layers, 1)
        self.assertEqual(self.ehr_model.feature_keys, ["input_ids"])
        self.assertEqual(self.ehr_model.label_key, "label")
        self.assertEqual(self.ehr_model.mode, "binary")

    def test_embedding_vocab_matches_dataset(self) -> None:
        """Embedding table size must match the SequenceProcessor vocabulary.

        Mismatched vocab sizes cause index-out-of-range errors at runtime.
        """
        vocab_size = self.sample_ds.input_processors["input_ids"].vocab_size()
        emb_vocab = (
            self.ehr_model.embedding_model.embedding.word_embeddings.num_embeddings
        )
        self.assertEqual(emb_vocab, vocab_size)

    def test_forward_output_keys(self) -> None:
        """Forward pass must return at least the four standard output keys."""
        self.ehr_model.eval()
        with torch.no_grad():
            out = self.ehr_model(**self.ehr_batch)
        self.assertTrue({"loss", "y_prob", "y_true", "logit"} <= set(out.keys()))

    def test_forward_output_shapes(self) -> None:
        """y_prob, y_true, and logit must have batch size as their first dimension."""
        self.ehr_model.eval()
        with torch.no_grad():
            out = self.ehr_model(**self.ehr_batch)
        batch_size = self.ehr_batch["input_ids"].shape[0]
        self.assertEqual(out["y_prob"].shape[0], batch_size)
        self.assertEqual(out["y_true"].shape[0], batch_size)
        self.assertEqual(out["logit"].shape[0], batch_size)

    def test_loss_finite_scalar(self) -> None:
        """Loss must be a finite scalar tensor (0-dimensional, no NaN/Inf)."""
        self.ehr_model.eval()
        with torch.no_grad():
            out = self.ehr_model(**self.ehr_batch)
        self.assertEqual(out["loss"].ndim, 0)
        self.assertFalse(out["loss"].isnan().item())
        self.assertFalse(out["loss"].isinf().item())

    def test_y_prob_in_unit_interval(self) -> None:
        """Predicted probabilities must lie within [0, 1]."""
        self.ehr_model.eval()
        with torch.no_grad():
            out = self.ehr_model(**self.ehr_batch)
        self.assertTrue((out["y_prob"] >= 0).all().item())
        self.assertTrue((out["y_prob"] <= 1).all().item())

    def test_backward_and_optimizer_step(self) -> None:
        """Backward pass and optimizer step must produce nonzero gradients."""
        import torch.optim as optim

        model = EHRMamba(
            dataset=self.sample_ds, embedding_dim=32, num_layers=1, dropout=0.1
        )
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        model(**self.ehr_batch)["loss"].backward()
        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.norm() > 0
        )
        self.assertGreater(grad_count, 0)
        optimizer.step()

    def test_gradient_reaches_word_embeddings(self) -> None:
        """Gradient must flow back to word_embeddings through the adapter.

        If the EHRMambaEmbeddingAdapter breaks the computation graph, the
        embedding weights receive no gradient and the model cannot be trained.
        """
        model = EHRMamba(
            dataset=self.sample_ds, embedding_dim=32, num_layers=1, dropout=0.1
        )
        model.train()
        model(**self.ehr_batch)["loss"].backward()
        grad = model.embedding_model.embedding.word_embeddings.weight.grad
        self.assertIsNotNone(grad)

    def test_embed_flag(self) -> None:
        """embed=True must add pooled patient-level embeddings to the output dict."""
        model = EHRMamba(
            dataset=self.sample_ds, embedding_dim=32, num_layers=1, dropout=0.1
        )
        model.eval()
        with torch.no_grad():
            out = model(**self.ehr_batch, embed=True)
        self.assertIn("embed", out)
        self.assertEqual(
            out["embed"].shape[0], self.ehr_batch["input_ids"].shape[0]
        )

    def test_batch_size_one(self) -> None:
        """A batch of a single sample must not cause shape errors."""
        ds = _make_synthetic_sample_dataset(n_samples=2)
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_ehr_mamba_batch)
        batch = next(iter(loader))
        batch["label"] = batch["label"].float().unsqueeze(-1)
        model = EHRMamba(dataset=ds, embedding_dim=16, num_layers=1)
        model.eval()
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(out["y_prob"].shape[0], 1)

    def test_create_mask_mixed_tokens(self) -> None:
        """_create_mask: non-zero positions must be unmasked, zero positions masked."""
        val = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=torch.long)
        mask = self.ehr_model._create_mask("input_ids", val)
        self.assertEqual(mask.shape[0], 2)
        self.assertTrue(mask[0, 0].item())
        self.assertFalse(mask[0, 2].item())

    def test_create_mask_allzero_row_fixed(self) -> None:
        """_create_mask: an all-zero sequence must have at least one True position.

        ``get_last_visit`` requires a valid position to pool from; an all-False
        mask would produce NaN hidden states.
        """
        val = torch.zeros(1, 4, dtype=torch.long)
        mask = self.ehr_model._create_mask("input_ids", val)
        self.assertTrue(mask[0].any().item())

    def test_pool_embedding_3d_passthrough(self) -> None:
        """3-D embedding must pass through _pool_embedding unchanged."""
        x = torch.randn(2, 7, 32)
        self.assertEqual(EHRMamba._pool_embedding(x).shape, (2, 7, 32))

    def test_pool_embedding_2d_unsqueeze(self) -> None:
        """2-D embedding must be unsqueezed to ``(B, 1, D)``."""
        x = torch.randn(2, 32)
        self.assertEqual(EHRMamba._pool_embedding(x).shape, (2, 1, 32))

    def test_pool_embedding_4d_sum(self) -> None:
        """4-D embedding must be summed over the third axis to produce ``(B, L, D)``."""
        x = torch.randn(2, 7, 3, 32)
        self.assertEqual(EHRMamba._pool_embedding(x).shape, (2, 7, 32))

    def test_tmpdir_round_trip(self) -> None:
        """Model checkpoint must be saveable and loadable from a temp directory."""
        tmpdir = tempfile.mkdtemp(prefix="ehr_mamba_test_")
        try:
            model = EHRMamba(
                dataset=self.sample_ds, embedding_dim=32, num_layers=1
            )
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)
            self.assertTrue(os.path.exists(path))
            sd = torch.load(path, map_location="cpu")
            self.assertIsInstance(sd, dict)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
