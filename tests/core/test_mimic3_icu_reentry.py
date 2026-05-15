"""
Synthetic test fixtures and unit tests for ICUReEntryClassification.

Fixtures
--------
Provides a minimal set of 17 ICU stays across 9 patients with known
re-entry labels, designed to exercise all branches of the task logic:

    - Single-stay patients (no re-entry possible)
    - Multi-stay patients with qualifying re-entry (gap > 24h, <= 168h)
    - Multi-stay patients with gap > 7 days (not a re-entry)
    - Multi-stay patients with direct transfer (gap < 24h, same episode)
    - Transfer chain followed by qualifying re-entry

All feature tensors are synthetic float32 arrays of shape (24, 65),
populated with physiologically plausible ranges for key vitals so that
spot-checks are meaningful. Labels are deterministic and verified by hand
against the episode grouping logic in ICUReEntryClassification.

Unit tests
----------
TestICUReEntryClassification covers:
    - Task class schema attributes
    - Constructor defaults and validation (ValueError on bad args)
    - from_arrays() happy path and shape/length error cases
    - __call__() episode grouping and re-entry labeling for each patient
      scenario defined in the fixture set

Usage:
    >>> from tests.test_mimic3_icu_reentry import SYNTHETIC_STAYS, EXPECTED_LABELS
    >>> task = ICUReEntryClassification(feature_set="clinical")
    >>> samples = task.from_arrays(
    ...     features=np.stack([s["vitals_labs"] for s in SYNTHETIC_STAYS]),
    ...     labels=np.array([s["reentry_7day"] for s in SYNTHETIC_STAYS]),
    ...     stay_ids=[s["icustay_id"] for s in SYNTHETIC_STAYS],
    ...     patient_ids=[s["subject_id"] for s in SYNTHETIC_STAYS],
    ... )
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Feature tensor construction helpers
# ---------------------------------------------------------------------------

# Index positions in the 65-category feature vector for key vitals,
# used to populate synthetic tensors with plausible values.
# Matches the ordering in CLINICAL_CATEGORIES from mimic3_icu_reentry.py.
_FEATURE_INDICES = {
    "heart rate":             0,
    "systolic blood pressure": 1,
    "diastolic blood pressure": 2,
    "mean blood pressure":    3,
    "respiratory rate":       4,
    "temperature":            5,
    "oxygen saturation":      6,
}

N_FEATURES = 65
N_HOURS    = 24


def make_vitals_tensor(
    heart_rate: float = 80.0,
    systolic_bp: float = 120.0,
    diastolic_bp: float = 75.0,
    mean_bp: float = 90.0,
    resp_rate: float = 16.0,
    temperature: float = 37.0,
    spo2: float = 97.0,
    noise_scale: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Constructs a (24, 65) float32 feature tensor with plausible vital signs.

    Core vitals are populated with the provided base values plus small
    Gaussian noise across hours. All other features are set to 0.0
    (representing no observation, consistent with forward-fill imputation
    of missing values).

    Args:
        heart_rate: Base heart rate (bpm). Normal: 60-100.
        systolic_bp: Base systolic blood pressure (mmHg). Normal: 90-140.
        diastolic_bp: Base diastolic blood pressure (mmHg). Normal: 60-90.
        mean_bp: Base mean arterial pressure (mmHg). Normal: 70-100.
        resp_rate: Base respiratory rate (breaths/min). Normal: 12-20.
        temperature: Base temperature (Celsius). Normal: 36.5-37.5.
        spo2: Base oxygen saturation (%). Normal: 95-100.
        noise_scale: Standard deviation of per-hour Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (24, 65), dtype float32.
    """
    rng    = np.random.default_rng(seed)
    tensor = np.zeros((N_HOURS, N_FEATURES), dtype=np.float32)

    base_values = {
        "heart rate":              heart_rate,
        "systolic blood pressure": systolic_bp,
        "diastolic blood pressure": diastolic_bp,
        "mean blood pressure":     mean_bp,
        "respiratory rate":        resp_rate,
        "temperature":             temperature,
        "oxygen saturation":       spo2,
    }

    for feat_name, base_val in base_values.items():
        idx = _FEATURE_INDICES[feat_name]
        noise = rng.normal(0, noise_scale, size=N_HOURS).astype(np.float32)
        tensor[:, idx] = np.clip(base_val + noise, 0, None)

    return tensor


# ---------------------------------------------------------------------------
# Patient definitions
# ---------------------------------------------------------------------------
# Each patient dict defines one or more ICU stays with INTIME, OUTTIME,
# and a pre-computed vitals tensor. The expected re-entry label for the
# index stay (first stay per episode) is included for test assertion.
#
# Episode grouping rule: stays separated by < 24h are the same episode.
# Re-entry rule: next episode starts > 24h and <= 168h after episode_end.
#
# Patient roster:
#   P001 — single stay, no re-entry (label=0)
#   P002 — two stays, gap=72h -> re-entry (label=1)
#   P003 — two stays, gap=200h -> no re-entry, gap too long (label=0)
#   P004 — two stays, gap=6h -> direct transfer, one episode (label=0)
#   P005 — three stays: stays 1+2 are a transfer chain, stay 3 is
#           re-entry 96h after chain end (chain label=1, stay 3 is new ep)
#   P006 — single stay, no subsequent admission (label=0)
#   P007 — two stays, gap=168h -> re-entry, exactly at boundary (label=1)
#   P008 — two stays, gap=169h -> no re-entry, just over boundary (label=0)
#   P009 — three stays: all three are separate episodes with qualifying
#           re-entry between ep1->ep2 (label=1) and no re-entry ep2->ep3
#           because gap > 7 days (ep2 label=0)
#   P010 — two stays, gap=25h -> re-entry, just over transfer threshold
#           (label=1)
# ---------------------------------------------------------------------------

T0 = datetime(2100, 1, 1, 8, 0, 0)   # Arbitrary base time


def date_time(days: float = 0, hours: float = 0) -> datetime:
    """Returns _T0 offset by the given days and hours."""
    return T0 + timedelta(days=days, hours=hours)


# ── P001: Single stay, no re-entry ───────────────────────────────────────
P001_STAY1 = {
    "subject_id":  1001,
    "icustay_id":  100101,
    "intime":      date_time(0),
    "outtime":     date_time(3),           # 3-day stay
    "vitals_labs": make_vitals_tensor(seed=1),
    # No subsequent stay -> label 0
    "reentry_7day": 0,
}

# ── P002: Two stays, gap = 72h -> qualifying re-entry ────────────────────
P002_STAY1 = {
    "subject_id":  1002,
    "icustay_id":  100201,
    "intime":      date_time(0),
    "outtime":     date_time(2),           # discharged after 2 days
    "vitals_labs": make_vitals_tensor(heart_rate=95.0, seed=2),
    "reentry_7day": 1,               # stay2 starts 72h later -> re-entry
}
P002_STAY2 = {
    "subject_id":  1002,
    "icustay_id":  100202,
    "intime":      date_time(5),           # gap = 72h from stay1 outtime
    "outtime":     date_time(7),
    "vitals_labs": make_vitals_tensor(heart_rate=88.0, seed=3),
    "reentry_7day": 0,               # no subsequent stay
}

# ── P003: Two stays, gap = 200h -> too long, not a re-entry ──────────────
P003_STAY1 = {
    "subject_id":  1003,
    "icustay_id":  100301,
    "intime":      date_time(0),
    "outtime":     date_time(2),
    "vitals_labs": make_vitals_tensor(systolic_bp=145.0, seed=4),
    "reentry_7day": 0,               # gap 200h > 168h -> not re-entry
}
P003_STAY2 = {
    "subject_id":  1003,
    "icustay_id":  100302,
    "intime":      date_time(2, hours=200),  # gap = 200h
    "outtime":     date_time(2, hours=210),
    "vitals_labs": make_vitals_tensor(seed=5),
    "reentry_7day": 0,
}

# ── P004: Two stays, gap = 6h -> direct transfer, same episode ───────────
# Both stays belong to one episode. The index stay's label reflects
# whether the episode has a qualifying re-entry, which it does not.
P004_STAY1 = {
    "subject_id":  1004,
    "icustay_id":  100401,
    "intime":      date_time(0),
    "outtime":     date_time(1),
    "vitals_labs": make_vitals_tensor(spo2=91.0, seed=6),
    "reentry_7day": 0,               # stay2 is a transfer, no re-entry after
}
P004_STAY2 = {
    "subject_id":  1004,
    "icustay_id":  100402,
    "intime":      date_time(1, hours=6),  # gap = 6h -> same episode
    "outtime":     date_time(4),
    "vitals_labs": make_vitals_tensor(seed=7),
    "reentry_7day": 0,               # part of same episode as stay1
}

# ── P005: Transfer chain (stays 1+2) followed by re-entry (stay 3) ───────
# Stay1 + Stay2 form one episode (gap=10h). Stay3 arrives 96h after
# episode end -> qualifying re-entry for the episode.
P005_STAY1 = {
    "subject_id":  1005,
    "icustay_id":  100501,
    "intime":      date_time(0),
    "outtime":     date_time(1),
    "vitals_labs": make_vitals_tensor(resp_rate=22.0, seed=8),
    "reentry_7day": 1,               # episode (stays 1+2) -> re-entry via stay3
}
P005_STAY2 = {
    "subject_id":  1005,
    "icustay_id":  100502,
    "intime":      date_time(1, hours=10),  # gap = 10h -> transfer, same episode
    "outtime":     date_time(3),
    "vitals_labs": make_vitals_tensor(seed=9),
    "reentry_7day": 1,               # also part of episode with re-entry
}
P005_STAY3 = {
    "subject_id":  1005,
    "icustay_id":  100503,
    "intime":      date_time(3, hours=96),  # gap = 96h from episode end (stay2 outtime)
    "outtime":     date_time(5, hours=96),
    "vitals_labs": make_vitals_tensor(heart_rate=105.0, seed=10),
    "reentry_7day": 0,               # no subsequent stay
}

# ── P006: Single stay, no re-entry ───────────────────────────────────────
P006_STAY1 = {
    "subject_id":  1006,
    "icustay_id":  100601,
    "intime":      date_time(0),
    "outtime":     date_time(5),
    "vitals_labs": make_vitals_tensor(temperature=38.5, seed=11),
    "reentry_7day": 0,
}

# ── P007: Gap = 168h exactly -> re-entry (boundary, inclusive) ───────────
P007_STAY1 = {
    "subject_id":  1007,
    "icustay_id":  100701,
    "intime":      date_time(0),
    "outtime":     date_time(2),
    "vitals_labs": make_vitals_tensor(seed=12),
    "reentry_7day": 1,               # gap = exactly 168h -> qualifies
}
P007_STAY2 = {
    "subject_id":  1007,
    "icustay_id":  100702,
    "intime":      date_time(2, hours=168),  # gap = exactly 168h
    "outtime":     date_time(4, hours=168),
    "vitals_labs": make_vitals_tensor(seed=13),
    "reentry_7day": 0,
}

# ── P008: Gap = 169h -> no re-entry (just over boundary) ─────────────────
P008_STAY1 = {
    "subject_id":  1008,
    "icustay_id":  100801,
    "intime":      date_time(0),
    "outtime":     date_time(2),
    "vitals_labs": make_vitals_tensor(seed=14),
    "reentry_7day": 0,               # gap = 169h > 168h -> not re-entry
}
P008_STAY2 = {
    "subject_id":  1008,
    "icustay_id":  100802,
    "intime":      date_time(2, hours=169),
    "outtime":     date_time(4, hours=169),
    "vitals_labs": make_vitals_tensor(seed=15),
    "reentry_7day": 0,
}

# ── P009: Three stays, three separate episodes ───────────────────────────────
# ep1 -> ep2: gap = 72h -> qualifying re-entry (ep1 label=1)
# ep2 -> ep3: gap = 200h -> too long, not a re-entry (ep2 label=0)
P009_STAY1 = {
    "subject_id":  1009,
    "icustay_id":  100901,
    "intime":      date_time(0),
    "outtime":     date_time(2),
    "vitals_labs": make_vitals_tensor(seed=18),
    "reentry_7day": 1,               # gap to stay2 = 72h -> re-entry
}
P009_STAY2 = {
    "subject_id":  1009,
    "icustay_id":  100902,
    "intime":      date_time(5),           # gap = 72h from stay1 outtime
    "outtime":     date_time(7),
    "vitals_labs": make_vitals_tensor(seed=19),
    "reentry_7day": 0,               # gap to stay3 = 200h -> not re-entry
}
P009_STAY3 = {
    "subject_id":  1009,
    "icustay_id":  100903,
    "intime":      date_time(7, hours=200),  # gap = 200h from stay2 outtime
    "outtime":     date_time(7, hours=210),
    "vitals_labs": make_vitals_tensor(seed=20),
    "reentry_7day": 0,               # no subsequent stay
}

# ── P010: Gap = 25h -> re-entry (just over transfer threshold) ───────────
P010_STAY1 = {
    "subject_id":  1010,
    "icustay_id":  101001,
    "intime":      date_time(0),
    "outtime":     date_time(2),
    "vitals_labs": make_vitals_tensor(mean_bp=65.0, seed=16),
    "reentry_7day": 1,               # gap = 25h > 24h and <= 168h -> re-entry
}
P010_STAY2 = {
    "subject_id":  1010,
    "icustay_id":  101002,
    "intime":      date_time(2, hours=25),  # gap = 25h
    "outtime":     date_time(4, hours=25),
    "vitals_labs": make_vitals_tensor(seed=17),
    "reentry_7day": 0,
}

# ---------------------------------------------------------------------------
# Collected stays and expected labels
# ---------------------------------------------------------------------------

SYNTHETIC_STAYS: List[Dict] = [
    P001_STAY1,
    P002_STAY1, P002_STAY2,
    P003_STAY1, P003_STAY2,
    P004_STAY1, P004_STAY2,
    P005_STAY1, P005_STAY2, P005_STAY3,
    P006_STAY1,
    P007_STAY1, P007_STAY2,
    P008_STAY1, P008_STAY2,
    P009_STAY1, P009_STAY2, P009_STAY3,
    P010_STAY1, P010_STAY2,
]

# Ground-truth labels keyed by icustay_id for use in assertions
EXPECTED_LABELS: Dict[int, int] = {
    s["icustay_id"]: s["reentry_7day"]
    for s in SYNTHETIC_STAYS
}

# Ground-truth episode groupings for structural validation
EXPECTED_EPISODES: Dict[int, List[int]] = {
    # subject_id -> list of icustay_ids per episode (grouped)
    1001: [[100101]],
    1002: [[100201], [100202]],
    1003: [[100301], [100302]],
    1004: [[100401, 100402]],        # transfer chain -> one episode
    1005: [[100501, 100502], [100503]],  # chain then separate episode
    1006: [[100601]],
    1007: [[100701], [100702]],
    1008: [[100801], [100802]],
    1009: [[100901], [100902], [100903]],  # three episodes: ep1->ep2 qualifies, ep2->ep3 does not
    1010: [[101001], [101002]],
}

# ---------------------------------------------------------------------------
# Mock helpers for __call__() testing
# ---------------------------------------------------------------------------

class _MockStay:
    """Minimal ICU stay object accepted by ICUReEntryClassification.__call__."""
    def __init__(self, stay_dict: Dict):
        self.icustay_id  = stay_dict["icustay_id"]
        self.intime      = stay_dict["intime"]
        self.outtime     = stay_dict["outtime"]
        self.vitals_labs = stay_dict.get("vitals_labs")


class _MockPatient:
    """Minimal patient object accepted by ICUReEntryClassification.__call__."""
    def __init__(self, subject_id: int, stay_dicts: List[Dict]):
        self.patient_id = subject_id
        self._stays     = [_MockStay(s) for s in stay_dicts]

    def get_events(self, event_type: str) -> List[_MockStay]:  # noqa: ARG002
        return self._stays


# ---------------------------------------------------------------------------
# Shared test utilities
# ---------------------------------------------------------------------------

class _PassMessageMixin:
    """
    Mixin that prints a one-line success message for every passing test.

    Uses self._outcome.success, which CPython's TestCase.run() sets to True
    after setUp and the test body both complete without error, before tearDown
    is invoked.  Falls back silently if _outcome is unavailable (e.g. under
    alternative test runners).
    """

    def tearDown(self):
        super().tearDown()
        outcome = getattr(self, "_outcome", None)
        if outcome is not None and getattr(outcome, "success", False):
            print(f"  PASS  {self._testMethodName}")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestICUReEntryClassification(_PassMessageMixin, unittest.TestCase):

    # ── Imports ──────────────────────────────────────────────────────────────
    # Deferred to class level so import errors surface as test failures,
    # not module-load errors.
    @classmethod
    def setUpClass(cls):

        from pyhealth.tasks.mimic3_icu_reentry import ICUReEntryClassification
        from pyhealth.tasks.mimic3_clinical_aggregation import (
            CLINICAL_CATEGORIES,
            apply_clinical_aggregation,
        )
        cls.Task = ICUReEntryClassification
        cls.clinical_categories = CLINICAL_CATEGORIES
        cls.apply_clinical_aggregation = staticmethod(apply_clinical_aggregation)

    # ── Schema ───────────────────────────────────────────────────────────────

    def test_task_schema_attributes_present(self):
        self.assertIn("task_name",     vars(self.Task))
        self.assertIn("input_schema",  vars(self.Task))
        self.assertIn("output_schema", vars(self.Task))

    def test_task_schema_values(self):
        self.assertEqual("ICUReEntryClassification", self.Task.task_name)
        self.assertIn("vitals_labs",  self.Task.input_schema)
        self.assertEqual("tensor",    self.Task.input_schema["vitals_labs"])
        self.assertIn("reentry_7day", self.Task.output_schema)
        self.assertEqual("binary",    self.Task.output_schema["reentry_7day"])

    # ── Constructor defaults and validation ──────────────────────────────────

    def test_defaults(self):
        task = self.Task()
        self.assertEqual(task.feature_set,              "clinical")
        self.assertEqual(task.reentry_window_hours,     168)
        self.assertEqual(task.transfer_threshold_hours, 24)

    def test_feature_set_raw_instantiates(self):
        """feature_set='raw' is a valid value and must not raise."""
        task = self.Task(feature_set="raw")
        self.assertEqual(task.feature_set, "raw")

    def test_invalid_feature_set(self):
        with self.assertRaises(ValueError):
            self.Task(feature_set="unknown")

    def test_zero_reentry_window_raises(self):
        with self.assertRaises(ValueError):
            self.Task(reentry_window_hours=0)

    def test_negative_reentry_window_raises(self):
        with self.assertRaises(ValueError):
            self.Task(reentry_window_hours=-1)

    def test_zero_transfer_threshold_raises(self):
        with self.assertRaises(ValueError):
            self.Task(transfer_threshold_hours=0)

    def test_transfer_threshold_equals_reentry_window_raises(self):
        with self.assertRaises(ValueError):
            self.Task(reentry_window_hours=48, transfer_threshold_hours=48)

    def test_transfer_threshold_exceeds_reentry_window_raises(self):
        with self.assertRaises(ValueError):
            self.Task(reentry_window_hours=24, transfer_threshold_hours=48)

    # ── from_arrays() happy path ─────────────────────────────────────────────

    def _make_from_arrays_samples(self):
        task     = self.Task()
        features = np.stack([s["vitals_labs"] for s in SYNTHETIC_STAYS])
        labels   = np.array([s["reentry_7day"] for s in SYNTHETIC_STAYS])
        stay_ids = [s["icustay_id"] for s in SYNTHETIC_STAYS]
        pids     = [s["subject_id"]  for s in SYNTHETIC_STAYS]
        return task.from_arrays(features, labels, stay_ids, patient_ids=pids)

    def test_from_arrays_returns_correct_count(self):
        samples = self._make_from_arrays_samples()
        self.assertEqual(len(samples), len(SYNTHETIC_STAYS))

    def test_from_arrays_sample_schema(self):
        for sample in self._make_from_arrays_samples():
            self.assertIn("patient_id",   sample)
            self.assertIn("visit_id",     sample)
            self.assertIn("vitals_labs",  sample)
            self.assertIn("reentry_7day", sample)

    def test_from_arrays_ids_are_strings(self):
        for sample in self._make_from_arrays_samples():
            self.assertIsInstance(sample["patient_id"], str)
            self.assertIsInstance(sample["visit_id"],   str)

    def test_from_arrays_tensor_shape(self):
        for sample in self._make_from_arrays_samples():
            self.assertEqual(sample["vitals_labs"].shape, (N_HOURS, N_FEATURES))

    def test_from_arrays_tensor_dtype(self):
        """vitals_labs must be float32 — the dtype required by the LSTM model."""
        for sample in self._make_from_arrays_samples():
            self.assertEqual(sample["vitals_labs"].dtype, np.float32)

    def test_from_arrays_label_values_match_expected(self):
        for sample in self._make_from_arrays_samples():
            stay_id  = int(sample["visit_id"])
            expected = EXPECTED_LABELS[stay_id]
            self.assertEqual(
                sample["reentry_7day"], expected,
                msg=f"icustay_id={stay_id}: expected label {expected}, "
                    f"got {sample['reentry_7day']}"
            )

    def test_from_arrays_label_type_is_int(self):
        """reentry_7day must be a plain Python int, not numpy.int64."""
        for sample in self._make_from_arrays_samples():
            self.assertIsInstance(
                sample["reentry_7day"], int,
                msg=f"visit_id={sample['visit_id']}: reentry_7day is "
                    f"{type(sample['reentry_7day'])}, expected int"
            )

    def test_from_arrays_labels_are_binary(self):
        """reentry_7day must be exactly 0 or 1 — never another integer."""
        for sample in self._make_from_arrays_samples():
            self.assertIn(
                sample["reentry_7day"], (0, 1),
                msg=f"visit_id={sample['visit_id']}: reentry_7day="
                    f"{sample['reentry_7day']} is not a binary label"
            )

    def test_call_labels_are_binary(self):
        """__call__() must only produce labels of 0 or 1 across all patient scenarios."""
        scenarios = [
            (1001, [P001_STAY1]),
            (1002, [P002_STAY1, P002_STAY2]),
            (1004, [P004_STAY1, P004_STAY2]),
            (1005, [P005_STAY1, P005_STAY2, P005_STAY3]),
            (1009, [P009_STAY1, P009_STAY2, P009_STAY3]),
        ]
        for pid, stay_dicts in scenarios:
            patient = _MockPatient(pid, stay_dicts)
            for sample in self.Task()(patient):
                self.assertIn(
                    sample["reentry_7day"], (0, 1),
                    msg=f"patient={pid}, visit_id={sample['visit_id']}: "
                        f"reentry_7day={sample['reentry_7day']} is not binary"
                )

    def test_from_arrays_patient_ids_default_to_stay_ids(self):
        """When patient_ids is None, visit_id and patient_id should match."""
        task     = self.Task()
        features = np.zeros((3, 24, 65), dtype="float32")
        labels   = np.array([0, 1, 0])
        stay_ids = [200001, 200002, 200003]
        samples  = task.from_arrays(features, labels, stay_ids)
        for s in samples:
            self.assertEqual(s["patient_id"], s["visit_id"])

    def test_from_arrays_vital_values_preserved(self):
        """Values in the tensor must be passed through unchanged."""
        task     = self.Task()
        features = np.arange(24 * 65, dtype="float32").reshape(1, 24, 65)
        samples  = task.from_arrays(features, np.array([0]), [999])
        np.testing.assert_array_equal(samples[0]["vitals_labs"], features[0])

    # ── from_arrays() error cases ────────────────────────────────────────────

    def test_from_arrays_features_row_mismatch_raises(self):
        task = self.Task()
        with self.assertRaises(ValueError):
            task.from_arrays(
                np.zeros((3, 24, 65), dtype="float32"),
                np.array([0, 1]),       # 2 labels for 3 feature rows
                [1, 2],
            )

    def test_from_arrays_labels_length_mismatch_raises(self):
        task = self.Task()
        with self.assertRaises(ValueError):
            task.from_arrays(
                np.zeros((2, 24, 65), dtype="float32"),
                np.array([0, 1, 0]),    # 3 labels for 2 feature rows
                [1, 2],
            )

    def test_from_arrays_patient_ids_length_mismatch_raises(self):
        task = self.Task()
        with self.assertRaises(ValueError):
            task.from_arrays(
                np.zeros((2, 24, 65), dtype="float32"),
                np.array([0, 1]),
                [1, 2],
                patient_ids=[100],      # only 1 for 2 stays
            )

    def test_from_arrays_wrong_ndim_raises(self):
        task = self.Task()
        with self.assertRaises(ValueError):
            task.from_arrays(
                np.zeros((2, 65), dtype="float32"),  # 2-D instead of 3-D
                np.array([0, 1]),
                [1, 2],
            )

    def test_from_arrays_wrong_time_steps_raises(self):
        task = self.Task()
        with self.assertRaises(ValueError):
            task.from_arrays(
                np.zeros((2, 12, 65), dtype="float32"),  # 12h not 24h
                np.array([0, 1]),
                [1, 2],
            )

    # ── __call__() – empty patient ───────────────────────────────────────────

    def test_call_empty_patient_returns_empty(self):
        patient = _MockPatient(9999, [])
        samples = self.Task()(patient)
        self.assertEqual(samples, [])

    def test_call_stay_without_vitals_is_skipped(self):
        stay = {**P001_STAY1, "vitals_labs": None}
        patient = _MockPatient(1001, [stay])
        samples = self.Task()(patient)
        self.assertEqual(samples, [])

    # ── __call__() sample schema ─────────────────────────────────────────────

    def test_call_sample_schema(self):
        patient = _MockPatient(1002, [P002_STAY1, P002_STAY2])
        for sample in self.Task()(patient):
            self.assertIn("patient_id",   sample)
            self.assertIn("visit_id",     sample)
            self.assertIn("vitals_labs",  sample)
            self.assertIn("reentry_7day", sample)
            self.assertIsInstance(sample["patient_id"], str)
            self.assertIsInstance(sample["visit_id"],   str)

    # ── __call__() – single-stay patients (no re-entry possible) ─────────────

    def test_call_p001_single_stay_label_0(self):
        """P001: one stay, no re-entry."""
        patient = _MockPatient(1001, [P001_STAY1])
        samples = self.Task()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["reentry_7day"], 0)
        self.assertEqual(samples[0]["visit_id"], str(P001_STAY1["icustay_id"]))

    def test_call_p006_single_stay_label_0(self):
        """P006: one stay, no re-entry."""
        patient = _MockPatient(1006, [P006_STAY1])
        samples = self.Task()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["reentry_7day"], 0)

    # ── __call__() – qualifying re-entry ─────────────────────────────────────

    def test_call_p002_gap_72h_label_1(self):
        """P002: gap = 72h between episodes -> re-entry."""
        patient = _MockPatient(1002, [P002_STAY1, P002_STAY2])
        samples = self.Task()(patient)
        # Two episodes -> two samples
        self.assertEqual(len(samples), 2)
        ep1 = next(s for s in samples if s["visit_id"] == str(P002_STAY1["icustay_id"]))
        ep2 = next(s for s in samples if s["visit_id"] == str(P002_STAY2["icustay_id"]))
        self.assertEqual(ep1["reentry_7day"], 1)
        self.assertEqual(ep2["reentry_7day"], 0)

    def test_call_p007_boundary_inclusive_168h_label_1(self):
        """P007: gap = exactly 168h -> qualifies as re-entry (boundary inclusive)."""
        patient = _MockPatient(1007, [P007_STAY1, P007_STAY2])
        samples = self.Task()(patient)
        self.assertEqual(len(samples), 2)
        ep1 = next(s for s in samples if s["visit_id"] == str(P007_STAY1["icustay_id"]))
        self.assertEqual(ep1["reentry_7day"], 1)

    def test_call_p010_gap_25h_label_1(self):
        """P010: gap = 25h (just over transfer threshold) -> re-entry."""
        patient = _MockPatient(1010, [P010_STAY1, P010_STAY2])
        samples = self.Task()(patient)
        self.assertEqual(len(samples), 2)
        ep1 = next(s for s in samples if s["visit_id"] == str(P010_STAY1["icustay_id"]))
        self.assertEqual(ep1["reentry_7day"], 1)

    # ── __call__() – non-qualifying gaps ─────────────────────────────────────

    def test_call_p003_gap_too_long_label_0(self):
        """P003: gap = 200h > 168h -> not a re-entry."""
        patient = _MockPatient(1003, [P003_STAY1, P003_STAY2])
        samples = self.Task()(patient)
        self.assertEqual(len(samples), 2)
        ep1 = next(s for s in samples if s["visit_id"] == str(P003_STAY1["icustay_id"]))
        self.assertEqual(ep1["reentry_7day"], 0)

    def test_call_p008_boundary_exclusive_169h_label_0(self):
        """P008: gap = 169h (just over boundary) -> not a re-entry."""
        patient = _MockPatient(1008, [P008_STAY1, P008_STAY2])
        samples = self.Task()(patient)
        self.assertEqual(len(samples), 2)
        ep1 = next(s for s in samples if s["visit_id"] == str(P008_STAY1["icustay_id"]))
        self.assertEqual(ep1["reentry_7day"], 0)

    # ── __call__() – episode grouping ────────────────────────────────────────

    def test_call_p004_direct_transfer_one_episode(self):
        """P004: gap = 6h -> direct transfer; both stays become one episode."""
        patient = _MockPatient(1004, [P004_STAY1, P004_STAY2])
        samples = self.Task()(patient)
        # One episode -> one sample, indexed by STAY1
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["visit_id"],     str(P004_STAY1["icustay_id"]))
        self.assertEqual(samples[0]["reentry_7day"], 0)

    def test_call_p005_transfer_chain_then_reentry(self):
        """P005: stays 1+2 form one episode (gap=10h); stay3 is re-entry 96h later."""
        patient = _MockPatient(1005, [P005_STAY1, P005_STAY2, P005_STAY3])
        samples = self.Task()(patient)
        # Two episodes: [STAY1, STAY2] and [STAY3]
        self.assertEqual(len(samples), 2)
        ep1 = next(s for s in samples if s["visit_id"] == str(P005_STAY1["icustay_id"]))
        ep2 = next(s for s in samples if s["visit_id"] == str(P005_STAY3["icustay_id"]))
        # Episode 1 re-enters via stay3 (gap from stay2.outtime=3d to stay3.intime=3d+96h)
        self.assertEqual(ep1["reentry_7day"], 1)
        self.assertEqual(ep2["reentry_7day"], 0)

    def test_call_p004_transfer_stay2_not_an_index_stay(self):
        """STAY2 of a transfer chain must not appear as a separate sample."""
        patient  = _MockPatient(1004, [P004_STAY1, P004_STAY2])
        visit_ids = [s["visit_id"] for s in self.Task()(patient)]
        self.assertNotIn(str(P004_STAY2["icustay_id"]), visit_ids)

    def test_call_p009_three_episodes_mixed_labels(self):
        """P009: three stays, three episodes. ep1->ep2 gap=72h (label=1); ep2->ep3 gap=200h (label=0)."""
        patient = _MockPatient(1009, [P009_STAY1, P009_STAY2, P009_STAY3])
        samples = self.Task()(patient)
        # Three separate episodes -> three samples
        self.assertEqual(len(samples), 3)
        ep1 = next(s for s in samples if s["visit_id"] == str(P009_STAY1["icustay_id"]))
        ep2 = next(s for s in samples if s["visit_id"] == str(P009_STAY2["icustay_id"]))
        ep3 = next(s for s in samples if s["visit_id"] == str(P009_STAY3["icustay_id"]))
        self.assertEqual(ep1["reentry_7day"], 1)   # 72h gap -> qualifies
        self.assertEqual(ep2["reentry_7day"], 0)   # 200h gap -> too long
        self.assertEqual(ep3["reentry_7day"], 0)   # no subsequent episode

    def test_call_unsorted_stays_produce_correct_labels(self):
        """__call__() must sort stays by intime internally; reversed input must match forward input."""
        forward = _MockPatient(1002, [P002_STAY1, P002_STAY2])
        reversed_ = _MockPatient(1002, [P002_STAY2, P002_STAY1])
        samples_forward  = self.Task()(forward)
        samples_reversed = self.Task()(reversed_)
        labels_forward  = {s["visit_id"]: s["reentry_7day"] for s in samples_forward}
        labels_reversed = {s["visit_id"]: s["reentry_7day"] for s in samples_reversed}
        self.assertEqual(labels_forward, labels_reversed)

    # ── Clinical feature index spot-check ────────────────────────────────────

    def test_clinical_categories_count(self):
        """CLINICAL_CATEGORIES must export exactly 65 entries."""
        self.assertEqual(len(self.clinical_categories), 65)

    def test_feature_index_heart_rate_nonzero(self):
        """Heart-rate column (index 0) must be non-zero in tensors built by make_vitals_tensor."""
        tensor = make_vitals_tensor(heart_rate=80.0, seed=42)
        self.assertTrue(np.all(tensor[:, _FEATURE_INDICES["heart rate"]] > 0))

    def test_make_vitals_tensor_shape(self):
        self.assertEqual(make_vitals_tensor().shape, (N_HOURS, N_FEATURES))

    def test_make_vitals_tensor_dtype(self):
        self.assertEqual(make_vitals_tensor().dtype, np.float32)

    # ── End-to-end sample output ─────────────────────────────────────────────

    def test_end_to_end_from_arrays_sample_output(self):
        """
        Complete input → output trace for from_arrays().

        Verifies that a single known feature tensor and label, fed through
        from_arrays(), produces a sample dict whose every field has the
        exact expected value and type — exercising data loading, integrity,
        and output format in one narrative test.
        """
        task    = self.Task(feature_set="clinical")
        feature = np.arange(N_HOURS * N_FEATURES, dtype="float32").reshape(
            1, N_HOURS, N_FEATURES
        )
        samples = task.from_arrays(
            features    = feature,
            labels      = np.array([1]),
            stay_ids    = [999001],
            patient_ids = [888001],
        )

        self.assertEqual(len(samples), 1)
        s = samples[0]

        # All four required keys are present and nothing extra
        self.assertEqual(set(s.keys()), {"patient_id", "visit_id", "vitals_labs", "reentry_7day"})

        # IDs are strings with correct values
        self.assertEqual(s["patient_id"], "888001")
        self.assertEqual(s["visit_id"],   "999001")

        # Feature tensor is passed through unchanged, with the right shape and dtype
        self.assertEqual(s["vitals_labs"].shape, (N_HOURS, N_FEATURES))
        self.assertEqual(s["vitals_labs"].dtype, np.float32)
        np.testing.assert_array_equal(s["vitals_labs"], feature[0])

        # Label is correct value, type, and in the binary range
        self.assertEqual(s["reentry_7day"], 1)
        self.assertIsInstance(s["reentry_7day"], int)
        self.assertIn(s["reentry_7day"], (0, 1))

    def test_end_to_end_call_sample_output(self):
        """
        Complete input → output trace for __call__().

        Drives a two-stay patient (P002: gap=72h, qualifying re-entry) through
        the full __call__() pipeline and verifies every field of the episode-1
        sample: patient_id, visit_id, vitals_labs, and the re-entry label.
        """
        task    = self.Task(feature_set="clinical")
        patient = _MockPatient(1002, [P002_STAY1, P002_STAY2])
        samples = task(patient)

        self.assertEqual(len(samples), 2)

        ep1 = next(s for s in samples if s["visit_id"] == str(P002_STAY1["icustay_id"]))

        # All required keys present
        self.assertEqual(set(ep1.keys()), {"patient_id", "visit_id", "vitals_labs", "reentry_7day"})

        # Patient and visit IDs are strings
        self.assertIsInstance(ep1["patient_id"], str)
        self.assertIsInstance(ep1["visit_id"],   str)
        self.assertEqual(ep1["patient_id"], "1002")
        self.assertEqual(ep1["visit_id"],   str(P002_STAY1["icustay_id"]))

        # vitals_labs is the index stay's tensor, shape and dtype intact
        self.assertEqual(ep1["vitals_labs"].shape, (N_HOURS, N_FEATURES))
        self.assertEqual(ep1["vitals_labs"].dtype, np.float32)
        np.testing.assert_array_equal(ep1["vitals_labs"], P002_STAY1["vitals_labs"])

        # Label: 72h gap -> re-entry
        self.assertEqual(ep1["reentry_7day"], 1)
        self.assertIsInstance(ep1["reentry_7day"], int)


# ---------------------------------------------------------------------------
# Helpers for apply_clinical_aggregation() tests
# ---------------------------------------------------------------------------

def _make_mock_hourly_data(
    stay_ids: List[int] = None,
    n_hours: int = 24,
    col_names: List[str] = None,
    value: float = 80.0,
) -> "object":
    """
    Builds a minimal MultiIndex DataFrame matching the MIMIC_Extract
    all_hourly_data.h5 'vitals_labs' structure.

    Column MultiIndex: (LEVEL2, Aggregation Function) with 'mean' as the
    aggregation level.  Row MultiIndex: (subject_id, hadm_id, icustay_id,
    hours_in).  Requires pandas; the caller must skip the test if pandas is
    not available.

    Args:
        stay_ids: ICU stay IDs. Defaults to [300001].
        n_hours: Number of hourly rows per stay. Defaults to 24.
        col_names: LEVEL2 column names to include. Defaults to
            ['heart rate', 'systolic blood pressure'].
        value: Constant value to fill all cells with. Defaults to 80.0.

    Returns:
        pd.DataFrame with MultiIndex rows and columns.
    """
    import pandas as pd

    if stay_ids is None:
        stay_ids = [300001]
    if col_names is None:
        col_names = ["heart rate", "systolic blood pressure"]

    rows = [
        (9001, 8001, stay_id, h)
        for stay_id in stay_ids
        for h in range(n_hours)
    ]
    index = pd.MultiIndex.from_tuples(
        rows,
        names=["subject_id", "hadm_id", "icustay_id", "hours_in"],
    )
    columns = pd.MultiIndex.from_tuples(
        [(col, "mean") for col in col_names],
        names=["LEVEL2", "Aggregation Function"],
    )
    data = np.full((len(rows), len(col_names)), value, dtype=np.float32)
    return pd.DataFrame(data, index=index, columns=columns)


class TestApplyClinicalAggregation(_PassMessageMixin, unittest.TestCase):
    """Tests for apply_clinical_aggregation() in mimic3_clinical_aggregation.py."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.tasks.mimic3_clinical_aggregation import (
            apply_clinical_aggregation,
            CLINICAL_CATEGORIES,
        )
        cls.fn = staticmethod(apply_clinical_aggregation)
        cls.clinical_categories = CLINICAL_CATEGORIES

        import importlib.util
        cls.pandas_available = importlib.util.find_spec("pandas") is not None

    def setUp(self):
        if not self.pandas_available:
            self.skipTest("pandas not installed — skipping apply_clinical_aggregation tests")

    # ── Output shape and types ────────────────────────────────────────────────

    def test_output_array_shape(self):
        """Array shape must be (n_stays, 24, 65)."""
        df = _make_mock_hourly_data(stay_ids=[300001, 300002])
        arr, _, _ = self.fn(df)
        self.assertEqual(arr.shape, (2, 24, 65))

    def test_output_category_names_match_clinical_categories(self):
        """Returned category_names must equal CLINICAL_CATEGORIES exactly."""
        df = _make_mock_hourly_data()
        _, cats, _ = self.fn(df)
        self.assertEqual(cats, self.clinical_categories)

    def test_output_stay_ids_match_input(self):
        """Returned stay_ids must match the icustay_ids in the input index."""
        input_stay_ids = [300001, 300002, 300003]
        df = _make_mock_hourly_data(stay_ids=input_stay_ids)
        _, _, stay_ids = self.fn(df)
        self.assertEqual(sorted(stay_ids), sorted(input_stay_ids))

    def test_output_dtype_is_float32(self):
        df = _make_mock_hourly_data()
        arr, _, _ = self.fn(df)
        self.assertEqual(arr.dtype, np.float32)

    # ── Column mapping ────────────────────────────────────────────────────────

    def test_present_column_produces_nonnan_values(self):
        """A column that exists in the input must produce non-NaN output values."""
        df = _make_mock_hourly_data(col_names=["heart rate"], value=75.0)
        arr, cats, _ = self.fn(df)
        hr_idx = cats.index("heart rate")
        self.assertFalse(
            np.all(np.isnan(arr[:, :, hr_idx])),
            msg="heart rate column should be non-NaN when present in input",
        )

    def test_absent_column_produces_nan_values(self):
        """A category whose LEVEL2 columns are all absent must produce NaN values."""
        # Only supply 'heart rate'; all other categories are absent.
        df = _make_mock_hourly_data(col_names=["heart rate"])
        arr, cats, _ = self.fn(df)
        # "systolic blood pressure" (index 1) should be all NaN
        sbp_idx = cats.index("systolic blood pressure")
        self.assertTrue(
            np.all(np.isnan(arr[:, :, sbp_idx])),
            msg="systolic blood pressure should be NaN when absent from input",
        )

    def test_multi_source_column_averaged(self):
        """A category mapped to multiple LEVEL2 columns must average their values."""
        # "partial pressure of carbon dioxide" maps to three LEVEL2 names.
        # Supply two of them with values 60.0 and 100.0; the output must be ~80.0.
        col_a = "partial pressure of carbon dioxide"
        col_b = "co2 (etco2, pco2, etc.)"
        df = _make_mock_hourly_data(col_names=[col_a, col_b], value=0.0)
        # Overwrite column values
        df[(col_a, "mean")] = 60.0
        df[(col_b, "mean")] = 100.0

        arr, cats, _ = self.fn(df)
        pco2_idx = cats.index("partial pressure of carbon dioxide")
        expected_mean = 80.0
        np.testing.assert_allclose(
            arr[:, :, pco2_idx],
            expected_mean,
            rtol=1e-4,
            err_msg="Multi-source category should average its source columns",
        )

    # ── Flat (non-MultiIndex) columns ─────────────────────────────────────────

    def test_flat_columns_accepted(self):
        """DataFrame with plain (non-MultiIndex) columns must work without error."""
        # Build the mock then flatten the column MultiIndex.
        df_multi = _make_mock_hourly_data(col_names=["heart rate"])
        df_flat = df_multi.copy()
        df_flat.columns = [col for col, _ in df_multi.columns]
        arr, _, stay_ids = self.fn(df_flat)
        self.assertEqual(arr.shape[2], 65)
        self.assertEqual(len(stay_ids), 1)

    # ── Hour indexing ─────────────────────────────────────────────────────────

    def test_values_placed_at_correct_hour(self):
        """Values for hours_in=0..23 must land in the correct time-step slice."""
        df = _make_mock_hourly_data(col_names=["heart rate"], value=0.0)
        # Set a distinct sentinel value only at hour 5.
        mask = df.index.get_level_values("hours_in") == 5
        df.loc[mask, ("heart rate", "mean")] = 999.0

        arr, cats, _ = self.fn(df)
        hr_idx = cats.index("heart rate")
        self.assertAlmostEqual(float(arr[0, 5, hr_idx]), 999.0, places=1)

    def test_complete_pipeline_sample_output(self):
        """
        Complete input → output trace for apply_clinical_aggregation().

        Builds a single-stay mock DataFrame with one known column ('heart rate'
        = 75.0 at every hour), runs it through the function, and verifies every
        aspect of the output: shape, dtype, category order, correct value at the
        expected feature index, and NaN for an absent feature.
        """
        df = _make_mock_hourly_data(
            stay_ids  = [400001],
            n_hours   = 24,
            col_names = ["heart rate"],
            value     = 75.0,
        )
        arr, cats, stay_ids = self.fn(df, n_hours=24)

        # Shape: (1 stay, 24 hours, 65 categories)
        self.assertEqual(arr.shape, (1, 24, 65))

        # Dtype is float32
        self.assertEqual(arr.dtype, np.float32)

        # Category list matches the authoritative CLINICAL_CATEGORIES
        self.assertEqual(cats, self.clinical_categories)

        # Stay IDs round-trip
        self.assertEqual(stay_ids, [400001])

        # heart rate (index 0) should be 75.0 at every hour
        hr_idx = cats.index("heart rate")
        np.testing.assert_allclose(
            arr[0, :, hr_idx], 75.0, rtol=1e-5,
            err_msg="heart rate column should be 75.0 at all 24 hours",
        )

        # An absent category (e.g., sodium, index >0) should be NaN
        sodium_idx = cats.index("sodium")
        self.assertTrue(
            np.all(np.isnan(arr[0, :, sodium_idx])),
            msg="sodium should be NaN because it was not in the input DataFrame",
        )

    def test_out_of_range_hours_ignored(self):
        """Rows with hours_in >= n_hours must not raise and must not corrupt the array."""
        import pandas as pd

        df = _make_mock_hourly_data(n_hours=24, col_names=["heart rate"], value=1.0)
        # Inject a row with hours_in = 99 (out of range).
        extra_index = pd.MultiIndex.from_tuples(
            [(9001, 8001, 300001, 99)],
            names=["subject_id", "hadm_id", "icustay_id", "hours_in"],
        )
        extra_cols = pd.MultiIndex.from_tuples(
            [("heart rate", "mean")],
            names=["LEVEL2", "Aggregation Function"],
        )
        extra_row = pd.DataFrame([[999.0]], index=extra_index, columns=extra_cols)
        df_extended = pd.concat([df, extra_row])

        arr, cats, _ = self.fn(df_extended)
        hr_idx = cats.index("heart rate")
        # No time step should contain 999.0
        self.assertFalse(
            np.any(arr[:, :, hr_idx] == 999.0),
            msg="Out-of-range hours_in=99 should be silently ignored",
        )


if __name__ == "__main__":
    unittest.main()
