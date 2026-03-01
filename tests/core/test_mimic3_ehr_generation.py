"""Tests for EHRGenerationMIMIC3 task and the sequence helper utilities.

All tests use a fully synthetic mock dataset so no real MIMIC-III files or
PyHealth's set_task() / litdata pipeline are required.
"""

import unittest

import pandas as pd

from pyhealth.tasks import EHRGenerationMIMIC3

# ── visit-delimiter helpers (mirrored from the example script) ─────────────────

VISIT_DELIM = "VISIT_DELIM"


def samples_to_sequences(samples: list) -> list:
    """Nested visit list → VISIT_DELIM-delimited text string per patient."""
    sequences = []
    for sample in samples:
        visit_texts = [" ".join(visit_codes) for visit_codes in sample["conditions"]]
        sequences.append(f" {VISIT_DELIM} ".join(visit_texts))
    return sequences


def sequences_to_dataframe(sequences: list) -> pd.DataFrame:
    """Text sequences → long-form (SUBJECT_ID, HADM_ID, ICD9_CODE) DataFrame."""
    rows = []
    for subj_idx, seq in enumerate(sequences):
        for hadm_idx, visit_str in enumerate(seq.strip().split(VISIT_DELIM)):
            for code in visit_str.strip().split():
                if code:
                    rows.append(
                        {"SUBJECT_ID": subj_idx, "HADM_ID": hadm_idx, "ICD9_CODE": code}
                    )
    return pd.DataFrame(rows)


# ── minimal mock objects that mimic PyHealth's Patient/Event interface ─────────


class _MockAdmission:
    """Lightweight stand-in for a MIMIC-III admission event."""

    def __init__(self, hadm_id: str) -> None:
        self.hadm_id = hadm_id


class _MockDiagnosis:
    """Lightweight stand-in for a diagnoses_icd event."""

    def __init__(self, hadm_id: str, icd9_code: str) -> None:
        self.hadm_id = hadm_id
        self.icd9_code = icd9_code


class _MockPatient:
    """Mimics BasePatient.get_events() for admissions and diagnoses_icd tables."""

    def __init__(self, patient_id: str, visits: list) -> None:
        """
        Args:
            patient_id: Synthetic subject_id string.
            visits: List of visits; each visit is a list of ICD-9 code strings.
                    Duplicates within a visit are intentional to test dedup logic.
        """
        self.patient_id = patient_id
        self._admissions = [
            _MockAdmission(hadm_id=str(100 + i)) for i in range(len(visits))
        ]
        self._diagnoses = []
        for admission, codes in zip(self._admissions, visits):
            for code in codes:
                self._diagnoses.append(_MockDiagnosis(admission.hadm_id, code))

    def get_events(self, event_type: str, filters=None):
        if event_type == "admissions":
            return list(self._admissions)
        if event_type == "diagnoses_icd":
            result = list(self._diagnoses)
            if filters:
                for field, op, value in filters:
                    if op == "==":
                        result = [e for e in result if getattr(e, field) == value]
            return result
        return []


# ── synthetic patient corpus ───────────────────────────────────────────────────

_PATIENTS = {
    # 3 visits, no duplicates
    "P001": [
        ["250.00", "401.9", "278.00"],
        ["250.00", "272.0"],
        ["428.0", "401.9", "285.9"],
    ],
    # 2 visits with intentional within-visit duplicates
    "P002": [
        ["410.01", "410.01", "412"],   # 410.01 duplicated intentionally
        ["414.01", "V45.81"],
    ],
    # 1 visit (used to test min_visits filtering)
    "P003": [
        ["486", "518.81"],
    ],
    # 4 visits with long codes (used for truncate_icd tests)
    "P004": [
        ["250.40", "250.00"],
        ["401.10", "401.90"],
        ["428.00"],
        ["272.00", "272.10"],
    ],
    # patient with some empty visits (should be silently skipped)
    "P005": [
        [],            # empty → skipped
        ["V15.82"],
        [],            # empty → skipped
        ["401.9"],
    ],
}

ALL_PATIENTS = [_MockPatient(pid, visits) for pid, visits in _PATIENTS.items()]


# ── test class ─────────────────────────────────────────────────────────────────


class TestEHRGenerationMIMIC3Task(unittest.TestCase):
    """Unit tests for EHRGenerationMIMIC3 using synthetic mock patients."""

    def _run_task(self, task, patients=None):
        """Helper: run task over a list of mock patients, flatten results."""
        if patients is None:
            patients = ALL_PATIENTS
        samples = []
        for p in patients:
            samples.extend(task(p))
        return samples

    # ── schema / init ──────────────────────────────────────────────────────────

    def test_task_name(self):
        self.assertEqual(EHRGenerationMIMIC3.task_name, "EHRGenerationMIMIC3")

    def test_input_schema(self):
        # nested_sequence required so PyHealth's processor handles variable-length visits
        self.assertEqual(EHRGenerationMIMIC3.input_schema, {"conditions": "nested_sequence"})

    def test_output_schema(self):
        self.assertEqual(EHRGenerationMIMIC3.output_schema, {})

    def test_default_init(self):
        task = EHRGenerationMIMIC3()
        self.assertEqual(task.min_visits, 1)
        self.assertFalse(task.truncate_icd)

    def test_custom_init(self):
        task = EHRGenerationMIMIC3(min_visits=3, truncate_icd=True)
        self.assertEqual(task.min_visits, 3)
        self.assertTrue(task.truncate_icd)

    # ── per-patient __call__ output ────────────────────────────────────────────

    def test_returns_one_sample_per_patient(self):
        """Each qualifying patient produces exactly one sample dict."""
        task = EHRGenerationMIMIC3()
        for patient in ALL_PATIENTS:
            result = task(patient)
            self.assertIn(len(result), (0, 1))

    def test_sample_keys_present(self):
        """Each sample must have patient_id and conditions keys."""
        task = EHRGenerationMIMIC3()
        samples = self._run_task(task)
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertIn("patient_id", sample)
            self.assertIn("conditions", sample)

    def test_patient_id_matches(self):
        """sample['patient_id'] must equal the originating patient id."""
        task = EHRGenerationMIMIC3()
        for patient in ALL_PATIENTS:
            for sample in task(patient):
                self.assertEqual(sample["patient_id"], patient.patient_id)

    def test_conditions_is_nested_list_of_strings(self):
        """conditions must be List[List[str]] with no empty inner lists."""
        task = EHRGenerationMIMIC3()
        samples = self._run_task(task)
        for sample in samples:
            conds = sample["conditions"]
            self.assertIsInstance(conds, list)
            self.assertGreater(len(conds), 0)
            for visit in conds:
                self.assertIsInstance(visit, list)
                self.assertGreater(len(visit), 0, "Empty visits must be dropped")
                for code in visit:
                    self.assertIsInstance(code, str)
                    self.assertGreater(len(code), 0)

    def test_empty_visits_skipped(self):
        """Admissions with no ICD-9 codes are silently skipped."""
        task = EHRGenerationMIMIC3()
        p005 = _MockPatient("P005", _PATIENTS["P005"])
        result = task(p005)
        self.assertEqual(len(result), 1)
        # 4 admissions, 2 empty → 2 valid visits
        self.assertEqual(len(result[0]["conditions"]), 2)

    def test_within_visit_deduplication(self):
        """Duplicate ICD-9 codes within a single visit are removed."""
        task = EHRGenerationMIMIC3()
        p002 = _MockPatient("P002", _PATIENTS["P002"])
        result = task(p002)
        self.assertEqual(len(result), 1)
        for visit in result[0]["conditions"]:
            self.assertEqual(len(visit), len(set(visit)),
                             f"Duplicate codes in visit: {visit}")

    def test_visit_order_preserved(self):
        """Visits appear in the same order they were supplied."""
        task = EHRGenerationMIMIC3()
        p001 = _MockPatient("P001", _PATIENTS["P001"])
        result = task(p001)
        self.assertIn("250.00", result[0]["conditions"][0])
        self.assertIn("428.0", result[0]["conditions"][2])

    def test_conditions_length_matches_nonempty_visits(self):
        """len(conditions) equals number of non-empty visits."""
        task = EHRGenerationMIMIC3()
        self.assertEqual(len(task(_MockPatient("P001", _PATIENTS["P001"]))[0]["conditions"]), 3)
        self.assertEqual(len(task(_MockPatient("P005", _PATIENTS["P005"]))[0]["conditions"]), 2)

    # ── min_visits filtering ───────────────────────────────────────────────────

    def test_min_visits_1_includes_single_visit_patient(self):
        task = EHRGenerationMIMIC3(min_visits=1)
        self.assertEqual(len(task(_MockPatient("P003", _PATIENTS["P003"]))), 1)

    def test_min_visits_2_excludes_single_visit_patient(self):
        task = EHRGenerationMIMIC3(min_visits=2)
        self.assertEqual(len(task(_MockPatient("P003", _PATIENTS["P003"]))), 0)

    def test_min_visits_2_keeps_multi_visit_patient(self):
        task = EHRGenerationMIMIC3(min_visits=2)
        self.assertEqual(len(task(_MockPatient("P001", _PATIENTS["P001"]))), 1)

    def test_min_visits_too_high_returns_empty_for_all(self):
        task = EHRGenerationMIMIC3(min_visits=10)
        self.assertEqual(self._run_task(task), [])

    # ── truncate_icd ───────────────────────────────────────────────────────────

    def test_truncate_icd_shortens_codes_to_3_chars(self):
        """All codes must be ≤ 3 characters when truncate_icd=True."""
        task = EHRGenerationMIMIC3(truncate_icd=True)
        for sample in self._run_task(task):
            for visit in sample["conditions"]:
                for code in visit:
                    self.assertLessEqual(len(code), 3,
                                         f"Code '{code}' exceeds 3 chars")

    def test_truncate_icd_false_preserves_full_codes(self):
        """Codes longer than 3 chars must survive when truncate_icd=False."""
        task = EHRGenerationMIMIC3(truncate_icd=False)
        result = task(_MockPatient("P004", _PATIENTS["P004"]))
        all_codes = [c for visit in result[0]["conditions"] for c in visit]
        self.assertTrue(any(len(c) > 3 for c in all_codes),
                        "Expected full-length codes like '250.40'")

    def test_truncate_icd_dedup_after_merge(self):
        """After truncation, merged codes are deduplicated within each visit."""
        # visit 0 of P004: "250.40" and "250.00" both → "250" (only one should survive)
        task = EHRGenerationMIMIC3(truncate_icd=True)
        result = task(_MockPatient("P004", _PATIENTS["P004"]))
        visit_0 = result[0]["conditions"][0]
        self.assertEqual(visit_0, ["250"])

    # ── edge cases ─────────────────────────────────────────────────────────────

    def test_all_empty_visits_returns_empty(self):
        task = EHRGenerationMIMIC3()
        self.assertEqual(task(_MockPatient("PEMPTY", [[], [], []])), [])

    def test_no_visits_returns_empty(self):
        task = EHRGenerationMIMIC3()
        self.assertEqual(task(_MockPatient("PNONE", [])), [])

    # ── sequence helper: samples_to_sequences ─────────────────────────────────

    def test_samples_to_sequences_one_string_per_sample(self):
        samples = self._run_task(EHRGenerationMIMIC3())
        seqs = samples_to_sequences(samples)
        self.assertEqual(len(seqs), len(samples))
        for seq in seqs:
            self.assertIsInstance(seq, str)
            self.assertGreater(len(seq.strip()), 0)

    def test_samples_to_sequences_delimiter_present_for_multi_visit(self):
        sample = EHRGenerationMIMIC3()(_MockPatient("P001", _PATIENTS["P001"]))[0]
        self.assertIn(VISIT_DELIM, samples_to_sequences([sample])[0])

    def test_samples_to_sequences_no_delimiter_for_single_visit(self):
        sample = EHRGenerationMIMIC3()(_MockPatient("P003", _PATIENTS["P003"]))[0]
        self.assertNotIn(VISIT_DELIM, samples_to_sequences([sample])[0])

    # ── sequence helper: sequences_to_dataframe ───────────────────────────────

    def test_sequences_to_dataframe_columns(self):
        samples = self._run_task(EHRGenerationMIMIC3())
        df = sequences_to_dataframe(samples_to_sequences(samples))
        for col in ("SUBJECT_ID", "HADM_ID", "ICD9_CODE"):
            self.assertIn(col, df.columns)

    def test_round_trip_all_codes_preserved(self):
        """Every code in the original samples must appear in the recovered DataFrame."""
        samples = self._run_task(EHRGenerationMIMIC3())
        df = sequences_to_dataframe(samples_to_sequences(samples))
        original = {c for s in samples for visit in s["conditions"] for c in visit}
        recovered = set(df["ICD9_CODE"].tolist())
        self.assertEqual(original, recovered)

    def test_round_trip_visit_count_per_patient(self):
        """The DataFrame must reconstruct the correct visit count per patient."""
        samples = self._run_task(EHRGenerationMIMIC3())
        df = sequences_to_dataframe(samples_to_sequences(samples))
        for idx, sample in enumerate(samples):
            syn_visits = df[df["SUBJECT_ID"] == idx]["HADM_ID"].nunique()
            self.assertEqual(syn_visits, len(sample["conditions"]),
                             f"Visit count mismatch at sample index {idx}")


if __name__ == "__main__":
    unittest.main()
