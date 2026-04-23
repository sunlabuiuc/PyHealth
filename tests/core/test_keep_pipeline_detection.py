"""Tests for KEEP pipeline's MIMIC-III vs MIMIC-IV schema auto-detection.

Covers the ``detect_mimic_schema`` and ``extract_diagnoses_for_schema``
helpers in ``run_pipeline.py`` that let the KEEP pipeline auto-detect
which MIMIC version is in use and route ICD-9 vs ICD-10 codes through
the correct mapping.

Uses small synthetic Polars DataFrames mimicking PyHealth's
``global_event_df`` output. No real MIMIC or Athena data required.
"""

import unittest

import polars as pl


ICD9_MAP = {
    "428.0": [300],    # Heart failure
    "401.9": [400],    # Hypertension
    "250.00": [600],   # Diabetes
}

ICD10_MAP = {
    "I50.9": [300],    # Heart failure
    "E11.9": [600],    # Diabetes
}


def _make_mimic3_diag_df():
    """MIMIC-III-style diagnoses DataFrame (ICD-9 only, no version column)."""
    return pl.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P2", "P2"],
        "diagnoses_icd/icd9_code": [
            "428.0", "428.0", "250.00", "401.9", "401.9",
        ],
    })


def _make_mimic4_diag_df():
    """MIMIC-IV-style diagnoses DataFrame (mixed ICD-9/ICD-10 + version)."""
    return pl.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P1", "P2", "P2"],
        "diagnoses_icd/icd_code": [
            "428.0", "428.0",
            "I50.9", "I50.9",
            "E11.9", "E11.9",
        ],
        "diagnoses_icd/icd_version": ["9", "9", "10", "10", "10", "10"],
    })


class TestDetectMimicSchema(unittest.TestCase):
    """Tests for detect_mimic_schema()."""

    def test_mimic3_columns(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        cols = ["patient_id", "diagnoses_icd/icd9_code"]
        self.assertEqual(detect_mimic_schema(cols), "mimic3")

    def test_mimic4_columns(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        cols = [
            "patient_id",
            "diagnoses_icd/icd_code",
            "diagnoses_icd/icd_version",
        ]
        self.assertEqual(detect_mimic_schema(cols), "mimic4")

    def test_mimic4_signal_is_version_column(self):
        """icd_version column is the definitive MIMIC-IV signal."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        cols = [
            "patient_id",
            "some_other_code_col",
            "diagnoses_icd/icd_version",
        ]
        self.assertEqual(detect_mimic_schema(cols), "mimic4")

    def test_empty_columns_defaults_to_mimic3(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        self.assertEqual(detect_mimic_schema([]), "mimic3")

    def test_extra_columns_do_not_affect_detection(self):
        """Detection is based solely on icd_version presence; extra cols ignored."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        cols = [
            "patient_id",
            "diagnoses_icd/icd9_code",
            "diagnoses_icd/seq_num",
            "some/other/column",
        ]
        self.assertEqual(detect_mimic_schema(cols), "mimic3")


class TestExtractDiagnosesForSchema(unittest.TestCase):
    """Tests for extract_diagnoses_for_schema()."""

    def setUp(self):
        self.icd9_map = dict(ICD9_MAP)
        self.icd10_map = dict(ICD10_MAP)
        self.mimic3_diag_df = _make_mimic3_diag_df()
        self.mimic4_diag_df = _make_mimic4_diag_df()

    def test_mimic3_path(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        result = extract_diagnoses_for_schema(
            self.mimic3_diag_df,
            icd9_map=self.icd9_map,
            icd10_map=None,
            min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})
        self.assertEqual(result["P2"], {400})

    def test_mimic4_path_routes_both_versions(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        result = extract_diagnoses_for_schema(
            self.mimic4_diag_df,
            icd9_map=self.icd9_map,
            icd10_map=self.icd10_map,
            min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})
        self.assertEqual(result["P2"], {600})

    def test_mimic4_requires_icd10_map(self):
        """Calling with MIMIC-IV data but no ICD-10 map should raise."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        with self.assertRaisesRegex(ValueError, "icd10_map is None"):
            extract_diagnoses_for_schema(
                self.mimic4_diag_df,
                icd9_map=self.icd9_map,
                icd10_map=None,
            )

    def test_mimic3_ignores_icd10_map(self):
        """Passing icd10_map with MIMIC-III data is harmless (just not used)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        result = extract_diagnoses_for_schema(
            self.mimic3_diag_df,
            icd9_map=self.icd9_map,
            icd10_map=self.icd10_map,
            min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})
        self.assertEqual(result["P2"], {400})

    def test_mimic4_cross_version_same_concept(self):
        """Patient with same concept coded in both ICD-9 and ICD-10 merges."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        diag_df = pl.DataFrame({
            "patient_id": ["P1", "P1"],
            "diagnoses_icd/icd_code": ["428.0", "I50.9"],
            "diagnoses_icd/icd_version": ["9", "10"],
        })
        result = extract_diagnoses_for_schema(
            diag_df,
            icd9_map=self.icd9_map,
            icd10_map=self.icd10_map,
            min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})

    def test_mimic4_min_occurrences_not_met(self):
        """ICD-9 once + ICD-10 once for different concepts = no qualifying codes."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        diag_df = pl.DataFrame({
            "patient_id": ["P1", "P1"],
            "diagnoses_icd/icd_code": ["428.0", "E11.9"],
            "diagnoses_icd/icd_version": ["9", "10"],
        })
        result = extract_diagnoses_for_schema(
            diag_df,
            icd9_map=self.icd9_map,
            icd10_map=self.icd10_map,
            min_occurrences=2,
        )
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
