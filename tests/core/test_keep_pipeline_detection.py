"""Tests for KEEP pipeline's MIMIC-III vs MIMIC-IV schema auto-detection.

Covers the ``detect_mimic_schema`` and ``extract_diagnoses_for_schema``
helpers in ``run_pipeline.py`` that let the KEEP pipeline auto-detect
which MIMIC version is in use and route ICD-9 vs ICD-10 codes through
the correct mapping.

Uses small synthetic Polars DataFrames mimicking PyHealth's
``global_event_df`` output. No real MIMIC or Athena data required.
"""

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def icd9_map():
    """Minimal ICD-9 → SNOMED mapping (multi-target format)."""
    return {
        "428.0": [300],    # Heart failure
        "401.9": [400],    # Hypertension
        "250.00": [600],   # Diabetes
    }


@pytest.fixture
def icd10_map():
    """Minimal ICD-10 → SNOMED mapping (multi-target format)."""
    return {
        "I50.9": [300],    # Heart failure
        "E11.9": [600],    # Diabetes
    }


@pytest.fixture
def mimic3_diag_df():
    """MIMIC-III-style diagnoses DataFrame (ICD-9 only, no version column)."""
    return pl.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P2", "P2"],
        "diagnoses_icd/icd9_code": [
            "428.0", "428.0", "250.00", "401.9", "401.9",
        ],
    })


@pytest.fixture
def mimic4_diag_df():
    """MIMIC-IV-style diagnoses DataFrame (mixed ICD-9/ICD-10 + version)."""
    return pl.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P1", "P2", "P2"],
        "diagnoses_icd/icd_code": [
            "428.0", "428.0",          # ICD-9 heart failure
            "I50.9", "I50.9",          # ICD-10 heart failure
            "E11.9", "E11.9",          # ICD-10 diabetes
        ],
        "diagnoses_icd/icd_version": ["9", "9", "10", "10", "10", "10"],
    })


# ---------------------------------------------------------------------------
# detect_mimic_schema
# ---------------------------------------------------------------------------

class TestDetectMimicSchema:
    """Tests for detect_mimic_schema()."""

    def test_mimic3_columns(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        cols = ["patient_id", "diagnoses_icd/icd9_code"]
        assert detect_mimic_schema(cols) == "mimic3"

    def test_mimic4_columns(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        cols = [
            "patient_id",
            "diagnoses_icd/icd_code",
            "diagnoses_icd/icd_version",
        ]
        assert detect_mimic_schema(cols) == "mimic4"

    def test_mimic4_signal_is_version_column(self):
        """icd_version column is the definitive MIMIC-IV signal."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        # Even with a differently named code column, presence of icd_version
        # indicates MIMIC-IV schema
        cols = ["patient_id", "some_other_code_col", "diagnoses_icd/icd_version"]
        assert detect_mimic_schema(cols) == "mimic4"

    def test_empty_columns_defaults_to_mimic3(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            detect_mimic_schema,
        )
        # Empty list should default to mimic3 (safer fallback; will fail
        # downstream in extract_diagnoses_for_schema with clearer error)
        assert detect_mimic_schema([]) == "mimic3"

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
        assert detect_mimic_schema(cols) == "mimic3"


# ---------------------------------------------------------------------------
# extract_diagnoses_for_schema
# ---------------------------------------------------------------------------

class TestExtractDiagnosesForSchema:
    """Tests for extract_diagnoses_for_schema()."""

    def test_mimic3_path(self, mimic3_diag_df, icd9_map):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        result = extract_diagnoses_for_schema(
            mimic3_diag_df,
            icd9_map=icd9_map,
            icd10_map=None,
            min_occurrences=2,
        )
        # P1 has 428.0 twice (→ SNOMED 300) and 250.00 once (filtered out by min=2)
        # P2 has 401.9 twice (→ SNOMED 400)
        assert result["P1"] == {300}
        assert result["P2"] == {400}

    def test_mimic4_path_routes_both_versions(
        self, mimic4_diag_df, icd9_map, icd10_map,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        result = extract_diagnoses_for_schema(
            mimic4_diag_df,
            icd9_map=icd9_map,
            icd10_map=icd10_map,
            min_occurrences=2,
        )
        # P1 has 428.0 twice (ICD-9, SNOMED 300) AND I50.9 twice (ICD-10, SNOMED 300)
        #   → 4 total hits on SNOMED 300 → passes min=2
        # P2 has E11.9 twice (ICD-10, SNOMED 600) → passes min=2
        assert result["P1"] == {300}
        assert result["P2"] == {600}

    def test_mimic4_requires_icd10_map(self, mimic4_diag_df, icd9_map):
        """Calling with MIMIC-IV data but no ICD-10 map should raise."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        with pytest.raises(ValueError, match="icd10_map is None"):
            extract_diagnoses_for_schema(
                mimic4_diag_df,
                icd9_map=icd9_map,
                icd10_map=None,
            )

    def test_mimic3_ignores_icd10_map(
        self, mimic3_diag_df, icd9_map, icd10_map,
    ):
        """Passing icd10_map with MIMIC-III data is harmless (just not used)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        result = extract_diagnoses_for_schema(
            mimic3_diag_df,
            icd9_map=icd9_map,
            icd10_map=icd10_map,   # should be ignored on the MIMIC-III path
            min_occurrences=2,
        )
        # Same result as test_mimic3_path
        assert result["P1"] == {300}
        assert result["P2"] == {400}

    def test_mimic4_cross_version_same_concept(self, icd9_map, icd10_map):
        """Patient with same concept coded in both ICD-9 and ICD-10 merges."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        # P1: one ICD-9 heart failure + one ICD-10 heart failure (both → SNOMED 300)
        # Together = 2 occurrences of SNOMED 300, passes min=2
        diag_df = pl.DataFrame({
            "patient_id": ["P1", "P1"],
            "diagnoses_icd/icd_code": ["428.0", "I50.9"],
            "diagnoses_icd/icd_version": ["9", "10"],
        })
        result = extract_diagnoses_for_schema(
            diag_df,
            icd9_map=icd9_map,
            icd10_map=icd10_map,
            min_occurrences=2,
        )
        # Single patient with 2 cross-version occurrences of same SNOMED concept
        assert result["P1"] == {300}

    def test_mimic4_min_occurrences_not_met(self, icd9_map, icd10_map):
        """ICD-9 once + ICD-10 once for different concepts = no qualifying codes."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            extract_diagnoses_for_schema,
        )
        diag_df = pl.DataFrame({
            "patient_id": ["P1", "P1"],
            "diagnoses_icd/icd_code": ["428.0", "E11.9"],  # different SNOMEDs
            "diagnoses_icd/icd_version": ["9", "10"],
        })
        result = extract_diagnoses_for_schema(
            diag_df,
            icd9_map=icd9_map,
            icd10_map=icd10_map,
            min_occurrences=2,
        )
        # Each concept only appears once → filtered by min=2 → patient excluded
        assert result == {}
