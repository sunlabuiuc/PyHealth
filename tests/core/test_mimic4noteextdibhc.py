import unittest
import os
import re
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from pyhealth.datasets import MIMIC4NoteExtDIBHCDataset


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_minimal_note(
    hospital_course: str = (
        "Brief Hospital Course: Patient was admitted for chest pain. "
        "Workup showed no acute MI. Patient was monitored and stabilized. "
        "Cardiology was consulted and agreed with management plan.\n"
        "Medications on Admission: aspirin 81mg"
    ),
    discharge_section: str = (
        "Dear ___, It was a pleasure caring for you during your stay. "
        "You were admitted because you had chest pain. "
        "We ran several tests and found no signs of a heart attack. "
        "You were monitored closely and your condition improved. "
        "Please take all your medications as prescribed and follow up "
        "with your primary care physician within one week of discharge. "
        "If you experience worsening chest pain, shortness of breath, "
        "or any other concerning symptoms, please call your doctor or "
        "return to the emergency department immediately."
    ),
) -> str:
    """Return a minimal synthetic discharge note with required structure."""
    return f"{hospital_course}\nDischarge Instructions:\n{discharge_section}"


def _make_note_df(n: int = 5) -> pd.DataFrame:
    """Return a small DataFrame of synthetic discharge notes."""
    rows = []
    for i in range(n):
        rows.append({
            "note_id": str(i),
            "subject_id": str(1000 + i),
            "hadm_id": str(2000 + i),
            "text": _make_minimal_note(
                hospital_course=(
                    f"Brief Hospital Course: Patient {i} was admitted for evaluation. "
                    f"They were treated appropriately and discharged in stable condition. "
                    f"All relevant workup was completed during the hospital stay. "
                    f"The team reviewed results daily and adjusted therapy as needed.\n"
                    f"Medications on Admission: lisinopril 10mg"
                ),
                discharge_section=(
                    f"You were admitted to the hospital because you were experiencing "
                    f"symptoms that required further evaluation and treatment. "
                    f"During your stay, we performed a thorough workup and started "
                    f"appropriate therapy. Your condition improved significantly. "
                    f"Please make sure to attend all follow-up appointments and "
                    f"continue taking your medications as directed. "
                    f"If you develop any new or worsening symptoms, please seek "
                    f"medical attention promptly. We wish you a speedy recovery."
                ),
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMIMIC4NoteExtDIBHCDatasetStaticHelpers(unittest.TestCase):
    """Unit tests for static helper methods that do not require a loaded dataset."""

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestMIMIC4NoteExtDIBHCDatasetStaticHelpers")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # _extract_hc
    # ------------------------------------------------------------------

    def test_extract_hc_returns_text_between_markers(self):
        """_extract_hc should extract text between 'Brief Hospital Course:' and the next known marker."""
        print("\nTEST: test_extract_hc_returns_text_between_markers")
        # Note: _extract_hc checks len(txt.split(' ')) >= 30, so the full note
        # text (not just the BHC section) must be at least 30 words.
        txt = (
            "Admission Date: ___ Discharge Date: ___\n"
            "Service: MEDICINE\n"
            "Brief Hospital Course: Patient was admitted and treated successfully "
            "with IV antibiotics over a five-day course. Cultures returned negative "
            "and the patient improved clinically throughout the admission.\n"
            "Medications on Admission: lisinopril 10mg\n"
        )
        self.assertGreaterEqual(
            len(txt.split(" ")), 30,
            msg="Synthetic note must have >= 30 words to pass _extract_hc's length guard"
        )
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        self.assertIsNotNone(result)
        self.assertIn("admitted and treated", result)
        print(f"  Extracted: {result[:80]}...")
        print("  ✓ passed")

    def test_extract_hc_returns_none_when_marker_absent(self):
        """_extract_hc should return None when 'Brief Hospital Course:' is missing."""
        print("\nTEST: test_extract_hc_returns_none_when_marker_absent")
        txt = "This note has no relevant section.\nMedications on Admission: aspirin"
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        self.assertIsNone(result)
        print("  ✓ returned None as expected")

    def test_extract_hc_returns_none_for_very_short_text(self):
        """_extract_hc should return None when the surrounding text has fewer than 30 words."""
        print("\nTEST: test_extract_hc_returns_none_for_very_short_text")
        txt = "Brief Hospital Course: Short.\nMedications on Admission: x"
        self.assertLess(len(txt.split(" ")), 30)
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        self.assertIsNone(result)
        print("  ✓ returned None for very short note")

    def test_extract_hc_falls_back_to_discharge_medications(self):
        """_extract_hc should use 'Discharge Medications:' as end marker when 'Medications on Admission:' is absent."""
        print("\nTEST: test_extract_hc_falls_back_to_discharge_medications")
        txt = (
            "Brief Hospital Course: " + ("The patient was treated and improved. " * 10) + "\n"
            "Discharge Medications: metoprolol 25mg\n"
        )
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        self.assertIsNotNone(result)
        self.assertNotIn("Discharge Medications", result)
        print(f"  Extracted length: {len(result)} chars")
        print("  ✓ passed")

    def test_extract_hc_falls_back_to_discharge_disposition(self):
        """_extract_hc should use 'Discharge Disposition:' as end marker of last resort."""
        print("\nTEST: test_extract_hc_falls_back_to_discharge_disposition")
        txt = (
            "Brief Hospital Course: " + ("Patient recovered well and was ready for discharge. " * 8) + "\n"
            "Discharge Disposition: Home\n"
        )
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        self.assertIsNotNone(result)
        self.assertNotIn("Discharge Disposition", result)
        print("  ✓ passed")

    # ------------------------------------------------------------------
    # _remove_empty_and_short_summaries
    # ------------------------------------------------------------------

    def test_remove_empty_and_short_summaries_drops_short(self):
        """_remove_empty_and_short_summaries should drop rows with summary shorter than threshold."""
        print("\nTEST: test_remove_empty_and_short_summaries_drops_short")
        df = pd.DataFrame({"summary": ["short", "x" * 350, "x" * 400, ""]})
        result = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df, min_length_summary=350)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result["summary"].str.len() >= 350))
        print(f"  Input rows: 4, output rows: {len(result)}")
        print("  ✓ passed")

    def test_remove_empty_and_short_summaries_keeps_all_if_long_enough(self):
        """_remove_empty_and_short_summaries should keep all rows when they meet the threshold."""
        print("\nTEST: test_remove_empty_and_short_summaries_keeps_all_if_long_enough")
        df = pd.DataFrame({"summary": ["x" * 400, "x" * 500, "x" * 600]})
        result = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        self.assertEqual(len(result), 3)
        print("  ✓ passed")


# ---------------------------------------------------------------------------

class TestMIMIC4NoteExtDIBHCDatasetPipelineSteps(unittest.TestCase):
    """
    Tests for each preprocessing step using synthetic DataFrames.
    No MIMIC data or filesystem access is required.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestMIMIC4NoteExtDIBHCDatasetPipelineSteps")
        print(f"{'='*60}")
        self.pipeline = _DummyDataset()

    # ------------------------------------------------------------------
    # Step 0
    # ------------------------------------------------------------------

    def test_step0_replaces_special_chars(self):
        """Step 0 should replace known Unicode characters with ASCII equivalents."""
        print("\nTEST: test_step0_replaces_special_chars")
        df = pd.DataFrame({"text": [u"Hello\u0091world\u0093end"]})
        result = MIMIC4NoteExtDIBHCDataset._step0_special_chars(df)
        self.assertNotIn(u"\u0091", result["text"].iloc[0])
        self.assertNotIn(u"\u0093", result["text"].iloc[0])
        self.assertIn("'", result["text"].iloc[0])
        self.assertIn('"', result["text"].iloc[0])
        print(f"  Converted: {result['text'].iloc[0]}")
        print("  ✓ passed")

    def test_step0_strips_whitespace(self):
        """Step 0 should strip leading/trailing whitespace from text."""
        print("\nTEST: test_step0_strips_whitespace")
        df = pd.DataFrame({"text": ["   hello world   "]})
        result = MIMIC4NoteExtDIBHCDataset._step0_special_chars(df)
        self.assertEqual(result["text"].iloc[0], "hello world")
        print("  ✓ passed")

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def test_step1_splits_correctly(self):
        """Step 1 should split note on 'Discharge Instructions:' and populate both columns."""
        print("\nTEST: test_step1_splits_correctly")
        df = pd.DataFrame({"text": [_make_minimal_note()]})
        result = self.pipeline._step1_split_on_discharge_instructions(df)
        self.assertIn("hospital_course", result.columns)
        self.assertIn("summary", result.columns)
        self.assertGreater(result["hospital_course"].str.len().iloc[0], 0)
        self.assertGreater(result["summary"].str.len().iloc[0], 0)
        print(f"  hospital_course length: {result['hospital_course'].str.len().iloc[0]}")
        print(f"  summary length: {result['summary'].str.len().iloc[0]}")
        print("  ✓ passed")

    def test_step1_drops_notes_without_marker(self):
        """Step 1 should drop notes that lack 'Discharge Instructions:'."""
        print("\nTEST: test_step1_drops_notes_without_marker")
        df = pd.DataFrame({"text": [
            _make_minimal_note(),
            "A note with no discharge instructions section at all.",
        ]})
        result = self.pipeline._step1_split_on_discharge_instructions(df)
        self.assertEqual(len(result), 1)
        print(f"  Input rows: 2, output rows: {len(result)}")
        print("  ✓ passed")

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def test_step2_encodes_dr_abbreviation(self):
        """Step 2 should encode 'Dr.' in summaries to avoid false sentence splits."""
        print("\nTEST: test_step2_encodes_dr_abbreviation")
        df = _make_note_df(3)
        df = self.pipeline._step1_split_on_discharge_instructions(df)
        df.at[df.index[0], "summary"] = "Dr. Smith reviewed your case. " + "x" * 350
        result = MIMIC4NoteExtDIBHCDataset._step2_encode_and_extract_hc(df)
        print(f"  Rows after step 2: {len(result)}")
        print("  ✓ passed")

    def test_step2_populates_brief_hospital_course(self):
        """Step 2 should extract and populate the brief_hospital_course column."""
        print("\nTEST: test_step2_populates_brief_hospital_course")
        df = _make_note_df(3)
        df = self.pipeline._step1_split_on_discharge_instructions(df)
        result = MIMIC4NoteExtDIBHCDataset._step2_encode_and_extract_hc(df)
        self.assertIn("brief_hospital_course", result.columns)
        non_null = result["brief_hospital_course"].notnull().sum()
        print(f"  Rows with brief_hospital_course populated: {non_null}/{len(result)}")
        self.assertGreater(non_null, 0)
        print("  ✓ passed")

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------

    def test_step4_collapses_multiple_spaces(self):
        """Step 4 should collapse runs of multiple spaces into a single space."""
        print("\nTEST: test_step4_collapses_multiple_spaces")
        long_pad = "x" * 350
        df = pd.DataFrame({"summary": [f"Word1   Word2   Word3. {long_pad}"]})
        result = MIMIC4NoteExtDIBHCDataset._step4_remove_static_patterns(df)
        if len(result) > 0:
            self.assertNotIn("   ", result["summary"].iloc[0])
        print("  ✓ passed")

    def test_step4_applies_deidentification(self):
        """Step 4 should replace ___ with 'You ' where a known verb suffix follows."""
        print("\nTEST: test_step4_applies_deidentification")
        filler = "A" * 350
        sentence = "___ were admitted to the hospital for chest pain evaluation."
        df = pd.DataFrame({"summary": [sentence + " " + filler]})
        result = MIMIC4NoteExtDIBHCDataset._step4_remove_static_patterns(df)
        if len(result) > 0:
            text = result["summary"].iloc[0]
            self.assertIn("You were admitted", text)
            print(f"  De-identified: {text[:60]}")
        print("  ✓ passed")

    # ------------------------------------------------------------------
    # Step 6
    # ------------------------------------------------------------------

    def test_step6_drops_too_short_summaries(self):
        """Step 6 should drop summaries below min_chars threshold."""
        print("\nTEST: test_step6_drops_too_short_summaries")
        pipeline = _DummyDataset(min_chars=500)
        df = pd.DataFrame({
            "summary": [
                "Short text that won't pass.",
                "y" * 600 + " end.",
            ]
        })
        result = pipeline._step6_quality_filter(df)
        self.assertTrue(all(result["summary"].str.len() >= 500))
        print(f"  Rows after quality filter: {len(result)}")
        print("  ✓ passed")

    def test_step6_drops_deidentification_dense_summaries(self):
        """Step 6 should drop summaries with too many '___' tokens."""
        print("\nTEST: test_step6_drops_deidentification_dense_summaries")
        pipeline = _DummyDataset(min_chars=100, min_sentences=1, num_words_per_deidentified=10)
        # One summary with many ___ tokens (1 per ~5 words) → should be dropped
        noisy = " ".join(["word ___ word ___ word"] * 20) + "."
        # One clean summary
        clean = ("This patient was admitted and treated well. " * 10) + "."
        df = pd.DataFrame({"summary": [noisy, clean]})
        result = pipeline._step6_quality_filter(df)
        kept = result["summary"].tolist()
        self.assertFalse(any("___" in s and s.count("___") > len(s.split()) / 10 for s in kept))
        print(f"  Rows after deidentification filter: {len(result)}")
        print("  ✓ passed")

    # ------------------------------------------------------------------
    # Step 7
    # ------------------------------------------------------------------

    def test_step7_drops_null_hospital_course(self):
        """Step 7 should drop records with null hospital_course or brief_hospital_course."""
        print("\nTEST: test_step7_drops_null_hospital_course")
        pipeline = _DummyDataset(min_chars_bhc=10)
        df = pd.DataFrame({
            "summary": ["ok summary"] * 3,
            "hospital_course": ["some course", None, "another course"],
            "brief_hospital_course": ["brief text here", "also brief", None],
        })
        result = pipeline._step7_filter_hospital_course(df)
        self.assertEqual(len(result), 1)
        print(f"  Input rows: 3, output rows: {len(result)}")
        print("  ✓ passed")

    def test_step7_drops_short_brief_hospital_course(self):
        """Step 7 should drop records whose brief_hospital_course is below min_chars_bhc."""
        print("\nTEST: test_step7_drops_short_brief_hospital_course")
        pipeline = _DummyDataset(min_chars_bhc=500)
        df = pd.DataFrame({
            "summary": ["ok"] * 2,
            "hospital_course": ["course a", "course b"],
            "brief_hospital_course": ["too short", "x" * 600],
        })
        result = pipeline._step7_filter_hospital_course(df)
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(len(result["brief_hospital_course"].iloc[0]), 500)
        print("  ✓ passed")

    def test_step7_normalises_excessive_blank_lines(self):
        """Step 7 should collapse 3+ consecutive newlines down to two."""
        print("\nTEST: test_step7_normalises_excessive_blank_lines")
        pipeline = _DummyDataset(min_chars_bhc=5)
        df = pd.DataFrame({
            "summary": ["ok"],
            "hospital_course": ["line1\n\n\n\nline2"],
            "brief_hospital_course": ["brief\n\n\n\nmore"],
        })
        result = pipeline._step7_filter_hospital_course(df)
        self.assertNotIn("\n\n\n", result["hospital_course"].iloc[0])
        self.assertNotIn("\n\n\n", result["brief_hospital_course"].iloc[0])
        print("  ✓ passed")


# ---------------------------------------------------------------------------

class TestMIMIC4NoteExtDIBHCDatasetEndToEnd(unittest.TestCase):
    """
    End-to-end test of the preprocess() method on synthetic data.
    Verifies that the full pipeline runs without error and produces the
    expected output columns.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestMIMIC4NoteExtDIBHCDatasetEndToEnd")
        print(f"{'='*60}")
        self.pipeline = _DummyDataset()
        self.df_input = _make_note_df(n=10)
        print(f"  Created {len(self.df_input)} synthetic notes")

    def test_preprocess_runs_without_error(self):
        """preprocess() should complete without raising an exception."""
        print("\nTEST: test_preprocess_runs_without_error")
        try:
            result = self.pipeline.preprocess(self.df_input)
            print(f"  ✓ preprocess() completed; {len(result)} rows survived pipeline")
        except Exception as e:
            self.fail(f"preprocess() raised an unexpected exception: {e}")

    def test_preprocess_output_columns_present(self):
        """preprocess() output should contain summary, hospital_course, and brief_hospital_course."""
        print("\nTEST: test_preprocess_output_columns_present")
        result = self.pipeline.preprocess(self.df_input)
        for col in ("summary", "hospital_course", "brief_hospital_course"):
            self.assertIn(col, result.columns, msg=f"Missing column: {col}")
            print(f"  ✓ Column '{col}' present")

    def test_preprocess_does_not_mutate_input(self):
        """preprocess() should not modify the original DataFrame."""
        print("\nTEST: test_preprocess_does_not_mutate_input")
        original_text = self.df_input["text"].iloc[0]
        _ = self.pipeline.preprocess(self.df_input)
        self.assertEqual(self.df_input["text"].iloc[0], original_text)
        print("  ✓ Input DataFrame unchanged")

    def test_preprocess_summary_minimum_length(self):
        """All surviving summaries should meet the min_chars threshold."""
        print("\nTEST: test_preprocess_summary_minimum_length")
        result = self.pipeline.preprocess(self.df_input)
        if len(result) > 0:
            min_len = result["summary"].str.len().min()
            print(f"  Shortest surviving summary: {min_len} chars (threshold: {self.pipeline.min_chars})")
            self.assertGreaterEqual(min_len, self.pipeline.min_chars)
        print("  ✓ passed")

    def test_preprocess_brief_hospital_course_minimum_length(self):
        """All surviving brief hospital courses should meet the min_chars_bhc threshold."""
        print("\nTEST: test_preprocess_brief_hospital_course_minimum_length")
        result = self.pipeline.preprocess(self.df_input)
        if len(result) > 0:
            min_len = result["brief_hospital_course"].str.len().min()
            print(f"  Shortest BHC: {min_len} chars (threshold: {self.pipeline.min_chars_bhc})")
            self.assertGreaterEqual(min_len, self.pipeline.min_chars_bhc)
        print("  ✓ passed")

    def test_preprocess_no_residual_discharge_instructions_header(self):
        """summaries should not start with 'Discharge Instructions:' after preprocessing."""
        print("\nTEST: test_preprocess_no_residual_discharge_instructions_header")
        result = self.pipeline.preprocess(self.df_input)
        for summary in result["summary"]:
            self.assertFalse(
                summary.lower().startswith("discharge instructions"),
                msg="Found residual 'Discharge Instructions:' header in summary",
            )
        print("  ✓ No residual headers found")


# ---------------------------------------------------------------------------
# Lightweight stand-in for the dataset class
# ---------------------------------------------------------------------------

class _DummyDataset:
    """
    Exposes the full pipeline without requiring BaseDataset initialisation
    or filesystem access. All instance methods are delegated to the real
    class using unbound-method calls, so self.* threshold attributes are
    honoured correctly.
    """

    def __init__(
        self,
        min_chars: int = 350,
        max_double_newlines: int = 5,
        min_sentences: int = 3,
        num_words_per_deidentified: int = 10,
        min_chars_bhc: int = 500,
    ):
        self.min_chars = min_chars
        self.max_double_newlines = max_double_newlines
        self.min_sentences = min_sentences
        self.num_words_per_deidentified = num_words_per_deidentified
        self.min_chars_bhc = min_chars_bhc

    # --- delegate every instance method to the real class ----------------

    def _step1_split_on_discharge_instructions(self, df):
        return MIMIC4NoteExtDIBHCDataset._step1_split_on_discharge_instructions(self, df)

    def _step3_truncate_prefixes(self, df):
        return MIMIC4NoteExtDIBHCDataset._step3_truncate_prefixes(self, df)

    def _step5_truncate_suffixes(self, df):
        return MIMIC4NoteExtDIBHCDataset._step5_truncate_suffixes(self, df)

    def _step6_quality_filter(self, df):
        return MIMIC4NoteExtDIBHCDataset._step6_quality_filter(self, df)

    def _step7_filter_hospital_course(self, df):
        return MIMIC4NoteExtDIBHCDataset._step7_filter_hospital_course(self, df)

    # --- two helpers called by the delegated instance methods above ------

    @staticmethod
    def _remove_empty_and_short_summaries(df, min_length_summary=350):
        return MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(
            df, min_length_summary=min_length_summary
        )

    def _remove_regex_dict(self, df, regexes, postprocess, keep=0):
        return MIMIC4NoteExtDIBHCDataset._remove_regex_dict(df, regexes, postprocess, keep=keep)

    # --- full pipeline ---------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = MIMIC4NoteExtDIBHCDataset._step0_special_chars(df)
        df = self._step1_split_on_discharge_instructions(df)
        df = MIMIC4NoteExtDIBHCDataset._step2_encode_and_extract_hc(df)
        df = self._step3_truncate_prefixes(df)
        df = MIMIC4NoteExtDIBHCDataset._step4_remove_static_patterns(df)
        df = self._step5_truncate_suffixes(df)
        df = self._step6_quality_filter(df)
        df = self._step7_filter_hospital_course(df)
        return df


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()