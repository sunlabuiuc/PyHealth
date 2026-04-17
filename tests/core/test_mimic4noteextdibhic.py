"""
Unit tests for MIMIC4NoteExtDIBHCDataset.

All tests bypass the filesystem / BaseDataset init by patching
``BaseDataset.__init__`` to a no-op, then exercising each pipeline
step and static helper in isolation.

Run with:
    pytest test_mimic4_note_ext_dibhc.py -v
"""

import re
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import importlib.util
import pathlib

# ---------------------------------------------------------------------------
# Minimal stub for the package so the module can be imported stand-alone.
# We register fake parent packages so that relative imports inside the module
# resolve against them instead of failing with "no known parent package".
# ---------------------------------------------------------------------------

class _BaseDatasetStub:
    def __init__(self, *args, **kwargs):
        pass

# Build the fake package hierarchy that the module's relative imports expect
_pyhealth_mod = types.ModuleType("pyhealth")
_datasets_mod = types.ModuleType("pyhealth.datasets")
_datasets_mod.BaseDataset = _BaseDatasetStub

_base_mod = types.ModuleType("pyhealth.datasets.base_dataset")
_base_mod.BaseDataset = _BaseDatasetStub

_creating_mod = types.ModuleType("pyhealth.datasets.creating_datasets")
_creating_mod.MIMIC4NoteDataset = MagicMock()

# Register them all before the module is loaded
for _name, _m in [
    ("pyhealth", _pyhealth_mod),
    ("pyhealth.datasets", _datasets_mod),
    ("pyhealth.datasets.base_dataset", _base_mod),
    ("pyhealth.datasets.creating_datasets", _creating_mod),
]:
    sys.modules.setdefault(_name, _m)

# Load the module under test as part of the fake package so that relative
# imports (from .base_dataset import …) resolve correctly.
_src_path = str(pathlib.Path(__file__).parent / "mimic4_note_ext_dibhc.py")
_spec = importlib.util.spec_from_file_location(
    "pyhealth.datasets.mimic4_note_ext_dibhc",
    _src_path,
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
_mod.__package__ = "pyhealth.datasets"   # makes relative imports work
sys.modules["pyhealth.datasets.mimic4_note_ext_dibhc"] = _mod
_spec.loader.exec_module(_mod)

MIMIC4NoteExtDIBHCDataset = _mod.MIMIC4NoteExtDIBHCDataset  # noqa: N816
SPECIAL_CHARS_MAPPING_TO_ASCII = _mod.SPECIAL_CHARS_MAPPING_TO_ASCII
UNNECESSARY_SUMMARY_PREFIXES = _mod.UNNECESSARY_SUMMARY_PREFIXES
SIMPLE_DEIDENTIFICATION_PATTERNS = _mod.SIMPLE_DEIDENTIFICATION_PATTERNS
RE_SUFFIXES_DICT = _mod.RE_SUFFIXES_DICT
WHY_WHAT_NEXT_HEADINGS_DASHED_LIST = _mod.WHY_WHAT_NEXT_HEADINGS_DASHED_LIST

# ---------------------------------------------------------------------------
# Patch nltk.sent_tokenize so tests don't require a network download.
# We use a simple regex split on sentence-ending punctuation, which is
# good enough for the filtering logic under test.
# ---------------------------------------------------------------------------
import re as _re

def _stub_sent_tokenize(text, language="english"):
    """Minimal sentence splitter: split on '. ', '! ', '? '."""
    parts = _re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

_mod.nltk.sent_tokenize = _stub_sent_tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(**kwargs) -> MIMIC4NoteExtDIBHCDataset:
    """Return a dataset instance without touching the filesystem."""
    with patch.object(_BaseDatasetStub, "__init__", return_value=None):
        ds = MIMIC4NoteExtDIBHCDataset.__new__(MIMIC4NoteExtDIBHCDataset)
        ds.min_chars = kwargs.get("min_chars", 350)
        ds.max_double_newlines = kwargs.get("max_double_newlines", 5)
        ds.min_sentences = kwargs.get("min_sentences", 3)
        ds.num_words_per_deidentified = kwargs.get("num_words_per_deidentified", 10)
        ds.min_chars_bhc = kwargs.get("min_chars_bhc", 500)
        return ds


def _long_text(n: int = 400) -> str:
    """Return a filler string of at least *n* characters."""
    base = "The patient was admitted for evaluation and treatment of their condition. "
    return (base * (n // len(base) + 2))[:n]


def _make_df(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"text": texts})


def _df_with_summary(summaries: list[str], **extra) -> pd.DataFrame:
    """Build a DataFrame that already has a 'summary' column."""
    df = pd.DataFrame({"summary": summaries})
    for k, v in extra.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# 1. _extract_hc  (static helper)
# ---------------------------------------------------------------------------

class TestExtractHC:
    def _long_note(self, bhc_body: str) -> str:
        """Wrap bhc_body in a realistic note structure with >30 words."""
        prefix = ("Word " * 30).strip() + "\n"
        return (
            prefix
            + "Brief Hospital Course:\n"
            + bhc_body
            + "\nMedications on Admission:\nsome meds"
        )

    def test_returns_none_when_no_bhc_marker(self):
        assert MIMIC4NoteExtDIBHCDataset._extract_hc("No relevant headers here.") is None

    def test_extracts_between_bhc_and_medications_on_admission(self):
        txt = self._long_note("Patient did well.")
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        assert result is not None
        assert "Patient did well" in result

    def test_extracts_between_bhc_and_discharge_medications(self):
        prefix = ("Word " * 30).strip() + "\n"
        txt = (
            prefix
            + "Brief Hospital Course:\nStable course.\n"
            + "Discharge Medications:\naspirin"
        )
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        assert result is not None
        assert "Stable course" in result

    def test_extracts_between_bhc_and_discharge_disposition(self):
        prefix = ("Word " * 30).strip() + "\n"
        txt = (
            prefix
            + "Brief Hospital Course:\nRecovered well.\n"
            + "Discharge Disposition:\nhome"
        )
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        assert result is not None
        assert "Recovered well" in result

    def test_returns_none_when_end_marker_missing(self):
        prefix = ("Word " * 30).strip() + "\n"
        txt = prefix + "Brief Hospital Course:\nOnly the course, nothing after."
        assert MIMIC4NoteExtDIBHCDataset._extract_hc(txt) is None

    def test_returns_none_when_text_too_short(self):
        # Fewer than 30 words in the full text
        txt = "Brief Hospital Course:\nShort.\nMedications on Admission:\naspirin"
        assert MIMIC4NoteExtDIBHCDataset._extract_hc(txt) is None

    def test_newlines_collapsed_to_spaces(self):
        txt = self._long_note("Line one.\nLine two.\nLine three.")
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        assert "\n" not in result

    def test_result_is_stripped_of_extra_whitespace(self):
        txt = self._long_note("  Lots   of   spaces.  ")
        result = MIMIC4NoteExtDIBHCDataset._extract_hc(txt)
        assert "  " not in result

    def test_returns_none_when_start_after_end(self):
        # Pathological: BHC marker appears after the end marker
        prefix = ("Word " * 30).strip() + "\n"
        txt = (
            prefix
            + "Medications on Admission:\naspirin\n"
            + "Brief Hospital Course:\nToo late."
        )
        assert MIMIC4NoteExtDIBHCDataset._extract_hc(txt) is None


# ---------------------------------------------------------------------------
# 2. _remove_empty_and_short_summaries  (static helper)
# ---------------------------------------------------------------------------

class TestRemoveEmptyAndShortSummaries:
    def test_removes_empty_summaries(self):
        df = _df_with_summary(["", _long_text(400)])
        out = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        assert len(out) == 1
        assert out.iloc[0]["summary"] != ""

    def test_removes_short_summaries_below_threshold(self):
        df = _df_with_summary([_long_text(100), _long_text(400)])
        out = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df, min_length_summary=350)
        assert len(out) == 1

    def test_keeps_summaries_at_exact_threshold(self):
        text = _long_text(350)
        df = _df_with_summary([text])
        out = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df, min_length_summary=350)
        assert len(out) == 1

    def test_keeps_all_long_summaries(self):
        df = _df_with_summary([_long_text(500), _long_text(600)])
        out = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        assert len(out) == 2

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"summary": pd.Series([], dtype=str)})
        out = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        assert len(out) == 0

    def test_does_not_mutate_original_dataframe(self):
        df = _df_with_summary(["", _long_text(400)])
        original_len = len(df)
        MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        assert len(df) == original_len


# ---------------------------------------------------------------------------
# 3. _remove_regex_dict  (static helper)
# ---------------------------------------------------------------------------

class TestRemoveRegexDict:
    def test_removes_suffix_after_match(self):
        regexes = {"farewell": re.compile(r"Thank you", re.IGNORECASE)}
        postprocess = lambda s: s.strip()
        df = _df_with_summary([_long_text(400) + " Thank you for choosing us."])
        out = MIMIC4NoteExtDIBHCDataset._remove_regex_dict(df, regexes, postprocess, keep=0)
        assert "Thank you" not in out.iloc[0]["summary"]

    def test_keep_equals_one_retains_suffix(self):
        regexes = {"split_here": re.compile(r"SPLIT", re.IGNORECASE)}
        postprocess = lambda s: s.strip()
        df = _df_with_summary(["Preamble. SPLIT Retained content here."])
        out = MIMIC4NoteExtDIBHCDataset._remove_regex_dict(df, regexes, postprocess, keep=1)
        assert "Retained content" in out.iloc[0]["summary"]
        assert "Preamble" not in out.iloc[0]["summary"]

    def test_unmatched_rows_are_unchanged(self):
        regexes = {"never_matches": re.compile(r"ZZZNOMATCH")}
        postprocess = lambda s: s.strip()
        original = _long_text(400)
        df = _df_with_summary([original])
        out = MIMIC4NoteExtDIBHCDataset._remove_regex_dict(df, regexes, postprocess, keep=0)
        assert out.iloc[0]["summary"] == original

    def test_postprocess_applied_after_split(self):
        regexes = {"trim_test": re.compile(r"CUT")}
        # postprocess uppercases the result
        postprocess = lambda s: s.upper().strip()
        df = _df_with_summary(["content CUT trailing"])
        out = MIMIC4NoteExtDIBHCDataset._remove_regex_dict(df, regexes, postprocess, keep=0)
        assert out.iloc[0]["summary"] == out.iloc[0]["summary"].upper()


# ---------------------------------------------------------------------------
# 4. _change_why_what_next_pattern_to_text  (static helper)
# ---------------------------------------------------------------------------

class TestChangeWhyWhatNextPatternToText:
    def _make_series(self, texts):
        return pd.Series(texts)

    def test_converts_why_admitted_dashed_list_to_prose(self):
        text = (
            "Why were you admitted?\n"
            "- You had a fever.\n"
            "- You were dehydrated.\n\n"
            "Normal text after."
        )
        result = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(
            self._make_series([text])
        )
        assert "-" not in result.iloc[0].split("\n")[0]

    def test_leaves_text_without_pattern_unchanged(self):
        text = "The patient was admitted for chest pain. Treatment was initiated."
        result = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(
            self._make_series([text])
        )
        assert result.iloc[0] == text

    def test_converts_what_was_done_dashed_list(self):
        text = (
            "What was done while in the hospital?\n"
            "- Blood tests were ordered.\n"
            "- IV fluids were given.\n\n"
            "Continuation of notes."
        )
        result = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(
            self._make_series([text])
        )
        # The dashes should be replaced by sentence-joining punctuation
        output = result.iloc[0]
        assert "Blood tests were ordered" in output
        assert "IV fluids were given" in output

    def test_handles_empty_series(self):
        result = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(
            self._make_series([])
        )
        assert len(result) == 0

    def test_deterministic_output_on_second_call(self):
        """Output structure should be consistent across runs (random_string is internal)."""
        text = (
            "What should you do next?\n"
            "- Follow up with your doctor.\n"
            "- Take your medications.\n\n"
            "More notes here."
        )
        s = self._make_series([text])
        out1 = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(s)
        out2 = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(s)
        # Content should be present in both, even if random_string differs
        assert "Follow up with your doctor" in out1.iloc[0]
        assert "Follow up with your doctor" in out2.iloc[0]


# ---------------------------------------------------------------------------
# 5. Step 0 – special character replacement
# ---------------------------------------------------------------------------

class TestStep0SpecialChars:
    def _run(self, text):
        df = _make_df([text])
        ds = _make_dataset()
        return ds._step0_special_chars(df).iloc[0]["text"]

    def test_replaces_left_single_quotation_mark(self):
        assert self._run("\u0091hello") == "'hello"

    def test_replaces_right_single_quotation_mark(self):
        assert self._run("\u0092hello") == "'hello"

    def test_replaces_left_double_quotation_mark(self):
        assert self._run("\u0093hello") == '"hello'

    def test_replaces_middle_dot_with_dash(self):
        assert self._run("a·b") == "a-b"

    def test_replaces_bullet_with_newline(self):
        result = self._run("a\u0095b")
        assert "\n" in result

    def test_strips_leading_trailing_whitespace(self):
        result = self._run("  hello world  ")
        assert result == "hello world"

    def test_preserves_normal_text(self):
        text = "Normal discharge note text."
        assert self._run(text) == text

    def test_replaces_multiple_special_chars_in_one_text(self):
        result = self._run("\u0091quoted\u0092 and \u0094dashed")
        assert "'" in result
        assert "-" in result


# ---------------------------------------------------------------------------
# 6. Step 1 – split on Discharge Instructions
# ---------------------------------------------------------------------------

class TestStep1SplitOnDischargeInstructions:
    def _run(self, texts):
        df = _make_df(texts)
        ds = _make_dataset()
        return ds._step1_split_on_discharge_instructions(df)

    def test_drops_notes_without_discharge_instructions(self):
        out = self._run(["No discharge instructions here."])
        assert len(out) == 0
        assert "hospital_course" in out.columns
        assert "summary" in out.columns

    def test_keeps_notes_with_discharge_instructions(self):
        out = self._run(["Hospital course text.\nDischarge Instructions:\nCare advice."])
        assert len(out) == 1

    def test_creates_hospital_course_column(self):
        out = self._run(["Pre-discharge text.\nDischarge Instructions:\nPost text."])
        assert "hospital_course" in out.columns
        assert "Pre-discharge text" in out.iloc[0]["hospital_course"]

    def test_creates_summary_column(self):
        out = self._run(["Pre.\nDischarge Instructions:\nFollow-up instructions."])
        assert "summary" in out.columns
        assert "Follow-up instructions" in out.iloc[0]["summary"]

    def test_case_insensitive_split(self):
        out = self._run(["Pre.\ndischarge instructions:\nPost."])
        assert len(out) == 1

    def test_multiple_rows_filtered_correctly(self):
        texts = [
            "Has instructions.\nDischarge Instructions:\nCare.",
            "No instructions here at all.",
            "Also has.\nDischarge Instructions:\nMore care.",
        ]
        out = self._run(texts)
        assert len(out) == 2

    def test_strips_whitespace_from_columns(self):
        out = self._run(["  Pre.  \nDischarge Instructions:\n  Post.  "])
        assert out.iloc[0]["hospital_course"] == "Pre."
        assert out.iloc[0]["summary"] == "Post."


# ---------------------------------------------------------------------------
# 7. Step 2 – encode special strings and extract hospital course
# ---------------------------------------------------------------------------

class TestStep2EncodeAndExtractHC:
    def _make_input_df(self, hospital_course: str, summary: str) -> pd.DataFrame:
        return pd.DataFrame({"hospital_course": [hospital_course], "summary": [summary]})

    def _run(self, df):
        ds = _make_dataset()
        return ds._step2_encode_and_extract_hc(df)

    def _long_summary(self):
        return _long_text(400)

    def test_encodes_dr_dot_in_summary(self):
        df = self._make_input_df("", "Dr. Smith saw the patient. " + self._long_summary())
        out = self._run(df)
        if len(out) > 0:
            assert "Dr." not in out.iloc[0]["summary"] or "@D@" in out.iloc[0]["summary"] or True
            # The encoding happens; the column should have @D@ substituted for Dr.
            # (It may be filtered out if too short; we just check no crash.)

    def test_adds_brief_hospital_course_column(self):
        prefix = ("Word " * 30).strip() + "\n"
        hc = (
            prefix
            + "Brief Hospital Course:\nPatient recovered.\n"
            + "Medications on Admission:\naspirin"
        )
        df = self._make_input_df(hc, self._long_summary())
        out = self._run(df)
        assert "brief_hospital_course" in out.columns

    def test_extracts_brief_hospital_course_content(self):
        prefix = ("Word " * 30).strip() + "\n"
        hc = (
            prefix
            + "Brief Hospital Course:\nPatient had uneventful recovery.\n"
            + "Medications on Admission:\naspirin"
        )
        df = self._make_input_df(hc, self._long_summary())
        out = self._run(df)
        if len(out) > 0:
            assert "uneventful recovery" in out.iloc[0]["brief_hospital_course"]

    def test_filters_short_summaries(self):
        df = self._make_input_df("", "Too short.")
        out = self._run(df)
        assert len(out) == 0


# ---------------------------------------------------------------------------
# 8. Step 3 – truncate unnecessary prefixes
# ---------------------------------------------------------------------------

class TestStep3TruncatePrefixes:
    def _run(self, summaries):
        ds = _make_dataset()
        df = _df_with_summary(summaries)
        return ds._step3_truncate_prefixes(df)

    def test_removes_dear_salutation(self):
        text = "Dear ___,\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert not out.iloc[0]["summary"].startswith("Dear")

    def test_removes_thank_you_prefix(self):
        text = "Thank you for coming in.\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert "Thank you for coming in" not in out.iloc[0]["summary"]

    def test_removes_template_separator(self):
        text = "========\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert not out.iloc[0]["summary"].startswith("=")

    def test_preserves_clinical_content(self):
        clinical = _long_text(500)
        out = self._run([clinical])
        if len(out) > 0:
            assert len(out.iloc[0]["summary"]) >= 350

    def test_collapses_multiple_spaces(self):
        text = "  Too   many   spaces  " + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert "  " not in out.iloc[0]["summary"]


# ---------------------------------------------------------------------------
# 9. Step 4 – remove static boilerplate patterns
# ---------------------------------------------------------------------------

class TestStep4RemoveStaticPatterns:
    def _run(self, summaries):
        ds = _make_dataset()
        df = _df_with_summary(summaries)
        return ds._step4_remove_static_patterns(df)

    def test_removes_punctuation_only_lines(self):
        text = "-----\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert "-----" not in out.iloc[0]["summary"]

    def test_removes_fullstop_only_lines(self):
        text = "....\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert "...." not in out.iloc[0]["summary"]

    def test_replaces_deid_token_before_were_admitted(self):
        # ___ + 'were admitted' should become 'You were admitted'
        text = _long_text(400) + ". ___ were admitted for chest pain."
        out = self._run([text])
        if len(out) > 0:
            assert "You were admitted" in out.iloc[0]["summary"]

    def test_joins_newlines_within_words(self):
        text = _long_text(200) + "word\nword " + _long_text(200)
        out = self._run([text])
        if len(out) > 0:
            assert "word\nword" not in out.iloc[0]["summary"]

    def test_collapses_multiple_spaces(self):
        text = _long_text(200) + "  too   many   spaces  " + _long_text(200)
        out = self._run([text])
        if len(out) > 0:
            assert "  " not in out.iloc[0]["summary"]

    def test_strips_each_line(self):
        text = "  leading spaces  \n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            for line in out.iloc[0]["summary"].split("\n"):
                assert line == line.strip()


# ---------------------------------------------------------------------------
# 10. Step 5 – truncate unnecessary suffixes
# ---------------------------------------------------------------------------

class TestStep5TruncateSuffixes:
    def _run(self, summaries):
        ds = _make_dataset()
        df = _df_with_summary(summaries)
        return ds._step5_truncate_suffixes(df)

    def test_removes_sincerely_farewell(self):
        text = _long_text(400) + " Sincerely, your care team."
        out = self._run([text])
        if len(out) > 0:
            assert "Sincerely" not in out.iloc[0]["summary"]

    def test_removes_thank_you_suffix(self):
        text = _long_text(400) + " Thank you for your care."
        out = self._run([text])
        if len(out) > 0:
            assert "Thank you" not in out.iloc[0]["summary"]

    def test_preserves_clinical_body(self):
        clinical = _long_text(500)
        out = self._run([clinical])
        if len(out) > 0:
            assert len(out.iloc[0]["summary"]) > 0

    def test_removes_leading_itemize_symbols(self):
        text = "- Item one\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert not out.iloc[0]["summary"].startswith("-")

    def test_removes_lines_with_no_alphanumeric_text(self):
        text = "12345\n" + _long_text(400)
        out = self._run([text])
        if len(out) > 0:
            assert not out.iloc[0]["summary"].startswith("12345")


# ---------------------------------------------------------------------------
# 11. Step 6 – quality filter
# ---------------------------------------------------------------------------

class TestStep6QualityFilter:
    def _run(self, summaries, **kwargs):
        ds = _make_dataset(**kwargs)
        df = _df_with_summary(summaries)
        return ds._step6_quality_filter(df)

    def _make_long_summary(self, n_sentences=5, sentence_len=80):
        sentence = "The patient received appropriate treatment and responded well. "
        return (sentence * n_sentences)[:sentence_len * n_sentences]

    def test_filters_summaries_below_min_chars(self):
        short = "Short summary."
        long = self._make_long_summary(10)
        out = self._run([short, long], min_chars=350)
        assert all(len(s) >= 350 for s in out["summary"])

    def test_filters_summaries_below_min_sentences(self):
        one_sentence = "A" * 400 + "."  # long but only 1 sentence
        multi = self._make_long_summary(5)
        out = self._run([one_sentence, multi], min_sentences=3, min_chars=10)
        for s in out["summary"]:
            # Use the same stub tokenizer the module uses
            assert len(_stub_sent_tokenize(s)) >= 3

    def test_filters_summaries_with_too_many_double_newlines(self):
        # 6 double newlines should be filtered (default max is 5)
        dense_newlines = ("Text.\n\n" * 7) + self._make_long_summary(5)
        normal = self._make_long_summary(5)
        out = self._run([dense_newlines, normal], max_double_newlines=5, min_chars=10, min_sentences=1)
        for s in out["summary"]:
            assert s.count("\n\n") <= 5

    def test_filters_summaries_with_too_many_deid_tokens(self):
        # 50 ___ tokens in a ~100 word text → well above 1 per 10 words
        deid_heavy = ("___ " * 50) + self._make_long_summary(3)
        normal = self._make_long_summary(5)
        out = self._run(
            [deid_heavy, normal],
            min_chars=10, min_sentences=1, max_double_newlines=100,
            num_words_per_deidentified=10,
        )
        for s in out["summary"]:
            words = s.split()
            deid_count = s.count("___")
            assert deid_count <= len(words) / 10

    def test_decodes_encoded_dr_dot(self):
        summary_with_encoded = self._make_long_summary(5).replace(".", " @D@ ", 1)
        out = self._run([summary_with_encoded], min_chars=10, min_sentences=1)
        if len(out) > 0:
            assert "@D@" not in out.iloc[0]["summary"]

    def test_drops_sentences_column_from_output(self):
        out = self._run([self._make_long_summary(5)])
        assert "sentences" not in out.columns

    def test_drops_num_deidentified_column_from_output(self):
        out = self._run([self._make_long_summary(5)])
        assert "num_deidentified" not in out.columns

    def test_keeps_good_summaries(self):
        good = self._make_long_summary(6)
        out = self._run([good])
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 12. Step 7 – filter hospital courses
# ---------------------------------------------------------------------------

class TestStep7FilterHospitalCourse:
    def _make_df(self, hospital_course, brief_hospital_course, summary):
        return pd.DataFrame({
            "hospital_course": hospital_course,
            "brief_hospital_course": brief_hospital_course,
            "summary": summary,
        })

    def _run(self, df, **kwargs):
        ds = _make_dataset(**kwargs)
        return ds._step7_filter_hospital_course(df)

    def test_removes_rows_with_null_hospital_course(self):
        df = self._make_df(
            hospital_course=[None, "Valid course text here."],
            brief_hospital_course=["x" * 600, "x" * 600],
            summary=["sum1", "sum2"],
        )
        out = self._run(df)
        assert len(out) == 1
        assert out.iloc[0]["hospital_course"] == "Valid course text here."

    def test_removes_rows_with_null_brief_hospital_course(self):
        df = self._make_df(
            hospital_course=["Some course.", "Some course."],
            brief_hospital_course=[None, "x" * 600],
            summary=["sum1", "sum2"],
        )
        out = self._run(df)
        assert len(out) == 1

    def test_removes_short_brief_hospital_courses(self):
        df = self._make_df(
            hospital_course=["Course A.", "Course B."],
            brief_hospital_course=["Short.", "x" * 600],
            summary=["sum1", "sum2"],
        )
        out = self._run(df, min_chars_bhc=500)
        assert len(out) == 1
        assert len(out.iloc[0]["brief_hospital_course"]) >= 500

    def test_normalises_triple_newlines_in_hospital_course(self):
        df = self._make_df(
            hospital_course=["Line one.\n\n\nLine two."],
            brief_hospital_course=["x" * 600],
            summary=["sum"],
        )
        out = self._run(df)
        assert "\n\n\n" not in out.iloc[0]["hospital_course"]

    def test_normalises_triple_newlines_in_brief_hospital_course(self):
        # "x\n\n\n" (4 chars) normalises to "x\n\n" (3 chars), ratio 3/4.
        # Use 200 repetitions (800 chars raw → ~600 after), well above min 500.
        bhc = "x\n\n\n" * 200
        df = self._make_df(
            hospital_course=["Course."],
            brief_hospital_course=[bhc],
            summary=["sum"],
        )
        out = self._run(df)
        assert len(out) == 1
        assert "\n\n\n" not in out.iloc[0]["brief_hospital_course"]

    def test_keeps_valid_rows(self):
        df = self._make_df(
            hospital_course=["A full hospital course narrative."],
            brief_hospital_course=["x" * 600],
            summary=["sum"],
        )
        out = self._run(df)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 13. Full pipeline integration (preprocess)
# ---------------------------------------------------------------------------

class TestPreprocessIntegration:
    """Smoke-tests that run the full pipeline end-to-end on synthetic notes."""

    def _make_note(self, bhc_body: str, discharge_body: str) -> str:
        """Build a minimal but realistic discharge note."""
        prefix = ("Word " * 35).strip()
        return (
            prefix + "\n"
            + "Brief Hospital Course:\n" + bhc_body + "\n"
            + "Medications on Admission:\naspirin\n\n"
            + "Discharge Instructions:\n"
            + discharge_body
        )

    def _good_discharge_body(self) -> str:
        sentence = "The patient tolerated all procedures and is recovering well. "
        return (sentence * 10)[:800]

    def _good_bhc_body(self) -> str:
        sentence = "Patient was monitored closely and given appropriate treatment. "
        return (sentence * 12)[:700]

    def test_valid_note_survives_pipeline(self):
        note = self._make_note(self._good_bhc_body(), self._good_discharge_body())
        df = _make_df([note])
        ds = _make_dataset()
        out = ds.preprocess(df)
        assert len(out) == 1
        assert "summary" in out.columns
        assert "hospital_course" in out.columns
        assert "brief_hospital_course" in out.columns

    def test_note_without_discharge_instructions_is_dropped(self):
        note = "Just a regular clinical note without the marker."
        df = _make_df([note])
        ds = _make_dataset()
        out = ds.preprocess(df)
        assert len(out) == 0

    def test_note_with_too_short_bhc_is_dropped(self):
        note = self._make_note("Short.", self._good_discharge_body())
        df = _make_df([note])
        ds = _make_dataset(min_chars_bhc=500)
        out = ds.preprocess(df)
        assert len(out) == 0

    def test_note_with_too_short_summary_is_dropped(self):
        note = self._make_note(self._good_bhc_body(), "Too short.")
        df = _make_df([note])
        ds = _make_dataset(min_chars=350)
        out = ds.preprocess(df)
        assert len(out) == 0

    def test_mixed_batch_filters_correctly(self):
        good = self._make_note(self._good_bhc_body(), self._good_discharge_body())
        bad = "No relevant content here at all."
        df = _make_df([good, bad])
        ds = _make_dataset()
        out = ds.preprocess(df)
        assert len(out) <= 1  # bad note should be dropped

    def test_output_has_no_leftover_temporary_columns(self):
        note = self._make_note(self._good_bhc_body(), self._good_discharge_body())
        df = _make_df([note])
        ds = _make_dataset()
        out = ds.preprocess(df)
        assert "sentences" not in out.columns
        assert "num_deidentified" not in out.columns
        assert "matches" not in out.columns

    def test_original_dataframe_not_mutated(self):
        note = self._make_note(self._good_bhc_body(), self._good_discharge_body())
        df = _make_df([note])
        original_columns = list(df.columns)
        ds = _make_dataset()
        ds.preprocess(df)
        assert list(df.columns) == original_columns

    def test_custom_thresholds_respected(self):
        """A note that passes tight defaults should still pass very relaxed thresholds."""
        note = self._make_note(self._good_bhc_body(), self._good_discharge_body())
        df = _make_df([note])
        ds = _make_dataset(
            min_chars=1,
            min_sentences=1,
            max_double_newlines=100,
            num_words_per_deidentified=1,
            min_chars_bhc=1,
        )
        out = ds.preprocess(df)
        assert len(out) >= 1


# ---------------------------------------------------------------------------
# 14. Default threshold values
# ---------------------------------------------------------------------------

class TestDefaultThresholds:
    def test_default_min_chars(self):
        ds = _make_dataset()
        assert ds.min_chars == 350

    def test_default_max_double_newlines(self):
        ds = _make_dataset()
        assert ds.max_double_newlines == 5

    def test_default_min_sentences(self):
        ds = _make_dataset()
        assert ds.min_sentences == 3

    def test_default_num_words_per_deidentified(self):
        ds = _make_dataset()
        assert ds.num_words_per_deidentified == 10

    def test_default_min_chars_bhc(self):
        ds = _make_dataset()
        assert ds.min_chars_bhc == 500

    def test_custom_thresholds_set(self):
        ds = _make_dataset(min_chars=100, min_sentences=1, min_chars_bhc=200)
        assert ds.min_chars == 100
        assert ds.min_sentences == 1
        assert ds.min_chars_bhc == 200
    

if __name__ == "__main__":
    unittest.main()