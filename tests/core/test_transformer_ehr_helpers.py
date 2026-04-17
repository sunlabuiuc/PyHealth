"""
Tests for the utility functions and classes defined in
pyhealth/models/generators/gpt_baseline.py

Covered:
* ``samples_to_sequences``   – nested visit lists -> text strings
* ``sequences_to_dataframe`` – text strings -> long-form DataFrame
* ``build_tokenizer``        – word-level HuggingFace tokenizer
* ``EHRTextDataset``         – PyTorch Dataset wrapping tokenized EHR sequences
"""

import unittest

import torch
from transformers import PreTrainedTokenizerFast

from pyhealth.models.generators import (
    VISIT_DELIM,
    EHRTextDataset,
    build_tokenizer,
    samples_to_sequences,
    sequences_to_dataframe,
)

# ── shared test data ───────────────────────────────────────────────────────────

_SINGLE_VISIT_SAMPLE = {"conditions": [["250.00", "401.9"]]}
_MULTI_VISIT_SAMPLE = {"conditions": [["250.00", "401.9"], ["272.0", "428.0"], ["250.00"]]}
_EMPTY_VISIT_SAMPLE = {"conditions": []}

_CORPUS = [
    "250.00 401.9 VISIT_DELIM 272.0",
    "428.0 VISIT_DELIM 250.00",
    "401.9 272.0 428.0",
]

_SEQUENCES = [
    "250.00 401.9 VISIT_DELIM 272.0",
    "428.0",
    "401.9 272.0 428.0 VISIT_DELIM 250.00 VISIT_DELIM 272.0",
]
_MAX_LENGTH = 16


# ── 1. samples_to_sequences ────────────────────────────────────────────────────


class TestSamplesToSequences(unittest.TestCase):
    def test_returns_one_string_per_sample(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE, _MULTI_VISIT_SAMPLE])
        self.assertEqual(len(result), 2)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(samples_to_sequences([]), [])

    def test_single_visit_no_delimiter(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE])
        self.assertNotIn(VISIT_DELIM, result[0])

    def test_multi_visit_delimiter_count_matches(self):
        # 3 visits -> 2 VISIT_DELIM occurrences
        result = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        self.assertEqual(result[0].count(VISIT_DELIM), 2)

    def test_codes_present_in_output(self):
        result = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        for visit in _MULTI_VISIT_SAMPLE["conditions"]:
            for code in visit:
                self.assertIn(code, result[0])

    def test_single_visit_codes_space_separated(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE])
        self.assertEqual(result[0], "250.00 401.9")

    def test_multi_visit_format(self):
        result = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        expected = f"250.00 401.9 {VISIT_DELIM} 272.0 428.0 {VISIT_DELIM} 250.00"
        self.assertEqual(result[0], expected)

    def test_empty_conditions_yields_empty_string(self):
        result = samples_to_sequences([_EMPTY_VISIT_SAMPLE])
        self.assertEqual(result[0], "")

    def test_single_code_per_visit(self):
        sample = {"conditions": [["A"], ["B"], ["C"]]}
        result = samples_to_sequences([sample])
        self.assertEqual(result[0], f"A {VISIT_DELIM} B {VISIT_DELIM} C")

    def test_multiple_samples_independent(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE, _MULTI_VISIT_SAMPLE])
        self.assertNotEqual(result[0], result[1])

    def test_output_is_list_of_strings(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE])
        self.assertIsInstance(result, list)
        for s in result:
            self.assertIsInstance(s, str)


# ── 2. sequences_to_dataframe ─────────────────────────────────────────────────


class TestSequencesToDataframe(unittest.TestCase):
    _SEQ_SINGLE = "250.00 401.9"
    _SEQ_MULTI = f"250.00 401.9 {VISIT_DELIM} 272.0 428.0"

    def test_required_columns_present(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE])
        self.assertEqual(set(df.columns), {"SUBJECT_ID", "HADM_ID", "ICD9_CODE"})

    def test_empty_input_returns_empty_dataframe(self):
        df = sequences_to_dataframe([])
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), [])

    def test_single_visit_produces_correct_codes(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE])
        self.assertEqual(set(df["ICD9_CODE"].tolist()), {"250.00", "401.9"})

    def test_single_visit_single_hadm_id(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE])
        self.assertEqual(df["HADM_ID"].nunique(), 1)
        self.assertEqual(df["HADM_ID"].iloc[0], 0)

    def test_multi_visit_hadm_ids(self):
        df = sequences_to_dataframe([self._SEQ_MULTI])
        self.assertEqual(set(df["HADM_ID"].tolist()), {0, 1})

    def test_subject_ids_sequential(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE, self._SEQ_SINGLE])
        self.assertEqual(set(df["SUBJECT_ID"].tolist()), {0, 1})

    def test_multi_patient_subject_id_mapping(self):
        df = sequences_to_dataframe([self._SEQ_MULTI, self._SEQ_SINGLE])
        self.assertEqual(df[df["SUBJECT_ID"] == 0]["HADM_ID"].nunique(), 2)
        self.assertEqual(df[df["SUBJECT_ID"] == 1]["HADM_ID"].nunique(), 1)

    def test_row_count_matches_codes(self):
        df = sequences_to_dataframe([self._SEQ_MULTI])
        self.assertEqual(len(df), 4)

    def test_whitespace_only_sequence_returns_empty(self):
        df = sequences_to_dataframe(["   "])
        self.assertTrue(df.empty)

    def test_round_trip_from_samples(self):
        seqs = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        df = sequences_to_dataframe(seqs)
        all_codes = {c for visit in _MULTI_VISIT_SAMPLE["conditions"] for c in visit}
        self.assertEqual(all_codes, set(df["ICD9_CODE"].tolist()))

    def test_round_trip_visit_count(self):
        seqs = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        df = sequences_to_dataframe(seqs)
        n_visits = df.groupby("SUBJECT_ID")["HADM_ID"].nunique().iloc[0]
        self.assertEqual(n_visits, len(_MULTI_VISIT_SAMPLE["conditions"]))


# ── 3. build_tokenizer ────────────────────────────────────────────────────────


class TestBuildTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = build_tokenizer(_CORPUS)

    def test_special_tokens_in_vocab(self):
        vocab = self.tokenizer.get_vocab()
        for tok in ("[UNK]", "[PAD]", "[BOS]", "[EOS]"):
            self.assertIn(tok, vocab, f"{tok!r} missing from vocab")

    def test_visit_delim_in_vocab(self):
        self.assertIn(VISIT_DELIM, self.tokenizer.get_vocab())

    def test_medical_codes_in_vocab(self):
        # Whitespace pre-tokenizer splits "250.00" -> ["250", ".", "00"]
        vocab = self.tokenizer.get_vocab()
        for sub in ["250", "00", "401", "9", "272", "0", "428", "."]:
            self.assertIn(sub, vocab, f"sub-token {sub!r} missing from vocab")

    def test_vocab_size_at_least_corpus_tokens(self):
        self.assertGreaterEqual(len(self.tokenizer), 10)

    def test_bos_eos_token_ids_set(self):
        self.assertIsNotNone(self.tokenizer.bos_token_id)
        self.assertIsNotNone(self.tokenizer.eos_token_id)

    def test_pad_token_id_set(self):
        self.assertIsNotNone(self.tokenizer.pad_token_id)

    def test_encode_includes_bos_eos(self):
        ids = self.tokenizer("250.00 401.9")["input_ids"]
        self.assertEqual(ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_token_id)

    def test_encode_decode_roundtrip(self):
        text = "250.00 401.9 VISIT_DELIM 272.0"
        ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        # Whitespace splits on '.', so check sub-tokens
        for sub in ["250", "00", "401", "9", VISIT_DELIM, "272", "0"]:
            self.assertIn(sub, decoded.split(), f"{sub!r} missing from decoded")

    def test_unknown_token_maps_to_unk_id(self):
        enc = self.tokenizer("UNKNOWN_CODE_XYZ")["input_ids"]
        inner = enc[1:-1]  # strip BOS/EOS
        self.assertIn(self.tokenizer.unk_token_id, inner)

    def test_returns_pretrained_tokenizer_fast(self):
        self.assertIsInstance(self.tokenizer, PreTrainedTokenizerFast)


# ── 4. EHRTextDataset ─────────────────────────────────────────────────────────


class TestEHRTextDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = build_tokenizer(_SEQUENCES)
        cls.dataset = EHRTextDataset(_SEQUENCES, cls.tokenizer, max_length=_MAX_LENGTH)

    def test_len_matches_sequences(self):
        self.assertEqual(len(self.dataset), len(_SEQUENCES))

    def test_getitem_returns_dict(self):
        self.assertIsInstance(self.dataset[0], dict)

    def test_getitem_has_input_ids_key(self):
        self.assertIn("input_ids", self.dataset[0])

    def test_getitem_has_labels_key(self):
        self.assertIn("labels", self.dataset[0])

    def test_input_ids_are_tensors(self):
        self.assertIsInstance(self.dataset[0]["input_ids"], torch.Tensor)

    def test_labels_are_tensors(self):
        self.assertIsInstance(self.dataset[0]["labels"], torch.Tensor)

    def test_input_ids_length_equals_max_length(self):
        for i in range(len(self.dataset)):
            self.assertEqual(self.dataset[i]["input_ids"].shape[0], _MAX_LENGTH)

    def test_labels_equal_input_ids(self):
        item = self.dataset[0]
        self.assertTrue(torch.equal(item["input_ids"], item["labels"]))

    def test_all_items_same_length(self):
        lengths = {self.dataset[i]["input_ids"].shape[0] for i in range(len(self.dataset))}
        self.assertEqual(len(lengths), 1)

    def test_empty_sequences_list(self):
        ds = EHRTextDataset([], self.tokenizer, max_length=_MAX_LENGTH)
        self.assertEqual(len(ds), 0)

    def test_single_sequence(self):
        ds = EHRTextDataset(["250.00"], self.tokenizer, max_length=_MAX_LENGTH)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["input_ids"].shape[0], _MAX_LENGTH)

    def test_long_sequence_truncated(self):
        long_seq = " ".join(["250.00"] * 100)
        ds = EHRTextDataset([long_seq], self.tokenizer, max_length=_MAX_LENGTH)
        self.assertEqual(ds[0]["input_ids"].shape[0], _MAX_LENGTH)

    def test_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            _ = self.dataset[len(_SEQUENCES)]


if __name__ == "__main__":
    unittest.main()
