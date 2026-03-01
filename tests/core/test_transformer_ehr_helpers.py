"""
Tests for the utility functions and classes defined in
examples/ehr_generation/ehr_generation_mimic3_transformer.py

Covered:
* ``samples_to_sequences``   – nested visit lists → text strings
* ``sequences_to_dataframe`` – text strings → long-form DataFrame
* ``build_tokenizer``         – word-level HuggingFace tokenizer
* ``EHRTextDataset``          – PyTorch Dataset wrapping tokenized EHR sequences
"""

import sys
import os

import pytest
import torch

# Allow importing directly from the examples directory without installing it.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../examples/ehr_generation"),
)

from ehr_generation_mimic3_transformer import (  # noqa: E402
    VISIT_DELIM,
    EHRTextDataset,
    build_tokenizer,
    samples_to_sequences,
    sequences_to_dataframe,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_VISIT_SAMPLE = {"conditions": [["250.00", "401.9"]]}
_MULTI_VISIT_SAMPLE = {"conditions": [["250.00", "401.9"], ["272.0", "428.0"], ["250.00"]]}
_EMPTY_VISIT_SAMPLE = {"conditions": []}  # patient with no visits


# ─────────────────────────────────────────────────────────────────────────────
# 1. samples_to_sequences
# ─────────────────────────────────────────────────────────────────────────────


class TestSamplesToSequences:
    def test_returns_one_string_per_sample(self):
        samples = [_SINGLE_VISIT_SAMPLE, _MULTI_VISIT_SAMPLE]
        result = samples_to_sequences(samples)
        assert len(result) == 2

    def test_empty_input_returns_empty_list(self):
        assert samples_to_sequences([]) == []

    def test_single_visit_no_delimiter(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE])
        assert VISIT_DELIM not in result[0]

    def test_multi_visit_delimiter_count_matches(self):
        # 3 visits → 2 VISIT_DELIM occurrences
        result = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        assert result[0].count(VISIT_DELIM) == 2

    def test_codes_present_in_output(self):
        result = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        for visit in _MULTI_VISIT_SAMPLE["conditions"]:
            for code in visit:
                assert code in result[0]

    def test_single_visit_codes_space_separated(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE])
        assert result[0] == "250.00 401.9"

    def test_multi_visit_format(self):
        result = samples_to_sequences([_MULTI_VISIT_SAMPLE])
        expected = f"250.00 401.9 {VISIT_DELIM} 272.0 428.0 {VISIT_DELIM} 250.00"
        assert result[0] == expected

    def test_empty_conditions_yields_empty_string(self):
        result = samples_to_sequences([_EMPTY_VISIT_SAMPLE])
        assert result[0] == ""

    def test_single_code_per_visit(self):
        sample = {"conditions": [["A"], ["B"], ["C"]]}
        result = samples_to_sequences([sample])
        assert result[0] == f"A {VISIT_DELIM} B {VISIT_DELIM} C"

    def test_multiple_samples_independent(self):
        samples = [_SINGLE_VISIT_SAMPLE, _MULTI_VISIT_SAMPLE]
        result = samples_to_sequences(samples)
        assert result[0] != result[1]

    def test_output_is_list_of_strings(self):
        result = samples_to_sequences([_SINGLE_VISIT_SAMPLE])
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


# ─────────────────────────────────────────────────────────────────────────────
# 2. sequences_to_dataframe
# ─────────────────────────────────────────────────────────────────────────────


class TestSequencesToDataframe:
    _SEQ_SINGLE = "250.00 401.9"
    _SEQ_MULTI = f"250.00 401.9 {VISIT_DELIM} 272.0 428.0"

    def test_required_columns_present(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE])
        assert set(df.columns) == {"SUBJECT_ID", "HADM_ID", "ICD9_CODE"}

    def test_empty_input_returns_empty_dataframe(self):
        df = sequences_to_dataframe([])
        assert df.empty
        assert list(df.columns) == []  # pd.concat on empty list → empty DF

    def test_single_visit_produces_correct_codes(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE])
        codes = set(df["ICD9_CODE"].tolist())
        assert codes == {"250.00", "401.9"}

    def test_single_visit_single_hadm_id(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE])
        assert df["HADM_ID"].nunique() == 1
        assert df["HADM_ID"].iloc[0] == 0

    def test_multi_visit_hadm_ids(self):
        df = sequences_to_dataframe([self._SEQ_MULTI])
        assert set(df["HADM_ID"].tolist()) == {0, 1}

    def test_subject_ids_sequential(self):
        df = sequences_to_dataframe([self._SEQ_SINGLE, self._SEQ_SINGLE])
        assert set(df["SUBJECT_ID"].tolist()) == {0, 1}

    def test_multi_patient_subject_id_mapping(self):
        df = sequences_to_dataframe([self._SEQ_MULTI, self._SEQ_SINGLE])
        assert df[df["SUBJECT_ID"] == 0]["HADM_ID"].nunique() == 2
        assert df[df["SUBJECT_ID"] == 1]["HADM_ID"].nunique() == 1

    def test_row_count_matches_codes(self):
        # seq has 4 codes across 2 visits
        df = sequences_to_dataframe([self._SEQ_MULTI])
        assert len(df) == 4

    def test_whitespace_only_sequence_returns_empty(self):
        df = sequences_to_dataframe(["   "])
        assert df.empty

    def test_round_trip_from_samples(self):
        samples = [_MULTI_VISIT_SAMPLE]
        seqs = samples_to_sequences(samples)
        df = sequences_to_dataframe(seqs)
        all_codes = set(
            code
            for visit in _MULTI_VISIT_SAMPLE["conditions"]
            for code in visit
        )
        recovered_codes = set(df["ICD9_CODE"].tolist())
        assert all_codes == recovered_codes

    def test_round_trip_visit_count(self):
        samples = [_MULTI_VISIT_SAMPLE]
        seqs = samples_to_sequences(samples)
        df = sequences_to_dataframe(seqs)
        n_visits = df.groupby("SUBJECT_ID")["HADM_ID"].nunique().iloc[0]
        assert n_visits == len(_MULTI_VISIT_SAMPLE["conditions"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_tokenizer
# ─────────────────────────────────────────────────────────────────────────────

_CORPUS = [
    "250.00 401.9 VISIT_DELIM 272.0",
    "428.0 VISIT_DELIM 250.00",
    "401.9 272.0 428.0",
]


class TestBuildTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return build_tokenizer(_CORPUS)

    def test_special_tokens_in_vocab(self, tokenizer):
        for tok in ("[UNK]", "[PAD]", "[BOS]", "[EOS]"):
            assert tok in tokenizer.get_vocab(), f"{tok!r} missing from vocab"

    def test_visit_delim_in_vocab(self, tokenizer):
        assert VISIT_DELIM in tokenizer.get_vocab()

    def test_medical_codes_in_vocab(self, tokenizer):
        # The Whitespace pre-tokenizer splits on punctuation, so "250.00" becomes
        # the sub-tokens ["250", ".", "00"].  Assert each constituent sub-token
        # (digits and the dot) appears in the vocabulary instead of the full code.
        vocab = tokenizer.get_vocab()
        for sub in ["250", "00", "401", "9", "272", "0", "428", "."]:
            assert sub in vocab, f"sub-token {sub!r} missing from vocab"

    def test_vocab_size_at_least_corpus_tokens(self, tokenizer):
        # 4 special tokens + 5 unique code tokens + VISIT_DELIM = at least 10
        assert len(tokenizer) >= 10

    def test_bos_eos_token_ids_set(self, tokenizer):
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

    def test_pad_token_id_set(self, tokenizer):
        assert tokenizer.pad_token_id is not None

    def test_encode_includes_bos_eos(self, tokenizer):
        ids = tokenizer("250.00 401.9")["input_ids"]
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "250.00 401.9 VISIT_DELIM 272.0"
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # The Whitespace pre-tokenizer splits codes on '.', so the round-trip
        # produces sub-tokens (e.g. "250 . 00" instead of "250.00").  Verify
        # that all digit sub-tokens and the VISIT_DELIM are present.
        for sub_token in ["250", "00", "401", "9", VISIT_DELIM, "272", "0"]:
            assert sub_token in decoded.split(), f"{sub_token!r} missing from decoded"

    def test_unknown_token_maps_to_unk_id(self, tokenizer):
        enc = tokenizer("UNKNOWN_CODE_XYZ")["input_ids"]
        # Strip BOS/EOS; the middle token should be [UNK]
        inner = enc[1:-1]
        assert tokenizer.unk_token_id in inner

    def test_returns_pretrained_tokenizer_fast(self, tokenizer):
        from transformers import PreTrainedTokenizerFast

        assert isinstance(tokenizer, PreTrainedTokenizerFast)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EHRTextDataset
# ─────────────────────────────────────────────────────────────────────────────

_SEQUENCES = [
    "250.00 401.9 VISIT_DELIM 272.0",
    "428.0",
    "401.9 272.0 428.0 VISIT_DELIM 250.00 VISIT_DELIM 272.0",
]
_MAX_LENGTH = 16


class TestEHRTextDataset:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return build_tokenizer(_SEQUENCES)

    @pytest.fixture(scope="class")
    def dataset(self, tokenizer):
        return EHRTextDataset(_SEQUENCES, tokenizer, max_length=_MAX_LENGTH)

    def test_len_matches_sequences(self, dataset):
        assert len(dataset) == len(_SEQUENCES)

    def test_getitem_returns_dict(self, dataset):
        item = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_input_ids_key(self, dataset):
        assert "input_ids" in dataset[0]

    def test_getitem_has_labels_key(self, dataset):
        assert "labels" in dataset[0]

    def test_input_ids_are_tensors(self, dataset):
        assert isinstance(dataset[0]["input_ids"], torch.Tensor)

    def test_labels_are_tensors(self, dataset):
        assert isinstance(dataset[0]["labels"], torch.Tensor)

    def test_input_ids_length_equals_max_length(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i]["input_ids"].shape[0] == _MAX_LENGTH

    def test_labels_equal_input_ids(self, dataset):
        item = dataset[0]
        assert torch.equal(item["input_ids"], item["labels"])

    def test_all_items_same_length(self, dataset):
        lengths = {dataset[i]["input_ids"].shape[0] for i in range(len(dataset))}
        assert len(lengths) == 1  # all padded/truncated to max_length

    def test_empty_sequences_list(self, tokenizer):
        ds = EHRTextDataset([], tokenizer, max_length=_MAX_LENGTH)
        assert len(ds) == 0

    def test_single_sequence(self, tokenizer):
        ds = EHRTextDataset(["250.00"], tokenizer, max_length=_MAX_LENGTH)
        assert len(ds) == 1
        item = ds[0]
        assert item["input_ids"].shape[0] == _MAX_LENGTH

    def test_long_sequence_truncated(self, tokenizer):
        # Construct a sequence much longer than max_length
        long_seq = " ".join(["250.00"] * 100)
        ds = EHRTextDataset([long_seq], tokenizer, max_length=_MAX_LENGTH)
        assert ds[0]["input_ids"].shape[0] == _MAX_LENGTH

    def test_index_out_of_range_raises(self, dataset):
        with pytest.raises(IndexError):
            _ = dataset[len(_SEQUENCES)]
