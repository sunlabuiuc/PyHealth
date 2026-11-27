"""Unit test for TupleTimeTextProcessor."""
import torch
from pyhealth.processors import TupleTimeTextProcessor


def test_tuple_time_text_processor():
    """Test TupleTimeTextProcessor basic functionality."""
    # Test default initialization
    processor = TupleTimeTextProcessor()
    assert processor.type_tag == "note"
    
    # Test custom type tag
    processor = TupleTimeTextProcessor(type_tag="clinical_note")
    assert processor.type_tag == "clinical_note"
    
    # Test processing
    texts = ["Patient admitted", "Follow-up visit", "Discharge"]
    time_diffs = [0.0, 24.0, 72.0]
    result_texts, time_tensor, tag = processor.process((texts, time_diffs))
    
    # Verify outputs
    assert result_texts == texts
    assert isinstance(time_tensor, torch.Tensor)
    assert time_tensor.shape == (3,)
    assert torch.equal(time_tensor, torch.tensor([0.0, 24.0, 72.0]))
    assert tag == "clinical_note"
    
    # Test registration
    from pyhealth.processors import get_processor
    ProcessorClass = get_processor("tuple_time_text")
    assert ProcessorClass is TupleTimeTextProcessor


def test_tuple_time_text_cleans_malformed_entries():
    """Null, blank, and misaligned entries are dropped with their times."""
    from pyhealth.processors import TupleTimeTextProcessor

    processor = TupleTimeTextProcessor()
    texts = ["  real note  ", None, "   ", "second note"]
    time_diffs = [0.0, 1.0, 2.0, 3.0]
    result_texts, time_tensor, tag = processor.process((texts, time_diffs))

    assert result_texts == ["real note", "second note"]
    assert torch.equal(time_tensor, torch.tensor([0.0, 3.0]))
    assert tag == "note"


def test_tuple_time_text_empty_is_zero_events_with_stub_tokenizer():
    """Tokenizer path returns (0, 1) empties instead of a fake row."""
    from pyhealth.processors import TupleTimeTextProcessor

    processor = TupleTimeTextProcessor()
    calls = []

    def fake_tokenizer(texts, **kwargs):
        calls.append((list(texts), kwargs.get("padding"), kwargs.get("max_length")))
        raise AssertionError("tokenizer must not run on zero notes")

    processor.tokenizer = fake_tokenizer
    input_ids, attn, token_types, time_tensor, tag = processor.process(([], []))

    assert calls == []
    assert input_ids.shape == (0, 1)
    assert attn.shape == (0, 1)
    assert token_types.shape == (0, 1)
    assert time_tensor.shape == (0,)
    assert tag == "note"


def test_tuple_time_text_longest_padding_bounded_by_max_length():
    """Padding follows the longest note in the sample, capped at max_length."""
    from pyhealth.processors import TupleTimeTextProcessor

    processor = TupleTimeTextProcessor(max_length=16)
    seen = {}

    def fake_tokenizer(texts, **kwargs):
        seen["padding"] = kwargs.get("padding")
        seen["max_length"] = kwargs.get("max_length")
        width = max(len(t.split()) + 2 for t in texts)
        n = len(texts)
        return {
            "input_ids": torch.ones((n, width), dtype=torch.long),
            "attention_mask": torch.ones((n, width), dtype=torch.long),
        }

    processor.tokenizer = fake_tokenizer
    input_ids, attn, token_types, time_tensor, tag = processor.process(
        (["hi", "a much longer note here"], [0.0, 1.0])
    )

    assert seen["padding"] == "longest"
    assert seen["max_length"] == 16
    assert input_ids.shape == (2, 7)
    assert attn.shape == (2, 7)
    assert token_types.shape == (2, 7)
    assert torch.equal(time_tensor, torch.tensor([0.0, 1.0]))
