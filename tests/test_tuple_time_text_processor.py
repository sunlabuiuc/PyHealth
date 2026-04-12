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


def test_tuple_time_text_processor_empty_input_fallback():
    """Empty or whitespace-only text lists should not crash processing."""
    processor = TupleTimeTextProcessor(type_tag="clinical_note")

    texts = ["   ", None, ""]
    time_diffs = [1.0, 2.0, 3.0]
    result_texts, time_tensor, tag = processor.process((texts, time_diffs))

    assert result_texts == ["[MISSING_TEXT]"]
    assert isinstance(time_tensor, torch.Tensor)
    assert time_tensor.shape == (1,)
    assert torch.equal(time_tensor, torch.tensor([0.0]))
    assert tag == "clinical_note"
