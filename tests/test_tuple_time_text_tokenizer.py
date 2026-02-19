"""Unit test for TupleTimeTextProcessor with Tokenizer support."""
import pytest
import torch
import shutil
from pyhealth.processors import TupleTimeTextProcessor

# Check if transformers is installed
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def test_tuple_time_text_processor_no_tokenizer():
    """Test TupleTimeTextProcessor without tokenizer (backward compatibility)."""
    processor = TupleTimeTextProcessor()
    assert processor.type_tag == "note"
    assert processor.tokenizer is None
    
    texts = ["Note 1", "Note 2"]
    time_diffs = [0.0, 24.0]
    
    result = processor.process((texts, time_diffs))
    assert len(result) == 3
    assert result[0] == texts
    assert torch.equal(result[1], torch.tensor([0.0, 24.0]))
    assert result[2] == "note"
    
    assert processor.schema() == ("text", "time", "type_tag")
    assert processor.dim() == (0, 1, 0)
    assert processor.is_token() is False
    assert "tokenizer" not in str(processor)

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not installed")
def test_tuple_time_text_processor_with_tokenizer():
    """Test TupleTimeTextProcessor with a real tokenizer."""
    # Use a small model for testing
    model_name = "prajjwal1/bert-tiny"
    
    processor = TupleTimeTextProcessor(
        tokenizer_model=model_name,
        max_length=16,
        padding=True,
        truncation=True
    )
    
    assert processor.tokenizer is not None
    assert processor.is_token() is True
    
    texts = ["Hello world", "Test sentence"]
    time_diffs = [0.0, 10.0]
    
    # Run processing
    result = processor.process((texts, time_diffs))
    
    # Expect 5 outputs: input_ids, attention_mask, token_type_ids, time, tag
    assert len(result) == 5
    
    input_ids, attention_mask, token_type_ids, time_tensor, tag = result
    
    # Check shapes
    N = 2 # number of sentences
    L = 16 # max_length
    
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (N, L)
    
    assert isinstance(attention_mask, torch.Tensor)
    assert attention_mask.shape == (N, L)
    
    assert isinstance(token_type_ids, torch.Tensor)
    assert token_type_ids.shape == (N, L)
    
    assert isinstance(time_tensor, torch.Tensor)
    assert time_tensor.shape == (N,)
    
    assert tag == "note"
    
    # Check schema
    assert processor.schema() == ("value", "mask", "token_type_ids", "time", "type_tag")
    # Check dims (input_ids: 2D, attention_mask: 2D, token_type_ids: 2D, time: 1D)
    assert processor.dim() == (2, 2, 2, 1)
    
    assert f"tokenizer='{model_name}'" in str(processor)

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not installed")
def test_tokenizer_integration_in_pyhealth_workflow():
    """Simulate how this would work in a PyHealth task flow."""
    # This just safeguards against basic runtime errors in likely usage
    try:
        proc = TupleTimeTextProcessor(tokenizer_model="prajjwal1/bert-tiny")
        # Simulate PyHealth Dataset loading (fit not really used for this proc, but good to check)
        # fit is inherited from FeatureProcessor and just passes
        proc.fit([{}], "dummy_field") 
    except Exception as e:
        pytest.fail(f"Integration smoke test failed: {e}")
