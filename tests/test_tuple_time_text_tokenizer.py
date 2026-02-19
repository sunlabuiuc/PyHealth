"""Unit test for TupleTimeTextProcessor with Tokenizer support."""
import pytest
import torch
import shutil
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pyhealth.processors import TupleTimeTextProcessor

# Check if transformers is installed
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _collate_smart_processor(batch):
    """Collate function that handles the 5-element tuple from tokenized TupleTimeTextProcessor."""
    result = {}
    for key in batch[0].keys():
        vals = [s[key] for s in batch]
        if isinstance(vals[0], tuple):
            collated = []
            for elem_vals in zip(*vals):
                if isinstance(elem_vals[0], torch.Tensor):
                    if all(e.shape == elem_vals[0].shape for e in elem_vals):
                        collated.append(torch.stack(list(elem_vals)))
                    else:
                        collated.append(pad_sequence(list(elem_vals), batch_first=True))
                else:
                    collated.append(list(elem_vals))
            result[key] = tuple(collated)
        elif isinstance(vals[0], torch.Tensor):
            if all(v.shape == vals[0].shape for v in vals):
                result[key] = torch.stack(vals)
            else:
                result[key] = pad_sequence(vals, batch_first=True)
        else:
            result[key] = vals
    return result

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
    """Smoke test: fit() and process() don't raise for the tokenized path."""
    proc = TupleTimeTextProcessor(tokenizer_model="prajjwal1/bert-tiny")
    proc.fit([{}], "dummy_field")   # fit is a no-op for this processor
    result = proc.process((["Hello"], [0.0]))
    assert len(result) == 5        # (input_ids, mask, type_ids, time, tag)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not installed")
def test_tuple_in_schema_canonical_form():
    """Canonical usage: declare processor via ('tuple_time_text', kwargs) in input_schema.

    Users should never need to import TupleTimeTextProcessor directly; the
    registry alias + kwargs tuple is the recommended pattern:

        input_schema = {
            "conditions": "sequence",
            "notes": ("tuple_time_text", {"tokenizer_model": "bert-base-uncased"}),
        }
    """
    from pyhealth.datasets import create_sample_dataset
    from pyhealth.models import MLP

    samples = [
        {
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "conditions": ["I21.0", "I10"] if i % 2 == 0 else ["R07.9", "I50.9"],
            "notes": (
                ["Chest pain on exertion", "Follow-up stable"],
                [0.0, 48.0],
            ),
            "label": i % 2,
        }
        for i in range(4)
    ]

    # ── Canonical input_schema form ────────────────────────────────────
    input_schema = {
        "conditions": "sequence",
        "notes": (
            "tuple_time_text",
            {"tokenizer_model": "prajjwal1/bert-tiny", "max_length": 16},
        ),
    }

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema={"label": "binary"},
        in_memory=True,
    )

    # Processor was instantiated from registry — verify type and config
    notes_proc = dataset.input_processors["notes"]
    assert isinstance(notes_proc, TupleTimeTextProcessor)
    assert notes_proc.is_token() is True
    assert notes_proc.tokenizer_model == "prajjwal1/bert-tiny"

    # One sample: notes should be the 5-element tuple
    sample = dataset[0]
    assert isinstance(sample["notes"], tuple)
    assert len(sample["notes"]) == 5
    input_ids, mask, type_ids, time, tag = sample["notes"]
    assert input_ids.shape == (2, 16)   # (N_notes, L)
    assert mask.shape == (2, 16)
    assert time.shape == (2,)
    assert tag == "note"

    # Forward pass through MLP using collate helper
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False,
        collate_fn=_collate_smart_processor,
    )
    batch = next(iter(loader))
    model = MLP(dataset, embedding_dim=64, hidden_dim=64)
    model.eval()
    import torch
    with torch.no_grad():
        out = model(**batch)
    assert "loss" in out
    assert out["y_prob"].shape == (2, 1)

