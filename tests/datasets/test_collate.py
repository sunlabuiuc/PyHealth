"""Tests for the collate_temporal function."""
import torch
from torch.nn.utils.rnn import pad_sequence
from pyhealth.datasets.collate import collate_temporal


def test_collate_temporal_tuple_handling():
    """Test that collate_temporal correctly handles tuple outputs from processors like TupleTimeTextProcessor."""
    # Simulate a batch of two samples where a field returns a tuple (text_list, time_tensor, tag)
    # This mimics the output of TupleTimeTextProcessor without tokenizer.
    batch = [
        {
            "patient_id": ["p1", "p2"],
            "notes": (["Note 1", "Note 2"], torch.tensor([0.0, 10.0]), "note"),
            "label": 0,
        },
        {
            "patient_id": ["p3"],
            "notes": (["Note 3"], torch.tensor([5.0]), "note"),
            "label": 1,
        },
    ]

    collated = collate_temporal(batch)

    # Check that the notes field is a tuple of three elements: padded text list, padded time tensor, and list of tags
    assert isinstance(collated["notes"], tuple)
    assert len(collated["notes"]) == 3

    # First element: list of text lists (should not be converted to tensor)
    assert collated["notes"][0] == [["Note 1", "Note 2"], ["Note 3"]]

    # Second element: time tensor padded to longest sequence
    expected_times = pad_sequence(
        [torch.tensor([0.0, 10.0]), torch.tensor([5.0])], batch_first=True, padding_value=0.0
    )
    assert torch.equal(collated["notes"][1], expected_times)

    # Third element: list of tags
    assert collated["notes"][2] == ["note", "note"]


def test_collate_temporal_tuple_with_tokenizer():
    """Test collation when tuple contains tensors (tokenized text output)."""
    # Simulate tokenized output: (input_ids, attention_mask, token_type_ids, time_tensor, tag)
    batch = [
        {
            "patient_id": ["p1"],
            "notes": (
                torch.tensor([[101, 2023, 2003, 102], [101, 2003, 2023, 102]]),  # input_ids (2,4)
                torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),                   # attention_mask (2,4)
                torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]),                   # token_type_ids (2,4)
                torch.tensor([0.0, 10.0]),                                    # time (2,)
                "note",
            ),
            "label": 0,
        },
        {
            "patient_id": ["p2"],
            "notes": (
                torch.tensor([[101, 2023, 102, 0], [0, 0, 0, 0]]),            # input_ids (2,4) - padded
                torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]]),                 # attention_mask (2,4)
                torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]),                 # token_type_ids (2,4)
                torch.tensor([5.0, 0.0]),                                     # time (2,)
                "note",
            ),
            "label": 1,
        },
    ]

    collated = collate_temporal(batch)

    assert isinstance(collated["notes"], tuple)
    assert len(collated["notes"]) == 5

    # input_ids should be stacked (both sequences now length 4)
    expected_input_ids = torch.stack([
        torch.tensor([[101, 2023, 2003, 102], [101, 2003, 2023, 102]]),  # (2,4)
        torch.tensor([[101, 2023, 102, 0], [0, 0, 0, 0]])                 # (2,4) padded
    ])
    assert torch.equal(collated["notes"][0], expected_input_ids)

    # attention_mask similarly stacked
    expected_attention_mask = torch.stack([
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),  # (2,4)
        torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]])    # (2,4) padded
    ])
    assert torch.equal(collated["notes"][1], expected_attention_mask)

    # token_type_ids stacked
    expected_token_type_ids = torch.stack([
        torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]),  # (2,4)
        torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])    # (2,4) padded
    ])
    assert torch.equal(collated["notes"][2], expected_token_type_ids)

    # time tensor stacked
    expected_time = torch.stack([
        torch.tensor([0.0, 10.0]),    # (2,)
        torch.tensor([5.0, 0.0])      # (2,) padded
    ])
    assert torch.equal(collated["notes"][3], expected_time)

    # tags
    assert collated["notes"][4] == ["note", "note"]


def test_collate_temporal_mixed():
    """Test collation with a mix of tensor, tuple, and scalar fields."""
    batch = [
        {
            "id": 1,
            "value": torch.tensor([1.0, 2.0]),
            "meta": (["a", "b"], torch.tensor([0.0, 1.0])),
        },
        {
            "id": 2,
            "value": torch.tensor([3.0]),
            "meta": (["c"], torch.tensor([2.0])),
        },
    ]

    collated = collate_temporal(batch)

    assert torch.equal(collated["id"], torch.tensor([1, 2]))
    # value tensor padded to length 2
    assert torch.equal(collated["value"], torch.tensor([[1.0, 2.0], [3.0, 0.0]]))
    # meta tuple: first element list of lists, second element padded time tensor
    assert collated["meta"][0] == [["a", "b"], ["c"]]
    assert torch.equal(
        collated["meta"][1],
        pad_sequence([torch.tensor([0.0, 1.0]), torch.tensor([2.0])], batch_first=True, padding_value=0.0),
    )