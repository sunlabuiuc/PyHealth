"""Ragged text tuples collate positionally like (time, values) pairs."""

from __future__ import annotations

import unittest

import torch


class TestP1TextTupleCollate(unittest.TestCase):
    """Ragged text tuples collate positionally like (time, values) pairs.

    Regression tests: tokenized note fields collate as one 5-tuple per
    sample ``(value, mask, token_type_ids, time, tag)`` with a ragged note
    count, which the old 2-tuple-only branch kept as a nested list. Every
    unified backbone then crashed in ``_build_unified_inputs`` indexing
    into per-sample tuples (``'tuple' object has no attribute 'to'``).
    """

    @staticmethod
    def _note(n, l, t0):
        return (
            torch.ones(n, l, dtype=torch.long),
            torch.ones(n, l, dtype=torch.long),
            torch.zeros(n, l, dtype=torch.long),
            torch.tensor([t0 + i for i in range(n)], dtype=torch.float),
            "note",
        )

    def test_ragged_text_tuples_stack_with_pad_mask(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

        batch = [
            {"notes": self._note(2, 4, 0.0)},
            {"notes": self._note(1, 4, 5.0)},
            {"notes": self._note(0, 1, 0.0)},
        ]
        collated = collate_fn_dict_with_padding(batch)
        value, mask, type_ids, time, tag = collated["notes"]
        self.assertEqual(tuple(value.shape), (3, 2, 4))
        self.assertEqual(tuple(mask.shape), (3, 2, 4))
        self.assertEqual(tuple(type_ids.shape), (3, 2, 4))
        self.assertEqual(tuple(time.shape), (3, 2))
        self.assertEqual(tag, ["note", "note", "note"])
        self.assertEqual(
            collated[f"notes{PAD_MASK_SUFFIX}"].tolist(),
            [[True, True], [True, False], [False, False]],
        )

    def test_two_tuple_contract_unchanged(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

        same = [
            {"x": (torch.tensor([1.0, 2.0]), torch.tensor([[1.0], [2.0]]))},
            {"x": (torch.tensor([3.0, 4.0]), torch.tensor([[3.0], [4.0]]))},
        ]
        collated = collate_fn_dict_with_padding(same)
        self.assertEqual(tuple(collated["x"][0].shape), (2, 2))
        self.assertEqual(tuple(collated["x"][1].shape), (2, 2, 1))
        self.assertNotIn(f"x{PAD_MASK_SUFFIX}", collated)

        ragged = [
            {"x": (torch.tensor([1.0, 2.0]), torch.tensor([[1.0], [2.0]]))},
            {"x": (torch.tensor([3.0]), torch.tensor([[3.0]]))},
        ]
        collated = collate_fn_dict_with_padding(ragged)
        self.assertEqual(tuple(collated["x"][1].shape), (2, 2, 1))
        self.assertEqual(
            collated[f"x{PAD_MASK_SUFFIX}"].tolist(), [[True, True], [True, False]]
        )


if __name__ == "__main__":
    unittest.main()
