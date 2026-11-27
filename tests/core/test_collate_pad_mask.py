"""Collators emit {field}__pad_mask for legacy tuple fields."""

import unittest

import torch

from pyhealth.datasets.collate import collate_temporal
from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding


class TestCollatePadMask(unittest.TestCase):
    def test_temporal_collator_marks_time_padding(self):
        batch = [
            {"labs": {"value": torch.ones(3, 2), "time": torch.tensor([0.0, 1.0, 2.0])}},
            {"labs": {"value": torch.ones(1, 2), "time": torch.tensor([0.0])}},
        ]
        out = collate_temporal(batch)
        self.assertEqual(out["labs"]["value"].shape, (2, 3, 2))
        self.assertEqual(
            out["labs"]["pad_mask"].tolist(),
            [[True, True, True], [True, False, False]],
        )

    def test_dict_collator_emits_suffix_mask(self):
        batch = [
            {"notes": (torch.ones(3, 4, dtype=torch.long), torch.ones(3, 4))},
            {"notes": (torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        self.assertEqual(tuple(collated["notes"][1].shape), (2, 3, 4))
        self.assertEqual(
            collated[f"notes{PAD_MASK_SUFFIX}"].tolist(),
            [[True, True, True], [True, False, False]],
        )

    def test_equal_shapes_emit_no_mask(self):
        batch = [
            {"labs": (torch.tensor([0.0, 1.0]), torch.ones(2, 2))},
            {"labs": (torch.tensor([0.0, 2.0]), torch.ones(2, 2))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        self.assertNotIn(f"labs{PAD_MASK_SUFFIX}", collated)

    def test_collator_keys_skip_padding_and_mask(self):
        batch = [{"x": torch.tensor([1, 2])}, {"x": torch.tensor([3])}]
        out = collate_fn_dict_with_padding(batch, collators={"x": len})
        self.assertEqual(out["x"], 2)
        self.assertNotIn(f"x{PAD_MASK_SUFFIX}", out)


if __name__ == "__main__":
    unittest.main()
