import unittest

import torch

from pyhealth.datasets.utils import collate_fn_dict_with_padding


class TestCollateFnDictWithPadding(unittest.TestCase):
    def test_collators_take_only_their_keys(self):
        batch = [
            {"x": (torch.tensor([1.0]), torch.tensor([0, 1])), "y": torch.tensor([1, 2]), "id": "a"},
            {"x": (torch.tensor([2.0, 3.0]), torch.tensor([0, 2])), "y": torch.tensor([3]), "id": "b"},
        ]
        calls = []

        def collate_x(items):
            calls.append(items)
            return "collated"

        collated = collate_fn_dict_with_padding(batch, collators={"x": collate_x})
        self.assertEqual(collated["x"], "collated")
        self.assertEqual(calls, [[batch[0]["x"], batch[1]["x"]]])
        # Keys without a collator are padded or listed as before.
        self.assertEqual(collated["y"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(collated["id"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
