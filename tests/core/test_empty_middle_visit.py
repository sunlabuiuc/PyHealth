"""A visit with no codes in the middle of a record must not hide later visits.

Sequence models used to take a record's length as its number of non-empty
visits. An empty visit in the middle then cut the record short, and the
most recent visit was ignored.
"""

import unittest
import warnings

import torch

import pyhealth.models as models
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.utils import get_last_visit, mask_lengths

# The two records differ only in their last visit, after an empty one.
SAMPLES = [
    {"patient_id": "p0", "label": 0, "visits": [["a"], [], ["b"]]},
    {"patient_id": "p1", "label": 1, "visits": [["a"], [], ["c"]]},
    {"patient_id": "p2", "label": 1, "visits": [["a"], ["d"], ["c"]]},
]
MODELS = ["RNN", "MultimodalRNN", "RETAIN", "MultimodalRETAIN", "TCN",
          "StageNet", "Agent"]


class TestMaskLengths(unittest.TestCase):
    def test_length_runs_to_the_last_valid_step(self):
        mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(mask_lengths(mask).tolist(), [3, 2, 0])

    def test_last_visit_skips_only_trailing_padding(self):
        hidden = torch.arange(12.0).reshape(2, 3, 2)
        mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
        last = get_last_visit(hidden, mask)
        self.assertEqual(last.tolist(), [[4.0, 5.0], [8.0, 9.0]])


class TestEmptyMiddleVisit(unittest.TestCase):
    def test_the_last_visit_changes_the_prediction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = create_sample_dataset(
                SAMPLES, {"visits": "nested_sequence"}, {"label": "binary"},
                dataset_name="empty_middle_visit",
            )
            batch = next(iter(get_dataloader(dataset, batch_size=3)))
            for name in MODELS:
                with self.subTest(model=name):
                    torch.manual_seed(0)
                    model = getattr(models, name)(dataset=dataset).eval()
                    with torch.no_grad():
                        prob = model(**batch)["y_prob"][:, 0]
                    self.assertFalse(
                        torch.isclose(prob[0], prob[1]),
                        f"{name} ignores the visit after an empty one",
                    )


if __name__ == "__main__":
    unittest.main()
