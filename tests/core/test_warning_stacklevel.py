"""Warnings point at the user's code, not at PyHealth or PyTorch internals.

Each test triggers a warning the way a user would and checks that the
warning's file is this test file.
"""

import os
import unittest
import warnings
from unittest import mock

import torch

from pyhealth.data import Visit
from pyhealth.datasets import (
    BaseEHRDataset,
    BaseSignalDataset,
    MIMIC3Dataset,
    SampleEHRDataset,
    SampleSignalDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.mimic4 import MIMIC4NoteDataset
from pyhealth.models import ConCare, StageAttentionNet, StageNet
from pyhealth.models.text_embedding import TextEmbedding

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "test-resources", "core")


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return caught


class TestWarningLocation(unittest.TestCase):
    def assert_points_here(self, caught, text):
        matches = [w for w in caught if text in str(w.message)]
        self.assertTrue(matches, f"no warning containing {text!r}")
        for w in matches:
            self.assertEqual(os.path.abspath(w.filename), os.path.abspath(__file__))

    def test_deprecated_stubs(self):
        for cls in (Visit, BaseEHRDataset, BaseSignalDataset, SampleEHRDataset,
                    SampleSignalDataset):
            with self.subTest(cls=cls.__name__):
                caught = _caught(cls)
                self.assert_points_here(caught, "deprecated")
                self.assertIs(caught[0].category, DeprecationWarning)

    def test_dataset_constructor_warnings(self):
        # the base constructor would build the event table; only the warning
        # raised before it is under test
        with mock.patch.object(BaseDataset, "__init__", return_value=None):
            caught = _caught(lambda: MIMIC3Dataset(
                root=os.path.join(ROOT, "mimic3demo"), tables=["prescriptions"]))
            self.assert_points_here(caught, "prescriptions table")
            caught = _caught(lambda: MIMIC4NoteDataset(
                root=ROOT, tables=["discharge", "discharge_detail"]))
            self.assert_points_here(caught, "discharge table")
            self.assert_points_here(caught, "discharge_detail table")

    def test_model_constructor_warning(self):
        samples = [
            {"patient_id": f"p{i}", "visit_id": f"v{i}",
             "conditions": [["a", "b"], ["c"]], "label": i % 2}
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples, input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"}, dataset_name="warn_concare",
        )
        caught = _caught(lambda: ConCare(dataset=dataset, embedding_dim=8,
                                         hidden_dim=8))
        self.assert_points_here(caught, "uses nested_sequence")

    def test_forward_warnings(self):
        # raised inside forward(), reached through nn.Module.__call__
        samples = [
            {"patient_id": f"p{i}", "visit_id": f"v{i}",
             "codes": ([0.0, 1.0], ["c1", "c2"]),
             "lab_values": (None, [[1.0, 2.0], [3.0, 4.0]]),
             "label": i % 2}
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "stagenet", "lab_values": "stagenet_tensor"},
            output_schema={"label": "binary"}, dataset_name="warn_stagenet",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        for cls in (StageNet, StageAttentionNet):
            with self.subTest(model=cls.__name__):
                model = cls(dataset=dataset)
                caught = _caught(lambda model=model: model(**batch))
                self.assert_points_here(caught, "does not have time")

    def test_text_embedding_truncation(self):
        class Stop(Exception):
            pass

        class Tokenizer:
            cls_token_id, sep_token_id, pad_token_id = 1, 2, 0

            def __call__(self, text, **kwargs):
                return {"input_ids": list(range(3, 53))}

        def transformer(**kwargs):
            raise Stop  # the encoding after the warning is not under test

        model = TextEmbedding.__new__(TextEmbedding)
        torch.nn.Module.__init__(model)
        model.tokenizer, model.transformer = Tokenizer(), transformer
        model.chunk_size, model.max_chunks = 10, 2

        def encode():
            with self.assertRaises(Stop):
                model._chunk_and_encode("long text", torch.device("cpu"))

        self.assert_points_here(_caught(encode), "truncated to 2")


if __name__ == "__main__":
    unittest.main()
