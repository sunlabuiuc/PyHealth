"""Supplying some processors must still fit the others.

SampleBuilder used to fit input (or output) processors only when none were
supplied. Supplying a processor for one field left every other field of that
side without a processor, so its raw value (e.g. a Python list) reached the
model without any error.
"""

import pickle
import unittest

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.processors import (
    BinaryLabelProcessor,
    SequenceProcessor,
    TensorProcessor,
)


def _samples():
    return [
        {"patient_id": "p1", "codes": ["a", "b"], "age": [40.0], "label": 1},
        {"patient_id": "p2", "codes": ["c"], "age": [55.0], "label": 0},
    ]


class TestPartialProcessors(unittest.TestCase):
    def setUp(self):
        self.codes = SequenceProcessor()
        self.codes.fit(_samples(), "codes")

    def test_missing_input_processor_is_fitted(self):
        builder = SampleBuilder(
            input_schema={"codes": "sequence", "age": "tensor"},
            output_schema={"label": "binary"},
            input_processors={"codes": self.codes},
        )
        builder.fit(_samples())
        self.assertIs(builder.input_processors["codes"], self.codes)
        self.assertIsInstance(builder.input_processors["age"], TensorProcessor)
        out = builder.transform({"sample": pickle.dumps(_samples()[0])})
        self.assertIsInstance(out["age"], torch.Tensor)

    def test_supplied_processor_is_not_refitted(self):
        vocab = dict(self.codes.code_vocab)
        samples = _samples() + [
            {"patient_id": "p3", "codes": ["z"], "age": [60.0], "label": 1}
        ]
        builder = SampleBuilder(
            input_schema={"codes": "sequence", "age": "tensor"},
            output_schema={"label": "binary"},
            input_processors={"codes": self.codes},
        )
        builder.fit(samples)
        self.assertEqual(self.codes.code_vocab, vocab)  # "z" was not added

    def test_missing_output_processor_is_fitted(self):
        samples = [dict(s, label2=s["label"]) for s in _samples()]
        label = BinaryLabelProcessor()
        label.fit(samples, "label")
        builder = SampleBuilder(
            input_schema={"codes": "sequence"},
            output_schema={"label": "binary", "label2": "binary"},
            output_processors={"label": label},
        )
        builder.fit(samples)
        self.assertIs(builder.output_processors["label"], label)
        self.assertIn("label2", builder.output_processors)

    def test_create_sample_dataset_with_partial_processors(self):
        dataset = create_sample_dataset(
            samples=_samples(),
            input_schema={"codes": "sequence", "age": "tensor"},
            output_schema={"label": "binary"},
            input_processors={"codes": self.codes},
            dataset_name="test_partial_processors",
        )
        self.assertIsInstance(dataset[0]["age"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
