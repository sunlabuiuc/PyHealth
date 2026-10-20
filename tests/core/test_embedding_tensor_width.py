"""EmbeddingModel must size tensor features without re-processing samples.

Samples read from a SampleDataset are already processed. Calling
``processor.process()`` on them again is wrong for any processor that is not
idempotent (one that selects columns, scales or imputes).
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import MLP
from pyhealth.processors import TensorProcessor


def _samples():
    return [
        {"patient_id": f"p{i}", "x": [float(i), 1.0, 2.0, 3.0], "label": i % 2}
        for i in range(6)
    ]


class FirstTwoColumns(TensorProcessor):
    """Keeps the first two columns: output width 2, not idempotent."""

    def process(self, value):
        tensor = super().process(value)
        if tensor.shape[-1] != 4:
            raise ValueError("process() called on an already-processed sample")
        return tensor[..., :2]

    def size(self):
        return 2


class TestTensorProcessorSize(unittest.TestCase):
    def test_fit_records_feature_width(self):
        p = TensorProcessor()
        p.fit(_samples(), "x")
        self.assertEqual(p.size(), 4)

    def test_size_is_none_before_fit(self):
        self.assertIsNone(TensorProcessor().size())

    def test_scalar_feature_has_width_one(self):
        p = TensorProcessor()
        p.fit([{"x": 3.0}], "x")
        self.assertEqual(p.size(), 1)


class TestEmbeddingTensorWidth(unittest.TestCase):
    def test_builds_without_reprocessing_samples(self):
        dataset = create_sample_dataset(
            samples=_samples(),
            input_schema={"x": FirstTwoColumns()},
            output_schema={"label": "binary"},
            dataset_name="test_embedding_tensor_width",
        )
        self.assertEqual(tuple(dataset[0]["x"].shape), (2,))
        model = MLP(dataset=dataset)  # raised ValueError before the fix
        layer = model.embedding_model.embedding_layers["x"]
        self.assertEqual(layer.in_features, 2)
        batch = {"x": torch.stack([dataset[i]["x"] for i in range(3)])}
        out = model.embedding_model(batch)
        self.assertEqual(out["x"].shape[0], 3)

    def test_plain_tensor_feature_width(self):
        dataset = create_sample_dataset(
            samples=_samples(),
            input_schema={"x": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_embedding_tensor_width_plain",
        )
        model = MLP(dataset=dataset)
        self.assertEqual(model.embedding_model.embedding_layers["x"].in_features, 4)


if __name__ == "__main__":
    unittest.main()
