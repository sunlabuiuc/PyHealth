import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets.sample_dataset import create_sample_dataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    RawProcessor,
    RegressionLabelProcessor,
)


class DummyBinaryModel(BaseModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.fc = nn.Linear(8, 1)

    def forward(self, x=None, **kwargs):  # x unused; focus on loss/mode logic
        batch = len(kwargs.get("y", [])) if "y" in kwargs else 4
        logits = self.fc(torch.randn(batch, 8, device=self.device))
        y_true = torch.randint(
            0, 2, (batch, 1), dtype=torch.float32, device=self.device
        )
        loss = self.label_processor.loss(logits, y_true)
        y_prob = self.label_processor.activation(logits)
        return {"loss": loss, "y_true": y_true, "y_prob": y_prob}


class TestLegacyModeResolution(unittest.TestCase):
    """Every output_schema form gives the label processor its loss comes from."""

    def _build_dataset(self, output_processor, key="label"):
        samples = [
            {key: 0, "text": "a"},
            {key: 1, "text": "b"},
        ]
        input_schema = {"text": "raw"}
        output_schema = {key: output_processor}
        return create_sample_dataset(samples, input_schema, output_schema)

    def _assert_loss(self, model, logits, y_true, expected_fn):
        torch.testing.assert_close(
            model.label_processor.loss(logits, y_true), expected_fn(logits, y_true)
        )

    def test_string_schema_sets_mode(self):
        model = DummyBinaryModel(self._build_dataset("binary"))
        self.assertEqual(model.mode, "binary")
        logits, y = torch.randn(4, 1), torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        self._assert_loss(model, logits, y, F.binary_cross_entropy_with_logits)

    def test_processor_class_schema_sets_mode(self):
        model = DummyBinaryModel(self._build_dataset(BinaryLabelProcessor))
        self.assertEqual(model.mode, "binary")
        self.assertIsInstance(model.label_processor, BinaryLabelProcessor)

    def test_unregistered_processor_leaves_mode_none(self):
        model = DummyBinaryModel(self._build_dataset(RawProcessor))
        self.assertIsNone(model.mode)
        with self.assertRaises(ValueError):
            model.label_processor

    def test_multiclass_loss_selection(self):
        samples = [{"label": i % 3, "text": f"row{i}"} for i in range(6)]
        ds = create_sample_dataset(
            samples, {"text": "raw"}, {"label": MultiClassLabelProcessor}
        )
        model = BaseModel(dataset=ds)
        self.assertEqual(model.mode, "multiclass")
        self._assert_loss(model, torch.randn(3, 3), torch.tensor([0, 2, 1]), F.cross_entropy)

    def test_multilabel_loss_selection(self):
        samples = [
            {"label": [0, 2], "text": "row0"},
            {"label": [1], "text": "row1"},
        ]
        ds = create_sample_dataset(samples, {"text": "raw"}, {"label": MultiLabelProcessor})
        model = BaseModel(dataset=ds)
        self.assertEqual(model.mode, "multilabel")
        y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self._assert_loss(model, torch.randn(2, 3), y, F.binary_cross_entropy_with_logits)

    def test_regression_loss_selection(self):
        samples = [
            {"label": 0.5, "text": "r0"},
            {"label": 1.2, "text": "r1"},
        ]
        ds = create_sample_dataset(
            samples, {"text": "raw"}, {"label": RegressionLabelProcessor}
        )
        model = BaseModel(dataset=ds)
        self.assertEqual(model.mode, "regression")
        self._assert_loss(model, torch.randn(2, 1), torch.randn(2, 1), F.mse_loss)


if __name__ == "__main__":
    unittest.main()
