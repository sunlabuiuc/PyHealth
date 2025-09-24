import unittest
import torch
import torch.nn as nn

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    RegressionLabelProcessor,
    RawProcessor,
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
        loss_fn = self.get_loss_function()
        loss = loss_fn(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {"loss": loss, "y_true": y_true, "y_prob": y_prob}


class TestLegacyModeResolution(unittest.TestCase):
    def _build_dataset(self, output_processor, key="label"):
        samples = [
            {key: 0, "text": "a"},
            {key: 1, "text": "b"},
        ]
        input_schema = {"text": "raw"}
        output_schema = {key: output_processor}
        return SampleDataset(samples, input_schema, output_schema)

    def test_string_schema_sets_mode(self):
        ds = self._build_dataset("binary")
        model = DummyBinaryModel(ds)
        self.assertEqual(model.mode, "binary")
        self.assertEqual(
            model.get_loss_function().__name__, "binary_cross_entropy_with_logits"
        )

    def test_processor_class_schema_sets_mode(self):
        ds = self._build_dataset(BinaryLabelProcessor)
        model = DummyBinaryModel(ds)
        self.assertEqual(model.mode, "binary")
        self.assertEqual(
            model.get_loss_function().__name__, "binary_cross_entropy_with_logits"
        )

    def test_unregistered_processor_leaves_mode_none(self):
        ds = self._build_dataset(RawProcessor)
        model = DummyBinaryModel(ds)
        # Expect legacy mode attribute not set
        self.assertIsNone(model.mode)
        with self.assertRaises(ValueError):
            model.get_loss_function()

    def test_multiclass_loss_selection(self):
        samples = [{"label": i % 3, "text": f"row{i}"} for i in range(6)]
        ds = SampleDataset(
            samples, {"text": "raw"}, {"label": MultiClassLabelProcessor}
        )
        model = BaseModel(dataset=ds)
        self.assertEqual(model.mode, "multiclass")
        self.assertEqual(model.get_loss_function().__name__, "cross_entropy")

    def test_multilabel_loss_selection(self):
        samples = [
            {"label": [0, 2], "text": "row0"},
            {"label": [1], "text": "row1"},
        ]
        ds = SampleDataset(samples, {"text": "raw"}, {"label": MultiLabelProcessor})
        model = BaseModel(dataset=ds)
        self.assertEqual(model.mode, "multilabel")
        self.assertEqual(
            model.get_loss_function().__name__, "binary_cross_entropy_with_logits"
        )

    def test_regression_loss_selection(self):
        samples = [
            {"label": 0.5, "text": "r0"},
            {"label": 1.2, "text": "r1"},
        ]
        ds = SampleDataset(
            samples, {"text": "raw"}, {"label": RegressionLabelProcessor}
        )
        model = BaseModel(dataset=ds)
        self.assertEqual(model.mode, "regression")
        self.assertEqual(model.get_loss_function().__name__, "mse_loss")


if __name__ == "__main__":
    unittest.main()
