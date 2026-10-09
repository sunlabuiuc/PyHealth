"""Removal-based metrics on code-sequence inputs (integer code ids)."""

import math
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.metrics.interpretability import (
    ComprehensivenessMetric,
    evaluate_attribution,
    threshold_sample_filter,
)
from pyhealth.metrics.interpretability.base import _ablate_codes
from pyhealth.models import MLP, Transformer


def _dataset():
    samples = [
        {
            "patient_id": f"p{i}",
            "codes": [f"c{(i + j) % 7}" for j in range(3 + i % 3)],
            "label": i % 2,
        }
        for i in range(20)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="ablation_codes",
    )


class TestAblateCodes(unittest.TestCase):
    def test_masked_codes_become_padding_and_keep_dtype(self):
        x = torch.tensor([[3, 4, 5, 0]])
        out = _ablate_codes(x, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(out.tolist(), [[0, 4, 0, 0]])

    def test_fully_ablated_sample_keeps_first_code(self):
        out = _ablate_codes(torch.tensor([[0, 6, 7]]), torch.ones(1, 3))
        self.assertEqual(out.tolist(), [[0, 6, 0]])

    def test_float_inputs_still_use_the_strategy(self):
        model = MLP(dataset=_dataset())
        metric = ComprehensivenessMetric(
            model, ablation_strategy="zero", sample_filter=threshold_sample_filter()
        )
        x = torch.tensor([[1.5, 2.5]])
        out = metric._apply_ablation({"x": x}, {"x": torch.tensor([[1.0, 0.0]])})
        self.assertTrue(torch.equal(out["x"], torch.tensor([[0.0, 2.5]])))


class TestRemovalMetricsOnCodeSequences(unittest.TestCase):
    """Comprehensiveness/sufficiency used to crash: code ids became floats."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = _dataset()

    def _check(self, model_class):
        model = model_class(dataset=self.dataset)
        model.eval()
        loader = get_dataloader(self.dataset, batch_size=8)
        for strategy in ["zero", "mean", "noise"]:
            with self.subTest(model=model_class.__name__, strategy=strategy):
                scores = evaluate_attribution(
                    model,
                    loader,
                    IntegratedGradients(model, steps=5),
                    ablation_strategy=strategy,
                    sample_filter=threshold_sample_filter(0.0),
                )
                for name, value in scores.items():
                    self.assertTrue(math.isfinite(value), (name, value))

    def test_mlp(self):
        self._check(MLP)

    def test_transformer(self):
        self._check(Transformer)


if __name__ == "__main__":
    unittest.main()
