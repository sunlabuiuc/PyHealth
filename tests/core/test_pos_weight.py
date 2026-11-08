"""Class imbalance: the ``pos_weight`` of the binary and multilabel processors."""

import unittest

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.processors import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    TimeseriesBinaryLabelProcessor,
)


def _binary_dataset(labels, label_spec="binary", **kwargs):
    samples = [
        {"patient_id": f"p{i}", "codes": ["a", "b", "c"][: 1 + i % 3], "label": y}
        for i, y in enumerate(labels)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": label_spec},
        dataset_name="test_pos_weight",
        **kwargs,
    )


class TestPosWeight(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.tensor([[0.3], [-1.2], [2.0]])
        self.y = torch.tensor([[1.0], [0.0], [1.0]])

    def test_default_loss_is_unweighted(self):
        model = RNN(dataset=_binary_dataset([1, 0, 0, 0, 0]))
        self.assertIsNone(model.label_processor.weight)
        torch.testing.assert_close(
            model.label_processor.loss(self.logits, self.y),
            F.binary_cross_entropy_with_logits(self.logits, self.y),
        )

    def test_fixed_weight_from_the_task(self):
        dataset = _binary_dataset([1, 0, 0, 0, 0], ("binary", {"pos_weight": 3.0}))
        loss = RNN(dataset=dataset).label_processor.loss(self.logits, self.y)
        expected = F.binary_cross_entropy_with_logits(
            self.logits, self.y, pos_weight=torch.tensor([3.0])
        )
        torch.testing.assert_close(loss, expected)

    def test_balanced_is_learned_from_the_fitted_samples(self):
        processor = BinaryLabelProcessor(pos_weight="balanced")
        processor.fit([{"y": y} for y in [1, 0, 0, 0, 0]], "y")  # 1 of 5
        torch.testing.assert_close(processor.weight, torch.tensor([4.0]))

    def test_balanced_warns_when_fitted_on_all_samples(self):
        with self.assertLogs("pyhealth.datasets.sample_dataset", "WARNING") as logs:
            _binary_dataset([1, 0, 0, 0], ("binary", {"pos_weight": "balanced"}))
        self.assertIn("label", "\n".join(logs.output))

    def test_balanced_from_a_training_fit(self):
        # A processor fitted on the training labels only, then used as is.
        processor = BinaryLabelProcessor(pos_weight="balanced")
        processor.fit([{"label": y} for y in [1, 0, 0, 0, 0]], "label")
        dataset = _binary_dataset(
            [1, 1, 0, 0], output_processors={"label": processor}
        )  # 50% positive here, but the weight stays 4
        torch.testing.assert_close(
            RNN(dataset=dataset).label_processor.weight, torch.tensor([4.0])
        )

    def test_balanced_needs_a_fit(self):
        with self.assertRaisesRegex(ValueError, "fit first"):
            BinaryLabelProcessor(pos_weight="balanced").weight

    def test_invalid_weight(self):
        with self.assertRaisesRegex(ValueError, "pos_weight must be"):
            BinaryLabelProcessor(pos_weight="heavy")

    def test_forward_uses_the_weight(self):
        labels = [1, 0, 0, 0, 0]
        plain_ds = _binary_dataset(labels)
        weighted_ds = _binary_dataset(labels, ("binary", {"pos_weight": 4.0}))
        batch = next(iter(get_dataloader(plain_ds, batch_size=5, shuffle=False)))
        torch.manual_seed(0)
        plain_model = RNN(dataset=plain_ds).eval()
        torch.manual_seed(0)
        weighted_model = RNN(dataset=weighted_ds).eval()
        with torch.no_grad():
            plain = plain_model(**batch)
            weighted = weighted_model(**batch)
        expected = F.binary_cross_entropy_with_logits(
            plain["logit"], plain["y_true"], pos_weight=torch.tensor([4.0])
        )
        torch.testing.assert_close(weighted["loss"], expected)
        self.assertFalse(torch.allclose(weighted["loss"], plain["loss"]))

    def test_multilabel_balanced_is_per_label(self):
        processor = MultiLabelProcessor(pos_weight="balanced")
        processor.fit(
            [{"y": labels} for labels in [["x"], ["x", "y"], ["x"], []]], "y"
        )
        weights = {k: float(processor.weight[i]) for k, i in processor.label_vocab.items()}
        self.assertAlmostEqual(weights["x"], 1 / 3)  # 3 positives, 1 negative
        self.assertAlmostEqual(weights["y"], 3.0)  # 1 positive, 3 negatives

    def test_timeseries_binary_balanced_counts_label_records(self):
        processor = TimeseriesBinaryLabelProcessor(pos_weight="balanced")
        processor.fit(
            [
                {"y": {"time": [0.0, 1.0, 2.0], "values": [0, 0, 1]}},
                {"y": {"time": [0.0], "values": [0]}},
            ],
            "y",
        )  # 1 positive of 4 records
        torch.testing.assert_close(processor.base.weight, torch.tensor([3.0]))

    def test_multiclass_has_no_pos_weight(self):
        with self.assertRaises(TypeError):
            MultiClassLabelProcessor(pos_weight=2.0)


if __name__ == "__main__":
    unittest.main()
