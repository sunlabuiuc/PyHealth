"""Class imbalance: BaseModel.set_pos_weight for the default BCE loss."""

import unittest

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN


def _binary_dataset(labels):
    samples = [
        {"patient_id": f"p{i}", "codes": ["a", "b", "c"][: 1 + i % 3], "label": y}
        for i, y in enumerate(labels)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="test_pos_weight",
    )


class TestPosWeight(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.train = _binary_dataset([1, 0, 0, 0, 0])  # 1 positive, 4 negatives
        self.logits = torch.tensor([[0.3], [-1.2], [2.0]])
        self.y = torch.tensor([[1.0], [0.0], [1.0]])

    def test_default_loss_is_unweighted(self):
        model = RNN(dataset=self.train)
        loss = model.get_loss_function()(self.logits, self.y)
        self.assertTrue(torch.allclose(loss, F.binary_cross_entropy_with_logits(self.logits, self.y)))
        self.assertIsNone(model.pos_weight)

    def test_fixed_weight(self):
        model = RNN(dataset=self.train)
        model.set_pos_weight(3.0)
        loss = model.get_loss_function()(self.logits, self.y)
        expected = F.binary_cross_entropy_with_logits(
            self.logits, self.y, pos_weight=torch.tensor([3.0])
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_balanced_uses_the_given_training_data(self):
        model = RNN(dataset=_binary_dataset([1, 1, 0, 0]))  # model's own data: 50%
        model.set_pos_weight("balanced", self.train)  # training data: 1 of 5
        self.assertTrue(torch.allclose(model.pos_weight, torch.tensor([4.0])))

    def test_balanced_needs_a_dataset(self):
        model = RNN(dataset=self.train)
        with self.assertRaisesRegex(ValueError, "training dataset"):
            model.set_pos_weight("balanced")

    def test_clear_weight(self):
        model = RNN(dataset=self.train)
        model.set_pos_weight(2.0)
        model.set_pos_weight(None)
        self.assertIsNone(model.pos_weight)

    def test_forward_uses_the_weight(self):
        model = RNN(dataset=self.train)
        batch = next(iter(get_dataloader(self.train, batch_size=5, shuffle=False)))
        model.eval()
        with torch.no_grad():
            plain = model(**batch)
            model.set_pos_weight(4.0)
            weighted = model(**batch)
        expected = F.binary_cross_entropy_with_logits(
            plain["logit"], plain["y_true"], pos_weight=torch.tensor([4.0])
        )
        self.assertTrue(torch.allclose(weighted["loss"], expected))
        self.assertFalse(torch.allclose(weighted["loss"], plain["loss"]))

    def test_multilabel_balanced_is_per_label(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a"], "label": labels}
            for i, labels in enumerate([["x"], ["x", "y"], ["x"], []])
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="test_pos_weight_multilabel",
        )
        model = RNN(dataset=dataset)
        model.set_pos_weight("balanced", dataset)
        vocab = dataset.output_processors["label"].label_vocab
        weights = {k: float(model.pos_weight[i]) for k, i in vocab.items()}
        self.assertAlmostEqual(weights["x"], 1 / 3)  # 3 positives, 1 negative
        self.assertAlmostEqual(weights["y"], 3.0)  # 1 positive, 3 negatives

    def test_multiclass_is_rejected(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a"], "label": i % 3} for i in range(6)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_pos_weight_multiclass",
        )
        model = RNN(dataset=dataset)
        with self.assertRaisesRegex(ValueError, "binary or multilabel"):
            model.set_pos_weight(2.0)


if __name__ == "__main__":
    unittest.main()
