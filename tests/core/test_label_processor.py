"""LabelProcessor: mode, activation, loss and an optional fixed label set."""

import pickle
import unittest

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import RNN
from pyhealth.processors import (
    BinaryLabelProcessor,
    LabelProcessor,
    MultiClassLabelProcessor,
    MultiHotProcessor,
    MultiLabelProcessor,
    RegressionLabelProcessor,
    TimeseriesBinaryLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)


class TestActivationAndLoss(unittest.TestCase):
    """Each processor gives the activation and loss BaseModel used to switch on."""

    def setUp(self):
        torch.manual_seed(0)

    def test_binary(self):
        p, logits = BinaryLabelProcessor(), torch.randn(4, 1)
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        weight = torch.tensor([3.0])
        self.assertEqual((p.mode, p.is_timeseries, p.supports_pos_weight), ("binary", False, True))
        self.assertTrue(torch.equal(p.activation(logits), torch.sigmoid(logits)))
        self.assertTrue(torch.equal(p.loss(logits, y), F.binary_cross_entropy_with_logits(logits, y)))
        self.assertTrue(
            torch.equal(
                p.loss(logits, y, pos_weight=weight),
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight),
            )
        )

    def test_multiclass(self):
        p, logits, y = MultiClassLabelProcessor(), torch.randn(4, 3), torch.tensor([0, 2, 1, 2])
        self.assertEqual((p.mode, p.supports_pos_weight), ("multiclass", False))
        self.assertTrue(torch.equal(p.activation(logits), F.softmax(logits, dim=-1)))
        self.assertTrue(torch.equal(p.loss(logits, y), F.cross_entropy(logits, y)))
        with self.assertRaisesRegex(ValueError, "pos_weight"):
            p.loss(logits, y, pos_weight=torch.tensor([1.0]))

    def test_multilabel(self):
        p, logits = MultiLabelProcessor(), torch.randn(4, 3)
        y = torch.randint(0, 2, (4, 3)).float()
        weight = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual((p.mode, p.supports_pos_weight), ("multilabel", True))
        self.assertTrue(torch.equal(p.activation(logits), torch.sigmoid(logits)))
        self.assertTrue(
            torch.equal(
                p.loss(logits, y, pos_weight=weight),
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight),
            )
        )

    def test_regression(self):
        p, logits, y = RegressionLabelProcessor(), torch.randn(4, 1), torch.randn(4, 1)
        self.assertEqual((p.mode, p.supports_pos_weight), ("regression", False))
        self.assertTrue(torch.equal(p.activation(logits), logits))
        self.assertTrue(torch.equal(p.loss(logits, y), F.mse_loss(logits, y)))

    def test_timeseries_labels_apply_the_static_loss_per_record(self):
        values = torch.tensor([[1.0], [0.0], [1.0]])
        packed = (values, torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0, 2, 3]))
        logits = torch.randn(3, 1)
        for p, fn in (
            (TimeseriesBinaryLabelProcessor(), F.binary_cross_entropy_with_logits),
            (TimeseriesRegressionLabelProcessor(), F.mse_loss),
        ):
            with self.subTest(processor=p):
                self.assertTrue(p.is_timeseries)
                self.assertFalse(p.supports_pos_weight)
                self.assertTrue(torch.equal(p.loss(logits, packed), fn(logits, values)))
                self.assertTrue(
                    torch.equal(p.activation(logits), p.base_cls().activation(logits))
                )
                with self.assertRaisesRegex(ValueError, "one prediction per label record"):
                    p.loss(torch.randn(2, 1), packed)

    def test_timeseries_batch_without_records_has_zero_loss(self):
        logits = torch.zeros(0, 1, requires_grad=True)
        packed = (torch.zeros(0, 1), torch.zeros(0), torch.tensor([0, 0]))
        loss = TimeseriesBinaryLabelProcessor().loss(logits, packed)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(logits.grad).all())


    def test_timeseries_fit_does_not_read_the_samples(self):
        class Unreadable:
            def __iter__(self):
                raise AssertionError("fit read the samples")

        for p in (TimeseriesBinaryLabelProcessor(), TimeseriesRegressionLabelProcessor()):
            with self.subTest(processor=p):
                p.fit(Unreadable(), "y")
                values, time = p.process({"time": [0.0, 1.0], "values": [1, 0]})
                self.assertEqual(values.shape, (2, 1))


class TestHierarchy(unittest.TestCase):
    def test_only_label_processors_are_labels(self):
        self.assertIsInstance(MultiLabelProcessor(), LabelProcessor)
        self.assertNotIsInstance(MultiHotProcessor(), LabelProcessor)


class TestGivenLabels(unittest.TestCase):
    def test_multiclass_labels_fix_indices_and_size(self):
        p = MultiClassLabelProcessor(labels=["low", "mid", "high"])
        p.fit([{"y": "high"}, {"y": "low"}], "y")  # "mid" absent from the data
        self.assertEqual(p.size(), 3)
        self.assertEqual(p.process("high").item(), 2)

    def test_multilabel_labels_fix_indices_and_size(self):
        p = MultiLabelProcessor(labels=["x", "y", "z"])
        p.fit([{"y": ["z"]}], "y")
        self.assertEqual(p.size(), 3)
        self.assertTrue(torch.equal(p.process(["z"]), torch.tensor([0.0, 0.0, 1.0])))

    def test_count_means_range(self):
        for cls in (MultiClassLabelProcessor, MultiLabelProcessor):
            with self.subTest(cls=cls):
                self.assertEqual(cls(labels=3).label_vocab, cls(labels=range(3)).label_vocab)
                self.assertEqual(cls(labels=3).label_vocab, {0: 0, 1: 1, 2: 2})

    def test_unknown_label_raises(self):
        with self.assertRaisesRegex(ValueError, "'d'"):
            MultiClassLabelProcessor(labels=["a", "b"]).fit([{"y": "a"}, {"y": "d"}], "y")
        with self.assertRaisesRegex(ValueError, "'d'"):
            MultiLabelProcessor(labels=["a", "b"]).fit([{"y": ["a", "d"]}], "y")
        p = MultiClassLabelProcessor(labels=["a", "b"])
        with self.assertRaises(KeyError):
            p.process("d")

    def test_invalid_labels_raise(self):
        for labels in ([], ["a", "a"]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    MultiClassLabelProcessor(labels=labels)

    def test_omitted_labels_are_learned_as_before(self):
        p = MultiClassLabelProcessor()
        p.fit([{"y": "b"}, {"y": "a"}], "y")
        self.assertEqual(p.label_vocab, {"a": 0, "b": 1})
        p = MultiLabelProcessor()
        p.fit([{"y": [2, 0]}, {"y": [1]}], "y")
        self.assertEqual(p.label_vocab, {0: 0, 1: 1, 2: 2})

    def test_pickle_round_trip(self):
        for p in (MultiClassLabelProcessor(labels=["a", "b"]), MultiLabelProcessor()):
            with self.subTest(processor=p):
                restored = pickle.loads(pickle.dumps(p))
                self.assertEqual(restored.label_vocab, p.label_vocab)
                self.assertEqual(restored._labels_given, p._labels_given)

    def test_processor_pickled_before_labels_existed(self):
        p = MultiLabelProcessor.__new__(MultiLabelProcessor)
        p.__dict__.update({"label_vocab": {"a": 0}})  # the old instance state
        p.fit([{"y": ["a", "b"]}], "y")
        self.assertEqual(p.label_vocab, {"a": 0, "b": 1})

    def test_given_labels_through_a_task_schema(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a"], "label": label}
            for i, label in enumerate(["high", "low", "high"])
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": ("multiclass", {"labels": ["low", "mid", "high"]})},
            dataset_name="test_given_labels",
        )
        model = RNN(dataset=dataset)
        self.assertEqual(model.mode, "multiclass")
        self.assertEqual(model.get_output_size(), 3)


class TestBaseModelDelegation(unittest.TestCase):
    def test_default_loss_and_y_prob_come_from_the_processor(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a"], "label": [i % 3]} for i in range(6)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="test_delegation",
        )
        model = RNN(dataset=dataset)
        model.set_pos_weight([1.0, 2.0, 3.0])
        logits, y = torch.randn(2, 3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor([1.0, 2.0, 3.0])
        )
        self.assertTrue(torch.equal(model.get_loss_function()(logits, y), expected))
        self.assertTrue(torch.equal(model.prepare_y_prob(logits), torch.sigmoid(logits)))

    def test_pos_weight_is_rejected_without_support(self):
        samples = [{"patient_id": f"p{i}", "codes": ["a"], "label": i % 3} for i in range(6)]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_pos_weight_multiclass",
        )
        with self.assertRaisesRegex(ValueError, "pos_weight"):
            RNN(dataset=dataset).set_pos_weight(2.0)


if __name__ == "__main__":
    unittest.main()
