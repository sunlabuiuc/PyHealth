"""The LabelProcessor contract of the eight label processors."""

import pickle
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.processors import (
    PROCESSOR_REGISTRY,
    BinaryLabelProcessor,
    LabelProcessor,
    MultiClassLabelProcessor,
    MultiHotProcessor,
    MultiLabelProcessor,
    RegressionLabelProcessor,
    TimeseriesBinaryLabelProcessor,
    TimeseriesLabelProcessor,
    TimeseriesMultiClassLabelProcessor,
    TimeseriesMultiLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)
from pyhealth.trainer import Trainer, get_metrics_fn
from tests.core.label_record_model_utils import LabelRecordModel

STATIC = {
    "binary": BinaryLabelProcessor,
    "multiclass": MultiClassLabelProcessor,
    "multilabel": MultiLabelProcessor,
    "regression": RegressionLabelProcessor,
}
TIMESERIES = {
    "timeseries_binary": TimeseriesBinaryLabelProcessor,
    "timeseries_multiclass": TimeseriesMultiClassLabelProcessor,
    "timeseries_multilabel": TimeseriesMultiLabelProcessor,
    "timeseries_regression": TimeseriesRegressionLabelProcessor,
}


class TestRegistry(unittest.TestCase):
    def test_each_processor_is_registered_under_its_mode(self):
        for mode, cls in {**STATIC, **TIMESERIES}.items():
            with self.subTest(mode=mode):
                self.assertIs(PROCESSOR_REGISTRY[mode], cls)
                self.assertEqual(cls.mode, mode)
                self.assertTrue(issubclass(cls, LabelProcessor))

    def test_output_size_is_size(self):
        processors = [
            BinaryLabelProcessor(), MultiClassLabelProcessor(labels=3),
            MultiLabelProcessor(labels=4), RegressionLabelProcessor(),
            TimeseriesBinaryLabelProcessor(), TimeseriesMultiClassLabelProcessor(labels=3),
            TimeseriesMultiLabelProcessor(labels=4), TimeseriesRegressionLabelProcessor(),
        ]
        for processor in processors:
            with self.subTest(processor=type(processor).__name__):
                self.assertEqual(processor.output_size(), processor.size())

    def test_only_label_processors_are_labels(self):
        self.assertNotIsInstance(MultiHotProcessor(), LabelProcessor)

    def test_static_processors_do_not_subclass_each_other(self):
        for a in STATIC.values():
            for b in STATIC.values():
                if a is not b:
                    self.assertFalse(issubclass(a, b), (a, b))

    def test_get_metrics_fn_is_the_processors(self):
        for mode, cls in {**STATIC, **TIMESERIES}.items():
            with self.subTest(mode=mode):
                self.assertEqual(get_metrics_fn(mode), cls.metrics_fn)
        with self.assertRaisesRegex(ValueError, "not supported"):
            get_metrics_fn("multi_hot")


class TestTrainerFallback(unittest.TestCase):
    """A module that is not a BaseModel is scored by its ``mode`` attribute."""

    def test_plain_module(self):
        from pyhealth.trainer import _label_processor

        module = torch.nn.Linear(1, 1)
        self.assertIsNone(_label_processor(module))
        module.mode = "multiclass"
        self.assertIsInstance(_label_processor(module), MultiClassLabelProcessor)
        module.mode = "multi_hot"
        with self.assertRaisesRegex(ValueError, "not supported"):
            _label_processor(module)


class TestStaticActivationAndLoss(unittest.TestCase):
    def test_against_the_torch_functions(self):
        torch.manual_seed(0)
        cases = [
            (BinaryLabelProcessor(), torch.randn(4, 1), torch.tensor([[0.0], [1.0], [1.0], [0.0]]),
             torch.sigmoid, F.binary_cross_entropy_with_logits),
            (MultiClassLabelProcessor(), torch.randn(4, 3), torch.tensor([0, 2, 1, 2]),
             lambda x: F.softmax(x, dim=-1), F.cross_entropy),
            (MultiLabelProcessor(), torch.randn(4, 3), torch.randint(0, 2, (4, 3)).float(),
             torch.sigmoid, F.binary_cross_entropy_with_logits),
            (RegressionLabelProcessor(), torch.randn(4, 1), torch.randn(4, 1),
             lambda x: x, F.mse_loss),
        ]
        for processor, logits, y, activation, loss in cases:
            with self.subTest(processor=type(processor).__name__):
                torch.testing.assert_close(processor.activation(logits), activation(logits))
                torch.testing.assert_close(processor.loss(logits, y), loss(logits, y))

    def test_per_label_pos_weight(self):
        logits, y = torch.randn(3, 2), torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        processor = MultiLabelProcessor(pos_weight=[2.0, 0.5])
        expected = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor([2.0, 0.5])
        )
        torch.testing.assert_close(processor.loss(logits, y), expected)

    def test_unknown_label_raises(self):
        processor = BinaryLabelProcessor()
        processor.fit([{"y": 0}, {"y": 1}], "y")
        with self.assertRaisesRegex(ValueError, "Unknown label 2"):
            processor.process(2)


class TestStaticDetails(unittest.TestCase):
    def test_schema_and_repr(self):
        self.assertEqual(RegressionLabelProcessor().schema(), ("value",))
        self.assertEqual(repr(RegressionLabelProcessor()), "RegressionLabelProcessor()")

    def test_other_binary_labels_are_sorted(self):
        processor = BinaryLabelProcessor()
        processor.fit([{"y": "yes"}, {"y": "no"}], "y")
        self.assertEqual(processor.label_vocab, {"no": 0, "yes": 1})

    def test_balanced_needs_samples(self):
        with self.assertRaisesRegex(ValueError, "no samples"):
            MultiLabelProcessor(labels=2, pos_weight="balanced").fit([], "y")

    def test_balanced_warns_on_a_label_without_positives(self):
        processor = MultiLabelProcessor(labels=["a", "b"], pos_weight="balanced")
        with self.assertLogs("pyhealth.processors.label_processor", "WARNING"):
            processor.fit([{"y": ["a"]}, {"y": []}], "y")
        self.assertEqual(processor.weight.tolist(), [1.0, 1.0])

    def test_fixed_multilabel_labels_are_checked_in_fit(self):
        with self.assertRaisesRegex(ValueError, "Unknown label 'c'"):
            MultiLabelProcessor(labels=["a", "b"]).fit([{"y": ["a", "c"]}], "y")


class TestFixedLabels(unittest.TestCase):
    def test_given_labels_fix_indices_and_size(self):
        processor = MultiClassLabelProcessor(labels=["low", "mid", "high"])
        processor.fit([{"y": "high"}, {"y": "low"}], "y")  # "mid" not seen
        self.assertEqual(processor.size(), 3)
        self.assertEqual(processor.process("high").item(), 2)

    def test_count_means_range(self):
        self.assertEqual(MultiLabelProcessor(labels=4).label_vocab, {i: i for i in range(4)})

    def test_unseen_label_in_fit_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown label 'x'"):
            MultiClassLabelProcessor(labels=["a", "b"]).fit([{"y": "x"}], "y")

    def test_invalid_labels(self):
        for labels in ([], ["a", "a"]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                MultiLabelProcessor(labels=labels)

    def test_omitted_labels_are_learned(self):
        processor = MultiClassLabelProcessor()
        processor.fit([{"y": "b"}, {"y": "a"}], "y")
        self.assertEqual(processor.label_vocab, {"a": 0, "b": 1})

    def test_labels_through_a_task_schema(self):
        samples = [{"patient_id": f"p{i}", "codes": ["a"], "y": "a"} for i in range(2)]
        dataset = create_sample_dataset(
            samples, {"codes": "sequence"}, {"y": ("multiclass", {"labels": ["a", "b"]})}
        )
        self.assertEqual(dataset.output_processors["y"].size(), 2)

    def test_pickles_from_before_the_options_load(self):
        processor = MultiLabelProcessor()
        processor.fit([{"y": ["a"]}, {"y": ["b"]}], "y")
        for name in ("labels", "pos_weight", "_weight", "learns_statistics"):
            processor.__dict__.pop(name, None)  # as pickled before the options
        restored = pickle.loads(pickle.dumps(processor))
        self.assertIsNone(restored.weight)
        self.assertEqual(restored.process(["b"]).tolist(), [0.0, 1.0])


class TestTimeseriesLabels(unittest.TestCase):
    def test_multiclass_and_multilabel_values(self):
        multiclass = TimeseriesMultiClassLabelProcessor()
        multiclass.fit([{"y": {"time": [0.0, 1.0], "values": ["b", "a"]}}], "y")
        values, time = multiclass.process({"time": [0.0, 2.0], "values": ["a", "b"]})
        self.assertEqual(values.dtype, torch.long)
        self.assertEqual(values.tolist(), [0, 1])
        self.assertEqual(multiclass.size(), 2)

        multilabel = TimeseriesMultiLabelProcessor(labels=["x", "y"])
        values, _ = multilabel.process({"time": [0.0, 1.0], "values": [["y"], []]})
        self.assertEqual(values.tolist(), [[0.0, 1.0], [0.0, 0.0]])

    def test_empty_label_has_the_value_shape(self):
        for processor, shape in (
            (TimeseriesBinaryLabelProcessor(), (0, 1)),
            (TimeseriesMultiClassLabelProcessor(labels=3), (0,)),
            (TimeseriesMultiLabelProcessor(labels=3), (0, 3)),
        ):
            with self.subTest(processor=type(processor).__name__):
                values, time = processor.process({"time": [], "values": []})
                self.assertEqual(tuple(values.shape), shape)
                self.assertEqual(tuple(time.shape), (0,))

    def test_collate(self):
        processor = TimeseriesRegressionLabelProcessor()
        batch = processor.collate([
            processor.process({"time": [0.0, 1.0], "values": [1.0, 2.0]}),
            processor.process({"time": [], "values": []}),
            processor.process({"time": [5.0], "values": [3.0]}),
        ])
        self.assertEqual(batch[2].tolist(), [0, 2, 2, 3])
        self.assertEqual(batch[1].tolist(), [0.0, 1.0, 5.0])

    def test_loss_is_the_static_loss_per_record(self):
        processor = TimeseriesMultiClassLabelProcessor(labels=3)
        batch = processor.collate([
            processor.process({"time": [0.0, 1.0], "values": [0, 2]}),
            processor.process({"time": [0.0], "values": [1]}),
        ])
        logits = torch.randn(3, 3)
        torch.testing.assert_close(
            processor.loss(logits, batch), F.cross_entropy(logits, batch[0])
        )
        with self.assertRaisesRegex(ValueError, "one prediction per label record"):
            processor.loss(torch.randn(3, 1), batch)

    def test_to_numpy_gives_one_array_per_sample(self):
        processor = TimeseriesMultiLabelProcessor(labels=2)
        batch = processor.collate([
            processor.process({"time": [0.0, 1.0], "values": [[0], [1]]}),
            processor.process({"time": [0.0], "values": [[0, 1]]}),
        ])
        y_true, y_prob = processor.to_numpy(batch, torch.full((3, 2), 0.5))
        self.assertEqual([a.shape for a in y_true], [(2, 2), (1, 2)])
        self.assertEqual([a.shape for a in y_prob], [(2, 2), (1, 2)])
        self.assertEqual(processor.gather(y_true, y_prob), (y_true, y_prob))

    def test_metrics_pool_the_records(self):
        y_true = [np.array([0, 1, 1]), np.array([0])]
        y_prob = [np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]), np.array([[0.6, 0.4]])]
        scores = TimeseriesMultiClassLabelProcessor.metrics_fn(
            y_true, y_prob, metrics=["accuracy"]
        )
        self.assertEqual(scores, {"accuracy": 1.0})

    def test_values_are_checked(self):
        bad = [
            (TimeseriesMultiClassLabelProcessor(labels=["a"]), {"time": [0.0], "values": ["b"]}),
            (TimeseriesMultiLabelProcessor(labels=2), {"time": [0.0, 1.0], "values": [[0]]}),
            (TimeseriesBinaryLabelProcessor(), {"time": [0.0], "values": [0.5]}),
        ]
        for processor, records in bad:
            with self.subTest(records=records), self.assertRaises(ValueError):
                processor.process(records)

    def test_shape_metadata_follows_the_base(self):
        multiclass = TimeseriesMultiClassLabelProcessor(labels=3)
        self.assertTrue(multiclass.is_token())
        self.assertEqual(multiclass.dim(), (1, 1))
        self.assertEqual(multiclass.spatial(), (True,))
        multilabel = TimeseriesMultiLabelProcessor(labels=3)
        self.assertFalse(multilabel.is_token())
        self.assertEqual(multilabel.dim(), (2, 1))
        self.assertEqual(multilabel.spatial(), (True, False))

    def test_collate_needs_a_sample(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            TimeseriesRegressionLabelProcessor().collate([])

    def test_regression_needs_no_fit(self):
        processor = TimeseriesRegressionLabelProcessor()

        class Unreadable:
            def __iter__(self):
                raise AssertionError("fit read the samples")

        processor.fit(Unreadable(), "y")


def _static_dataset(mode):
    labels = {
        "binary": lambda i: i % 2,
        "multiclass": lambda i: i % 3,
        "multilabel": lambda i: [["x"], ["y"], ["x", "y"]][i % 3],
        "regression": lambda i: float(i),
    }[mode]
    samples = [
        {"patient_id": f"p{i}", "codes": ["a", "b", "c"][: 1 + i % 3], "y": labels(i)}
        for i in range(9)
    ]
    return create_sample_dataset(samples, {"codes": "sequence"}, {"y": mode})


def _timeseries_dataset(mode):
    value = {
        "timeseries_binary": lambda i, h: (i + h) % 2,
        "timeseries_multiclass": lambda i, h: (i + h) % 3,
        "timeseries_multilabel": lambda i, h: [["x"], ["y"], ["x", "y"]][(i + h) % 3],
        "timeseries_regression": lambda i, h: float(i + h),
    }[mode]
    samples = []
    for i, steps in enumerate([3, 2, 4, 3]):
        samples.append({
            "patient_id": f"p{i}",
            "codes": ["a"],
            "y": {"time": [float(h) for h in range(steps)],
                  "values": [value(i, h) for h in range(steps)]},
        })
    return create_sample_dataset(samples, {"codes": "sequence"}, {"y": mode})


class TestTrainerWithEveryLabel(unittest.TestCase):
    METRIC = {
        "binary": "roc_auc", "multiclass": "accuracy", "multilabel": "f1_micro",
        "regression": "mae", "timeseries_binary": "roc_auc",
        "timeseries_multiclass": "accuracy", "timeseries_multilabel": "f1_micro",
        "timeseries_regression": "mae",
    }

    def _run(self, model, dataset, metric):
        trainer = Trainer(model=model, metrics=[metric], enable_logging=False)
        loader = get_dataloader(dataset, batch_size=3)
        trainer.train(train_dataloader=loader, epochs=1)
        scores = trainer.evaluate(loader)
        self.assertIn(metric, scores)
        self.assertTrue(np.isfinite(scores["loss"]))
        return trainer

    def test_static_labels(self):
        for mode in STATIC:
            with self.subTest(mode=mode):
                dataset = _static_dataset(mode)
                model = LogisticRegression(dataset=dataset, embedding_dim=8)
                self.assertEqual(model.mode, mode)
                trainer = self._run(model, dataset, self.METRIC[mode])
                y_true, y_prob, _ = trainer.inference(get_dataloader(dataset, batch_size=4))
                self.assertEqual(len(y_true), 9)
                self.assertEqual(len(y_prob), 9)

    def test_timeseries_labels(self):
        for mode in TIMESERIES:
            with self.subTest(mode=mode):
                dataset = _timeseries_dataset(mode)
                model = LabelRecordModel(dataset)
                self.assertIsInstance(model.label_processor, TimeseriesLabelProcessor)
                trainer = self._run(model, dataset, self.METRIC[mode])
                y_true, _, _ = trainer.inference(get_dataloader(dataset, batch_size=3))
                self.assertEqual([len(a) for a in y_true], [3, 2, 4, 3])


if __name__ == "__main__":
    unittest.main()
