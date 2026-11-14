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
        processor = MultiLabelProcessor(labels=["a", "b"], pos_weight=[2.0, 0.5])
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
    def test_repr(self):
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


class TestFitTest(unittest.TestCase):
    """fit on the training labels, then fit_test adds the labels of every split."""

    ALL = [{"y": "c"}, {"y": "b"}, {"y": "a"}]
    TRAIN = [{"y": "a"}, {"y": "c"}]

    def test_multiclass_adds_the_labels_of_every_split(self):
        processor = MultiClassLabelProcessor()
        processor.fit(self.TRAIN, "y")
        self.assertEqual(processor.label_vocab, {"a": 0, "c": 1})
        processor.fit_test(self.ALL, "y")
        # The order is that of all labels, whatever the split.
        self.assertEqual(processor.label_vocab, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(processor.process("c").item(), 2)

    def test_order_does_not_depend_on_the_split(self):
        labels = ["d", "b", "a", "c"]
        for cls, value in ((MultiClassLabelProcessor, str), (MultiLabelProcessor, list)):
            everything = [{"y": value(label)} for label in labels]
            expected = cls()
            expected.fit(everything, "y")
            # Every non-empty training subset, in every position.
            for mask in range(1, 2 ** len(labels)):
                train = [s for i, s in enumerate(everything) if mask >> i & 1]
                with self.subTest(cls=cls.__name__, train=train):
                    processor = cls()
                    processor.fit(train, "y")
                    processor.fit_test(everything, "y")
                    self.assertEqual(processor.label_vocab, expected.label_vocab)

    def test_range_labels_keep_their_values(self):
        processor = MultiClassLabelProcessor()
        processor.fit([{"y": 0}, {"y": 2}], "y")
        processor.fit_test([{"y": 1}], "y")
        self.assertEqual(processor.label_vocab, {0: 0, 1: 1, 2: 2})

    def test_running_it_again_changes_nothing(self):
        processor = MultiClassLabelProcessor()
        processor.fit(self.ALL, "y")
        for _ in range(2):
            processor.fit_test(self.ALL, "y")
            self.assertEqual(processor.label_vocab, {"a": 0, "b": 1, "c": 2})

    def test_need_fit_test_only_for_learned_vocabularies(self):
        self.assertTrue(MultiClassLabelProcessor().need_fit_test)
        self.assertTrue(MultiLabelProcessor().need_fit_test)
        self.assertTrue(TimeseriesMultiClassLabelProcessor().need_fit_test)
        for processor in (
            MultiClassLabelProcessor(labels=["c", "b", "a"]),
            MultiLabelProcessor(labels=2),
            BinaryLabelProcessor(),
            RegressionLabelProcessor(),
            TimeseriesBinaryLabelProcessor(),
            TimeseriesMultiClassLabelProcessor(labels=2),
            TimeseriesRegressionLabelProcessor(),
        ):
            with self.subTest(processor=processor):
                self.assertFalse(processor.need_fit_test)
                processor.fit_test(None, "y")  # never reads the samples

    def test_given_labels_are_kept(self):
        processor = MultiClassLabelProcessor(labels=["c", "b", "a"])
        processor.fit_test([{"y": "d"}], "y")
        self.assertEqual(processor.label_vocab, {"c": 0, "b": 1, "a": 2})

    def test_balanced_weight_keeps_its_training_values(self):
        processor = MultiLabelProcessor(pos_weight="balanced")
        processor.fit([{"y": ["a"]}, {"y": ["c"]}, {"y": ["c"]}], "y")
        self.assertEqual(processor.weight.tolist(), [2.0, 0.5])
        with self.assertLogs("pyhealth.processors.label_processor", "WARNING"):
            processor.fit_test([{"y": ["a", "b"]}, {"y": ["c", "d"]}], "y")
        self.assertEqual(processor.label_vocab, {"a": 0, "b": 1, "c": 2, "d": 3})
        self.assertEqual(processor.weight.tolist(), [2.0, 1.0, 0.5, 1.0])

    def test_given_and_unweighted_multilabel_weights_are_kept(self):
        processor = MultiLabelProcessor(pos_weight=2.0)
        processor.fit([{"y": ["a"]}], "y")
        processor.fit_test([{"y": ["b"]}], "y")
        self.assertEqual(processor.size(), 2)
        self.assertEqual(processor.weight.tolist(), [2.0])
        self.assertIsNone(MultiLabelProcessor().weight)

    def test_per_label_weight_needs_given_labels(self):
        # Without labels, the order the weights are matched to is unknown.
        for cls in (MultiLabelProcessor, TimeseriesMultiLabelProcessor):
            for weight in ([2.0, 0.5], (2.0,), torch.tensor([2.0, 0.5])):
                with self.subTest(cls=cls.__name__, weight=weight):
                    with self.assertRaisesRegex(ValueError, "needs labels"):
                        cls(pos_weight=weight)
        with self.assertRaisesRegex(ValueError, "2 weights for 3 labels"):
            MultiLabelProcessor(labels=["a", "b", "c"], pos_weight=[2.0, 0.5])
        # One number, or a weight per given label, is fine.
        for weight in (2.0, 3, torch.tensor(2.0)):
            MultiLabelProcessor(pos_weight=weight)
        processor = TimeseriesMultiLabelProcessor(labels=["a", "b"], pos_weight=[2.0, 0.5])
        self.assertEqual(processor.base.weight.tolist(), [2.0, 0.5])

    def test_timeseries_labels(self):
        multilabel = TimeseriesMultiLabelProcessor(pos_weight="balanced")
        multilabel.fit([{"y": {"time": [0.0, 1.0], "values": [["a"], []]}}], "y")
        with self.assertLogs("pyhealth.processors.label_processor", "WARNING"):
            multilabel.fit_test(
                [{"y": {"time": [0.0, 1.0], "values": [["a"], ["b"]]}}], "y"
            )
        self.assertEqual(multilabel.size(), 2)
        self.assertEqual(multilabel.base.weight.tolist(), [1.0, 1.0])


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

    def test_collate_needs_a_sample(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            TimeseriesRegressionLabelProcessor().collate([])

    def test_regression_needs_no_fit(self):
        processor = TimeseriesRegressionLabelProcessor()

        class Unreadable:
            def __iter__(self):
                raise AssertionError("fit read the samples")

        processor.fit(Unreadable(), "y")


class TestRegressionScale(unittest.TestCase):
    SAMPLES = [{"y": 10.0}, {"y": 30.0}, {"y": 20.0}]

    def _minmax(self):
        processor = RegressionLabelProcessor(normalization="minmax")
        processor.fit(self.SAMPLES, "y")
        return processor

    def test_no_normalization_changes_nothing(self):
        processor = RegressionLabelProcessor()
        processor.fit(self.SAMPLES, "y")
        self.assertIsNone(processor.center)
        self.assertFalse(processor.learns_statistics)
        self.assertEqual(processor.process(20.0).tolist(), [20.0])
        (y_true,), (y_prob,) = processor.to_numpy(
            torch.tensor([[20.0]]), torch.tensor([[21.0]])
        )
        self.assertEqual((y_true.item(), y_prob.item()), (20.0, 21.0))

    def test_minmax_learns_the_range_of_the_fitted_samples(self):
        processor = self._minmax()
        self.assertEqual((processor.center, processor.scale), (10.0, 20.0))
        self.assertTrue(processor.learns_statistics)

    def test_process_scales_without_clipping(self):
        processor = self._minmax()
        self.assertEqual(processor.process(20.0).tolist(), [0.5])
        self.assertEqual(processor.process(50.0).tolist(), [2.0])

    def test_labels_outside_the_range_keep_their_true_value(self):
        # A held-out label beyond the fitted range is not made easier to predict.
        processor = self._minmax()
        y_true = torch.stack([processor.process(50.0), processor.process(-10.0)])
        (true,), _ = processor.to_numpy(y_true, y_true)
        self.assertEqual(true.ravel().tolist(), [50.0, -10.0])

    def test_loss_and_activation_use_the_processed_label_as_is(self):
        processor = self._minmax()
        logits = torch.tensor([[0.0], [0.25]])
        y_true = torch.stack([processor.process(20.0), processor.process(30.0)])
        self.assertIs(processor.activation(logits), logits)
        torch.testing.assert_close(
            processor.loss(logits, y_true), F.mse_loss(logits, y_true)
        )
        # a model's own loss sees logits and y_true in the same units
        self.assertEqual(processor.loss(y_true, y_true).item(), 0.0)

    def test_to_numpy_maps_labels_and_predictions_back_together(self):
        processor = self._minmax()
        y_true = processor.process(20.0)
        (true,), (prob,) = processor.to_numpy(y_true, y_true + 0.25)
        self.assertEqual(true.tolist(), [20.0])
        self.assertEqual(prob.tolist(), [25.0])

    def test_fixed_range_needs_no_fit(self):
        processor = RegressionLabelProcessor(normalization=(0, 15))
        self.assertEqual((processor.center, processor.scale), (0.0, 15.0))
        self.assertFalse(processor.learns_statistics)
        processor.fit(self.SAMPLES, "y")  # does not move them
        self.assertEqual((processor.center, processor.scale), (0.0, 15.0))
        self.assertEqual(processor.process(15).tolist(), [1.0])

    def test_minmax_must_be_fitted(self):
        with self.assertRaisesRegex(ValueError, "fit first"):
            RegressionLabelProcessor(normalization="minmax").process(1.0)

    def test_zscore(self):
        processor = RegressionLabelProcessor(normalization="zscore")
        self.assertTrue(processor.learns_statistics)
        processor.fit(self.SAMPLES, "y")
        self.assertAlmostEqual(processor.center, 20.0)
        self.assertAlmostEqual(processor.scale, float(np.std([10.0, 30.0, 20.0])))
        y_true = processor.process(30.0)
        (true,), _ = processor.to_numpy(y_true, y_true)
        self.assertAlmostEqual(true.item(), 30.0, places=4)

    def test_invalid_normalization(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            RegressionLabelProcessor(normalization="robust")
        for bad in [(5, 5), (0, float("inf")), (float("nan"), 1)]:
            with self.assertRaisesRegex(ValueError, "finite low < high"):
                RegressionLabelProcessor(normalization=bad)

    def test_learned_normalization_needs_samples(self):
        with self.assertRaisesRegex(ValueError, "no samples"):
            RegressionLabelProcessor(normalization="minmax").fit([], "y")

    def test_constant_label_gets_a_unit_scale(self):
        processor = RegressionLabelProcessor(normalization="minmax")
        with self.assertLogs("pyhealth.processors.label_processor", "WARNING"):
            processor.fit([{"y": 3.0}, {"y": 3.0}], "y")
        self.assertEqual((processor.center, processor.scale), (3.0, 1.0))

    def test_pickles_from_before_the_option_load(self):
        processor = RegressionLabelProcessor()
        processor.__dict__.clear()  # as pickled before the option
        restored = pickle.loads(pickle.dumps(processor))
        self.assertIsNone(restored.center)
        self.assertEqual(restored.process(1.5).tolist(), [1.5])

    def test_timeseries_fits_on_every_label_record(self):
        processor = TimeseriesRegressionLabelProcessor(normalization="minmax")
        self.assertTrue(processor.learns_statistics)
        processor.fit(
            [
                {"y": {"time": [0.0, 1.0], "values": [10.0, 20.0]}},
                {"y": {"time": [0.0], "values": [[30.0]]}},
            ],
            "y",
        )
        self.assertEqual((processor.base.center, processor.base.scale), (10.0, 20.0))
        batch = processor.collate(
            [
                processor.process({"time": [0.0, 1.0], "values": [20.0, 30.0]}),
                processor.process({"time": [0.0], "values": [10.0]}),
            ]
        )
        self.assertEqual(batch[0].tolist(), [[0.5], [1.0], [0.0]])
        y_true, y_prob = processor.to_numpy(batch, batch[0] + 0.5)
        self.assertEqual([part.tolist() for part in y_true], [[20.0, 30.0], [10.0]])
        self.assertEqual([part.tolist() for part in y_prob], [[30.0, 40.0], [20.0]])

    def test_timeseries_fixed_range_does_not_read_the_samples(self):
        processor = TimeseriesRegressionLabelProcessor(normalization=(0, 15))
        self.assertFalse(processor.learns_statistics)

        class Unreadable:
            def __iter__(self):
                raise AssertionError("fit read the samples")

        processor.fit(Unreadable(), "y")

    def test_dataset_warns_when_fitted_on_all_samples(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a"], "y": float(i)} for i in range(3)
        ]
        with self.assertLogs("pyhealth.datasets.sample_dataset", "WARNING") as logs:
            create_sample_dataset(
                samples,
                {"codes": "sequence"},
                {"y": ("regression", {"normalization": "minmax"})},
            )
        self.assertIn("fitted on all samples", "".join(logs.output))


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

    def test_scaled_labels_are_scored_in_original_units(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a", "b"][: 1 + i % 2], "y": 100.0 * i}
            for i in range(9)
        ]
        dataset = create_sample_dataset(
            samples, {"codes": "sequence"}, {"y": ("regression", {"normalization": (0, 800)})}
        )
        model = LogisticRegression(dataset=dataset, embedding_dim=8)
        trainer = self._run(model, dataset, "mae")
        loader = get_dataloader(dataset, batch_size=3)

        # inside the model, labels and predictions are both scaled
        batch = next(iter(loader))
        out = model(**batch)
        self.assertLessEqual(out["y_true"].max().item(), 1.0)
        torch.testing.assert_close(
            out["loss"], F.mse_loss(out["y_prob"], out["y_true"])
        )

        # outside it, both are in the label's units
        y_true, y_prob, loss = trainer.inference(loader)
        np.testing.assert_allclose(
            sorted(y_true.ravel()), [100.0 * i for i in range(9)], rtol=1e-5
        )
        self.assertAlmostEqual(
            trainer.evaluate(loader)["mae"], np.abs(y_true - y_prob).mean(), places=3
        )
        self.assertAlmostEqual(
            loss, np.mean(((y_true - y_prob) / 800) ** 2), places=5
        )


if __name__ == "__main__":
    unittest.main()
