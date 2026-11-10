import unittest
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import torch

from pyhealth import tasks
from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel, RNN_V2
from pyhealth.processors import (
    LevelConfig,
    LabelProcessor,
    SequenceProcessorV2,
    TimeseriesBinaryLabelProcessor,
    TimeseriesLabelProcessor,
    TimeseriesMultiClassLabelProcessor,
    TimeseriesMultiLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)
from pyhealth.trainer import Trainer
from tests.core.label_record_model_utils import LabelRecordModel

NAN = float("nan")
DYNAMIC: list[LevelConfig] = [{"ordering": "series", "values": [2]}]
# Static values: one record at t = 0, before the first label.
STATIC: list[LevelConfig] = [{"ordering": "series", "values": [2]}]


def _samples(label_fn) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    samples = []
    for i, n in enumerate([5, 3, 6, 4]):
        hours = [float(h) for h in range(n)]
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": {"time": hours, "values": rng.normal(size=(n, 2))},
                "static": {"time": [0.0], "values": [[float(i), 1.0]]},
                "label": {"time": hours, "values": [label_fn(i, h) for h in range(n)]},
            }
        )
    return samples


def _dataset(
    output="timeseries_binary",
    label_fn=lambda i, h: float((i + h) % 2),
    timeline=("dynamic",),
):
    return create_sample_dataset(
        samples=_samples(label_fn),
        input_schema={
            "dynamic": ("sequence_v2", {"levels": DYNAMIC}),
            "static": ("sequence_v2", {"levels": STATIC}),
        },
        output_schema={"label": (output, {"timeline": list(timeline)})},
        dataset_name="per_step",
    )


TIMESERIES_CLASSES = (
    TimeseriesBinaryLabelProcessor,
    TimeseriesMultiClassLabelProcessor,
    TimeseriesMultiLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)


class TestTimeseriesLabelProcessors(unittest.TestCase):
    def test_are_label_processors(self):
        processor = TimeseriesBinaryLabelProcessor(timeline=["x"])
        self.assertIsInstance(processor, LabelProcessor)
        self.assertIsInstance(processor, TimeseriesLabelProcessor)
        self.assertNotIsInstance(processor, SequenceProcessorV2)
        self.assertEqual(processor.schema(), ("value", "time"))
        self.assertEqual(processor.size(), 1)
        self.assertFalse(processor.learns_statistics)
        self.assertEqual(
            repr(processor), "TimeseriesBinaryLabelProcessor(timeline=['x'], label_vocab_size=2)"
        )

    def test_values_and_time(self):
        values, time = TimeseriesRegressionLabelProcessor(timeline=["x"]).process(
            {"time": [0.0, 3.0, 4.5], "values": [[2.5], [4.0], [1.0]]}
        )
        self.assertEqual(values.tolist(), [[2.5], [4.0], [1.0]])
        self.assertEqual(time.tolist(), [0.0, 3.0, 4.5])

    def test_needs_no_fit(self):
        self.assertEqual(
            TimeseriesBinaryLabelProcessor(timeline=["x"])
            .process({"time": [1.0], "values": [1]})[0]
            .tolist(),
            [[1.0]],
        )

    def test_rejects(self):
        bad: list[tuple[Any, Any]] = [
            (TimeseriesBinaryLabelProcessor(timeline=["x"]), {"time": [0.0, 1.0], "values": [0, 2]}),
            (
                TimeseriesBinaryLabelProcessor(timeline=["x"]),
                {"time": [0.0, 0.0], "values": [0, 1]},
            ),  # one time twice
            (
                TimeseriesRegressionLabelProcessor(timeline=["x"]),
                {"time": [1.0, 0.0], "values": [1.0, 2.0]},
            ),
            (
                TimeseriesRegressionLabelProcessor(timeline=["x"]),
                {"time": [0.0, 1.0], "values": [1.0, NAN]},
            ),
            (TimeseriesRegressionLabelProcessor(timeline=["x"]), {"values": [1.0]}),  # no time
            (
                TimeseriesRegressionLabelProcessor(timeline=["x"]),
                {"time": [0.0], "values": [[1.0, 2.0]]},
            ),
        ]
        for processor, records in bad:
            with self.subTest(records=records), self.assertRaises(ValueError):
                processor.process(records)

    def test_batches_are_packed(self):
        dataset = _dataset()
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        values, time, offsets = batch["label"]
        self.assertEqual(values.shape, (18, 1))
        self.assertEqual(time.shape, (18,))
        self.assertEqual(offsets.tolist(), [0, 5, 8, 14, 18])


class TestTimeline(unittest.TestCase):
    def test_keeps_the_declared_fields_in_order(self):
        for cls in TIMESERIES_CLASSES:
            with self.subTest(cls=cls.__name__):
                processor = cls(timeline=["static", "dynamic"])
                self.assertEqual(processor.timeline, ("static", "dynamic"))

    def test_is_required_and_keyword_only(self):
        for cls in TIMESERIES_CLASSES:
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(TypeError, "timeline"):
                    cls()  # type: ignore[call-arg]
                with self.assertRaises(TypeError):
                    cls(["dynamic"])  # type: ignore[misc]

    def test_rejects_bad_declarations(self):
        bad: list[tuple[Any, type[Exception]]] = [
            (None, TypeError),
            ("dynamic", TypeError),  # a field name, not a list of them
            ([], ValueError),
            (["dynamic", "dynamic"], ValueError),
            ([1], TypeError),
        ]
        for timeline, error in bad:
            for cls in TIMESERIES_CLASSES:
                with self.subTest(cls=cls.__name__, timeline=timeline):
                    with self.assertRaises(error):
                        cls(timeline=timeline)

    def test_fit_checks_the_fields(self):
        sample = {"dynamic": None, "label": {"time": [0.0], "values": [1]}}
        processors = [
            TimeseriesBinaryLabelProcessor(timeline=["typo"]),
            TimeseriesBinaryLabelProcessor(timeline=["typo"], pos_weight="balanced"),
            TimeseriesMultiClassLabelProcessor(timeline=["typo"]),
            TimeseriesMultiLabelProcessor(timeline=["typo"]),
            TimeseriesRegressionLabelProcessor(timeline=["typo"]),
            TimeseriesRegressionLabelProcessor(
                timeline=["typo"], normalization="minmax"
            ),
        ]
        for processor in processors:
            with self.subTest(processor=repr(processor)):
                with self.assertRaisesRegex(KeyError, "typo"):
                    processor.fit([sample], "label")
        for cls in TIMESERIES_CLASSES:
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, "label field"):
                    cls(timeline=["label"]).fit([sample], "label")
                cls(timeline=["dynamic"]).fit([], "label")  # nothing to check

    def test_dataset_keeps_the_timeline(self):
        dataset = _dataset(timeline=("dynamic", "static"))
        processor = dataset.output_processors["label"]
        assert isinstance(processor, TimeseriesLabelProcessor)
        self.assertEqual(processor.timeline, ("dynamic", "static"))
        with self.assertRaisesRegex(KeyError, "dynamics"):
            _dataset(timeline=("dynamics",))

    def test_yaib_per_step_tasks_put_dynamic_on_the_timeline(self):
        for source in ("MIMIC3", "MIMIC4", "EICU"):
            for name in ("LengthOfStay", "AKI", "Sepsis"):
                task = getattr(tasks, f"YAIB{name}{source}")
                with self.subTest(task=task.__name__):
                    ((_, (_, kwargs)),) = task.output_schema.items()
                    self.assertEqual(kwargs["timeline"], ["dynamic"])


class TestPerStepEndToEnd(unittest.TestCase):
    """The label processors with a per-step model and the Trainer.

    The loss over label records is covered in test_timeseries_modes.py.
    """

    def test_label_processors_train_and_evaluate(self):
        models: dict[str, Callable[[SampleDataset], BaseModel]] = {
            "LabelRecordModel": LabelRecordModel,
            "RNN_V2": lambda dataset: RNN_V2(
                dataset,
                embedding_dim={"dynamic": [{"values": 8}], "static": [{"values": 4}]},
                hidden_dim=8,
            ),
        }
        for (mode, metric), (name, make_model) in product(
            (("timeseries_binary", "roc_auc"), ("timeseries_regression", "mae")),
            models.items(),
        ):
            with self.subTest(mode=mode, model=name):
                dataset = _dataset(mode)
                model = make_model(dataset)
                self.assertEqual(model.mode, mode)
                batch = next(iter(get_dataloader(dataset, batch_size=4)))
                out = model(**batch)
                self.assertEqual(tuple(out["y_prob"].shape), (18, 1))
                self.assertTrue(torch.isfinite(out["loss"]))

                trainer = Trainer(model=model, metrics=[metric], enable_logging=False)
                loader = get_dataloader(dataset, batch_size=3)
                trainer.train(train_dataloader=loader, epochs=1)
                y_true, _, _ = trainer.inference(loader)
                self.assertEqual(sorted(len(y) for y in y_true), [3, 4, 5, 6])
                self.assertIn(metric, trainer.evaluate(loader))


if __name__ == "__main__":
    unittest.main()
