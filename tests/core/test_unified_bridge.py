import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from pyhealth.datasets import collate_fn_dict_with_padding
from pyhealth.processors import (
    AudioProcessor,
    BinaryLabelProcessor,
    BridgeDataset,
    CehrProcessor,
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
    GraphProcessor,
    ImageProcessor,
    JaggedFeatureProcessor,
    MultiHotProcessor,
    NestedFloatsProcessor,
    NestedMultiHotProcessor,
    NestedSequenceProcessor,
    RawProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TemporalTimeseriesProcessor,
    TensorProcessor,
    TimeImageProcessor,
    TimeseriesProcessor,
    TupleTimeTextProcessor,
    UnifiedBridge,
    transform_batch,
    transform_processor,
)
from pyhealth.processors.unified_bridge import (
    AudioToUnifiedBridge,
    CehrToUnifiedBridge,
    DeepNestedFloatsToUnifiedBridge,
    DeepNestedSequenceToUnifiedBridge,
    ImageToUnifiedBridge,
    MultiHotToUnifiedBridge,
    NestedFloatsToUnifiedBridge,
    NestedMultiHotToUnifiedBridge,
    NestedSequenceToUnifiedBridge,
    SequenceToUnifiedBridge,
    StageNetTensorToUnifiedBridge,
    StageNetToUnifiedBridge,
    TemporalTimeseriesToUnifiedBridge,
    TensorToUnifiedBridge,
    TimeImageToUnifiedBridge,
    TimeseriesToUnifiedBridge,
)

T0 = datetime(2020, 1, 1)


def _fit(processor, *raws, field="x"):
    processor.fit([{field: raw} for raw in raws], field)
    return processor


def _collate(processor, *raws):
    """The legacy batch of one field, as get_dataloader collates it."""
    return collate_fn_dict_with_padding([{"x": processor.process(raw)} for raw in raws])[
        "x"
    ]


def _decode(processor: JaggedFeatureProcessor, out: tuple) -> list[list[Any]]:
    """A batch tuple as nested Python records, one list per sample.

    A record is its token string, value list or child list when it has only
    one of them, and a dict of its roles otherwise. The batch is validated
    first.
    """
    processor.validate(out, batched=True)
    fields = dict(zip(processor.batch_schema(), out))
    words = {
        level: {index: token for token, index in processor.vocab(level).items()}
        for level, sizes in processor.vocab_size().items()
        if sizes is not None
    }

    def record(level: int, i: int) -> Any:
        parts: dict[str, Any] = {}
        if (level, "tokens") in fields:
            parts["tokens"] = words[level][int(fields[(level, "tokens")][i])]
        if (level, "values") in fields:
            parts["values"] = fields[(level, "values")][i].tolist()
        if (level, "mask") in fields:
            parts["mask"] = fields[(level, "mask")][i].tolist()
        if (level, "time") in fields:
            parts["time"] = float(fields[(level, "time")][i])
        if level > 0:
            offsets = fields[(level - 1, "offsets")]
            parts["children"] = [
                record(level - 1, j) for j in range(int(offsets[i]), int(offsets[i + 1]))
            ]
        return next(iter(parts.values())) if len(parts) == 1 else parts

    batch = fields[(processor.depth, "offsets")]
    return [
        [record(processor.depth, i) for i in range(int(batch[b]), int(batch[b + 1]))]
        for b in range(len(batch) - 1)
    ]


def _bridge_batch(bridge_cls, legacy, *raws):
    bridge = bridge_cls(legacy)
    return bridge, _decode(bridge, bridge.transform_batch(_collate(legacy, *raws)))


class TestBridgeBase(unittest.TestCase):
    def setUp(self):
        self.legacy = _fit(SequenceProcessor(), ["A", "B"])
        self.bridge = SequenceToUnifiedBridge(self.legacy)

    def test_is_a_jagged_processor_with_metadata(self):
        self.assertIsInstance(self.bridge, JaggedFeatureProcessor)
        self.assertIs(self.bridge.legacy, self.legacy)
        self.assertEqual(self.bridge.depth, 0)
        self.assertEqual(self.bridge.levels, [{"ordering": "sequence", "tokens": 1}])
        self.assertEqual(self.bridge.schema(), ((0, "tokens"),))
        self.assertEqual(self.bridge.batch_schema(), ((0, "tokens"), (0, "offsets")))
        self.assertEqual(self.bridge.ordering(), {0: "sequence"})
        self.assertEqual(self.bridge.vocab_size(), {0: (4,)})
        self.assertEqual(self.bridge.value_dim(), {0: None})
        self.assertEqual(self.bridge.spatial(), {0: None})
        self.assertEqual(self.bridge.vocab(0), self.legacy.code_vocab)
        self.assertFalse(self.bridge.learns_statistics)

    def test_processing_methods_raise(self):
        for call in (
            lambda: self.bridge.fit([{"x": ["A"]}], "x"),
            lambda: self.bridge.process(["A"]),
            lambda: self.bridge.collate([]),
        ):
            with self.assertRaisesRegex(NotImplementedError, "SequenceProcessor processes it"):
                call()

    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            UnifiedBridge(self.legacy, [{"tokens": 1}])  # type: ignore[abstract]

    def test_repr_names_the_legacy_processor(self):
        self.assertEqual(repr(self.bridge), f"SequenceToUnifiedBridge({self.legacy!r})")

    def test_vocabulary_without_pad_and_unk_at_0_and_1_is_rejected(self):
        legacy = _fit(SequenceProcessor(), ["A"])
        legacy.code_vocab = {"A": 0, "<pad>": 1, "<unk>": 2}
        with self.assertRaisesRegex(ValueError, "<pad> at 0"):
            SequenceToUnifiedBridge(legacy)
        legacy.code_vocab = {"<pad>": 0, "<unk>": 1, "A": 3}
        with self.assertRaisesRegex(ValueError, "contiguous"):
            SequenceToUnifiedBridge(legacy)


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.legacy = _fit(SequenceProcessor(), ["A", "B", "C"])

    def test_padding_is_dropped(self):
        _, got = _bridge_batch(
            SequenceToUnifiedBridge, self.legacy, ["A", "B", "C"], ["C"], ["B", "A"]
        )
        self.assertEqual(got, [["A", "B", "C"], ["C"], ["B", "A"]])

    def test_unknown_and_missing_tokens(self):
        _, got = _bridge_batch(SequenceToUnifiedBridge, self.legacy, ["Z", None, "A"])
        self.assertEqual(got, [["<unk>", "A"]])

    def test_empty_samples(self):
        _, got = _bridge_batch(SequenceToUnifiedBridge, self.legacy, [], ["A"], [])
        self.assertEqual(got, [[], ["A"], []])

    def test_pad_in_between_is_kept_and_trailing_is_dropped(self):
        _, got = _bridge_batch(
            SequenceToUnifiedBridge, self.legacy, ["A", "<pad>", "B", "<pad>"]
        )
        self.assertEqual(got, [["A", "<pad>", "B"]])

    def test_all_samples_empty(self):
        _, got = _bridge_batch(SequenceToUnifiedBridge, self.legacy, [], [])
        self.assertEqual(got, [[], []])

    def test_vocabulary_edits_carry_over(self):
        self.legacy.remove({"B"})
        bridge, got = _bridge_batch(SequenceToUnifiedBridge, self.legacy, ["A", "B", "C"])
        self.assertEqual(got, [["A", "<unk>", "C"]])
        self.assertEqual(bridge.vocab_size(), {0: (4,)})

    def test_matches_the_jagged_processor(self):
        raws = [["A", "B"], [], ["C", "A"]]
        unified = JaggedFeatureProcessor(levels=[{"ordering": "sequence", "tokens": 1}])
        records = [{"tokens": raw} for raw in raws]
        unified.fit([{"x": r} for r in records], "x")
        expected = _decode(unified, unified.collate([unified.process(r) for r in records]))
        _, got = _bridge_batch(SequenceToUnifiedBridge, self.legacy, *raws)
        self.assertEqual(got, expected)


class TestCehr(unittest.TestCase):
    def setUp(self):
        self.legacy = CehrProcessor()
        self.legacy.fit([{"x": ["A", "B"]}], "x")

    def test_left_padding_is_kept_and_trailing_padding_dropped(self):
        bridge, got = _bridge_batch(
            CehrToUnifiedBridge,
            self.legacy,
            ["<pad>", "<pad>", "<cls>", "A"],
            ["<cls>", "B", "Z"],
            ["<cls>", "A", "<pad>"],
        )
        self.assertEqual(
            got,
            [
                ["<pad>", "<pad>", "<cls>", "A"],
                ["<cls>", "B", "<unk>"],
                ["<cls>", "A"],  # a real trailing <pad> looks like batch padding
            ],
        )
        self.assertEqual(bridge.ordering(), {0: "sequence"})

    def test_vocabulary_includes_special_tokens(self):
        bridge = CehrToUnifiedBridge(self.legacy)
        self.assertEqual(bridge.vocab(0), self.legacy.vocab.token_to_id)
        self.assertEqual(bridge.vocab_size(), {0: (self.legacy.vocab.vocab_size,)})

    def test_vocabulary_with_unused_ids_is_rejected(self):
        self.legacy.vocab._next_id += 1
        with self.assertRaisesRegex(ValueError, "unused ids"):
            CehrToUnifiedBridge(self.legacy)


class TestNestedSequence(unittest.TestCase):
    def setUp(self):
        self.legacy = _fit(NestedSequenceProcessor(), [["A", "B", "C"], ["D"]])

    def test_visits_and_padding(self):
        bridge, got = _bridge_batch(
            NestedSequenceToUnifiedBridge,
            self.legacy,
            [["A", "B"], ["C"], ["D", "A", "B"]],
            [["D"]],
        )
        self.assertEqual(got, [[["A", "B"], ["C"], ["D", "A", "B"]], [["D"]]])
        self.assertEqual(bridge.ordering(), {0: "set", 1: "sequence"})
        self.assertEqual(
            bridge.batch_schema(), ((0, "tokens"), (0, "offsets"), (1, "offsets"))
        )

    def test_empty_visit_in_between_is_kept(self):
        _, got = _bridge_batch(
            NestedSequenceToUnifiedBridge, self.legacy, [["A"], [], None, ["B"]]
        )
        self.assertEqual(got, [[["A"], [], [], ["B"]]])

    def test_trailing_empty_visits_are_dropped(self):
        # A documented limitation: they look like the batch padding.
        _, got = _bridge_batch(NestedSequenceToUnifiedBridge, self.legacy, [["A"], [], []])
        self.assertEqual(got, [[["A"]]])

    def test_empty_samples(self):
        _, got = _bridge_batch(
            NestedSequenceToUnifiedBridge, self.legacy, [], [["A"]], [[]]
        )
        self.assertEqual(got, [[], [["A"]], []])

    def test_unknown_codes_and_truncation(self):
        _, got = _bridge_batch(
            NestedSequenceToUnifiedBridge, self.legacy, [["Z", None, "A", "B"]]
        )
        self.assertEqual(got, [[["<unk>", "<unk>", "A"]]])  # cut to the fitted width

    def test_extra_padding_width(self):
        legacy = _fit(NestedSequenceProcessor(padding=3), [["A"]])
        _, got = _bridge_batch(NestedSequenceToUnifiedBridge, legacy, [["A"], ["A"]])
        self.assertEqual(got, [[["A"], ["A"]]])

    def test_matches_the_jagged_processor(self):
        raws = [[["A", "B"], [], ["C"]], [["D"]]]
        unified = JaggedFeatureProcessor(
            levels=[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]
        )
        records = [{"children": [{"tokens": visit} for visit in raw]} for raw in raws]
        unified.fit([{"x": r} for r in records], "x")
        expected = _decode(unified, unified.collate([unified.process(r) for r in records]))
        _, got = _bridge_batch(NestedSequenceToUnifiedBridge, self.legacy, *raws)
        self.assertEqual(got, expected)


class TestDeepNestedSequence(unittest.TestCase):
    def setUp(self):
        self.legacy = _fit(
            DeepNestedSequenceProcessor(),
            [[["A", "B"], ["C"], ["D"]], [["A"]]],
        )

    def test_groups_visits_and_padding(self):
        bridge, got = _bridge_batch(
            DeepNestedSequenceToUnifiedBridge,
            self.legacy,
            [[["A", "B"], ["C"]], [["D"]]],
            [[["C"]]],
        )
        self.assertEqual(got, [[[["A", "B"], ["C"]], [["D"]]], [[["C"]]]])
        self.assertEqual(bridge.ordering(), {0: "set", 1: "sequence", 2: "sequence"})

    def test_empty_visits_and_groups_in_between_are_kept(self):
        _, got = _bridge_batch(
            DeepNestedSequenceToUnifiedBridge,
            self.legacy,
            [[["A"], [], ["B"]], [], [["C"]]],
        )
        self.assertEqual(got, [[[["A"], [], ["B"]], [], [["C"]]]])

    def test_trailing_empty_visits_and_groups_are_dropped(self):
        _, got = _bridge_batch(
            DeepNestedSequenceToUnifiedBridge, self.legacy, [[["A"], []], [], None]
        )
        self.assertEqual(got, [[[["A"]]]])

    def test_empty_samples(self):
        # The legacy processor gives an empty sample one visit per group, not
        # the fitted count, so the legacy collate only batches it with other
        # empty samples.
        _, got = _bridge_batch(DeepNestedSequenceToUnifiedBridge, self.legacy, [], None)
        self.assertEqual(got, [[], []])


class TestNestedFloats(unittest.TestCase):
    def test_rows_and_padding(self):
        legacy = _fit(NestedFloatsProcessor(forward_fill=False), [[1.0, 2.0], [3.0]])
        bridge, got = _bridge_batch(
            NestedFloatsToUnifiedBridge,
            legacy,
            [[1.0, 2.0], [3.0, 4.0], [5.0]],
            [[6.0, 7.0]],
        )
        self.assertEqual(
            got, [[[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]], [[6.0, 7.0]]]
        )
        self.assertEqual(bridge.value_dim(), {0: (2,)})
        self.assertEqual(bridge.spatial(), {0: (False,)})
        self.assertEqual(bridge.ordering(), {0: "sequence"})

    def test_zero_row_in_between_is_kept_but_trailing_is_dropped(self):
        legacy = _fit(NestedFloatsProcessor(forward_fill=False), [[1.0]])
        _, got = _bridge_batch(
            NestedFloatsToUnifiedBridge, legacy, [[1.0], [0.0], [2.0], [0.0], []]
        )
        self.assertEqual(got, [[[1.0], [0.0], [2.0]]])

    def test_empty_sample_with_forward_fill_is_nan_padding(self):
        legacy = _fit(NestedFloatsProcessor(forward_fill=True), [[1.0, 2.0]])
        _, got = _bridge_batch(NestedFloatsToUnifiedBridge, legacy, [], [[1.0]])
        self.assertEqual(got, [[], [[1.0, 1.0]]])

    def test_forward_fill_carries_into_empty_visits(self):
        legacy = _fit(NestedFloatsProcessor(forward_fill=True), [[1.0]])
        _, got = _bridge_batch(NestedFloatsToUnifiedBridge, legacy, [[1.0], [], [None]])
        self.assertEqual(got, [[[1.0], [1.0], [1.0]]])

    def test_processor_without_fitted_values_is_rejected(self):
        # The width would default to 1 instead of coming from the data.
        for legacy in (
            NestedFloatsProcessor(),
            _fit(NestedFloatsProcessor(padding=2), [], [[], []], None),
        ):
            with self.assertRaisesRegex(ValueError, "fit\\(\\) saw no values"):
                NestedFloatsToUnifiedBridge(legacy)

    def test_processor_fitted_before_values_were_recorded(self):
        legacy = _fit(NestedFloatsProcessor(forward_fill=False), [[1.0, 2.0]])
        delattr(legacy, "_saw_values")  # falls back to the class default, None
        _, got = _bridge_batch(NestedFloatsToUnifiedBridge, legacy, [[3.0]])
        self.assertEqual(got, [[[3.0, 0.0]]])


class TestDeepNestedFloats(unittest.TestCase):
    def test_groups_and_padding(self):
        legacy = _fit(
            DeepNestedFloatsProcessor(forward_fill=False),
            [[[1.0, 2.0], [3.0], [4.0]]],
        )
        bridge, got = _bridge_batch(
            DeepNestedFloatsToUnifiedBridge,
            legacy,
            [[[1.0, 2.0], [3.0]], [[4.0]]],
            [[[5.0]]],
        )
        self.assertEqual(
            got,
            [[[[1.0, 2.0], [3.0, 0.0]], [[4.0, 0.0]]], [[[5.0, 0.0]]]],
        )
        self.assertEqual(bridge.ordering(), {0: "sequence", 1: "sequence"})
        self.assertEqual(bridge.value_dim(), {0: (2,), 1: None})

    def test_empty_groups(self):
        legacy = _fit(DeepNestedFloatsProcessor(forward_fill=False), [[[1.0], [2.0]]])
        _, got = _bridge_batch(
            DeepNestedFloatsToUnifiedBridge, legacy, [[[1.0]], [], [[2.0]], []], []
        )
        self.assertEqual(got, [[[[1.0]], [], [[2.0]]], []])

    def test_forward_filled_padding_visits_are_kept(self):
        # A documented limitation: forward fill makes padded visits nonzero.
        legacy = _fit(DeepNestedFloatsProcessor(forward_fill=True), [[[1.0], [2.0], [3.0]]])
        _, got = _bridge_batch(DeepNestedFloatsToUnifiedBridge, legacy, [[[5.0]]])
        self.assertEqual(got, [[[[5.0], [5.0], [5.0]]]])

    def test_processor_without_fitted_values_is_rejected(self):
        for legacy in (
            DeepNestedFloatsProcessor(),
            _fit(DeepNestedFloatsProcessor(), [], [[], [[], []]], None),
        ):
            with self.assertRaisesRegex(ValueError, "fit\\(\\) saw no values"):
                DeepNestedFloatsToUnifiedBridge(legacy)

    def test_processor_fitted_before_values_were_recorded(self):
        legacy = _fit(DeepNestedFloatsProcessor(forward_fill=False), [[[1.0, 2.0]]])
        delattr(legacy, "_saw_values")  # falls back to the class default, None
        _, got = _bridge_batch(DeepNestedFloatsToUnifiedBridge, legacy, [[[3.0]]])
        self.assertEqual(got, [[[[3.0, 0.0]]]])


class TestMultiHot(unittest.TestCase):
    def test_string_categories(self):
        legacy = _fit(MultiHotProcessor(), ["b", "a"], ["c"])
        bridge, got = _bridge_batch(
            MultiHotToUnifiedBridge, legacy, ["c", "a"], [], ["b"]
        )
        self.assertEqual(got, [["a", "c"], [], ["b"]])  # a set: column order
        self.assertEqual(
            bridge.vocab(0), {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4}
        )
        self.assertEqual(bridge.ordering(), {0: "set"})

    def test_integer_categories_are_shifted(self):
        legacy = _fit(MultiHotProcessor(), [0, 1], [2])
        bridge, got = _bridge_batch(MultiHotToUnifiedBridge, legacy, [0, 2])
        self.assertEqual(got, [[0, 2]])
        self.assertEqual(bridge.vocab(0), {"<pad>": 0, "<unk>": 1, 0: 2, 1: 3, 2: 4})


class TestNestedMultiHot(unittest.TestCase):
    def setUp(self):
        self.legacy = _fit(NestedMultiHotProcessor(), [["A", "B"], ["C"]])

    def test_visits_unknowns_and_repeats(self):
        bridge, got = _bridge_batch(
            NestedMultiHotToUnifiedBridge,
            self.legacy,
            [["B", "A", "A"], ["Z"], ["C"]],
            [["C"]],
        )
        self.assertEqual(got, [[["A", "B"], ["<unk>"], ["C"]], [["C"]]])
        self.assertEqual(bridge.vocab(0), self.legacy.code_vocab)

    def test_empty_visits_and_samples(self):
        _, got = _bridge_batch(
            NestedMultiHotToUnifiedBridge, self.legacy, [["A"], [], ["B"], []], []
        )
        self.assertEqual(got, [[["A"], [], ["B"]], []])


class TestStageNet(unittest.TestCase):
    def test_flat_codes_with_time(self):
        legacy = _fit(StageNetProcessor(), ([0.0, 1.0], ["A", "B"]))
        bridge, got = _bridge_batch(
            StageNetToUnifiedBridge,
            legacy,
            ([0.0, 1.5, 4.0], ["A", "Z", "B"]),
            ([2.0], ["B"]),
        )
        self.assertEqual(
            got,
            [
                [
                    {"tokens": "A", "time": 0.0},
                    {"tokens": "<unk>", "time": 1.5},
                    {"tokens": "B", "time": 4.0},
                ],
                [{"tokens": "B", "time": 2.0}],
            ],
        )
        self.assertEqual(bridge.ordering(), {0: "series"})

    def test_batch_without_time_gets_zero(self):
        legacy = _fit(StageNetProcessor(), ([0.0], ["A"]))
        _, got = _bridge_batch(
            StageNetToUnifiedBridge, legacy, ([1.0], ["A"]), (None, ["A", "A"])
        )
        self.assertEqual(
            got,
            [[{"tokens": "A", "time": 0.0}], [{"tokens": "A", "time": 0.0}] * 2],
        )

    def test_empty_sample_has_no_records(self):
        legacy = _fit(StageNetProcessor(), ([0.0], ["A"]))
        _, got = _bridge_batch(StageNetToUnifiedBridge, legacy, ([], []), ([3.0], ["A"]))
        self.assertEqual(got, [[], [{"tokens": "A", "time": 0.0}]])

    def test_nested_codes(self):
        legacy = _fit(StageNetProcessor(), ([0.0, 1.0], [["A", "B"], ["C"]]))
        bridge, got = _bridge_batch(
            StageNetToUnifiedBridge,
            legacy,
            ([0.0, 2.0, 5.0], [["A"], [], ["B", "C"]]),
            ([7.0], [["C"]]),
        )
        self.assertEqual(
            got,
            [
                [
                    {"children": ["A"], "time": 0.0},
                    {"children": [], "time": 2.0},
                    {"children": ["B", "C"], "time": 5.0},
                ],
                [{"children": ["C"], "time": 7.0}],
            ],
        )
        self.assertEqual(bridge.ordering(), {0: "set", 1: "series"})

    def test_time_shorter_than_codes_is_zero_filled(self):
        legacy = _fit(StageNetProcessor(), ([0.0], ["A"]))
        _, got = _bridge_batch(StageNetToUnifiedBridge, legacy, ([3.0], ["A", "B"]))
        self.assertEqual(
            got, [[{"tokens": "A", "time": 3.0}, {"tokens": "<unk>", "time": 0.0}]]
        )

    def test_unfitted_structure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nested is unknown"):
            StageNetToUnifiedBridge(StageNetProcessor())


class TestStageNetTensor(unittest.TestCase):
    def test_nested_values_with_time(self):
        legacy = _fit(StageNetTensorProcessor(), ([0.0], [[1.0, 2.0]]))
        bridge, got = _bridge_batch(
            StageNetTensorToUnifiedBridge,
            legacy,
            ([0.0, 1.0, 3.0], [[1.0, None], [None, 4.0], [5.0, 6.0]]),
            ([2.0], [[7.0, 8.0]]),
        )
        self.assertEqual(
            got,
            [
                [
                    {"values": [1.0, 0.0], "time": 0.0},
                    {"values": [1.0, 4.0], "time": 1.0},  # forward-filled
                    {"values": [5.0, 6.0], "time": 3.0},
                ],
                [{"values": [7.0, 8.0], "time": 2.0}],
            ],
        )
        self.assertEqual(bridge.value_dim(), {0: (2,)})

    def test_flat_values(self):
        legacy = _fit(StageNetTensorProcessor(), ([0.0], [1.0]))
        _, got = _bridge_batch(
            StageNetTensorToUnifiedBridge, legacy, ([0.0, 1.0], [1.0, 2.0]), ([5.0], [3.0])
        )
        self.assertEqual(
            got,
            [
                [{"values": [1.0], "time": 0.0}, {"values": [2.0], "time": 1.0}],
                [{"values": [3.0], "time": 5.0}],
            ],
        )

    def test_trailing_zero_rows_are_kept_when_their_time_is_not_zero(self):
        legacy = _fit(StageNetTensorProcessor(), ([0.0], [1.0]))
        _, got = _bridge_batch(
            StageNetTensorToUnifiedBridge,
            legacy,
            ([0.0, 1.0], [0.0, 0.0]),
            ([0.0], [0.0]),  # indistinguishable from padding: dropped
            ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]),
        )
        self.assertEqual(
            got,
            [
                [{"values": [0.0], "time": 0.0}, {"values": [0.0], "time": 1.0}],
                [],
                [
                    {"values": [1.0], "time": 0.0},
                    {"values": [2.0], "time": 1.0},
                    {"values": [3.0], "time": 2.0},
                ],
            ],
        )

    def test_batch_without_time(self):
        legacy = _fit(StageNetTensorProcessor(), (None, [[1.0, 2.0]]))
        _, got = _bridge_batch(
            StageNetTensorToUnifiedBridge,
            legacy,
            (None, [[1.0, 2.0], [3.0, 4.0]]),
            (None, [[5.0, 6.0]]),
        )
        self.assertEqual(
            got,
            [
                [{"values": [1.0, 2.0], "time": 0.0}, {"values": [3.0, 4.0], "time": 0.0}],
                [{"values": [5.0, 6.0], "time": 0.0}],
            ],
        )

    def test_unfitted_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "feature size is unknown"):
            StageNetTensorToUnifiedBridge(StageNetTensorProcessor())


class TestTimeseries(unittest.TestCase):
    @staticmethod
    def _raw(hours, values):
        return ([T0 + timedelta(hours=h) for h in hours], values)

    def test_grid_time_and_padding(self):
        raws = [self._raw([0, 4], [[1.0, 2.0], [3.0, 4.0]]), self._raw([0], [[5.0, 6.0]])]
        legacy = _fit(TimeseriesProcessor(sampling_rate=timedelta(hours=2)), *raws)
        bridge, got = _bridge_batch(TimeseriesToUnifiedBridge, legacy, *raws)
        self.assertEqual(
            got,
            [
                [
                    {"values": [1.0, 2.0], "time": 0.0},
                    {"values": [1.0, 2.0], "time": 2.0},  # forward-filled step
                    {"values": [3.0, 4.0], "time": 4.0},
                ],
                [{"values": [5.0, 6.0], "time": 0.0}],
            ],
        )
        self.assertEqual(bridge.ordering(), {0: "series"})
        self.assertEqual(bridge.value_dim(), {0: (2,)})

    def test_missing_indicators_become_the_mask(self):
        raw = self._raw([0, 2], [[1.0, float("nan")], [2.0, 3.0]])
        legacy = _fit(
            TimeseriesProcessor(
                sampling_rate=timedelta(hours=1),
                missing_indicators=True,
                impute_strategy="zero",
            ),
            raw,
        )
        bridge, got = _bridge_batch(TimeseriesToUnifiedBridge, legacy, raw)
        self.assertEqual(
            got,
            [
                [
                    {"values": [1.0, 0.0], "mask": [False, True], "time": 0.0},
                    {"values": [0.0, 0.0], "mask": [True, True], "time": 1.0},
                    {"values": [2.0, 3.0], "mask": [False, False], "time": 2.0},
                ]
            ],
        )
        self.assertEqual(
            bridge.schema(), ((0, "values"), (0, "mask"), (0, "time"))
        )
        self.assertEqual(bridge.levels[0]["impute"], {"method": "zero"})

    def test_normalized_values_pass_through(self):
        raws = [self._raw([0], [[1.0]]), self._raw([0], [[3.0]])]
        legacy = _fit(TimeseriesProcessor(normalization="zscore"), *raws)
        _, got = _bridge_batch(TimeseriesToUnifiedBridge, legacy, *raws)
        self.assertEqual(
            got, [[{"values": [-1.0], "time": 0.0}], [{"values": [1.0], "time": 0.0}]]
        )

    def test_trailing_zero_steps_are_dropped(self):
        # A documented limitation: a real trailing step of zeros looks like padding.
        raw = self._raw([0, 1], [[1.0], [0.0]])
        legacy = _fit(TimeseriesProcessor(), raw)
        _, got = _bridge_batch(TimeseriesToUnifiedBridge, legacy, raw)
        self.assertEqual(got, [[{"values": [1.0], "time": 0.0}]])

    def test_unfitted_processor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "feature count is unknown"):
            TimeseriesToUnifiedBridge(TimeseriesProcessor())


class TestTemporalTimeseries(unittest.TestCase):
    def test_records_are_exact(self):
        raws = [
            ([T0, T0 + timedelta(hours=2)], [[0.0, 1.0], [0.0, 0.0]]),
            ([T0], [[5.0, 6.0]]),
        ]
        legacy = _fit(TemporalTimeseriesProcessor(), *raws)
        _, got = _bridge_batch(TemporalTimeseriesToUnifiedBridge, legacy, *raws)
        self.assertEqual(
            got,
            [
                [
                    {"values": [0.0, 1.0], "time": 0.0},
                    {"values": [0.0, 1.0], "time": 1.0},
                    {"values": [0.0, 0.0], "time": 2.0},  # zeros kept: no guessing
                ],
                [{"values": [5.0, 6.0], "time": 0.0}],
            ],
        )

    def test_unfitted_processor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "feature count is unknown"):
            TemporalTimeseriesToUnifiedBridge(TemporalTimeseriesProcessor())


class TestTensor(unittest.TestCase):
    def test_vectors(self):
        legacy = _fit(TensorProcessor(), [1.0, 2.0])
        bridge, got = _bridge_batch(TensorToUnifiedBridge, legacy, [1.0, 2.0], [0.0, 0.0])
        self.assertEqual(got, [[[1.0, 2.0]], [[0.0, 0.0]]])  # static zeros are kept
        self.assertEqual(bridge.ordering(), {0: "set"})
        self.assertEqual(bridge.value_dim(), {0: (2,)})

    def test_scalars(self):
        legacy = _fit(TensorProcessor(), 1.0)
        bridge, got = _bridge_batch(TensorToUnifiedBridge, legacy, 1.0, 2.5)
        self.assertEqual(got, [[[1.0]], [[2.5]]])
        self.assertEqual(bridge.value_dim(), {0: (1,)})

    def test_missing_indicators_become_the_mask(self):
        legacy = _fit(
            TensorProcessor(missing_indicators=True, impute_strategy="zero"),
            [1.0, 2.0],
        )
        bridge, got = _bridge_batch(
            TensorToUnifiedBridge, legacy, [1.0, float("nan")], [3.0, 4.0]
        )
        self.assertEqual(
            got,
            [
                [{"values": [1.0, 0.0], "mask": [False, True]}],
                [{"values": [3.0, 4.0], "mask": [False, False]}],
            ],
        )
        self.assertEqual(bridge.value_dim(), {0: (2,)})

    def test_scalar_with_missing_indicator(self):
        legacy = _fit(TensorProcessor(missing_indicators=True), 1.0)
        _, got = _bridge_batch(TensorToUnifiedBridge, legacy, float("nan"), 2.0)
        # Without impute_strategy, NaN is left in the legacy value; it becomes 0.
        self.assertEqual(
            got,
            [[{"values": [0.0], "mask": [True]}], [{"values": [2.0], "mask": [False]}]],
        )

    def test_nan_without_imputation_becomes_zero(self):
        legacy = _fit(TensorProcessor(), [1.0, 2.0])
        _, got = _bridge_batch(TensorToUnifiedBridge, legacy, [float("nan"), 2.0])
        self.assertEqual(got, [[[0.0, 2.0]]])

    def test_normalized_values_pass_through(self):
        legacy = _fit(TensorProcessor(normalization="minmax"), [0.0], [4.0])
        _, got = _bridge_batch(TensorToUnifiedBridge, legacy, [2.0])
        self.assertEqual(got, [[[0.5]]])

    def test_integer_tensors_become_float_values(self):
        legacy = _fit(TensorProcessor(dtype=torch.long), [1, 2])
        _, got = _bridge_batch(TensorToUnifiedBridge, legacy, [3, 4])
        self.assertEqual(got, [[[3.0, 4.0]]])

    def test_spatial_flags_carry_over(self):
        legacy = _fit(TensorProcessor(spatial_dims=(True,)), [1.0, 2.0, 3.0])
        self.assertEqual(TensorToUnifiedBridge(legacy).spatial(), {0: (True,)})

    def test_fixed_matrix_is_one_record(self):
        legacy = _fit(TensorProcessor(), [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [5.0, 6.0]])
        bridge, got = _bridge_batch(
            TensorToUnifiedBridge, legacy, [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]
        )
        self.assertEqual(
            got, [[[[1.0, 2.0], [3.0, 4.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]
        )
        self.assertEqual(bridge.ordering(), {0: "set"})
        self.assertEqual(bridge.value_dim(), {0: (2, 2)})
        self.assertEqual(bridge.spatial(), {0: (False, False)})

    def test_fixed_spatial_flags_carry_over(self):
        legacy = _fit(TensorProcessor(spatial_dims=(True, False)), [[1.0, 2.0]])
        self.assertEqual(TensorToUnifiedBridge(legacy).spatial(), {0: (True, False)})

    def test_variable_rows_are_a_sequence_when_spatial(self):
        legacy = _fit(
            TensorProcessor(spatial_dims=(True, False)),
            [[1.0, 2.0]],
            [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        )
        bridge, got = _bridge_batch(
            TensorToUnifiedBridge,
            legacy,
            [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            [[5.0, 6.0]],
        )
        self.assertEqual(got, [[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], [[5.0, 6.0]]])
        self.assertEqual(bridge.ordering(), {0: "sequence"})
        self.assertEqual(bridge.value_dim(), {0: (2,)})
        self.assertEqual(bridge.spatial(), {0: (False,)})

    def test_variable_rows_are_a_set_otherwise(self):
        legacy = _fit(TensorProcessor(), [[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]])
        bridge = TensorToUnifiedBridge(legacy)
        self.assertEqual(bridge.ordering(), {0: "set"})

    def test_variable_rows_drop_trailing_zero_rows(self):
        # A documented limitation: a real trailing zero row looks like padding.
        legacy = _fit(TensorProcessor(), [[1.0]], [[2.0], [3.0]])
        _, got = _bridge_batch(TensorToUnifiedBridge, legacy, [[1.0], [0.0]], [[2.0]])
        self.assertEqual(got, [[[1.0]], [[2.0]]])

    def test_variable_length_vectors(self):
        legacy = _fit(TensorProcessor(spatial_dims=(True,)), [1.0], [2.0, 3.0])
        bridge, got = _bridge_batch(TensorToUnifiedBridge, legacy, [1.0, 2.0, 3.0], [4.0], [])
        self.assertEqual(got, [[[1.0], [2.0], [3.0]], [[4.0]], []])
        self.assertEqual(bridge.ordering(), {0: "sequence"})
        self.assertEqual(bridge.value_dim(), {0: (1,)})

    def test_variable_rows_with_missing_indicators(self):
        legacy = _fit(
            TensorProcessor(missing_indicators=True, impute_strategy="zero"),
            [[1.0, 2.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        )
        bridge, got = _bridge_batch(
            TensorToUnifiedBridge,
            legacy,
            [[1.0, float("nan")], [3.0, 4.0]],
            [[float("nan"), float("nan")]],
        )
        self.assertEqual(
            got,
            [
                [
                    {"values": [1.0, 0.0], "mask": [False, True]},
                    {"values": [3.0, 4.0], "mask": [False, False]},
                ],
                # all missing: the indicators keep the row from looking like padding
                [{"values": [0.0, 0.0], "mask": [True, True]}],
            ],
        )
        self.assertEqual(bridge.value_dim(), {0: (2,)})

    def test_fixed_matrix_with_missing_indicators(self):
        legacy = _fit(TensorProcessor(missing_indicators=True), [[1.0, 2.0], [3.0, 4.0]])
        _, got = _bridge_batch(
            TensorToUnifiedBridge, legacy, [[1.0, float("nan")], [3.0, 4.0]]
        )
        self.assertEqual(
            got,
            [[{"values": [[1.0, 0.0], [3.0, 4.0]], "mask": [[False, True], [False, False]]}]],
        )

    def test_batch_unlike_the_fitted_shape_is_rejected(self):
        fixed = _fit(TensorProcessor(), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, r"shape \(3,\) per record, but fit\(\) saw \(2,\)"):
            TensorToUnifiedBridge(fixed).transform_batch(torch.zeros(2, 3))
        rows = _fit(TensorProcessor(), [[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]])
        with self.assertRaisesRegex(ValueError, "fit"):
            TensorToUnifiedBridge(rows).transform_batch(torch.zeros(2, 4, 3))

    def test_unbridgeable_tensors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "varies between samples beyond"):
            TensorToUnifiedBridge(_fit(TensorProcessor(), [[1.0]], [[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "same axis"):
            TensorToUnifiedBridge(
                _fit(TensorProcessor(missing_indicators=True), [1.0], [1.0, 2.0])
            )
        with self.assertRaisesRegex(ValueError, "shape is unknown"):
            TensorToUnifiedBridge(TensorProcessor())

    def test_processors_fitted_before_the_shape_was_recorded(self):
        vector = _fit(TensorProcessor(), [1.0, 2.0])
        del vector.__dict__["_shape"]
        _, got = _bridge_batch(TensorToUnifiedBridge, vector, [3.0, 4.0])
        self.assertEqual(got, [[[3.0, 4.0]]])
        matrix = _fit(TensorProcessor(), [[1.0, 2.0]])
        del matrix.__dict__["_shape"]
        with self.assertRaisesRegex(ValueError, "did not record it"):
            TensorToUnifiedBridge(matrix)


class _Images(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.paths = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            path = Path(self._dir.name) / f"{i}.png"
            Image.new("RGB", (8, 6), color=color).save(path)
            self.paths.append(str(path))


class TestImage(_Images):
    def test_one_record_per_sample(self):
        legacy = ImageProcessor(image_size=4, mode="RGB")
        bridge = ImageToUnifiedBridge(legacy)
        out = bridge.transform_batch(_collate(legacy, *self.paths[:2]))
        bridge.validate(out, batched=True)
        values, batch = out
        self.assertEqual(values.shape, (2, 3, 4, 4))
        self.assertEqual(batch.tolist(), [0, 1, 2])
        torch.testing.assert_close(values[1], legacy.process(self.paths[1]))
        self.assertEqual(bridge.ordering(), {0: "set"})
        self.assertEqual(bridge.spatial(), {0: (False, True, True)})

    def test_channels_follow_the_mode(self):
        for mode, channels in (("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)):
            with self.subTest(mode):
                legacy = ImageProcessor(image_size=4, mode=mode)
                bridge = ImageToUnifiedBridge(legacy)
                self.assertEqual(bridge.value_dim(), {0: (channels, 4, 4)})
                out = bridge.transform_batch(_collate(legacy, self.paths[0]))
                bridge.validate(out, batched=True)

    def _gray_and_wide(self):
        gray = str(Path(self._dir.name) / "gray.png")
        Image.new("L", (8, 6), color=200).save(gray)
        wide = str(Path(self._dir.name) / "wide.png")
        Image.new("RGB", (10, 4), color=(9, 9, 9)).save(wide)
        return gray, wide

    def test_widest_mode_pads_narrower_batches(self):
        gray, _ = self._gray_and_wide()
        legacy = ImageProcessor(image_size=4)  # no mode: each image keeps its own
        legacy.fit([{"x": gray}, {"x": self.paths[0]}], "x")
        bridge = ImageToUnifiedBridge(legacy)
        self.assertEqual(bridge.value_dim(), {0: (3, 4, 4)})
        out = bridge.transform_batch(_collate(legacy, gray, gray))
        bridge.validate(out, batched=True)
        values, _ = out
        self.assertEqual(values.shape, (2, 3, 4, 4))
        torch.testing.assert_close(values[0, :1], legacy.process(gray))
        self.assertEqual(values[:, 1:].abs().sum().item(), 0.0)  # padded channels

    def test_widest_size_without_an_image_size(self):
        gray, wide = self._gray_and_wide()
        legacy = ImageProcessor(image_size=None, mode="L")
        legacy.fit([{"x": gray}, {"x": wide}], "x")
        bridge = ImageToUnifiedBridge(legacy)
        self.assertEqual(bridge.value_dim(), {0: (1, 6, 10)})  # max H, max W
        values, _ = bridge.transform_batch(_collate(legacy, gray))
        self.assertEqual(values.shape, (1, 1, 6, 10))
        torch.testing.assert_close(values[0, :, :6, :8], legacy.process(gray))
        self.assertEqual(values[0, :, :, 8:].abs().sum().item(), 0.0)  # padded width
        values, _ = bridge.transform_batch(_collate(legacy, wide))
        torch.testing.assert_close(values[0, :, :4, :], legacy.process(wide))

    def test_batch_wider_than_fitted_is_rejected(self):
        gray, _ = self._gray_and_wide()
        legacy = ImageProcessor(image_size=4)
        legacy.fit([{"x": gray}], "x")  # one channel seen
        bridge = ImageToUnifiedBridge(legacy)
        with self.assertRaisesRegex(ValueError, r"\(3, 4, 4\) per record, but fit\(\) saw \(1, 4, 4\)"):
            bridge.transform_batch(_collate(legacy, self.paths[0]))

    def test_processors_without_a_known_shape_are_rejected(self):
        for legacy, message in (
            (ImageProcessor(image_size=4), "fit\\(\\) has not recorded"),
            (ImageProcessor(image_size=None, mode="L"), "fit\\(\\) has not recorded"),
            (ImageProcessor(image_size=4, mode="L", to_tensor=False), "to_tensor"),
        ):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                ImageToUnifiedBridge(legacy)


class TestTimeImage(_Images):
    def test_series_of_images(self):
        legacy = TimeImageProcessor(image_size=4, mode="L")
        legacy.fit([{"x": (self.paths, [0.0, 1.0, 2.0])}], "x")
        bridge = TimeImageToUnifiedBridge(legacy)
        out = bridge.transform_batch(
            _collate(legacy, (self.paths[:2], [3.0, 1.0]), (self.paths[2:], [5.0]))
        )
        bridge.validate(out, batched=True)
        values, time, batch = out
        self.assertEqual(values.shape, (3, 1, 4, 4))
        self.assertEqual(time.tolist(), [1.0, 3.0, 5.0])  # sorted by the processor
        self.assertEqual(batch.tolist(), [0, 2, 3])
        self.assertEqual(bridge.ordering(), {0: "series"})
        self.assertEqual(bridge.spatial(), {0: (False, True, True)})

    def test_processors_without_a_fixed_shape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "channel count is unknown"):
            TimeImageToUnifiedBridge(TimeImageProcessor(image_size=4))
        legacy = TimeImageProcessor(image_size=None)
        legacy.n_channels = 3
        with self.assertRaisesRegex(ValueError, "image_size"):
            TimeImageToUnifiedBridge(legacy)


def _audio(**options) -> AudioProcessor:
    """An AudioProcessor's configuration, without needing torchaudio."""
    processor = AudioProcessor.__new__(AudioProcessor)
    defaults = {
        "sample_rate": 100,
        "duration": 2.0,
        "to_mono": True,
        "normalize": False,
        "mean": None,
        "std": None,
        "n_mels": None,
        "n_fft": 400,
        "hop_length": None,
    }
    processor.__dict__.update({**defaults, **options})
    return processor


class TestAudio(unittest.TestCase):
    def test_waveform(self):
        bridge = AudioToUnifiedBridge(_audio())
        self.assertEqual(bridge.value_dim(), {0: (1, 200)})
        self.assertEqual(bridge.spatial(), {0: (False, True)})
        out = bridge.transform_batch(torch.randn(3, 1, 200))
        bridge.validate(out, batched=True)
        self.assertEqual(out[1].tolist(), [0, 1, 2, 3])

    def test_mel_frames_match_torch_stft(self):
        for hop in (None, 40, 33):
            with self.subTest(hop=hop):
                legacy = _audio(n_mels=8, n_fft=64, hop_length=hop)
                frames = torch.stft(
                    torch.zeros(200),
                    n_fft=64,
                    hop_length=hop or 32,
                    window=torch.hann_window(64),
                    center=True,
                    return_complex=True,
                ).shape[-1]
                bridge = AudioToUnifiedBridge(legacy)
                self.assertEqual(bridge.value_dim(), {0: (1, 8, frames)})
                self.assertEqual(bridge.spatial(), {0: (False, True, True)})

    def test_processors_without_a_fixed_shape_are_rejected(self):
        for options in ({"to_mono": False}, {"duration": None}, {"sample_rate": None}):
            with self.subTest(options), self.assertRaisesRegex(ValueError, "fixed shape"):
                AudioToUnifiedBridge(_audio(**options))


class TestRNNV2Encoders(_Images):
    """Bridged batches run through RNN_V2's own field embedding and encoder."""

    def _cases(self):
        hours = lambda *h: [T0 + timedelta(hours=x) for x in h]
        series = [(hours(0, 2), [[1.0, 2.0], [3.0, float("nan")]]), (hours(0), [[5.0, 6.0]])]
        image = ImageProcessor(image_size=4, mode="L")
        unmoded = ImageProcessor(image_size=4)
        unmoded.fit([{"x": path} for path in self.paths], "x")
        time_image = TimeImageProcessor(image_size=4, mode="L")
        time_image.fit([{"x": (self.paths, [0.0, 1.0, 2.0])}], "x")
        cases = [
            (_fit(SequenceProcessor(), ["A", "B"]), [["A", "B"], [], ["B"]]),
            (_fit(CehrProcessor(), ["A"]), [["<pad>", "<cls>", "A"], ["A"]]),
            (_fit(NestedSequenceProcessor(), [["A", "B"]]), [[["A"], [], ["B"]], [], [["A"]]]),
            (_fit(DeepNestedSequenceProcessor(), [[["A"], ["B"]]]), [[[["A"], []]], [[["B"]]]]),
            (_fit(NestedFloatsProcessor(), [[1.0, 2.0]]), [[[1.0, 2.0], [3.0]], [[4.0]]]),
            (_fit(DeepNestedFloatsProcessor(), [[[1.0], [2.0]]]), [[[[1.0], [2.0]]], [[[3.0]]]]),
            (_fit(MultiHotProcessor(), ["a", "b"]), [["a"], [], ["a", "b"]]),
            (_fit(NestedMultiHotProcessor(), [["A"]]), [[["A"], []], []]),
            (_fit(StageNetProcessor(), ([0.0], ["A"])), [([0.0, 1.0], ["A", "A"]), ([2.0], ["A"])]),
            (
                _fit(StageNetProcessor(), ([0.0], [["A", "B"]])),
                [([0.0, 1.0], [["A"], ["B"]]), ([2.0], [["A"]])],
            ),
            (_fit(StageNetTensorProcessor(), ([0.0], [[1.0]])), [([0.0, 1.0], [[1.0], [2.0]]), ([3.0], [[4.0]])]),
            (
                _fit(TimeseriesProcessor(missing_indicators=True, impute_strategy="zero"), *series),
                series,
            ),
            (_fit(TemporalTimeseriesProcessor(), *series), series),
            (_fit(TensorProcessor(missing_indicators=True), [1.0, 2.0]), [[1.0, float("nan")], [3.0, 4.0]]),
            (_fit(TensorProcessor(), 1.0), [1.0, 2.0]),
            (image, self.paths[:2]),
            (_fit(TensorProcessor(), [[1.0, 2.0]]), [[[1.0, 2.0]], [[3.0, 4.0]]]),
            (
                _fit(TensorProcessor(spatial_dims=(True, False)), [[1.0]], [[2.0], [3.0]]),
                [[[1.0], [2.0]], [[3.0]]],
            ),
            (unmoded, self.paths[:2]),
            (time_image, [(self.paths[:2], [0.0, 1.0]), (self.paths[2:], [0.0])]),
        ]
        return cases

    def test_every_bridge_feeds_the_rnn_v2_encoder(self):
        from pyhealth.models.rnn import _FieldRNN
        from pyhealth.models.sequence_embedding import _FieldEmbedding

        for legacy, raws in self._cases():
            with self.subTest(type(legacy).__name__):
                bridge = transform_processor({"x": legacy})["x"]
                out = bridge.transform_batch(_collate(legacy, *raws))
                bridge.validate(out, batched=True)
                vocab, values = bridge.vocab_size(), bridge.value_dim()
                dims = [
                    {
                        **({"tokens": (3,)} if vocab[level] else {}),
                        **({"values": 4} if values[level] else {}),
                    }
                    for level in range(bridge.depth + 1)
                ]
                embedding = _FieldEmbedding(
                    "x", bridge, dims, [True] * (bridge.depth + 1), "concat"
                )
                encoder = _FieldRNN(
                    "x", bridge, embedding.sizes, 5, "GRU", 1, 0.0, False, True
                )
                summary, _, _ = encoder(embedding(out))
                # A set top level is mean-pooled, so its width is the embedding's.
                self.assertEqual(summary.shape, (len(raws), encoder.output_dim))
                self.assertTrue(torch.isfinite(summary).all())


class TestTransformProcessor(unittest.TestCase):
    def test_every_bridged_processor_class(self):
        expected = {
            SequenceProcessor: SequenceToUnifiedBridge,
            NestedSequenceProcessor: NestedSequenceToUnifiedBridge,
            NestedFloatsProcessor: NestedFloatsToUnifiedBridge,
            DeepNestedSequenceProcessor: DeepNestedSequenceToUnifiedBridge,
            DeepNestedFloatsProcessor: DeepNestedFloatsToUnifiedBridge,
            MultiHotProcessor: MultiHotToUnifiedBridge,
            NestedMultiHotProcessor: NestedMultiHotToUnifiedBridge,
            TensorProcessor: TensorToUnifiedBridge,
            TimeseriesProcessor: TimeseriesToUnifiedBridge,
            TemporalTimeseriesProcessor: TemporalTimeseriesToUnifiedBridge,
            StageNetProcessor: StageNetToUnifiedBridge,
            StageNetTensorProcessor: StageNetTensorToUnifiedBridge,
            CehrProcessor: CehrToUnifiedBridge,
            ImageProcessor: ImageToUnifiedBridge,
            TimeImageProcessor: TimeImageToUnifiedBridge,
            AudioProcessor: AudioToUnifiedBridge,
        }
        for legacy_cls, bridge_cls in expected.items():
            self.assertIs(bridge_cls.legacy_cls, legacy_cls)

    def test_bridges_legacy_and_keeps_the_rest(self):
        codes = _fit(SequenceProcessor(), ["A"])
        unified = JaggedFeatureProcessor(levels=[{"tokens": 1}])
        raw, label = RawProcessor(), BinaryLabelProcessor()
        processors = {"codes": codes, "unified": unified, "raw": raw, "label": label}
        bridged = transform_processor(processors)
        self.assertIsInstance(bridged["codes"], SequenceToUnifiedBridge)
        self.assertIs(bridged["codes"].legacy, codes)
        self.assertIs(bridged["unified"], unified)
        self.assertIs(bridged["raw"], raw)
        self.assertIs(bridged["label"], label)
        self.assertIs(processors["codes"], codes)  # the input is not changed

    def test_subclasses_use_the_nearest_bridge(self):
        class MyCodes(SequenceProcessor):
            pass

        bridged = transform_processor({"x": _fit(MyCodes(), ["A"])})
        self.assertIsInstance(bridged["x"], SequenceToUnifiedBridge)

    def test_bridges_become_passing_through_copies(self):
        bridge = SequenceToUnifiedBridge(_fit(SequenceProcessor(), ["A"]))
        passing = transform_processor({"x": bridge})["x"]
        self.assertIsInstance(passing, SequenceToUnifiedBridge)
        self.assertIsNot(passing, bridge)
        self.assertFalse(passing.converts)
        self.assertTrue(bridge.converts)  # the outer model keeps converting
        self.assertIs(passing.legacy, bridge.legacy)
        self.assertEqual(passing.batch_schema(), bridge.batch_schema())
        self.assertEqual(passing.vocab(0), bridge.vocab(0))
        self.assertEqual(passing.levels, bridge.levels)
        self.assertIn("converts=False", repr(passing))
        self.assertIs(transform_processor({"x": passing})["x"], passing)

    def test_processors_without_a_bridge_are_rejected(self):
        for legacy in (GraphProcessor.__new__(GraphProcessor), TupleTimeTextProcessor()):
            with (
                self.subTest(type(legacy).__name__),
                self.assertRaisesRegex(TypeError, "has no unified bridge"),
            ):
                transform_processor({"x": legacy})


class TestTransformBatch(unittest.TestCase):
    def test_converts_bridged_fields_only(self):
        codes = _fit(SequenceProcessor(), ["A", "B"])
        label = BinaryLabelProcessor()
        label.fit([{"y": 0}, {"y": 1}], "y")
        processors = transform_processor({"codes": codes, "y": label})
        batch = collate_fn_dict_with_padding(
            [
                {"codes": codes.process(["A", "B"]), "y": label.process(1), "id": "p1"},
                {"codes": codes.process(["B"]), "y": label.process(0), "id": "p2"},
            ]
        )
        out = transform_batch(processors, batch)
        self.assertEqual(_decode(processors["codes"], out["codes"]), [["A", "B"], ["B"]])
        self.assertIs(out["y"], batch["y"])
        self.assertEqual(out["id"], ["p1", "p2"])
        self.assertIsNot(out, batch)

    def test_keys_without_a_processor_pass_through(self):
        self.assertEqual(transform_batch({}, {"embed": True}), {"embed": True})

    def test_passing_through_bridges_leave_the_value(self):
        codes = _fit(SequenceProcessor(), ["A", "B"])
        outer = transform_processor({"codes": codes})
        inner = transform_processor(outer)
        batch = collate_fn_dict_with_padding([{"codes": codes.process(["A", "B"])}])
        converted = transform_batch(outer, batch)
        out = transform_batch(inner, converted)
        self.assertIs(out["codes"], converted["codes"])
        legacy = transform_batch(inner, batch)
        self.assertIs(legacy["codes"], batch["codes"])

    def test_native_unified_fields_are_untouched(self):
        unified = JaggedFeatureProcessor(levels=[{"tokens": 1}])
        value = (torch.tensor([2]), torch.tensor([0, 1]))
        out = transform_batch({"x": unified}, {"x": value})
        self.assertIs(out["x"], value)


class TestBridgeDataset(unittest.TestCase):
    def setUp(self):
        from pyhealth.datasets import create_sample_dataset

        samples = [
            {"patient_id": f"p{i}", "codes": ["A", "B"][: i + 1], "note": "n", "y": i % 2}
            for i in range(3)
        ]
        self.dataset = create_sample_dataset(
            samples, {"codes": "sequence", "note": "raw"}, {"y": "binary"}
        )
        self.processors = dict(self.dataset.input_processors)

    def test_bridges_the_input_processors_only(self):
        bridged = BridgeDataset(self.dataset)
        self.assertIsInstance(bridged.input_processors["codes"], SequenceToUnifiedBridge)
        self.assertTrue(bridged.input_processors["codes"].converts)
        self.assertIs(bridged.input_processors["codes"].legacy, self.processors["codes"])
        self.assertIs(bridged.input_processors["note"], self.processors["note"])
        self.assertIs(bridged.output_processors, self.dataset.output_processors)
        self.assertEqual(bridged.input_schema, self.dataset.input_schema)
        self.assertEqual(bridged.output_schema, self.dataset.output_schema)
        self.assertEqual(len(bridged), 3)
        self.assertTrue(bridged)  # BaseModel checks `if dataset:`
        self.assertIs(bridged.dataset, self.dataset)
        self.assertEqual(bridged.patient_to_index, self.dataset.patient_to_index)

    def test_original_is_unchanged(self):
        BridgeDataset(self.dataset)
        self.assertEqual(self.dataset.input_processors, self.processors)
        self.assertIsInstance(self.dataset.input_processors["codes"], SequenceProcessor)

    def test_nested_wrap_passes_through(self):
        outer = BridgeDataset(self.dataset)
        inner = BridgeDataset(outer)
        self.assertFalse(inner.input_processors["codes"].converts)
        self.assertTrue(outer.input_processors["codes"].converts)
        self.assertIs(inner.dataset, outer)
        self.assertIs(inner.output_processors, self.dataset.output_processors)
        self.assertEqual(len(inner), 3)

    def test_fields_without_a_bridge_are_rejected(self):
        dataset = SimpleNamespace(input_processors={"text": TupleTimeTextProcessor()})
        with self.assertRaisesRegex(TypeError, "has no unified bridge"):
            BridgeDataset(dataset)


if __name__ == "__main__":
    unittest.main()
