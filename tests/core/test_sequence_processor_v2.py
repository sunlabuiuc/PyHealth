import pickle
import unittest
from collections.abc import Sequence
from typing import Any, ClassVar
from unittest.mock import patch

import torch

from pyhealth.processors import (
    Impute,
    LevelConfig,
    Records,
    Resample,
    JaggedFeatureProcessor,
    get_processor,
)
from pyhealth.tasks import BaseTask

NAN = float("nan")
CODES: LevelConfig = {"tokens": 1}
SCALARS: LevelConfig = {"values": [1]}
NESTED: list[LevelConfig] = [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]
LAST: Resample = {"step": 1.0, "method": "last"}


def _fit(levels: Sequence[LevelConfig], *records: Any) -> JaggedFeatureProcessor:
    proc = JaggedFeatureProcessor(levels=levels)
    proc.fit([{"x": r} for r in records], "x")
    return proc


def _visits(*visits: list[str]) -> Records:
    return {"children": [{"tokens": list(v)} for v in visits]}


class _FakeMap:
    def __init__(self, table):
        self.table = table
        self.s_vocab, self.t_vocab = "ICD9CM", "CCSCM"
        self.has_many_targets = any(len(t) > 1 for t in table.values())

    def map(self, code):
        return self.table.get(code, [])


class TestConfiguration(unittest.TestCase):
    def test_levels_are_required(self):
        with self.assertRaises(TypeError):
            JaggedFeatureProcessor()  # type: ignore[call-arg]  # the missing argument is the test

    def test_registered(self):
        self.assertIs(get_processor("jagged"), JaggedFeatureProcessor)

    def test_repr_shows_levels(self):
        self.assertEqual(
            repr(JaggedFeatureProcessor(levels=[CODES])),
            "JaggedFeatureProcessor(levels=[{'tokens': 1}])",
        )

    def test_subclass_is_configuration_only(self):
        class NestedCodes(JaggedFeatureProcessor):
            def __init__(self):
                super().__init__(levels=NESTED)

        proc = NestedCodes()
        proc.fit([{"x": _visits(["A"])}], "x")
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets")))

    def test_rejects_bad_configurations(self):
        mapping = ("ICD9CM", "CCSCM")
        # Deliberately invalid, so typed Any rather than LevelConfig.
        cases: list[Any] = [
            [],
            ["set"],
            [{"order": "set", "tokens": 1}],
            [{"ordering": "bag", "tokens": 1}],
            [{}],  # level 0 declares nothing
            [{"tokens": 0}],
            [{"tokens": True}],
            [{"tokens": 1.0}],
            [{"values": []}],
            [{"values": 1}],
            [{"values": [0]}],
            [{"values": [True]}],
            [{"values": [1], "resample": LAST}],  # resample on a sequence level
            [{"ordering": "series", "values": [1], "resample": 1.0}],  # not a dict
            [{"ordering": "series", "values": [1], "resample": {"step": 1.0}}],
            [{"ordering": "series", "values": [1], "resample": {**LAST, "step": 0}}],
            [{"ordering": "series", "values": [1], "resample": {**LAST, "step": True}}],
            [
                {
                    "ordering": "series",
                    "values": [1],
                    "resample": {**LAST, "method": "sum"},
                }
            ],
            [
                {
                    "ordering": "series",
                    "values": [1],
                    "resample": {**LAST, "align": "mid"},
                }
            ],
            [
                {
                    "ordering": "series",
                    "values": [1],
                    "resample": {**LAST, "label": "left"},
                }
            ],
            # A resampled level has a record at every grid step, so time alone
            # would not say when the records were.
            [{"ordering": "series", "resample": LAST}],
            [{"values": [1, 1], "spatial": [1, 0]}],
            [{"tokens": 1, "spatial": [False]}],  # spatial without values
            [{"values": [1], "spatial": [False, True]}],  # one flag per axis
            [{"tokens": 1, "code_mapping": "ICD9CM"}],
            [{"tokens": 2, "code_mapping": [mapping, "x"]}],
            [{"tokens": 2, "code_mapping": mapping}],  # one entry per namespace
            [{"values": [1], "code_mapping": mapping}],  # mapping without tokens
        ]
        for levels in cases:
            with (
                self.subTest(levels=levels),
                self.assertRaises((ValueError, TypeError)),
            ):
                JaggedFeatureProcessor(levels=levels)

    def test_only_vocabulary_needs_fit(self):
        proc = JaggedFeatureProcessor(levels=[{"tokens": 2, "values": [3]}])
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "values")))
        self.assertEqual(proc.value_dim(), {0: (3,)})
        self.assertEqual(proc.spatial(), {0: (False,)})
        with self.assertRaises(RuntimeError):
            proc.vocab_size()
        with self.assertRaises(RuntimeError):
            proc.vocab(0)
        with self.assertRaises(ValueError):
            _fit([SCALARS]).vocab(0)  # no tokens
        with self.assertRaises(RuntimeError):
            proc.process({"tokens": [("A", "u")], "values": [[1.0, 2.0, 3.0]]})


class TestSpecExamples(unittest.TestCase):
    """The examples of Section 5.1, written as Records."""

    def assertOutput(self, output, expected):
        self.assertEqual(len(output), len(expected))
        for got, want in zip(output, expected):
            self.assertEqual(got.dtype, want.dtype)
            torch.testing.assert_close(got, want)

    def test_flat_sequence(self):
        raw: Records = {"tokens": ["D01", "D02", "D01"]}
        proc = _fit([CODES], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"),))
        self.assertOutput(proc.process(raw), (torch.tensor([2, 3, 2]),))
        self.assertEqual(proc.vocab_size(), {0: (4,)})
        self.assertEqual(proc.value_dim(), {0: None})
        self.assertEqual(proc.spatial(), {0: None})
        self.assertEqual(proc.ordering(), {0: "sequence"})

    def test_codes_in_visits(self):
        raw = _visits(["D01", "D02"], [], ["D03"])
        proc = _fit(NESTED, raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets")))
        self.assertOutput(
            proc.process(raw), (torch.tensor([2, 3, 4]), torch.tensor([0, 2, 2, 3]))
        )
        self.assertEqual(proc.vocab_size(), {0: (5,), 1: None})
        self.assertEqual(proc.ordering(), {0: "set", 1: "sequence"})

    def test_admissions_visits_codes(self):
        raw: Records = {"children": [_visits(["A", "B"], ["C"]), _visits(["A", "D"])]}
        proc = _fit([{"ordering": "set", "tokens": 1}, {}, {}], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets"), (1, "offsets")))
        self.assertOutput(
            proc.process(raw),
            (
                torch.tensor([2, 3, 4, 2, 5]),
                torch.tensor([0, 2, 3, 5]),
                torch.tensor([0, 2, 3]),
            ),
        )
        self.assertEqual(proc.vocab_size(), {0: (6,), 1: None, 2: None})

    def test_codes_values_and_time(self):
        raw: Records = {
            "time": [0.0, 1.5, 6.0],
            "tokens": ["Glucose", "WBC", "Glucose"],
            "values": [[105.0], [7.2], [110.0]],
        }
        proc = _fit([{"ordering": "series", "tokens": 1, "values": [1]}], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "values"), (0, "time")))
        self.assertOutput(
            proc.process(raw),
            (
                torch.tensor([2, 3, 2]),
                torch.tensor([[105.0], [7.2], [110.0]]),
                torch.tensor([0.0, 1.5, 6.0]),
            ),
        )
        self.assertEqual(proc.value_dim(), {0: (1,)})
        self.assertEqual(proc.spatial(), {0: (False,)})
        self.assertEqual(proc.ordering(), {0: "series"})

    def test_multi_namespace_tokens(self):
        raw: Records = {
            "tokens": [("Glucose", "mg/dL"), ("WBC", "10^3/uL")],
            "values": [[105.0], [7.2]],
        }
        proc = _fit([{"tokens": 2, "values": [1]}], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "values")))
        self.assertOutput(
            proc.process(raw),
            (torch.tensor([[2, 2], [3, 3]]), torch.tensor([[105.0], [7.2]])),
        )
        self.assertEqual(proc.vocab_size(), {0: (4, 4)})
        self.assertEqual(
            proc.vocab(0, 1), {"<pad>": 0, "<unk>": 1, "mg/dL": 2, "10^3/uL": 3}
        )

    def test_multichannel_time_series(self):
        raw: Records = {
            "time": [0.0, 1.0, 2.0],
            "values": [[80.0, 95.0], [82.0, 91.0], [78.0, 94.0]],
        }
        proc = _fit([{"ordering": "series", "values": [2]}], raw)
        self.assertEqual(proc.schema(), ((0, "values"), (0, "time")))
        output = proc.process(raw)
        self.assertEqual(output[0].shape, (3, 2))
        self.assertEqual(proc.vocab_size(), {0: None})
        self.assertEqual(proc.value_dim(), {0: (2,)})

    def test_notes_over_time(self):
        raw: Records = {
            "time": [0.0, 3.5],
            "children": [
                {"tokens": ["[CLS]", "pain", "[SEP]"]},
                {"tokens": ["[CLS]", "[SEP]"]},
            ],
        }
        proc = _fit([CODES, {"ordering": "series"}], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets"), (1, "time")))
        self.assertOutput(
            proc.process(raw),
            (
                torch.tensor([2, 3, 4, 2, 4]),
                torch.tensor([0, 3, 5]),
                torch.tensor([0.0, 3.5]),
            ),
        )
        self.assertEqual(proc.ordering(), {0: "sequence", 1: "series"})

    def test_spatial_values(self):
        raw: Records = {"time": [0.0, 24.0], "values": torch.zeros(2, 1, 4, 4).numpy()}
        config: LevelConfig = {
            "ordering": "series",
            "values": [1, 4, 4],
            "spatial": [False, True, True],
        }
        proc = _fit([config], raw)
        self.assertEqual(proc.process(raw)[0].shape, (2, 1, 4, 4))
        self.assertEqual(proc.value_dim(), {0: (1, 4, 4)})
        self.assertEqual(proc.spatial(), {0: (False, True, True)})


class TestProcess(unittest.TestCase):
    def test_unseen_tokens_are_unknown_and_nothing_is_truncated(self):
        proc = _fit(NESTED, _visits(["A"]))
        tokens, offsets = proc.process(_visits(["A", "B", "C", "D"], ["A"]))
        self.assertEqual(tokens.tolist(), [2, 1, 1, 1, 2])
        self.assertEqual(offsets.tolist(), [0, 4, 5])

    def test_fit_grows_vocabulary_across_samples(self):
        proc = _fit([CODES], {"tokens": ["A"]}, {"tokens": ["B", "A"]})
        self.assertEqual(proc.vocab(0), {"<pad>": 0, "<unk>": 1, "A": 2, "B": 3})

    def test_empty_inputs(self):
        proc = _fit([{"tokens": 2}], {"tokens": [("a", "b")]})
        self.assertEqual(proc.process({"tokens": []})[0].shape, (0, 2))

        proc = _fit(
            [{"ordering": "series", "values": [2]}],
            {"time": [0.0], "values": [[1.0, 2.0]]},
        )
        values, time = proc.process({"time": [], "values": []})
        self.assertEqual(values.shape, (0, 2))
        self.assertEqual(time.shape, (0,))

        proc = _fit(NESTED, _visits(["A"]))
        tokens, offsets = proc.process({"children": []})
        self.assertEqual(tokens.shape, (0,))
        self.assertEqual(offsets.tolist(), [0])

    def test_fit_needs_no_records(self):
        # Shapes come from the configuration, so empty training data is fine.
        proc = _fit(NESTED, {"children": []})
        self.assertEqual(proc.vocab_size(), {0: (2,), 1: None})
        self.assertEqual(proc.process(_visits(["A"]))[0].tolist(), [1])

        proc = _fit([SCALARS], {"values": []})
        self.assertEqual(proc.process({"values": [[1.0]]})[0].shape, (1, 1))

    def test_dtypes(self):
        raw: Records = {
            "time": [0, 1],
            "tokens": ["A", "B"],
            "values": [[1], [2]],
        }
        tokens, values, time = _fit(
            [{"ordering": "series", "tokens": 1, "values": [1]}], raw
        ).process(raw)
        self.assertEqual(
            (tokens.dtype, values.dtype, time.dtype),
            (torch.long, torch.float32, torch.float32),
        )

    def test_flat_scalars_become_one_column(self):
        proc = _fit([SCALARS], {"values": [[1.0]]})
        (values,) = proc.process({"values": [1.0, 2.0]})
        self.assertEqual(values.tolist(), [[1.0], [2.0]])
        with self.assertRaises(ValueError):  # only when S is (1,)
            _fit([{"values": [2]}], {"values": [[1.0, 2.0]]}).process(
                {"values": [1.0, 2.0]}
            )

    def test_undeclared_keys_are_ignored(self):
        proc = _fit([CODES], {"tokens": ["A"]})
        # Typed Any: "extra" is deliberately not a Records key.
        cases: tuple[Any, ...] = (
            {"tokens": ["A"], "values": [[1.0]]},  # values not declared
            {"tokens": ["A"], "extra": [1]},
            {"tokens": ["A"], "time": [0.0]},  # time on a sequence level
            {"tokens": ["A"], "children": [{"tokens": ["B"]}]},  # level 0
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(proc.process(raw)[0].tolist(), [2])

    def test_bad_records_raise_when_read(self):
        series: LevelConfig = {"ordering": "series", "tokens": 1}
        pairs: LevelConfig = {"tokens": 2}
        cases: list[tuple[list[LevelConfig], Any]] = [
            ([CODES], {"values": [[1.0]]}),  # declared tokens missing
            ([CODES], {"tokens": [("A", "u")]}),  # K = 1 takes strings
            ([CODES], {"tokens": [("A",)]}),
            ([pairs], {"tokens": ["A"]}),
            ([pairs], {"tokens": [("A", "u", "v")]}),
            ([CODES], {"tokens": [None]}),
            (
                [{"tokens": 1, "values": [1]}],
                {"tokens": ["A"], "values": [[1.0], [2.0]]},
            ),  # lengths differ
            ([CODES], ["A"]),
            ([CODES], {}),
            ([series], {"tokens": ["A"]}),  # series without time
            ([series], {"time": [1.0, 0.0], "tokens": ["A", "B"]}),
            ([series], {"time": [NAN], "tokens": ["A"]}),
            ([series], {"time": [[0.0]], "tokens": ["A"]}),
            ([SCALARS], {"values": [[1.0, 2.0]]}),
            ([SCALARS], {"values": [[[1.0]]]}),
            ([SCALARS], {"values": [[NAN]]}),
            (NESTED, {"tokens": ["A"]}),  # no children at level 1
        ]
        for levels, raw in cases:
            proc = _fit(levels)
            with (
                self.subTest(raw=raw),
                self.assertRaises((ValueError, TypeError)),
            ):
                proc.process(raw)

    def test_process_checks_records(self):
        proc = _fit(
            [{"tokens": 1, "values": [1]}], {"tokens": ["A"], "values": [[1.0]]}
        )
        cases: tuple[Records, ...] = (
            {"tokens": ["A"]},
            {"tokens": ["A"], "values": [[1.0, 2.0]]},
            {"tokens": ["A"], "values": [[NAN]]},
            {"tokens": ["A"], "values": [[float("inf")]]},
            {"tokens": ["A"], "values": [[1e39]]},  # overflows float32
        )
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                proc.process(raw)


class TestCodeMapping(unittest.TestCase):
    MAPPING = ("ICD9CM", "CCSCM")

    def setUp(self):
        self.use_table({"250.00": ["49"], "401.9": ["98", "99"]})  # multi-target

    def use_table(self, table):
        patcher = patch("pyhealth.medcode.CrossMap.load", return_value=_FakeMap(table))
        self.load = patcher.start()
        self.addCleanup(patcher.stop)

    def test_maps_and_keeps_unmapped_codes(self):
        proc = _fit(
            [{"ordering": "set", "tokens": 1, "code_mapping": self.MAPPING}],
            {"tokens": ["250.00", "X"]},
        )
        self.load.assert_called_once_with("ICD9CM", "CCSCM")
        self.assertEqual(proc.vocab(0), {"<pad>": 0, "<unk>": 1, "49": 2, "X": 3})
        self.assertEqual(
            proc.process({"tokens": ["250.00", "X", "Y"]})[0].tolist(), [2, 3, 1]
        )

    def test_one_to_many_expands_records(self):
        levels: list[LevelConfig] = [
            {"ordering": "set", "tokens": 1, "code_mapping": self.MAPPING},
            {},
        ]
        raw = _visits(["401.9", "250.00"], ["401.9"])
        tokens, offsets = _fit(levels, raw).process(raw)
        self.assertEqual(tokens.tolist(), [2, 3, 4, 2, 3])
        self.assertEqual(offsets.tolist(), [0, 3, 5])

    def test_per_namespace_mapping(self):
        self.use_table({"250.00": ["49"]})  # several namespaces need one target
        proc = _fit(
            [{"ordering": "set", "tokens": 2, "code_mapping": [None, ("A", "B")]}],
            {"tokens": [("401.9", "250.00")]},
        )
        self.assertEqual(proc.vocab(0, 0), {"<pad>": 0, "<unk>": 1, "401.9": 2})
        self.assertEqual(proc.vocab(0, 1), {"<pad>": 0, "<unk>": 1, "49": 2})

    def test_multi_target_mapping_needs_a_bottom_set_level_of_codes(self):
        mapping = self.MAPPING
        cases: list[list[LevelConfig]] = [
            [{"tokens": 1, "code_mapping": mapping}],  # sequence
            [{"ordering": "series", "tokens": 1, "code_mapping": mapping}],
            [{"ordering": "set", "tokens": 1, "values": [1], "code_mapping": mapping}],
            [CODES, {"ordering": "set", "tokens": 1, "code_mapping": mapping}],
            [{"ordering": "set", "tokens": 2, "code_mapping": [mapping, None]}],
        ]
        for levels in cases:
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                JaggedFeatureProcessor(levels=levels)

    def test_single_target_mapping_is_allowed_anywhere(self):
        self.use_table({"250.00": ["49"], "401.9": ["98"]})
        raw: Records = {
            "time": [0.0, 1.0],
            "tokens": ["250.00", "X"],
            "values": [[1.0], [2.0]],
        }
        levels: list[LevelConfig] = [
            {
                "ordering": "series",
                "tokens": 1,
                "values": [1],
                "code_mapping": self.MAPPING,
            }
        ]
        tokens, values, time = _fit(levels, raw).process(raw)
        self.assertEqual(tokens.tolist(), [2, 3])
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(time.tolist(), [0.0, 1.0])


class TestResample(unittest.TestCase):
    @staticmethod
    def series(method: Any, align: Any = "left", **config: Any) -> LevelConfig:
        # Typed Any: the tests also build invalid configurations. Left
        # alignment by default: step i holds [i, i + 1).
        resample = {"step": 1.0, "method": method, "align": align}
        level: Any = {"ordering": "series", "resample": resample}
        level.update(config)
        return level

    def test_methods_need_the_right_roles(self):
        series = self.series
        cases: list[list[LevelConfig]] = [
            [series("avg", tokens=1, values=[1])],  # reduces values only
            [series("min")],  # no values
            [CODES, series("max", values=[1])],  # level 1 has children
            [series("merge", tokens=1)],  # level 0 has no children
            [CODES, series("merge", values=[1])],
            [CODES, series("merge", tokens=1)],
            [{"tokens": 1}, series("merge")],  # child level is not a set
        ]
        for levels in cases:
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                JaggedFeatureProcessor(levels=levels)

    def test_first_and_last_keep_one_record_per_step(self):
        raw: Records = {
            "time": [0.0, 0.5, 1.2, 2.0],
            "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        }
        for method, expected in (
            ("first", [[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]]),
            ("last", [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        ):
            with self.subTest(method=method):
                proc = _fit([self.series(method, values=[2])], raw)
                self.assertEqual(proc.schema(), ((0, "values"), (0, "time")))
                self.assertEqual(proc.ordering(), {0: "series"})
                values, time = proc.process(raw)
                self.assertEqual(values.tolist(), expected)
                self.assertEqual(time.tolist(), [0.0, 1.0, 2.0])

    def test_last_keeps_tokens_and_children(self):
        raw: Records = {
            "time": [0.0, 0.5, 1.5],
            "tokens": ["a", "b", "c"],
            "children": [{"tokens": ["X"]}, {"tokens": ["Y", "Z"]}, {"tokens": []}],
        }
        proc = _fit([CODES, self.series("last", tokens=1)], raw)
        self.assertEqual(
            proc.schema(),
            ((0, "tokens"), (0, "offsets"), (1, "tokens"), (1, "time")),
        )
        child_tokens, offsets, tokens, time = proc.process(raw)
        self.assertEqual(tokens.tolist(), [3, 4])  # "b", "c"
        self.assertEqual(child_tokens.tolist(), [3, 4])  # "Y", "Z"
        self.assertEqual(offsets.tolist(), [0, 2, 2])
        self.assertEqual(time.tolist(), [0.0, 1.0])
        proc.validate((child_tokens, offsets, tokens, time))

    def test_steps_without_records_are_left_for_imputation(self):
        raw: Records = {
            "time": [0.0, 2.0],
            "tokens": ["a", "b"],
            "children": [{"tokens": ["X"]}, {"tokens": ["Y"]}],
        }
        proc = _fit([CODES, self.series("first", tokens=1)], raw)
        child_tokens, offsets, tokens, time = proc.process(raw)
        self.assertEqual(tokens.tolist(), [2, 0, 3])  # step 1 is <pad>
        self.assertEqual(child_tokens.tolist(), [2, 3])
        self.assertEqual(offsets.tolist(), [0, 1, 1, 2])  # and has no children
        self.assertEqual(time.tolist(), [0.0, 1.0, 2.0])  # but has its time
        proc.validate((child_tokens, offsets, tokens, time))

    def test_each_group_gets_its_own_grid(self):
        stay_a: Records = {"time": [0.0, 1.5], "values": [[1.0], [2.0]]}
        stay_b: Records = {"time": [], "values": []}
        stay_c: Records = {"time": [0.2, 0.7], "values": [[3.0], [4.0]]}
        raw: Records = {"children": [stay_a, stay_b, stay_c]}
        proc = _fit([self.series("last", values=[1]), {}], raw)
        values, time, offsets = proc.process(raw)
        self.assertEqual(values.tolist(), [[1.0], [2.0], [4.0]])
        self.assertEqual(time.tolist(), [0.0, 1.0, 0.0])  # each grid from 0
        self.assertEqual(offsets.tolist(), [0, 2, 2, 3])

    def test_min_max_avg_reduce_values(self):
        raw: Records = {
            "time": [0.0, 0.5, 1.0],
            "values": [[1.0, 6.0], [3.0, 2.0], [5.0, 5.0]],
        }
        for method, expected in (
            ("min", [[1.0, 2.0], [5.0, 5.0]]),
            ("max", [[3.0, 6.0], [5.0, 5.0]]),
            ("avg", [[2.0, 4.0], [5.0, 5.0]]),
        ):
            with self.subTest(method=method):
                proc = _fit([self.series(method, values=[2])], raw)
                self.assertEqual(proc.process(raw)[0].tolist(), expected)

    def test_reductions_skip_missing_readings(self):
        # Step 0: channel 0 has 1 and 3, channel 1 nothing. Step 1: one NaN
        # reading next to a value in each channel.
        raw: Records = {
            "time": [0.0, 0.5, 1.0, 1.5],
            "values": [[1.0, NAN], [3.0, NAN], [NAN, NAN], [5.0, 4.0]],
        }
        for method, expected in (
            ("min", [[1.0, 0.0], [5.0, 4.0]]),
            ("max", [[3.0, 0.0], [5.0, 4.0]]),
            ("avg", [[2.0, 0.0], [5.0, 4.0]]),
        ):
            with self.subTest(method=method):
                level: LevelConfig = {
                    **self.series(method, values=[2]),
                    "impute": {"method": "zero"},
                    "missing_indicator": True,
                }
                values, mask, _ = _fit([level], raw).process(raw)
                self.assertEqual(values.tolist(), expected)
                # Only a step without any reading in a channel is missing.
                self.assertEqual(mask.tolist(), [[False, True], [False, False]])

    def test_merge_unions_child_sets(self):
        raw: Records = {
            "time": [0.0, 0.5, 2.0],
            "children": [{"tokens": ["A"]}, {"tokens": ["B", "C"]}, {"tokens": ["D"]}],
        }
        proc = _fit([{"ordering": "set", "tokens": 1}, self.series("merge")], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets"), (1, "time")))
        tokens, offsets, time = proc.process(raw)
        self.assertEqual(tokens.tolist(), [2, 3, 4, 5])
        self.assertEqual(offsets.tolist(), [0, 3, 3, 4])  # step 1 is an empty set
        self.assertEqual(time.tolist(), [0.0, 1.0, 2.0])
        proc.validate((tokens, offsets, time))

    def test_merge_regroups_below_the_top_level(self):
        days: Records = {
            "time": [0.0, 0.5],
            "children": [{"tokens": ["A"]}, {"tokens": ["B"]}],
        }
        raw: Records = {"children": [days, days]}
        levels = [{"ordering": "set", "tokens": 1}, self.series("merge"), {}]
        proc = _fit(levels, raw)
        tokens, day_offsets, day_time, stay_offsets = proc.process(raw)
        self.assertEqual(tokens.tolist(), [2, 3, 2, 3])
        self.assertEqual(day_offsets.tolist(), [0, 2, 4])  # one merged day per stay
        self.assertEqual(day_time.tolist(), [0.0, 0.0])
        self.assertEqual(stay_offsets.tolist(), [0, 1, 2])
        batch = proc.collate([(tokens, day_offsets, day_time, stay_offsets)] * 2)
        proc.validate(batch, batched=True)

    def test_grid_starts_at_time_zero(self):
        proc = _fit([self.series("last", values=[1])])
        # Step 0 has no record, so its values are NaN, and nothing imputes them.
        with self.assertRaisesRegex(ValueError, "NaN"):
            proc.process({"time": [1.0, 2.0], "values": [[1.0], [2.0]]})
        self.assertEqual(
            proc.process({"time": [0.0, 1.0], "values": [[1.0], [2.0]]})[0].shape,
            (2, 1),
        )

    def test_value_gaps_and_negative_time_raise(self):
        proc = _fit([self.series("avg", values=[1])])
        cases: tuple[Records, ...] = (
            {"time": [0.0, 2.0], "values": [[1.0], [2.0]]},
            {"time": [-1.0, 0.0], "values": [[1.0], [2.0]]},
        )
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                proc.process(raw)

    def test_empty_series(self):
        proc = _fit([self.series("last", values=[1])])
        self.assertEqual(proc.process({"time": [], "values": []})[0].shape, (0, 1))

    def test_records_need_values(self):
        with self.assertRaises(ValueError):
            _fit([self.series("last", values=[1])]).process({"time": [0.0]})

    def test_right_holds_records_up_to_the_step_time(self):
        raw: Records = {
            "time": [0.0, 0.5, 1.0, 1.2],
            "values": [[1.0], [3.0], [5.0], [7.0]],
        }
        for align, expected, expected_time in (
            # (T - 1, T]: only time 0 is in step 0, and 1.0 closes step 1.
            ("right", [1.0, 4.0, 7.0], [0.0, 1.0, 2.0]),
            # [T, T + 1): each step looks ahead to just before T + 1.
            ("left", [2.0, 6.0], [0.0, 1.0]),
        ):
            with self.subTest(align=align):
                proc = _fit([self.series("avg", align, values=[1])], raw)
                values, time = proc.process(raw)
                self.assertEqual(values.reshape(-1).tolist(), expected)
                self.assertEqual(time.tolist(), expected_time)

    def test_align_defaults_to_right(self):
        level: LevelConfig = {
            "ordering": "series",
            "values": [1],
            "resample": {"step": 1.0, "method": "last"},
        }
        raw: Records = {"time": [0.0, 0.5], "values": [[1.0], [2.0]]}
        values, time = _fit([level], raw).process(raw)
        self.assertEqual(values.reshape(-1).tolist(), [1.0, 2.0])
        self.assertEqual(time.tolist(), [0.0, 1.0])

    def test_records_on_a_boundary_stay_on_it(self):
        # float32(0.3) / 0.1 is just above 3 and float32(0.7) / 0.1 just below
        # 7, so dividing alone puts them in the neighboring step.
        raw: Records = {"time": [0.0, 0.3, 0.7], "tokens": ["a", "b", "c"]}
        for align in ("left", "right"):
            with self.subTest(align=align):
                level: LevelConfig = {
                    "ordering": "series",
                    "tokens": 1,
                    "resample": {"step": 0.1, "method": "last", "align": align},
                }
                tokens, time = _fit([level], raw).process(raw)
                self.assertEqual(tokens.tolist(), [2, 0, 0, 3, 0, 0, 0, 4])
                torch.testing.assert_close(
                    time, torch.arange(8, dtype=torch.float32) * 0.1
                )


class TestImpute(unittest.TestCase):
    @staticmethod
    def level(*steps: Any, **config: Any) -> LevelConfig:
        # Typed Any: the tests also build invalid configurations.
        level: Any = {"impute": list(steps)}
        level.update(config)
        return level

    def test_rejects_bad_steps(self):
        level = self.level
        bad: list[Any] = [
            [{"values": [1], "impute": "zero"}],
            [level({"by_tokens": [0]}, values=[1])],  # no method
            [level({"method": "median"}, values=[1])],
            [level({"method": "zero", "extra": 1}, values=[1])],
            [level({"method": "mean", "by_tokens": [1]}, tokens=1, values=[1])],
            [level({"method": "mean", "by_tokens": []}, tokens=1, values=[1])],
            [level({"method": "mean", "by_dims": [1]}, values=[1])],
            [level({"method": "mean", "by_dims": [0, 0]}, values=[2, 2])],
            [level({"method": "mean"}, tokens=1)],  # no values
            [level({"method": "linear"}, ordering="series", tokens=1)],
            [level({"method": "forward_fill"}, ordering="set", values=[1])],
        ]
        for levels in bad:
            with (
                self.subTest(levels=levels),
                self.assertRaises((ValueError, TypeError)),
            ):
                JaggedFeatureProcessor(levels=levels)

    def test_learns_statistics_only_with_statistics(self):
        zero = JaggedFeatureProcessor(levels=[self.level({"method": "zero"}, values=[1])])
        mean = JaggedFeatureProcessor(levels=[self.level({"method": "mean"}, values=[1])])
        self.assertFalse(zero.learns_statistics)
        self.assertTrue(mean.learns_statistics)

    def test_zero(self):
        proc = _fit([self.level({"method": "zero"}, values=[2])])
        (values,) = proc.process({"values": [[1.0, NAN], [NAN, 4.0]]})
        self.assertEqual(values.tolist(), [[1.0, 0.0], [0.0, 4.0]])

    def test_statistics_by_tokens_and_dims(self):
        train: Records = {
            "tokens": ["A", "A", "B"],
            "values": [[1.0, 10.0], [3.0, NAN], [5.0, 50.0]],
        }
        test: Records = {"tokens": ["A", "B", "C"], "values": [[NAN, NAN]] * 3}
        cases: list[tuple[Impute, list[list[float]]]] = [
            # One mean of everything: (1 + 10 + 3 + 5 + 50) / 5.
            ({"method": "mean"}, [[13.8, 13.8]] * 3),
            # One mean per value position.
            ({"method": "mean", "by_dims": ()}, [[3.0, 30.0]] * 3),
            # One per token. "C" was never seen, so it keeps its NaN, which the
            # chained "zero" step then fills.
            (
                {"method": "max", "by_tokens": (0,)},
                [[10.0, 10.0], [50.0, 50.0], [0.0, 0.0]],
            ),
            (
                {"method": "min", "by_tokens": (0,), "by_dims": ()},
                [[1.0, 10.0], [5.0, 50.0], [0.0, 0.0]],
            ),
        ]
        for step, expected in cases:
            with self.subTest(step=step):
                level = self.level(step, {"method": "zero"}, tokens=1, values=[2])
                _, values = _fit([level], train).process(test)
                torch.testing.assert_close(values, torch.tensor(expected))

    def test_statistics_come_from_raw_values(self):
        # Resampling averages 1 and 5 into 3, but the max is taken from raw
        # values, so the empty step 1 becomes 5.
        level = self.level(
            {"method": "max"},
            ordering="series",
            values=[1],
            resample={"step": 1.0, "method": "avg", "align": "left"},
        )
        raw: Records = {"time": [0.0, 0.5, 2.0], "values": [[1.0], [5.0], [2.0]]}
        values, _ = _fit([level], raw).process(raw)
        self.assertEqual(values.tolist(), [[3.0], [5.0], [2.0]])

    def test_token_without_finite_values_keeps_its_nan(self):
        # Unlike normalization, imputation does not fall back to all tokens:
        # A keeps its NaN, which the chained "zero" step then fills.
        train: Records = {"tokens": ["A", "B"], "values": [[NAN], [4.0]]}
        step: Impute = {"method": "mean", "by_tokens": (0,)}
        level = self.level(step, {"method": "zero"}, tokens=1, values=[1])
        _, values = _fit([level], train).process(
            {"tokens": ["A", "B"], "values": [[NAN], [NAN]]}
        )
        self.assertEqual(values.reshape(-1).tolist(), [0.0, 4.0])

    def test_fills_never_cross_groups(self):
        raw: Records = {
            "children": [
                {"values": [[1.0], [NAN], [3.0]]},
                {"values": [[NAN], [2.0]]},
            ]
        }
        cases: list[tuple[Impute, list[float]]] = [
            ({"method": "forward_fill"}, [1.0, 1.0, 3.0, 0.0, 2.0]),
            ({"method": "backward_fill"}, [1.0, 3.0, 3.0, 2.0, 2.0]),
            ({"method": "linear"}, [1.0, 2.0, 3.0, 0.0, 2.0]),
        ]
        for step, expected in cases:
            with self.subTest(step=step):
                level = self.level(step, {"method": "zero"}, values=[1])
                values, _ = _fit([level, {}], raw).process(raw)
                self.assertEqual(values.reshape(-1).tolist(), expected)

    def test_fills_by_token(self):
        raw: Records = {
            "time": [0.0, 1.0, 2.0, 4.0],
            "tokens": ["A", "B", "A", "A"],
            "values": [[1.0], [10.0], [NAN], [4.0]],
        }
        cases: list[tuple[Impute, float]] = [
            ({"method": "forward_fill", "by_tokens": (0,)}, 1.0),
            ({"method": "backward_fill", "by_tokens": (0,)}, 4.0),
            ({"method": "linear", "by_tokens": (0,)}, 2.5),  # by time: 2 of 0..4
            ({"method": "forward_fill"}, 10.0),  # any token: the previous record
        ]
        for step, expected in cases:
            with self.subTest(step=step):
                level = self.level(step, ordering="series", tokens=1, values=[1])
                _, values, _ = _fit([level], raw).process(raw)
                self.assertAlmostEqual(values[2, 0].item(), expected)

    def test_fill_also_fills_pad_tokens(self):
        level = self.level(
            {"method": "forward_fill"},
            ordering="series",
            tokens=1,
            values=[1],
            resample={"step": 1.0, "method": "last"},
        )
        raw: Records = {
            "time": [0.0, 2.0],
            "tokens": ["A", "B"],
            "values": [[1.0], [2.0]],
        }
        tokens, values, _ = _fit([level], raw).process(raw)
        self.assertEqual(tokens.tolist(), [2, 2, 3])  # step 1 takes "A"
        self.assertEqual(values.reshape(-1).tolist(), [1.0, 1.0, 2.0])

    def test_linear_on_a_resampled_grid_uses_grid_time(self):
        level = self.level(
            {"method": "linear"},
            ordering="series",
            values=[1],
            resample={"step": 1.0, "method": "last", "align": "left"},
        )
        raw: Records = {"time": [0.0, 3.5], "values": [[0.0], [3.0]]}
        values, time = _fit([level], raw).process(raw)
        self.assertEqual(values.reshape(-1).tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(time.tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_levels_without_records(self):
        # A sample whose stays have no records at all, for every method.
        for method in ("zero", "mean", "forward_fill", "backward_fill", "linear"):
            with self.subTest(method=method):
                level = self.level(
                    {"method": method}, ordering="series", tokens=1, values=[2]
                )
                proc = _fit(
                    [level, {}],
                    {
                        "children": [
                            {"time": [0.0], "tokens": ["A"], "values": [[1.0, 2.0]]}
                        ]
                    },
                )
                empty: Records = {"time": [], "tokens": [], "values": []}
                tokens, values, time, offsets = proc.process(
                    {"children": [empty, empty]}
                )
                self.assertEqual(
                    (tokens.shape, values.shape, time.shape), ((0,), (0, 2), (0,))
                )
                self.assertEqual(offsets.tolist(), [0, 0, 0])

    def test_unfilled_gaps_still_raise(self):
        level = self.level({"method": "forward_fill"}, values=[1])
        with self.assertRaisesRegex(ValueError, "NaN"):
            _fit([level]).process({"values": [[NAN], [1.0]]})  # leading gap

    def test_pickle_keeps_statistics(self):
        level = self.level({"method": "mean"}, values=[1])
        proc = _fit([level], {"values": [[2.0], [4.0]]})
        restored = pickle.loads(pickle.dumps(proc))
        self.assertEqual(restored.process({"values": [[NAN]]})[0].tolist(), [[3.0]])


class TestNormalize(unittest.TestCase):
    @staticmethod
    def level(normalize: Any, **config: Any) -> LevelConfig:
        # Typed Any: the tests also build invalid configurations.
        level: Any = {"normalize": normalize}
        level.update(config)
        return level

    def test_rejects_bad_configurations(self):
        level = self.level
        bad: list[Any] = [
            [level("zscore", values=[1])],
            [level({"by_dims": ()}, values=[1])],  # no method
            [level({"method": "robust"}, values=[1])],
            [level({"method": "zscore", "extra": 1}, values=[1])],
            [level({"method": "zscore"}, tokens=1)],  # no values
            [level({"method": "zscore", "by_tokens": [1]}, tokens=1, values=[1])],
            [level({"method": "zscore", "by_dims": [0, 0]}, values=[2, 2])],
        ]
        for levels in bad:
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                JaggedFeatureProcessor(levels=levels)

    def test_learns_statistics(self):
        proc = JaggedFeatureProcessor(
            levels=[self.level({"method": "minmax"}, values=[1])]
        )
        self.assertTrue(proc.learns_statistics)

    def test_zscore_per_channel_by_default(self):
        raw: Records = {"values": [[1.0, 10.0], [3.0, 30.0]]}
        proc = _fit([self.level({"method": "zscore"}, values=[2])], raw)
        (values,) = proc.process(raw)
        self.assertEqual(values.tolist(), [[-1.0, -1.0], [1.0, 1.0]])

    def test_minmax_over_all_axes(self):
        raw: Records = {"values": [[1.0, 10.0], [3.0, 21.0]]}
        level = self.level({"method": "minmax", "by_dims": None}, values=[2])
        (values,) = _fit([level], raw).process(raw)
        torch.testing.assert_close(values, torch.tensor([[0.0, 0.45], [0.1, 1.0]]))

    def test_statistics_stream_over_samples(self):
        # Chan et al.'s update across samples equals one pass over all values.
        samples: list[Records] = [
            {"values": [[1.0], [2.0]]},
            {"values": [[3.0]]},
            {"values": []},
            {"values": [[4.0], [NAN]]},
        ]
        proc = _fit([self.level({"method": "zscore"}, values=[1])], *samples)
        (values,) = proc.process({"values": [[2.5]]})
        self.assertAlmostEqual(values.item(), 0.0, places=6)
        (values,) = proc.process({"values": [[2.5 + 1.25**0.5]]})
        self.assertAlmostEqual(values.item(), 1.0, places=6)

    def test_by_tokens_falls_back_to_all_tokens(self):
        train: Records = {
            "tokens": ["A", "A", "B", "B"],
            "values": [[0.0], [2.0], [10.0], [30.0]],
        }
        level = self.level(
            {"method": "minmax", "by_tokens": (0,)}, tokens=1, values=[1]
        )
        proc = _fit([level], train)
        _, values = proc.process(
            {"tokens": ["A", "B", "C"], "values": [[1.0], [20.0], [15.0]]}
        )
        # A: 0..2, B: 10..30, and unseen C: all tokens, 0..30.
        self.assertEqual(values.reshape(-1).tolist(), [0.5, 0.5, 0.5])

    def test_by_tokens_falls_back_per_value_position(self):
        # B never had a finite value: it uses the statistics over all tokens,
        # like the unseen C. A never had one in channel 1: only that channel
        # falls back. No token had one in channel 2: it is left as it is.
        train: Records = {
            "tokens": ["A", "A", "B", "D", "D"],
            "values": [
                [0.0, NAN, NAN],
                [2.0, NAN, NAN],
                [NAN, NAN, NAN],
                [10.0, 100.0, NAN],
                [30.0, 300.0, NAN],
            ],
        }
        level = self.level(
            {"method": "minmax", "by_tokens": (0,)}, tokens=1, values=[3]
        )
        proc = _fit([level], train)
        _, values = proc.process(
            {"tokens": ["A", "B", "C"], "values": [[1.0, 200.0, 7.0]] * 3}
        )
        # Over all tokens: channel 0 is 0..30, channel 1 is 100..300.
        torch.testing.assert_close(
            values,
            torch.tensor([[0.5, 0.5, 7.0], [1 / 30, 0.5, 7.0], [1 / 30, 0.5, 7.0]]),
        )

    def test_fit_checks_tokens_match_values(self):
        level = self.level(
            {"method": "zscore", "by_tokens": (0,)}, tokens=1, values=[1]
        )
        with self.assertRaises(ValueError):
            _fit([level], {"tokens": ["A", "B"], "values": [[1.0]]})

    def test_constant_divides_by_one(self):
        raw: Records = {"values": [[5.0], [5.0]]}
        proc = _fit([self.level({"method": "zscore"}, values=[1])], raw)
        self.assertEqual(proc.process({"values": [[7.0]]})[0].tolist(), [[2.0]])

    def test_position_never_seen_is_kept(self):
        train: Records = {"values": [[1.0, NAN], [3.0, NAN]]}
        proc = _fit([self.level({"method": "zscore"}, values=[2])], train)
        (values,) = proc.process({"values": [[3.0, 7.0]]})
        self.assertEqual(values.tolist(), [[1.0, 7.0]])

    def test_runs_after_imputation(self):
        # mean imputation, then zscore with the same keys: the gap becomes 0.
        level = self.level({"method": "zscore"}, values=[1], impute={"method": "mean"})
        proc = _fit([level], {"values": [[1.0], [3.0]]})
        (values,) = proc.process({"values": [[NAN], [3.0]]})
        self.assertEqual(values.tolist(), [[0.0], [1.0]])

    def test_statistics_come_from_raw_values(self):
        # Resampling averages 0 and 4 into 2; minmax still spans the raw 0..4.
        level = self.level(
            {"method": "minmax"},
            ordering="series",
            values=[1],
            resample={"step": 1.0, "method": "avg", "align": "left"},
        )
        raw: Records = {"time": [0.0, 0.5, 1.0], "values": [[0.0], [4.0], [1.0]]}
        values, _ = _fit([level], raw).process(raw)
        self.assertEqual(values.reshape(-1).tolist(), [0.5, 0.25])

    def test_pickle_keeps_statistics(self):
        proc = _fit(
            [self.level({"method": "zscore"}, values=[1])], {"values": [[1.0], [3.0]]}
        )
        restored = pickle.loads(pickle.dumps(proc))
        self.assertEqual(restored.process({"values": [[3.0]]})[0].tolist(), [[1.0]])


class TestMissingIndicator(unittest.TestCase):
    def test_rejects_bad_configurations(self):
        bad: list[Any] = [
            [{"values": [1], "missing_indicator": 1}],
            [{"tokens": 1, "missing_indicator": True}],  # no values
            [{"values": [1], "missing_indicator": True}],  # no impute
        ]
        for levels in bad:
            with (
                self.subTest(levels=levels),
                self.assertRaises((ValueError, TypeError)),
            ):
                JaggedFeatureProcessor(levels=levels)

    def test_mask_follows_values_in_the_schema(self):
        level: LevelConfig = {
            "ordering": "series",
            "tokens": 1,
            "values": [2],
            "impute": {"method": "zero"},
            "missing_indicator": True,
        }
        proc = JaggedFeatureProcessor(levels=[level])
        self.assertEqual(
            proc.schema(), ((0, "tokens"), (0, "values"), (0, "mask"), (0, "time"))
        )

    def test_mask_marks_values_missing_before_imputation(self):
        level: LevelConfig = {
            "values": [2],
            "impute": {"method": "zero"},
            "missing_indicator": True,
        }
        values, mask = _fit([level]).process({"values": [[1.0, NAN], [NAN, 4.0]]})
        self.assertEqual(values.tolist(), [[1.0, 0.0], [0.0, 4.0]])
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [[False, True], [True, False]])

    def test_resampled_gaps_are_missing(self):
        level: LevelConfig = {
            "ordering": "series",
            "values": [1],
            "resample": {"step": 1.0, "method": "last"},
            "impute": {"method": "forward_fill"},
            "missing_indicator": True,
        }
        raw: Records = {"time": [0.0, 2.0], "values": [[1.0], [3.0]]}
        values, mask, _ = _fit([level]).process(raw)
        self.assertEqual(values.reshape(-1).tolist(), [1.0, 1.0, 3.0])
        self.assertEqual(mask.reshape(-1).tolist(), [False, True, False])

    def test_collate_and_validate(self):
        level: LevelConfig = {
            "values": [1],
            "impute": {"method": "zero"},
            "missing_indicator": True,
        }
        proc = _fit([level, {}])
        a = proc.process({"children": [{"values": [[NAN]]}, {"values": [[2.0]]}]})
        b = proc.process({"children": [{"values": []}]})
        proc.validate(a)
        batch = proc.collate([a, b])
        proc.validate(batch, batched=True)
        self.assertEqual(batch[1].reshape(-1).tolist(), [True, False])
        values, mask, offsets = a
        with self.assertRaises(ValueError):
            proc.validate((values, mask.float(), offsets))  # mask must be bool
        with self.assertRaises(ValueError):
            proc.validate((values, mask[:, :0], offsets))  # and shaped like values


class TestCollateAndValidate(unittest.TestCase):
    def test_collate_nested(self):
        a, b = _visits(["A", "B"], ["C"]), _visits(["C"], [], ["A"])
        proc = _fit(NESTED, a, b)
        batch = proc.collate(
            [proc.process(a), proc.process(b), proc.process({"children": []})]
        )
        self.assertEqual(
            proc.batch_schema(), ((0, "tokens"), (0, "offsets"), (1, "offsets"))
        )
        self.assertEqual(batch[0].tolist(), [2, 3, 4, 4, 2])
        self.assertEqual(batch[1].tolist(), [0, 2, 3, 4, 4, 5])
        self.assertEqual(batch[2].tolist(), [0, 2, 5, 5])
        proc.validate(batch, batched=True)

    def test_collate_deep_and_flat_series(self):
        deep: Records = {"children": [_visits(["A"], ["B", "C"]), _visits(["D"])]}
        proc = _fit([{"ordering": "set", "tokens": 1}, {}, {}], deep)
        batch = proc.collate([proc.process(deep)] * 2)
        self.assertEqual(batch[1].tolist(), [0, 1, 3, 4, 5, 7, 8])
        self.assertEqual(batch[2].tolist(), [0, 2, 3, 5, 6])
        self.assertEqual(batch[3].tolist(), [0, 2, 4])
        proc.validate(batch, batched=True)

        raw: Records = {"time": [0.0, 1.0], "values": [[1.0], [2.0]]}
        proc = _fit([{"ordering": "series", "values": [1]}], raw)
        values, time, batch_offsets = proc.collate(
            [proc.process(raw), proc.process({"time": [5.0], "values": [[3.0]]})]
        )
        self.assertEqual(values.shape, (3, 1))
        self.assertEqual(time.tolist(), [0.0, 1.0, 5.0])
        self.assertEqual(batch_offsets.tolist(), [0, 2, 3])

    def test_collate_needs_items(self):
        with self.assertRaises(ValueError):
            _fit([CODES], {"tokens": ["A"]}).collate([])

    def test_validate_accepts_process_output(self):
        raw: Records = {
            "time": [0.0, 1.0],
            "children": [
                {"tokens": [("A", "u")], "values": [[1.0]]},
                {"tokens": [], "values": []},
            ],
        }
        proc = _fit([{"tokens": 2, "values": [1]}, {"ordering": "series"}], raw)
        proc.validate(proc.process(raw))

    def test_validate_rejects_broken_outputs(self):
        proc = _fit(NESTED, _visits(["A", "B"], ["C"]))
        tokens, offsets = proc.process(_visits(["A", "B"], ["C"]))
        cases = [
            (tokens,),
            [tokens, offsets],
            (tokens.int(), offsets),
            (torch.tensor([2, 3, 9]), offsets),
            (tokens, torch.tensor([1, 2, 3])),
            (tokens, torch.tensor([0, 3, 2])),
            (tokens, torch.tensor([0, 2, 4])),
            (tokens[:2], offsets),
        ]
        for output in cases:
            with self.subTest(output=output), self.assertRaises(ValueError):
                proc.validate(output)
        with self.assertRaises(ValueError):
            proc.validate((tokens, offsets, torch.tensor([0, 1])), batched=True)
        with self.assertRaises(TypeError):
            proc.validate((tokens.tolist(), offsets))  # type: ignore[arg-type]

    def test_validate_checks_shapes_values_and_time(self):
        level: LevelConfig = {"ordering": "series", "tokens": 1, "values": [2]}
        raw: Records = {"time": [0.0], "tokens": ["A"], "values": [[1.0, 2.0]]}
        proc = _fit([level], raw)
        tokens, values, time = proc.process(raw)
        cases = {
            "tokens [N, 1] for one namespace": (tokens[:, None], values, time),
            "values [N, 1] for [2]": (tokens, values[:, :1], time),
            "NaN value": (tokens, torch.full_like(values, NAN), time),
            "time [N, 1]": (tokens, values, time[:, None]),
            "infinite time": (tokens, values, torch.full_like(time, float("inf"))),
            "record counts differ": (tokens, values, torch.zeros(2)),
        }
        for name, output in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                proc.validate(output)


class TestStorage(unittest.TestCase):
    def test_dataloader_uses_the_processor_collate(self):
        from pyhealth.datasets import create_sample_dataset, get_dataloader

        deep: list[LevelConfig] = [{"ordering": "set", "tokens": 1}, {}, {}]
        samples = [
            {
                "patient_id": "p0",
                "visits": _visits(["A", "B"], ["C"]),
                "stays": {"children": [_visits(["A"])]},
                "label": 0,
            },
            {
                "patient_id": "p1",
                "visits": _visits(["C"]),
                "stays": {"children": [_visits(["B"], ["C"])]},
                "label": 1,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "visits": ("jagged", {"levels": NESTED}),  # two tensors
                "stays": ("jagged", {"levels": deep}),  # three tensors
            },
            output_schema={"label": "binary"},
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        for key in ("visits", "stays"):
            proc = dataset.input_processors[key]
            assert isinstance(proc, JaggedFeatureProcessor)
            expected = proc.collate([dataset[0][key], dataset[1][key]])
            self.assertEqual(len(batch[key]), len(expected))
            for got, want in zip(batch[key], expected):
                torch.testing.assert_close(got, want)
            proc.validate(batch[key], batched=True)
        self.assertEqual(batch["label"].shape[0], 2)  # other fields as before

    def test_pickle_round_trip(self):
        raw = _visits(["A", "B"], ["C"])
        proc = pickle.loads(pickle.dumps(_fit(NESTED, raw)))
        self.assertEqual(proc.process(raw)[0].tolist(), [2, 3, 4])

    def test_sample_dataset_round_trip(self):
        from pyhealth.datasets import create_sample_dataset

        samples = [
            {"patient_id": "p0", "visits": _visits(["A", "B"], ["C"]), "label": 0},
            {"patient_id": "p1", "visits": _visits(["C"]), "label": 1},
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"visits": ("jagged", {"levels": NESTED})},
            output_schema={"label": "binary"},
        )
        proc = dataset.input_processors["visits"]
        for sample, raw in zip(dataset, samples):
            expected = proc.process(raw["visits"])
            self.assertIsInstance(sample["visits"], tuple)
            for got, want in zip(sample["visits"], expected):
                torch.testing.assert_close(got, want)


class TestOldName(unittest.TestCase):
    """SequenceProcessorV2 and "sequence_v2" are a deprecated subclass of the new names."""

    def test_is_an_empty_subclass_with_its_own_registration(self):
        from pyhealth.processors import SequenceProcessorV2
        from pyhealth.processors.sequence_processor_v2 import (
            SequenceProcessorV2 as OldModuleName,
        )

        self.assertIs(OldModuleName, SequenceProcessorV2)
        self.assertTrue(issubclass(SequenceProcessorV2, JaggedFeatureProcessor))
        self.assertIsNot(SequenceProcessorV2, JaggedFeatureProcessor)
        # Nothing but the warning: it must keep behaving like the base class.
        defined = vars(SequenceProcessorV2)
        self.assertEqual([k for k, v in defined.items() if callable(v)], ["__init__"])
        self.assertEqual(
            [k for k in defined if not (k.startswith("__") and k.endswith("__"))],
            ["_abc_impl"],  # every ABC class has it
        )
        self.assertIs(get_processor("sequence_v2"), SequenceProcessorV2)
        self.assertIs(get_processor("jagged"), JaggedFeatureProcessor)

    def test_constructing_it_warns(self):
        from pyhealth.processors import SequenceProcessorV2

        with self.assertWarnsRegex(FutureWarning, "deprecated.*JaggedFeatureProcessor"):
            proc = SequenceProcessorV2(levels=NESTED)
        self.assertEqual(repr(proc), f"SequenceProcessorV2(levels={NESTED})")
        new = JaggedFeatureProcessor(levels=NESTED)
        records = [{"x": {"children": [{"tokens": ["A", "B"]}, {"tokens": ["C"]}]}}]
        proc.fit(records, "x")
        new.fit(records, "x")
        for got, want in zip(proc.process(records[0]["x"]), new.process(records[0]["x"])):
            torch.testing.assert_close(got, want)

    def test_schema_name_warns(self):
        from pyhealth.datasets import create_sample_dataset
        from pyhealth.processors import SequenceProcessorV2

        samples = [
            {"patient_id": f"p{i}", "x": {"tokens": ["A"]}, "y": i % 2} for i in range(2)
        ]
        with self.assertWarns(FutureWarning):
            dataset = create_sample_dataset(
                samples, {"x": ("sequence_v2", {"levels": [CODES]})}, {"y": "binary"}
            )
        self.assertIsInstance(dataset.input_processors["x"], SequenceProcessorV2)

    def test_pickles_under_the_old_module_load_without_warning(self):
        import warnings

        from pyhealth.processors import SequenceProcessorV2

        # Caches written before the rename refer to the class by its old path.
        old = b"cpyhealth.processors.sequence_processor_v2\nSequenceProcessorV2\n."
        self.assertIs(pickle.loads(old), SequenceProcessorV2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proc = JaggedFeatureProcessor(levels=[CODES])
            proc.__class__ = SequenceProcessorV2
            loaded = pickle.loads(pickle.dumps(proc))
        self.assertIsInstance(loaded, SequenceProcessorV2)
        self.assertEqual(caught, [])



class _CodesTask(BaseTask):
    task_name = "old_name_cache_key"
    input_schema: ClassVar[dict[str, Any]] = {"x": ("jagged", {"levels": [CODES]})}
    output_schema: ClassVar[dict[str, Any]] = {"y": "binary"}

    def __call__(self, patient):
        codes = [event.code for event in patient.get_events("events")]
        k = int(patient.patient_id[1:])
        return [{"patient_id": patient.patient_id, "x": {"tokens": codes}, "y": k % 2}]


class TestOldNameCacheKey(unittest.TestCase):
    """The processed-sample cache key of a pre-fitted processor names its class."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / "data"
        root.mkdir()
        rows = ["patient_id,time,code"] + [
            f"p{k},2020-01-0{1 + v} 00:00:00,c{(k + v) % 3}" for k in range(4) for v in range(2)
        ]
        (root / "events.csv").write_text("\n".join(rows) + "\n")
        (root / "config.yaml").write_text(
            'version: "1.0"\ntables:\n  events:\n    file_path: "events.csv"\n'
            '    patient_id: "patient_id"\n    timestamp: "time"\n    attributes:\n      - "code"\n'
        )
        self.root = root

    def samples_dir(self, processor) -> str:
        from pathlib import Path

        from pyhealth.datasets import BaseDataset

        processor.fit([{"x": {"tokens": ["c0", "c1", "c2"]}}], "x")
        dataset = BaseDataset(
            root=str(self.root),
            tables=["events"],
            dataset_name="OldNameCacheKey",
            config_path=str(self.root / "config.yaml"),
            cache_dir=str(Path(self.tmp.name) / "cache"),
            num_workers=1,
        )
        samples = dataset.set_task(
            _CodesTask(), num_workers=1, input_processors={"x": processor}
        )
        return Path(samples.path).name

    def test_a_pre_fitted_old_processor_keeps_its_key(self):
        import warnings

        from pyhealth.processors import SequenceProcessorV2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            old = SequenceProcessorV2(levels=[CODES])
            again = SequenceProcessorV2(levels=[CODES])
        new = JaggedFeatureProcessor(levels=[CODES])
        # The key hashes the class name and vars(): the subclass keeps the
        # name it had before the rename and adds no state, so its key is
        # the one it had then, and a new run finds the same cache.
        self.assertEqual(vars(old).keys(), vars(new).keys())
        first = self.samples_dir(old)
        self.assertEqual(self.samples_dir(again), first)
        self.assertNotEqual(self.samples_dir(new), first)


if __name__ == "__main__":
    unittest.main()
