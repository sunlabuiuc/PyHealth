import pickle
import unittest
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import torch

from pyhealth.processors import (
    LevelConfig,
    Records,
    Resample,
    SequenceProcessorV2,
    get_processor,
)

NAN = float("nan")
CODES: LevelConfig = {"tokens": 1}
SCALARS: LevelConfig = {"values": [1]}
NESTED: list[LevelConfig] = [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]
LAST: Resample = {"step": 1.0, "method": "last"}


def _fit(levels: Sequence[LevelConfig], *records: Any) -> SequenceProcessorV2:
    proc = SequenceProcessorV2(levels=levels)
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
            SequenceProcessorV2()  # type: ignore[call-arg]  # the missing argument is the test

    def test_registered(self):
        self.assertIs(get_processor("sequence_v2"), SequenceProcessorV2)

    def test_subclass_is_configuration_only(self):
        class NestedCodes(SequenceProcessorV2):
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
                SequenceProcessorV2(levels=levels)

    def test_reserved_options_not_implemented(self):
        reserved: list[tuple[str, Any]] = [
            ("normalize", "zscore"),
            ("impute", "zero"),
            ("missing_indicator", True),
        ]
        for key, value in reserved:
            config: Any = {"tokens": 1, key: value}  # deliberately invalid
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                SequenceProcessorV2(levels=[config])
        SequenceProcessorV2(
            levels=[
                {
                    "tokens": 1,
                    "normalize": None,
                    "impute": None,
                    "missing_indicator": False,
                }
            ]
        )

    def test_only_vocabulary_needs_fit(self):
        proc = SequenceProcessorV2(levels=[{"tokens": 2, "values": [3]}])
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "values")))
        self.assertEqual(proc.value_dim(), {0: (3,)})
        self.assertEqual(proc.spatial(), {0: (False,)})
        with self.assertRaises(RuntimeError):
            proc.vocab_size()
        with self.assertRaises(RuntimeError):
            proc.vocab(0)
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
        proc = _fit(
            [{"ordering": "set", "tokens": 2, "code_mapping": [None, ("A", "B")]}],
            {"tokens": [("401.9", "250.00")]},
        )
        self.assertEqual(proc.vocab(0, 0), {"<pad>": 0, "<unk>": 1, "401.9": 2})
        self.assertEqual(proc.vocab(0, 1), {"<pad>": 0, "<unk>": 1, "49": 2})

    def test_multi_target_mapping_needs_a_set_level_without_values(self):
        mapping = self.MAPPING
        cases: list[list[LevelConfig]] = [
            [{"tokens": 1, "code_mapping": mapping}],  # sequence
            [{"ordering": "series", "tokens": 1, "code_mapping": mapping}],
            [{"ordering": "set", "tokens": 1, "values": [1], "code_mapping": mapping}],
        ]
        for levels in cases:
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                SequenceProcessorV2(levels=levels)

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

    def test_expansion_raises_when_it_would_duplicate_data(self):
        mapping = self.MAPPING
        cases: list[tuple[list[LevelConfig], Records]] = [
            (
                [{"ordering": "set", "tokens": 2, "code_mapping": [mapping, None]}],
                {"tokens": [("401.9", "u")]},
            ),
            (
                [CODES, {"ordering": "set", "tokens": 1, "code_mapping": mapping}],
                {"tokens": ["401.9"], "children": [{"tokens": ["A"]}]},
            ),
        ]
        for levels, raw in cases:
            proc = _fit(levels, raw)  # fit only builds vocabularies
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                proc.process(raw)


class TestResample(unittest.TestCase):
    @staticmethod
    def series(method: Any, **config: Any) -> LevelConfig:
        # Typed Any: the tests also build invalid configurations.
        level: Any = {"ordering": "series", "resample": {"step": 1.0, "method": method}}
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
                SequenceProcessorV2(levels=levels)

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
                self.assertEqual(proc.schema(), ((0, "values"),))
                self.assertEqual(proc.ordering(), {0: "sequence"})
                (values,) = proc.process(raw)
                self.assertEqual(values.tolist(), expected)

    def test_last_keeps_tokens_and_children(self):
        raw: Records = {
            "time": [0.0, 0.5, 1.5],
            "tokens": ["a", "b", "c"],
            "children": [{"tokens": ["X"]}, {"tokens": ["Y", "Z"]}, {"tokens": []}],
        }
        proc = _fit([CODES, self.series("last", tokens=1)], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets"), (1, "tokens")))
        child_tokens, offsets, tokens = proc.process(raw)
        self.assertEqual(tokens.tolist(), [3, 4])  # "b", "c"
        self.assertEqual(child_tokens.tolist(), [3, 4])  # "Y", "Z"
        self.assertEqual(offsets.tolist(), [0, 2, 2])
        proc.validate((child_tokens, offsets, tokens))

    def test_steps_without_records_are_left_for_imputation(self):
        raw: Records = {
            "time": [0.0, 2.0],
            "tokens": ["a", "b"],
            "children": [{"tokens": ["X"]}, {"tokens": ["Y"]}],
        }
        proc = _fit([CODES, self.series("first", tokens=1)], raw)
        child_tokens, offsets, tokens = proc.process(raw)
        self.assertEqual(tokens.tolist(), [2, 0, 3])  # step 1 is <pad>
        self.assertEqual(child_tokens.tolist(), [2, 3])
        self.assertEqual(offsets.tolist(), [0, 1, 1, 2])  # and has no children
        proc.validate((child_tokens, offsets, tokens))

    def test_each_group_gets_its_own_grid(self):
        stay_a: Records = {"time": [0.0, 1.5], "values": [[1.0], [2.0]]}
        stay_b: Records = {"time": [], "values": []}
        stay_c: Records = {"time": [0.2, 0.7], "values": [[3.0], [4.0]]}
        raw: Records = {"children": [stay_a, stay_b, stay_c]}
        proc = _fit([self.series("last", values=[1]), {}], raw)
        values, offsets = proc.process(raw)
        self.assertEqual(values.tolist(), [[1.0], [2.0], [4.0]])
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

    def test_merge_unions_child_sets(self):
        raw: Records = {
            "time": [0.0, 0.5, 2.0],
            "children": [{"tokens": ["A"]}, {"tokens": ["B", "C"]}, {"tokens": ["D"]}],
        }
        proc = _fit([{"ordering": "set", "tokens": 1}, self.series("merge")], raw)
        self.assertEqual(proc.schema(), ((0, "tokens"), (0, "offsets")))
        tokens, offsets = proc.process(raw)
        self.assertEqual(tokens.tolist(), [2, 3, 4, 5])
        self.assertEqual(offsets.tolist(), [0, 3, 3, 4])  # step 1 is an empty set
        proc.validate((tokens, offsets))

    def test_merge_regroups_below_the_top_level(self):
        days: Records = {
            "time": [0.0, 0.5],
            "children": [{"tokens": ["A"]}, {"tokens": ["B"]}],
        }
        raw: Records = {"children": [days, days]}
        levels = [{"ordering": "set", "tokens": 1}, self.series("merge"), {}]
        proc = _fit(levels, raw)
        tokens, day_offsets, stay_offsets = proc.process(raw)
        self.assertEqual(tokens.tolist(), [2, 3, 2, 3])
        self.assertEqual(day_offsets.tolist(), [0, 2, 4])  # one merged day per stay
        self.assertEqual(stay_offsets.tolist(), [0, 1, 2])
        batch = proc.collate([(tokens, day_offsets, stay_offsets)] * 2)
        proc.validate(batch, batched=True)

    def test_grid_starts_at_time_zero(self):
        proc = _fit([self.series("last", values=[1])])
        # Step 0 has no record, so its values are NaN, which imputation (not
        # supported yet) would have to fill.
        with self.assertRaisesRegex(ValueError, "imputation"):
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


class TestStorage(unittest.TestCase):
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
            input_schema={"visits": ("sequence_v2", {"levels": NESTED})},
            output_schema={"label": "binary"},
        )
        proc = dataset.input_processors["visits"]
        for sample, raw in zip(dataset, samples):
            expected = proc.process(raw["visits"])
            self.assertIsInstance(sample["visits"], tuple)
            for got, want in zip(sample["visits"], expected):
                torch.testing.assert_close(got, want)


if __name__ == "__main__":
    unittest.main()
