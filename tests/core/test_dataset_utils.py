"""Tests for the shared dataset utilities."""

import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets.utils import (
    _convert_for_cache,
    _is_time_value_tuple,
    _restore_from_cache,
    collate_fn_dict,
    collate_fn_dict_with_padding,
    flatten_list,
    hash_str,
    is_homo_list,
    list_nested_levels,
    load_processors,
    padyear,
    save_processors,
    strptime,
)


class TestSmallHelpers(unittest.TestCase):
    def test_hash_and_dates(self):
        self.assertEqual(len(hash_str("abc")), 32)
        self.assertIsNone(strptime(float("nan")))
        self.assertEqual(strptime("2020-01-02").year, 2020)
        self.assertEqual(padyear("2020"), "2020-1-1")
        self.assertEqual(padyear("2020", "02", "03"), "2020-02-03")

    def test_list_helpers(self):
        self.assertEqual(flatten_list([[1], [2, 3]]), [1, 2, 3])
        self.assertEqual(set(list_nested_levels([1, [2]])), {1, 2})
        self.assertEqual(list_nested_levels([]), (1,))
        self.assertTrue(is_homo_list([1, 2.0]))
        self.assertFalse(is_homo_list([1, "2"]))
        self.assertTrue(is_homo_list([]))


class TestTemporalCache(unittest.TestCase):
    def test_time_value_detection(self):
        self.assertTrue(_is_time_value_tuple((None, [1])))
        self.assertTrue(_is_time_value_tuple(([1], [2], "extra")))
        self.assertFalse(
            _is_time_value_tuple(
                ([1], [2], "extra"), allow_additional_components=False
            )
        )
        self.assertFalse(_is_time_value_tuple(("bad", [1])))
        self.assertFalse(_is_time_value_tuple(([1], "bad")))
        self.assertFalse(_is_time_value_tuple([1, 2]))
        self.assertFalse(_is_time_value_tuple((1,)))

    def test_convert_and_restore(self):
        sample = {
            "t": (None, [1, 2]),
            "u": ([3], [4], "extra"),
            "plain": 1,
        }
        converted = _convert_for_cache(sample)
        self.assertTrue(converted["t"]["__stagenet_cache__"])
        self.assertEqual(converted["t"]["components"], [None, [1, 2]])
        self.assertEqual(converted["u"]["extras"], ["extra"])
        self.assertEqual(_restore_from_cache(converted), sample)

    def test_restore_component_variants(self):
        restored = _restore_from_cache(
            {
                "list": {
                    "__stagenet_cache__": True,
                    "components": [None, [1]],
                },
                "tuple": {
                    "__stagenet_cache__": True,
                    "components": (None, [1]),
                },
                "legacy": {
                    "__stagenet_cache__": True,
                    "time": None,
                    "values": [1],
                    "extras": ["x"],
                },
                "scalar_extra": {
                    "__stagenet_cache__": True,
                    "time": None,
                    "values": [1],
                    "extras": "x",
                },
                "plain": 2,
            }
        )
        self.assertEqual(restored["list"], (None, [1]))
        self.assertEqual(restored["tuple"], (None, [1]))
        self.assertEqual(restored["legacy"], (None, [1], "x"))
        self.assertEqual(restored["scalar_extra"], (None, [1], "x"))
        self.assertEqual(restored["plain"], 2)


class TestCollation(unittest.TestCase):
    def test_collate_dict(self):
        self.assertEqual(
            collate_fn_dict([{"x": 1}, {"x": 2}]), {"x": [1, 2]}
        )

    def test_tensor_and_non_tensor(self):
        batch = [
            {"x": torch.tensor([1, 2]), "id": "a"},
            {"x": torch.tensor([3, 4]), "id": "b"},
        ]
        collated = collate_fn_dict_with_padding(batch)
        self.assertEqual(tuple(collated["x"].shape), (2, 2))
        self.assertEqual(collated["id"], ["a", "b"])

    def test_variable_tensors_and_custom_collator(self):
        batch = [
            {"x": torch.tensor([1]), "raw": [1]},
            {"x": torch.tensor([2, 3]), "raw": [2, 3]},
        ]
        collated = collate_fn_dict_with_padding(batch)
        self.assertEqual(tuple(collated["x"].shape), (2, 2))
        self.assertEqual(
            collate_fn_dict_with_padding(batch, collators={"x": len})["x"], 2
        )

    def test_temporal_tuples(self):
        batch = [
            {"x": (torch.tensor([1.0]), torch.tensor([2.0]))},
            {"x": (torch.tensor([3.0]), torch.tensor([4.0]))},
        ]
        times, values = collate_fn_dict_with_padding(batch)["x"]
        self.assertEqual(tuple(times.shape), (2, 1))
        self.assertEqual(tuple(values.shape), (2, 1))

        scalar = [
            {"x": (torch.tensor([1.0]), torch.tensor(2.0))},
            {"x": (torch.tensor([3.0]), torch.tensor(4.0))},
        ]
        _, scalar_values = collate_fn_dict_with_padding(scalar)["x"]
        self.assertEqual(tuple(scalar_values.shape), (2,))

        no_time = [
            {"x": (None, torch.tensor([1.0]))},
            {"x": (torch.tensor([3.0]), torch.tensor([4.0]))},
        ]
        times, _ = collate_fn_dict_with_padding(no_time)["x"]
        self.assertIsNone(times)

    def test_temporal_variable_shapes(self):
        batch = [
            {"x": (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))},
            {"x": (torch.tensor([3.0]), torch.tensor([3.0]))},
        ]
        times, values = collate_fn_dict_with_padding(batch)["x"]
        self.assertEqual(tuple(times.shape), (2, 2))
        self.assertEqual(tuple(values.shape), (2, 2))


class TestProcessorFiles(unittest.TestCase):
    class _Dataset:
        def __init__(self):
            self.input_processors = {"a": 1}
            self.output_processors = {"b": 2}

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = save_processors(self._Dataset(), directory)
            self.assertTrue(Path(paths["input_processors"]).exists())
            input_processors, output_processors = load_processors(directory)
            self.assertEqual(input_processors, {"a": 1})
            self.assertEqual(output_processors, {"b": 2})

    def test_missing_files_raise(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileNotFoundError),
        ):
            load_processors(directory)
