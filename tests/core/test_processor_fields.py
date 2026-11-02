"""Keyed ``schema()`` / ``spatial()`` and the helpers that read them."""

import unittest

import torch

from pyhealth.processors import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    MultiHotProcessor,
    MultiLabelProcessor,
    RegressionLabelProcessor,
    SequenceProcessorV2,
    TensorProcessor,
    TimeseriesBinaryLabelProcessor,
    field_index,
    has_field,
)


def _fitted(processor, samples):
    processor.fit(samples, "x")
    return processor


class TestFieldHelpers(unittest.TestCase):
    def test_keyed_schema(self):
        schema = ((0, "time"), (0, "values"), (1, "offsets"))
        self.assertEqual(field_index(schema, "values"), 1)
        self.assertEqual(field_index(schema, "offsets", level=1), 2)
        self.assertTrue(has_field(schema, "time"))
        self.assertFalse(has_field(schema, "mask"))
        self.assertFalse(has_field(schema, "offsets"))

    def test_value_is_the_main_tensor(self):
        for schema, index in (
            (((0, "time"), (0, "values")), 1),
            (((0, "tokens"), (0, "offsets")), 0),
            (("time", "value"), 1),
            (((0, "tokens"), (0, "values")), 0),  # both: the first
        ):
            with self.subTest(schema=schema):
                self.assertTrue(has_field(schema, "value"))
                self.assertEqual(field_index(schema, "value"), index)
        self.assertFalse(has_field(((0, "time"),), "value"))
        self.assertFalse(has_field(((1, "values"),), "value"))

    def test_other_roles_match_exactly(self):
        self.assertFalse(has_field(("value",), "values"))
        self.assertFalse(has_field(((0, "tokens"),), "values"))

    def test_role_schema_is_level_zero(self):
        schema = ("value", "mask")
        self.assertEqual(field_index(schema, "mask"), 1)
        self.assertTrue(has_field(schema, "value"))
        self.assertFalse(has_field(schema, "value", level=1))

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            field_index(((0, "values"),), "mask")


class TestKeyedProcessors(unittest.TestCase):
    def _check(self, processor, role, spatial):
        schema = processor.schema()
        self.assertEqual(schema, ((0, role),))
        self.assertEqual(field_index(schema, "value"), 0)
        self.assertEqual(len(processor.dim()), len(schema))
        self.assertEqual(processor.spatial(), {0: spatial})

    def test_label_processors(self):
        self._check(
            _fitted(BinaryLabelProcessor(), [{"x": 0}, {"x": 1}]), "values", (False,)
        )
        self._check(
            _fitted(MultiClassLabelProcessor(), [{"x": "a"}, {"x": "b"}]), "tokens", ()
        )
        self._check(
            _fitted(MultiLabelProcessor(), [{"x": ["a"]}, {"x": ["b"]}]),
            "values",
            (False,),
        )
        self._check(RegressionLabelProcessor(), "values", (False,))

    def test_multi_hot_processor(self):
        processor = _fitted(MultiHotProcessor(), [{"x": ["b", "a"]}, {"x": ["c"]}])
        self._check(processor, "values", (False,))
        self.assertEqual(processor.label_vocab, {"a": 0, "b": 1, "c": 2})
        self.assertTrue(
            torch.equal(processor.process(["c", "a"]), torch.tensor([1.0, 0.0, 1.0]))
        )

    def test_multi_hot_and_multilabel_are_separate(self):
        self.assertFalse(issubclass(MultiHotProcessor, MultiLabelProcessor))
        self.assertFalse(issubclass(MultiLabelProcessor, MultiHotProcessor))

    def test_tensor_processor(self):
        self._check(
            _fitted(TensorProcessor(), [{"x": [[1.0, 2.0]]}]), "values", (False, False)
        )
        self._check(
            _fitted(TensorProcessor(dtype=torch.long), [{"x": [1, 2]}]),
            "tokens",
            (False,),
        )
        processor = _fitted(
            TensorProcessor(spatial_dims=(True, False)), [{"x": [[1.0, 2.0]]}]
        )
        self.assertEqual(processor.spatial(), {0: (True, False)})

    def test_packed_processors_share_the_form(self):
        processor = TimeseriesBinaryLabelProcessor()
        self.assertEqual(processor.schema(), ((0, "values"), (0, "time")))
        self.assertEqual(field_index(processor.schema(), "time"), 1)
        visits = SequenceProcessorV2(
            levels=[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]
        )
        self.assertEqual(field_index(visits.schema(), "offsets"), 1)
        self.assertEqual(visits.spatial(), {0: None, 1: None})


if __name__ == "__main__":
    unittest.main()
