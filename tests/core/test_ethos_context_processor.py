"""Tests for EthosContextProcessor. All data here is synthetic."""

import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch

from pyhealth.processors import EthosContextProcessor

BIRTH = datetime.fromisoformat("1980-01-01")
START = datetime.fromisoformat("2024-01-01")


def static(*rows, start=START):
    return {
        "time": [time for time, _ in rows],
        "code": [code for _, code in rows],
        "start": start,
    }


def fitted(*contexts, **kwargs):
    processor = EthosContextProcessor(**kwargs)
    processor.fit([{"context": context} for context in contexts], "context")
    return processor


def tokens_of(processor, context):
    return processor.decode(processor.process(context)[1])


class TestEthosContextProcessor(unittest.TestCase):
    def test_order_and_shape(self):
        context = static(
            (BIRTH, "MEDS_BIRTH"),
            (None, "RACE//WHITE"),
            (None, "GENDER//F"),
            (BIRTH, "MARITAL//SINGLE"),
        )
        processor = fitted(context)
        age, tokens = processor.process(context)
        self.assertEqual(processor.context_size, 5)
        self.assertEqual(processor.age_position, 2)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(age.dtype, torch.float64)
        self.assertEqual(age.shape, (1,))
        self.assertAlmostEqual(age.item(), (START - BIRTH).days / 365.25)
        self.assertEqual(
            processor.decode(tokens),
            ["GENDER//F", "MARITAL//SINGLE", "<pad>", "<pad>", "RACE//WHITE"],
        )

    def test_missing_attributes_are_unknown(self):
        processor = fitted(static())
        age, tokens = processor.process(static())
        self.assertTrue(math.isnan(age.item()))
        self.assertEqual(
            processor.decode(tokens),
            ["GENDER//UNKNOWN", "MARITAL//UNKNOWN", "<pad>", "<pad>", "RACE//UNKNOWN"],
        )

    def test_unseen_value_is_unk(self):
        processor = fitted(static((None, "GENDER//F")))
        out = tokens_of(processor, static((None, "GENDER//X")))
        self.assertEqual(out[0], "<unk>")

    def test_birth_without_a_time_gives_no_age(self):
        processor = fitted(static((None, "MEDS_BIRTH")))
        age, _ = processor.process(static((None, "MEDS_BIRTH")))
        self.assertTrue(math.isnan(age.item()))

    def test_changing_attribute_uses_the_latest_known_value(self):
        early, late = datetime.fromisoformat("2020-01-01"), datetime.fromisoformat("2030-01-01")
        context = static((early, "MARITAL//SINGLE"), (late, "MARITAL//MARRIED"))
        processor = fitted(context)
        self.assertEqual(tokens_of(processor, context)[1], "MARITAL//SINGLE")

    def test_changing_attribute_before_any_value_is_unknown(self):
        later = datetime.fromisoformat("2030-01-01"), datetime.fromisoformat("2031-01-01")
        context = static((later[0], "MARITAL//A"), (later[1], "MARITAL//B"))
        processor = fitted(context)
        self.assertEqual(tokens_of(processor, context)[1], "MARITAL//UNKNOWN")

    def test_single_value_applies_whenever_recorded(self):
        context = static((datetime.fromisoformat("2030-01-01"), "MARITAL//SINGLE"))
        processor = fitted(context)
        self.assertEqual(tokens_of(processor, context)[1], "MARITAL//SINGLE")

    def test_prefixes_must_include_birth(self):
        with self.assertRaises(ValueError):
            EthosContextProcessor(prefixes=("GENDER",))

    def test_process_needs_fit(self):
        with self.assertRaises(RuntimeError):
            EthosContextProcessor().process(static())

    def test_save_and_load(self):
        context = static((BIRTH, "MEDS_BIRTH"), (None, "GENDER//F"))
        processor = fitted(context, prefixes=("MEDS_BIRTH", "GENDER"))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "context.json")
            processor.save(path)
            restored = EthosContextProcessor()
            restored.load(path)
        self.assertEqual(restored.prefixes, processor.prefixes)
        self.assertTrue(
            torch.equal(restored.process(context)[1], processor.process(context)[1])
        )

    def test_schema_matches_the_output(self):
        processor = fitted(static())
        self.assertEqual(len(processor.schema()), len(processor.process(static())))
        self.assertTrue(processor.is_token())


if __name__ == "__main__":
    unittest.main()
