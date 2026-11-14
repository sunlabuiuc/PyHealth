"""Tests for EthosProcessor, on synthetic timelines only."""

import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.processors import EthosProcessor, get_processor
from pyhealth.processors.ethos_processor import TIME_INTERVALS

T0 = datetime.fromisoformat("2024-01-01T08:00:00")  # MEDS times are naive


def timeline(*rows):
    """Builds a timeline from ``(minutes after T0, code, value)`` rows."""
    return {
        "time": [T0 + timedelta(minutes=row[0]) for row in rows],
        "code": [row[1] for row in rows],
        "numeric_value": [row[2] for row in rows],
    }


def fitted(*rows, **kwargs):
    processor = EthosProcessor(**kwargs)
    processor.fit([{"x": timeline(*rows)}], "x")
    return processor


def tokens_of(processor, value):
    return processor.decode(processor.process(value)[1])


class TestQuantiles(unittest.TestCase):
    def test_ten_quantile_fit_on_uniform_values(self):
        rows = [(i, "LAB", float(i)) for i in range(1, 101)]
        processor = fitted(*rows)
        # Nine breaks: (n - 1) * q rounded, so 0.1 lands on sorted[10].
        self.assertEqual(
            processor.quantiles["LAB"],
            [11.0, 21.0, 31.0, 41.0, 51.0, 60.0, 70.0, 80.0, 90.0],
        )

    def test_value_on_a_break_takes_the_lower_bin(self):
        processor = fitted(*[(i, "LAB", float(i)) for i in range(1, 101)])
        for value, expected in [(11.0, "Q1"), (11.5, "Q2"), (90.0, "Q9"), (90.5, "Q10")]:
            out = tokens_of(processor, timeline((0, "LAB", value)))
            self.assertEqual(out, ["LAB", expected, "TIMELINE_END"])

    def test_unseen_values_fall_in_the_outer_bins(self):
        processor = fitted(*[(i, "LAB", float(i)) for i in range(1, 101)])
        low = tokens_of(processor, timeline((0, "LAB", -1e6)))
        high = tokens_of(processor, timeline((0, "LAB", 1e6)))
        self.assertEqual(low[1], "Q1")
        self.assertEqual(high[1], "Q10")

    def test_repeated_values_collapse_the_bins(self):
        rows = [(i, "LAB", 5.0) for i in range(8)] + [(9, "LAB", 7.0), (10, "LAB", 9.0)]
        processor = fitted(*rows)
        self.assertEqual(processor.quantiles["LAB"], [5.0, 7.0])
        out = tokens_of(processor, timeline((0, "LAB", 8.0)))
        self.assertEqual(out[1], "Q3")

    def test_breaks_are_per_code_and_tokens_shared(self):
        processor = fitted(
            *[(i, "A", float(i)) for i in range(10)],
            *[(i, "B", 100.0 + i) for i in range(10)],
        )
        self.assertNotEqual(processor.quantiles["A"], processor.quantiles["B"])
        out = tokens_of(processor, timeline((0, "A", 3.0), (0, "B", 103.0)))
        self.assertEqual(out[1], out[3])

    def test_fit_on_train_only(self):
        self.assertTrue(EthosProcessor.learns_statistics)
        processor = fitted(*[(i, "LAB", float(i)) for i in range(10)], num_quantiles=4)
        breaks = dict(processor.quantiles)
        size = processor.size()
        processor.process(timeline((0, "LAB", 1e6), (1, "OTHER", 3.0)))
        self.assertEqual(processor.quantiles, breaks)
        self.assertEqual(processor.size(), size)

    def test_unseen_code_is_unk_without_a_quantile(self):
        processor = fitted((0, "LAB", 1.0))
        out = tokens_of(processor, timeline((0, "NEW", 1.0)))
        self.assertEqual(out, ["<unk>", "TIMELINE_END"])

    def test_code_without_a_value_gets_no_quantile(self):
        processor = fitted((0, "LAB", 1.0), (1, "LAB", 2.0))
        for missing in (None, math.nan, math.inf):
            out = tokens_of(processor, timeline((0, "LAB", missing)))
            self.assertEqual(out, ["LAB", "TIMELINE_END"])

    def test_quantize_prefixes(self):
        processor = fitted(
            (0, "LAB//A", 1.0), (1, "AGE", 2.0), quantize_prefixes=["LAB//"]
        )
        self.assertEqual(list(processor.quantiles), ["LAB//A"])
        out = tokens_of(processor, timeline((0, "AGE", 2.0)))
        self.assertEqual(out, ["AGE", "TIMELINE_END"])

    def test_num_quantiles_must_allow_a_break(self):
        with self.assertRaises(ValueError):
            EthosProcessor(num_quantiles=1)


class TestTimeIntervals(unittest.TestCase):
    def gap_tokens(self, gap):
        processor = fitted((0, "A", None))
        value = {"time": [T0, T0 + gap], "code": ["A", "A"]}
        return tokens_of(processor, value)[1:-2]

    def test_nineteen_bins_as_in_the_paper(self):
        self.assertEqual(len(TIME_INTERVALS), 19)

    def test_lower_bounds_are_inclusive(self):
        for (label, bound), (previous, _) in zip(
            TIME_INTERVALS[1:], TIME_INTERVALS
        ):
            self.assertEqual(self.gap_tokens(bound), [label])
            self.assertEqual(
                self.gap_tokens(bound - timedelta(microseconds=1)), [previous]
            )

    def test_gap_under_five_minutes_has_no_token(self):
        self.assertEqual(self.gap_tokens(timedelta(minutes=5, microseconds=-1)), [])
        self.assertEqual(self.gap_tokens(timedelta(minutes=5)), ["5m-15m"])

    def test_six_months_repeat_rounded_half_up(self):
        for days, repeats in [(180, 1), (269, 1), (270, 2), (450, 3), (630, 4)]:
            self.assertEqual(self.gap_tokens(timedelta(days=days)), ["=6mt"] * repeats)

    def test_short_gaps_do_not_accumulate(self):
        processor = fitted((0, "A", None))
        rows = [(minute, "A", None) for minute in range(0, 40, 4)]
        out = tokens_of(processor, timeline(*rows))
        self.assertEqual(out, ["A"] * 10 + ["TIMELINE_END"])

    def test_interval_precedes_later_event_and_has_its_time(self):
        processor = fitted((0, "A", None), (1, "B", 1.0))
        time, tokens = processor.process(timeline((0, "A", None), (120, "B", 1.0)))
        self.assertEqual(
            processor.decode(tokens), ["A", "2h-3h", "B", "Q1", "TIMELINE_END"]
        )
        self.assertEqual(time.tolist(), [0.0, 2.0, 2.0, 2.0, 2.0])
        self.assertEqual(time.dtype, torch.float64)
        self.assertEqual(tokens.dtype, torch.long)


class TestOrdering(unittest.TestCase):
    def test_rows_are_sorted_and_ties_keep_input_order(self):
        rows = [(60, "D", None), (0, "C", None), (0, "A", None), (0, "B", None)]
        processor = fitted(*rows)
        self.assertEqual(
            tokens_of(processor, timeline(*rows)),
            ["C", "A", "B", "45m-1h15m", "D", "TIMELINE_END"],
        )

    def test_shared_timestamp_has_no_marker(self):
        rows = [(0, "A", 1.0), (0, "B", 1.0), (0, "C", 1.0)]
        processor = fitted(*rows)
        time, tokens = processor.process(timeline(*rows))
        self.assertEqual(
            processor.decode(tokens),
            ["A", "Q1", "B", "Q1", "C", "Q1", "TIMELINE_END"],
        )
        self.assertEqual(set(time.tolist()), {0.0})

    def test_death_goes_before_same_time_discharge(self):
        rows = [
            (0, "HOSPITAL_DISCHARGE//HOME", None),
            (0, "LAB", None),
            (0, "MEDS_DEATH", None),
            (0, "DRG//1", None),
        ]
        processor = fitted(*rows)
        self.assertEqual(
            tokens_of(processor, timeline(*rows)),
            ["LAB", "MEDS_DEATH", "HOSPITAL_DISCHARGE//HOME", "DRG//1", "TIMELINE_END"],
        )

    def test_death_at_another_time_stays(self):
        rows = [(0, "HOSPITAL_DISCHARGE", None), (1, "MEDS_DEATH", None)]
        processor = fitted(*rows)
        self.assertEqual(
            tokens_of(processor, timeline(*rows)),
            ["HOSPITAL_DISCHARGE", "MEDS_DEATH", "TIMELINE_END"],
        )


class TestInputs(unittest.TestCase):
    def test_empty_timelines(self):
        processor = fitted((0, "A", 1.0))
        for empty in (
            {"time": [], "code": [], "numeric_value": []},
            {"time": [], "code": []},
            {},
            None,
            timeline((0, "MEDS_BIRTH", None)),
        ):
            time, tokens = processor.process(empty)
            self.assertEqual((time.numel(), tokens.numel()), (0, 0))

    def test_fit_skips_empty_and_missing_timelines(self):
        processor = EthosProcessor()
        processor.fit(
            [{"x": {"time": [], "code": []}}, {"x": None}, {"x": timeline((0, "A", 1.0))}],
            "x",
        )
        self.assertIn("A", processor.vocab.token_to_id)

    def test_static_rows_are_dropped(self):
        value = timeline((0, "GENDER//F", None), (0, "A", None))
        value["time"].append(None)
        value["code"].append("HEIGHT")
        value["numeric_value"].append(170.0)
        processor = EthosProcessor()
        processor.fit([{"x": value}], "x")
        self.assertEqual(tokens_of(processor, value), ["A", "TIMELINE_END"])
        self.assertNotIn("GENDER//F", processor.vocab.token_to_id)
        self.assertNotIn("HEIGHT", processor.quantiles)

    def test_numeric_value_column_is_optional(self):
        processor = fitted((0, "A", None))
        value = {"time": [T0], "code": ["A"]}
        self.assertEqual(tokens_of(processor, value), ["A", "TIMELINE_END"])

    def test_timeline_end_can_be_left_out(self):
        processor = fitted((0, "A", None), add_timeline_end=False)
        self.assertEqual(tokens_of(processor, timeline((0, "A", None))), ["A"])

    def test_time_column_types(self):
        processor = fitted((0, "A", None))
        stamps = pd.to_datetime([T0, T0 + timedelta(minutes=5)])
        for times in (list(stamps), stamps.to_numpy(), [str(t) for t in stamps]):
            value = {"time": times, "code": ["A", "A"]}
            self.assertEqual(
                tokens_of(processor, value), ["A", "5m-15m", "A", "TIMELINE_END"]
            )

    def test_dates_past_the_pandas_nanosecond_range(self):
        processor = fitted((0, "A", None))
        value = {
            "time": [
                datetime.fromisoformat("2500-01-01T08:00:00"),
                datetime.fromisoformat("2500-01-01T09:00:00"),
                pd.NaT,
            ],
            "code": ["A", "A", "A"],
        }
        self.assertEqual(
            tokens_of(processor, value), ["A", "45m-1h15m", "A", "TIMELINE_END"]
        )

    def test_numeric_times_are_rejected(self):
        processor = fitted((0, "A", None))
        for times in ([0, 5], [0.0, 5.0], np.array([0, 5])):
            with self.assertRaises(TypeError):
                processor.process({"time": times, "code": ["A", "A"]})

    def test_mismatched_columns_are_rejected(self):
        processor = fitted((0, "A", None))
        with self.assertRaises(ValueError):
            processor.process({"time": [T0], "code": ["A", "A"]})

    def test_process_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            EthosProcessor().process(timeline((0, "A", None)))

    def test_input_is_not_modified(self):
        value = timeline((60, "A", math.inf), (0, "B", None))
        processor = fitted((0, "A", 1.0))
        processor.process(value)
        self.assertEqual(value, timeline((60, "A", math.inf), (0, "B", None)))


class TestVocabulary(unittest.TestCase):
    def test_every_quantile_and_interval_token_is_reserved(self):
        processor = EthosProcessor(num_quantiles=4)
        processor.fit([], "x")
        vocab = processor.vocab.token_to_id
        self.assertEqual((vocab["<pad>"], vocab["<unk>"]), (0, 1))
        for token in ["Q1", "Q4", "TIMELINE_END", *[t for t, _ in TIME_INTERVALS]]:
            self.assertIn(token, vocab)
        self.assertEqual(processor.size(), 2 + 4 + 19 + 1)

    def test_ids_do_not_depend_on_sample_order(self):
        first = fitted((0, "B", None), (1, "A", None))
        second = fitted((0, "A", None), (1, "B", None))
        self.assertEqual(first.vocab.token_to_id, second.vocab.token_to_id)

    def test_refit_replaces_state(self):
        processor = fitted((0, "OLD", 1.0))
        processor.fit([{"x": timeline((0, "NEW", 1.0))}], "x")
        self.assertNotIn("OLD", processor.vocab.token_to_id)
        self.assertNotIn("OLD", processor.quantiles)


class TestSaveLoad(unittest.TestCase):
    def test_round_trip(self):
        rows = [(i * 10, "LAB", i / 3) for i in range(30)]
        processor = fitted(
            *rows,
            num_quantiles=5,
            quantize_prefixes=["LAB"],
            exclude_prefixes=["X"],
            add_timeline_end=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "ethos.json")
            processor.save(path)
            loaded = EthosProcessor()
            loaded.load(path)

        self.assertEqual(loaded.quantiles, processor.quantiles)
        self.assertEqual(loaded.vocab.token_to_id, processor.vocab.token_to_id)
        self.assertEqual(loaded.num_quantiles, 5)
        self.assertEqual(loaded.quantize_prefixes, ("LAB",))
        self.assertEqual(loaded.exclude_prefixes, ("X",))
        self.assertFalse(loaded.add_timeline_end)
        for out, expected in zip(
            loaded.process(timeline(*rows)), processor.process(timeline(*rows))
        ):
            self.assertTrue(torch.equal(out, expected))

    def test_unfitted_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ethos.json")
            EthosProcessor().save(path)
            loaded = EthosProcessor()
            loaded.load(path)
        self.assertIsNone(loaded.vocab)
        self.assertIsNone(loaded.quantize_prefixes)


class TestPipeline(unittest.TestCase):
    def test_registered_and_contract(self):
        self.assertIs(get_processor("ethos"), EthosProcessor)
        processor = EthosProcessor()
        self.assertTrue(processor.is_token())
        self.assertEqual(processor.schema(), ("time", "value"))
        self.assertEqual(processor.dim(), (1, 1))
        self.assertEqual(processor.spatial(), (True,))
        self.assertIn("num_quantiles=10", repr(processor))

    def test_sample_dataset_and_batching(self):
        samples = [
            {
                "patient_id": "p0",
                "timeline": timeline((0, "A", 1.0), (120, "B", None)),
                "label": 0,
            },
            {"patient_id": "p1", "timeline": timeline((0, "A", 2.0)), "label": 1},
            {
                "patient_id": "p2",
                "timeline": {"time": [], "code": [], "numeric_value": []},
                "label": 1,
            },
        ]
        dataset = create_sample_dataset(
            samples,
            input_schema={"timeline": "ethos"},
            output_schema={"label": "binary"},
            dataset_name="ethos_test",
            in_memory=True,
        )
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        time, tokens = batch["timeline"]
        self.assertEqual(tokens.shape, (3, 5))
        self.assertEqual(time.shape, (3, 5))
        self.assertEqual(int((tokens == 0).sum()), 7)

    def test_interval_estimates_are_mean_hours_per_token(self):
        processor = EthosProcessor()
        timelines = [
            timeline((0, "A", None), (10, "A", None)),
            timeline((0, "A", None), (14, "A", None)),
        ]
        processor.fit([{"timeline": t} for t in timelines], "timeline")
        self.assertAlmostEqual(
            processor.interval_estimates["5m-15m"], (10 + 14) / 2 / 60
        )

    def test_interval_estimates_split_repeated_tokens(self):
        days = 24 * 60
        processor = fitted((0, "A", None), (400 * days, "A", None))
        self.assertAlmostEqual(processor.interval_estimates["=6mt"], 200 * 24.0)

    def test_interval_estimates_fall_back_to_the_lower_bound(self):
        processor = fitted((0, "A", None))
        self.assertEqual(len(processor.interval_estimates), 19)
        self.assertAlmostEqual(processor.interval_estimates["2h-3h"], 2.0)

    def test_interval_estimates_survive_save_and_load(self):
        processor = fitted((0, "A", None), (10, "A", None))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ethos.json")
            processor.save(path)
            restored = EthosProcessor()
            restored.load(path)
        self.assertEqual(restored.interval_estimates, processor.interval_estimates)


if __name__ == "__main__":
    unittest.main()
