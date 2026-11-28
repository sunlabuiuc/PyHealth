"""Tests for SleepStagingDREAMT on tiny synthetic DREAMT-style nights.

No real data is used: nights are generated in code and written to a temporary
folder, and the task is called on a dummy patient (the same pattern as the
other sleep staging tests), so the whole file runs in well under a second.
"""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.tasks.sleep_staging_dreamt import (
    IGNORE_INDEX,
    SleepStagingDREAMT,
    _beats_from_bvp,
    _beats_from_ibi_column,
    _epoch_labels,
    _ibi_to_grid,
    _map_stages,
)

FS = 64  # DREAMT wearable rate
ROWS = 30 * FS  # rows per 30 s epoch
POINTS = 750  # IBI grid points per epoch (30 s at 25 Hz)


def _bvp(n_rows: int, hz: float = 1.25) -> np.ndarray:
    """A clean pulse-like waveform at ``hz`` beats per second (75 bpm)."""
    t = np.arange(n_rows) / FS
    return np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(4 * np.pi * hz * t)


def _device_ibi(n_rows: int, every_rows: int = 48) -> np.ndarray:
    """Sparse device IBI column: 0.75 s (48 rows) between reported beats."""
    ibi = np.full(n_rows, np.nan)
    ibi[every_rows::every_rows] = every_rows / FS
    return ibi


def _stages(*epoch_stages: str) -> pd.Series:
    """One stage string per row, ``ROWS`` rows per listed epoch."""
    return pd.Series(np.repeat(epoch_stages, ROWS))


@dataclass
class _DummyEvent:
    file_64hz: object


class _DummyPatient:
    def __init__(self, patient_id: str, events: list) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> list:
        return self._events if event_type == "dreamt_sleep" else []


class TestHelpers(unittest.TestCase):
    """Pure functions behind the task."""

    def test_map_stages_three_class(self) -> None:
        stages = pd.Series(["W", "N1", "N2", "N3", "R", "P", "Missing"])
        out = _map_stages(stages)
        self.assertEqual(out.tolist(), [0, 1, 1, 1, 2, 0, IGNORE_INDEX])
        self.assertEqual(out.dtype, np.int64)

    def test_map_stages_other_schemes_and_prep(self) -> None:
        stages = pd.Series(["W", "N1", "N2", "N3", "R", "P"])
        self.assertEqual(
            _map_stages(stages, scheme="4class").tolist(), [0, 1, 1, 2, 3, 0]
        )
        self.assertEqual(
            _map_stages(stages, scheme="5class").tolist(), [0, 1, 2, 3, 4, 0]
        )
        ignored = _map_stages(stages, ignore_prep=True)
        self.assertEqual(ignored[-1], IGNORE_INDEX)

    def test_epoch_labels_majority_and_ignored_epochs(self) -> None:
        ign = IGNORE_INDEX
        rows = np.array([0, 0, 1, 2, 2, 2, 2, 1, ign, ign, ign, 0, 1, 1, ign, ign])
        labels = _epoch_labels(rows, rows_per_epoch=4, n_epochs=4, num_classes=3)
        # epoch 2 has a single valid row (< half), so it is ignored
        self.assertEqual(labels.tolist(), [0, 2, ign, 1])
        self.assertEqual(
            _epoch_labels(rows, 4, 2, 3).tolist(), [0, 2]
        )  # truncation to n_epochs

    def test_ibi_to_grid_regular_beats(self) -> None:
        times = np.arange(1, 40) * 0.8  # a beat every 0.8 s, up to 31.2 s
        series = _ibi_to_grid(times, np.full(len(times), 0.8), n_epochs=1)
        self.assertEqual(series.shape, (POINTS,))
        self.assertTrue(np.isnan(series[:20]).all())
        np.testing.assert_allclose(series[20:], 0.8)

    def test_ibi_to_grid_marks_gaps_and_long_intervals_missing(self) -> None:
        times = np.concatenate([np.arange(1, 13) * 0.8, 20 + np.arange(0, 20) * 0.8])
        intervals = np.full(len(times), 0.8)
        intervals[12] = 3.0  # first beat after the gap reports a 3 s interval
        series = _ibi_to_grid(times, intervals, n_epochs=1)
        self.assertAlmostEqual(series[int(9.0 * 25)], 0.8)
        self.assertTrue(np.isnan(series[int(14.0 * 25)]))
        self.assertTrue(np.isnan(series[int(20.5 * 25)]))

    def test_ibi_to_grid_without_beats_is_all_missing(self) -> None:
        series = _ibi_to_grid(np.array([]), np.array([]), n_epochs=2)
        self.assertEqual(series.shape, (2 * POINTS,))
        self.assertTrue(np.isnan(series).all())

    def test_beats_from_ibi_column(self) -> None:
        ibi = np.array([np.nan, 0.8, np.nan, np.nan, 0.9])
        times, intervals = _beats_from_ibi_column(ibi, fs=2.0)
        np.testing.assert_allclose(times, [0.5, 2.0])
        np.testing.assert_allclose(intervals, [0.8, 0.9])

    def test_beats_from_bvp_recovers_heart_rate(self) -> None:
        times, intervals = _beats_from_bvp(_bvp(60 * FS), FS)
        self.assertGreater(len(intervals), 50)
        self.assertEqual(len(times), len(intervals))
        self.assertAlmostEqual(float(np.median(intervals)), 0.8, delta=0.03)

    def test_beats_from_bvp_too_short_returns_nothing(self) -> None:
        times, intervals = _beats_from_bvp(_bvp(10 * FS), FS)  # shorter than 30 s
        self.assertEqual((len(times), len(intervals)), (0, 0))


class TestSleepStagingDREAMTConfig(unittest.TestCase):
    """Constructor validation and class attributes."""

    def test_defaults_and_schemas(self) -> None:
        task = SleepStagingDREAMT()
        self.assertEqual(task.task_name, "SleepStagingDREAMT")
        self.assertEqual(task.input_schema, {"ibi": "tensor"})
        self.assertEqual(task.output_schema, {"label": "tensor"})
        self.assertEqual(task.num_classes, 3)
        self.assertEqual(task.window_epochs, 100)
        self.assertEqual(task.max_epochs, 1100)

    def test_num_classes_follows_label_scheme(self) -> None:
        for scheme, n in [("3class", 3), ("4class", 4), ("5class", 5)]:
            self.assertEqual(SleepStagingDREAMT(label_scheme=scheme).num_classes, n)

    def test_invalid_options_raise(self) -> None:
        for bad in (
            {"ibi_source": "ppg"},
            {"fill": "mean"},
            {"normalize": "zscore"},
            {"label_scheme": "2class"},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                SleepStagingDREAMT(**bad)


class TestBuildNight(unittest.TestCase):
    """Conversion of one night of raw columns into epoch arrays."""

    def setUp(self) -> None:
        self.stages = _stages("W", "N2", "R")
        n = len(self.stages)
        self.bvp, self.ibi = _bvp(n), _device_ibi(n)

    def _build(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        task = SleepStagingDREAMT(**kwargs)
        return task.build_night(self.bvp, self.ibi, self.stages)

    def test_shapes_dtypes_and_labels(self) -> None:
        x, labels = self._build(ibi_source="e4")
        self.assertEqual(x.shape, (3, POINTS))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertTrue(np.isfinite(x).all())

    def test_device_ibi_values_land_on_the_grid(self) -> None:
        x, _ = self._build(ibi_source="e4")
        np.testing.assert_allclose(x[1], 0.75, atol=1e-6)

    def test_bvp_source_detects_beats(self) -> None:
        x, labels = self._build(ibi_source="bvp")
        valid = x[x > 0]
        self.assertGreater(valid.size, 0.8 * x.size)
        self.assertAlmostEqual(float(np.median(valid)), 0.8, delta=0.03)
        self.assertEqual(labels.tolist(), [0, 1, 2])

    def test_zero_fill_leaves_leading_gap_at_zero(self) -> None:
        x, _ = self._build(ibi_source="e4", fill="zero")
        self.assertEqual(float(x[0, 0]), 0.0)  # before the first beat

    def test_interpolate_and_median_fill_remove_zeros(self) -> None:
        for fill in ("interpolate", "median"):
            x, _ = self._build(ibi_source="e4", fill=fill)
            self.assertTrue((x > 0).all(), fill)

    def test_add_mask_channel(self) -> None:
        x, _ = self._build(ibi_source="e4", add_mask=True)
        self.assertEqual(x.shape, (3, 2, POINTS))
        self.assertEqual(set(np.unique(x[:, 1])), {0.0, 1.0})
        self.assertEqual(float(x[0, 1, 0]), 0.0)  # missing before the first beat
        self.assertEqual(float(x[1, 1, 100]), 1.0)

    def test_night_and_epoch_normalisation(self) -> None:
        self.ibi[::48] = np.nan
        rng = np.random.default_rng(0)
        beats = np.flatnonzero(~np.isnan(_device_ibi(len(self.ibi))))
        self.ibi[beats] = rng.uniform(0.6, 1.0, len(beats))
        for mode in ("night", "epoch"):
            x, _ = self._build(ibi_source="e4", normalize=mode, add_mask=True)
            valid = x[:, 1] == 1.0
            vals = x[:, 0][valid] if mode == "night" else x[1, 0][valid[1]]
            self.assertAlmostEqual(float(vals.mean()), 0.0, delta=0.05, msg=mode)
            self.assertAlmostEqual(float(vals.std()), 1.0, delta=0.1, msg=mode)

    def test_max_epochs_truncates(self) -> None:
        x, labels = self._build(ibi_source="e4", max_epochs=2)
        self.assertEqual((x.shape[0], labels.shape[0]), (2, 2))

    def test_prep_stage_is_wake_or_ignored(self) -> None:
        self.stages = _stages("P", "N2", "R")
        self.assertEqual(self._build(ibi_source="e4")[1].tolist(), [0, 1, 2])
        ignored = self._build(ibi_source="e4", ignore_prep=True)[1]
        self.assertEqual(ignored.tolist(), [IGNORE_INDEX, 1, 2])

    def test_label_schemes(self) -> None:
        self.stages = _stages("W", "N1", "N3", "R")
        self.bvp, self.ibi = _bvp(len(self.stages)), _device_ibi(len(self.stages))
        for scheme, expected in [
            ("3class", [0, 1, 1, 2]),
            ("4class", [0, 1, 2, 3]),
            ("5class", [0, 1, 3, 4]),
        ]:
            labels = self._build(ibi_source="e4", label_scheme=scheme)[1]
            self.assertEqual(labels.tolist(), expected, scheme)

    def test_night_shorter_than_one_epoch_is_empty(self) -> None:
        stages = pd.Series(["W"] * (ROWS - 1))
        x, labels = SleepStagingDREAMT().build_night(
            _bvp(len(stages)), _device_ibi(len(stages)), stages
        )
        self.assertEqual((x.shape, labels.shape), ((0, POINTS), (0,)))

    def test_night_without_any_beat_is_all_zero(self) -> None:
        self.ibi = np.full(len(self.stages), np.nan)
        x, labels = self._build(ibi_source="e4", fill="interpolate")
        self.assertTrue((x == 0).all())
        self.assertEqual(labels.tolist(), [0, 1, 2])

    def test_unknown_stage_labels_are_ignored(self) -> None:
        self.stages = _stages("W", "Missing", "R")
        labels = self._build(ibi_source="e4")[1]
        self.assertEqual(labels.tolist(), [0, IGNORE_INDEX, 2])


class TestSleepStagingDREAMTCall(unittest.TestCase):
    """Windowing of one participant read from a 64 Hz CSV."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _patient(self, epoch_stages: list, pid: str = "S001") -> _DummyPatient:
        stages = _stages(*epoch_stages)
        path = self.dir / f"{pid}_whole_df.csv"
        pd.DataFrame(
            {
                "BVP": _bvp(len(stages)),
                "IBI": _device_ibi(len(stages)),
                "Sleep_Stage": stages,
                "HR": 60.0,  # extra columns are ignored
            }
        ).to_csv(path)
        return _DummyPatient(pid, [_DummyEvent(str(path))])

    def test_windows_keys_and_shapes(self) -> None:
        patient = self._patient(["W", "N2", "N2", "R", "R", "W"])
        samples = SleepStagingDREAMT(window_epochs=2)(patient)
        self.assertEqual(len(samples), 3)
        self.assertEqual(set(samples[0]), {"patient_id", "window_idx", "ibi", "label"})
        self.assertEqual([s["window_idx"] for s in samples], [0, 1, 2])
        self.assertEqual({s["patient_id"] for s in samples}, {"S001"})
        for s in samples:
            self.assertEqual(s["ibi"].shape, (2, POINTS))
            self.assertEqual(s["label"].shape, (2,))
        self.assertEqual(samples[1]["label"].tolist(), [1, 2])

    def test_partial_trailing_window_is_dropped(self) -> None:
        patient = self._patient(["W", "N2", "N2", "R", "R"])
        self.assertEqual(len(SleepStagingDREAMT(window_epochs=2)(patient)), 2)

    def test_fully_ignored_window_is_skipped(self) -> None:
        patient = self._patient(["W", "N2", "Missing", "Missing", "R", "R"])
        samples = SleepStagingDREAMT(window_epochs=2)(patient)
        self.assertEqual([s["window_idx"] for s in samples], [0, 2])

    def test_mask_option_changes_sample_shape(self) -> None:
        patient = self._patient(["W", "N2"])
        task = SleepStagingDREAMT(window_epochs=2, ibi_source="e4", add_mask=True)
        self.assertEqual(task(patient)[0]["ibi"].shape, (2, 2, POINTS))

    def test_patient_ids_filter(self) -> None:
        patient = self._patient(["W", "N2"])
        task = SleepStagingDREAMT(window_epochs=2, patient_ids=["S999"])
        self.assertEqual(task(patient), [])
        task = SleepStagingDREAMT(window_epochs=2, patient_ids=["S001"])
        self.assertEqual(len(task(patient)), 1)

    def test_missing_data_returns_no_samples(self) -> None:
        task = SleepStagingDREAMT(window_epochs=2)
        self.assertEqual(task(_DummyPatient("S1", [])), [])  # no events
        missing = _DummyPatient("S2", [_DummyEvent(str(self.dir / "nope.csv"))])
        self.assertEqual(task(missing), [])  # file does not exist
        self.assertEqual(task(_DummyPatient("S3", [_DummyEvent(None)])), [])
        self.assertEqual(task(_DummyPatient("S4", [_DummyEvent(np.nan)])), [])

    def test_night_shorter_than_one_window_returns_no_samples(self) -> None:
        patient = self._patient(["W"])
        self.assertEqual(SleepStagingDREAMT(window_epochs=2)(patient), [])


if __name__ == "__main__":
    unittest.main()
