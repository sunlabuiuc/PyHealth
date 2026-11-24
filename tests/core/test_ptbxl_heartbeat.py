"""Tests for PTBXLHeartbeatMI on synthetic WFDB records (no real PTB-XL data)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import PTBXLDataset, split_by_strat_fold
from pyhealth.tasks import PTBXLHeartbeatMI
from pyhealth.tasks.ptbxl_heartbeat import detect_r_peaks, segment_heartbeats

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "test-resources" / "core" / "ptbxl"


def _beats(fs, seconds=9.8, rr=0.8, offset=0.4, width=0.012):
    """A spike train: one Gaussian 'R wave' every ``rr`` seconds.

    9.8 s ends between spikes, so every spike is whole: 12 at rr=0.8, 16 at 0.6.
    """
    t = np.arange(int(seconds * fs)) / fs
    return np.exp(-(((t % rr) - offset) ** 2) / (2 * width**2))


def _write_wfdb(record_base, signal, fs):
    """Writes ``(leads, samples)`` mV as a 16-bit WFDB record."""
    record_base.parent.mkdir(parents=True, exist_ok=True)
    leads, n = signal.shape
    data = np.round(signal.T * 1000).astype(np.int16)
    Path(f"{record_base}.dat").write_bytes(data.tobytes())
    lines = [f"{record_base.name} {leads} {fs} {n}"]
    lines += [
        f"{record_base.name}.dat 16 1000.0(0)/mV 16 0 0 0 0 {i}" for i in range(leads)
    ]
    Path(f"{record_base}.hea").write_text("\n".join(lines) + "\n")


class TestHelpers(unittest.TestCase):
    def test_r_peaks_follow_the_leads_that_agree(self):
        clean = _beats(250)
        noisy = clean + _beats(250, rr=0.27, offset=0.1)  # extra spikes
        peaks = detect_r_peaks(np.stack([clean, 0.8 * clean, noisy]), fs=250)
        self.assertEqual(len(peaks), 12)
        np.testing.assert_allclose(np.diff(peaks), 200, atol=1)
        np.testing.assert_allclose(peaks % 200, 100, atol=2)

    def test_no_beats_gives_no_peaks(self):
        self.assertEqual(len(detect_r_peaks(np.zeros((2, 2500)), fs=250)), 0)

    def test_windows_are_centred_and_padded(self):
        signal = np.arange(1000, dtype=np.float32)[None].repeat(2, 0)
        beats = segment_heartbeats(signal, np.array([50, 250, 450]), length=288)
        self.assertEqual(beats.shape, (3, 2, 288))
        # R-R is 200: 100 samples on each side, 44 zeros on each side.
        np.testing.assert_array_equal(beats[1, 0, 44:244], np.arange(150, 350))
        self.assertEqual(beats[1, 0, :44].sum() + beats[1, 0, 244:].sum(), 0)
        # The first beat is cut at sample 0, so 150 samples are padded to 288.
        np.testing.assert_array_equal(beats[0, 0, 69:219], np.arange(0, 150))

    def test_skip_partial_drops_beats_cut_by_the_edges(self):
        signal = np.arange(1000, dtype=np.float32)[None]
        peaks = np.array([50, 250, 450, 650, 900, 950])  # median R-R 200: half 100
        beats = segment_heartbeats(signal, peaks, length=288, skip_partial=True)
        # 50 starts before 0 and 950 ends after 1000; 900 ends exactly at it.
        np.testing.assert_array_equal(beats[:, 0, 144], [250, 450, 650, 900])

    def test_long_rr_is_cropped_to_the_length(self):
        beats = segment_heartbeats(np.ones((1, 3000)), np.array([500, 1500]), 288)
        self.assertEqual(beats[0, 0].sum(), 288)

    def test_fewer_than_two_peaks_gives_no_beat(self):
        self.assertEqual(
            segment_heartbeats(np.ones((2, 500)), np.array([9]), 8).shape, (0, 2, 8)
        )


class TestTask(unittest.TestCase):
    def setUp(self):
        self.task = PTBXLHeartbeatMI(FIXTURE_ROOT / "scp_statements.csv")

    def test_labels(self):
        self.assertEqual(self.task.label_of(["MI"]), 1)
        self.assertEqual(self.task.label_of(["HYP", "MI"]), 1)
        self.assertEqual(self.task.label_of(["NORM"]), 0)
        self.assertIsNone(self.task.label_of(["NORM", "STTC"]))
        self.assertIsNone(self.task.label_of([]))

    def test_heartbeats_resample_and_zscore(self):
        record = 2.0 * _beats(500)[None].repeat(12, 0) + 5.0
        beats = self.task.heartbeats(record, fs=500)
        self.assertEqual(beats.shape, (12, 12, 288))
        # At 250 Hz the R-R interval is 200 samples: 44 zeros on each side.
        self.assertTrue(np.all(beats[1:-1, :, :44] == 0))
        self.assertEqual(int(np.argmax(beats[3, 0])), 144)
        self.assertGreater(beats[3, 0, 144], 3.0)  # z-scored: spikes are outliers
        capped = PTBXLHeartbeatMI(
            FIXTURE_ROOT / "scp_statements.csv", max_beats_per_record=4
        )
        self.assertEqual(len(capped.heartbeats(record, fs=500)), 4)

    def test_skip_partial_beats_drops_the_edge_beats(self):
        # Spikes at 0.1 s, 0.9 s, ..., 9.7 s: the last is cut by the end.
        record = _beats(500, offset=0.1)[None].repeat(12, 0)
        padded = self.task.heartbeats(record, fs=500)
        skipping = PTBXLHeartbeatMI(
            FIXTURE_ROOT / "scp_statements.csv", skip_partial_beats=True
        )
        whole = skipping.heartbeats(record, fs=500)
        self.assertEqual(len(padded), len(whole) + 1)
        np.testing.assert_array_equal(whole, padded[:-1])
        # At 250 Hz the last spike is at 2425 of 2450 samples, so only 125
        # samples of its 200-sample window exist: 81 zeros, then the window.
        nonzero = np.flatnonzero(padded[-1, 0])
        self.assertEqual((nonzero[0], nonzero[-1]), (81, 205))

    def test_rejects_bad_arguments(self):
        path = FIXTURE_ROOT / "scp_statements.csv"
        with self.assertRaises(ValueError):
            PTBXLHeartbeatMI(path, length=1)
        with self.assertRaises(ValueError):
            PTBXLHeartbeatMI(path, max_beats_per_record=0)


class TestSetTask(unittest.TestCase):
    """The task on PTBXLDataset, with real set_task and split_by_strat_fold."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "data"
        root.mkdir()
        for name in ("ptbxl_database.csv", "scp_statements.csv"):
            shutil.copy(FIXTURE_ROOT / name, root / name)
        db = pd.read_csv(root / "ptbxl_database.csv")
        for i, row in db.iterrows():
            rr = 0.8 if i % 2 else 0.6  # 75 or 100 bpm
            record = _beats(500, rr=rr)[None].repeat(12, 0)
            _write_wfdb(root / row.filename_hr, record, 500)
            _write_wfdb(root / row.filename_lr, record[:, ::5], 100)
        dataset = PTBXLDataset(
            root=str(root),
            sampling_rate=500,
            metadata_cache_dir=cls.tmp / "meta",
            cache_dir=cls.tmp / "cache",
        )
        cls.samples = dataset.set_task(
            PTBXLHeartbeatMI(dataset.scp_statements_path), num_workers=1
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_one_sample_per_beat_of_the_mi_and_healthy_records(self):
        # Kept: ecg 1 NORM, ecg 2 MI, ecg 4 MI+HYP. Skipped: 3 PACE, 5 no label.
        records = {}
        for sample in self.samples:
            records.setdefault(sample["record_id"], []).append(sample)
            self.assertEqual(tuple(sample["signal"].shape), (12, 288))
        self.assertEqual(sorted(records), ["1", "2", "4"])
        labels = {r: {int(s["label"].item()) for s in b} for r, b in records.items()}
        self.assertEqual(labels, {"1": {0}, "2": {1}, "4": {1}})
        # 9.8 s at 100 bpm (ecg 1) or 75 bpm (ecg 2 and 4).
        self.assertEqual(len(records["1"]), 16)
        self.assertEqual(len(records["2"]), 12)
        self.assertEqual(sorted(int(s["beat"]) for s in records["2"]), list(range(12)))
        self.assertTrue(torch.isfinite(self.samples[0]["signal"]).all())

    def test_strat_fold_splits(self):
        train, val, test = split_by_strat_fold(
            self.samples, train_folds=(1,), val_folds=(2,), test_folds=(9,)
        )
        self.assertEqual((len(train), len(val), len(test)), (16, 12, 12))


if __name__ == "__main__":
    unittest.main()
