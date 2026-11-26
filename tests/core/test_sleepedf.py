"""Tests for SleepEDFDataset metadata and the Sleep EDF staging tasks.

All data is synthetic. The subject spreadsheets (``*.xls``) are replaced by
in-memory DataFrames, because reading ``.xls`` needs ``xlrd``, which PyHealth
does not install. EDF reads are replaced by small MNE objects.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.sleepedf import SleepEDFDataset
from pyhealth.tasks import sleep_staging_v2
from pyhealth.tasks.sleep_staging_v2 import (
    SleepStagingSequenceSleepEDF,
    SleepStagingSleepEDF,
)

CASSETTE_SHEET = pd.DataFrame(
    {
        "subject": [0, 0],
        "night": [1, 2],
        "age": [33, 33],
        "sex (F=1)": [1, 1],
        "LightsOff": ["00:38:00", "21:57:00"],
    }
)

# Same two-row header as the real ST-subjects.xls.
TELEMETRY_SHEET = pd.DataFrame(
    [[1, 60, 1, 1, "23:01:00", 2, "23:48:00"]],
    columns=pd.MultiIndex.from_tuples(
        [
            ("Subject - age - sex", "Nr"),
            ("Subject - age - sex", "Age"),
            ("Subject - age - sex", "M1/F2"),
            ("Placebo night", "night nr"),
            ("Placebo night", "lights off"),
            ("Temazepam night", "night nr"),
            ("Temazepam night", "lights off"),
        ]
    ),
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_root(root: Path) -> None:
    for name in (
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf",
        "SC4002E0-PSG.edf",
        "SC4002EC-Hypnogram.edf",
    ):
        _touch(root / "sleep-cassette" / name)
    for name in (
        "ST7011J0-PSG.edf",
        "ST7011JP-Hypnogram.edf",
        "ST7012J0-PSG.edf",
        "ST7012JP-Hypnogram.edf",
    ):
        _touch(root / "sleep-telemetry" / name)


def _fake_read_excel(path, *args, **kwargs):
    name = Path(path).name
    if name == "SC-subjects.xls":
        return CASSETTE_SHEET.copy()
    if name == "ST-subjects.xls":
        return TELEMETRY_SHEET.copy()
    raise FileNotFoundError(path)


def _read_only(root: Path):
    """Make ``DataFrame.to_csv`` fail under ``root``.

    This stands in for a read-only directory: ``chmod`` has no effect on
    Windows or when the tests run as root.
    """
    real_to_csv = pd.DataFrame.to_csv

    def to_csv(df, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf).startswith(str(root)):
            raise PermissionError(f"read-only: {path_or_buf}")
        return real_to_csv(df, path_or_buf, *args, **kwargs)

    return patch.object(pd.DataFrame, "to_csv", to_csv)


def _metadata_root(root: Path, subset: str) -> Path:
    """Run ``__init__`` up to ``BaseDataset`` and return the root it picks."""
    captured = {}

    def fake_base_init(dataset, root, **kwargs):
        captured["root"] = root

    with patch.object(BaseDataset, "__init__", fake_base_init):
        SleepEDFDataset(root=str(root), subset=subset)
    return Path(captured["root"])


@patch("pyhealth.datasets.sleepedf.pd.read_excel", side_effect=_fake_read_excel)
class TestSleepEDFMetadata(unittest.TestCase):
    def test_cassette_csv_pairs_psg_and_hypnogram(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_root(root)
            self.assertEqual(_metadata_root(root, "cassette"), root)
            df = pd.read_csv(root / "sleepedf-cassette-pyhealth.csv")
            self.assertEqual(len(df), 2)
            night1 = df[df["night"] == 1].iloc[0]
            self.assertTrue(night1["signal_file"].endswith("SC4001E0-PSG.edf"))
            self.assertTrue(night1["label_file"].endswith("SC4001EC-Hypnogram.edf"))

    def test_telemetry_csv_has_one_row_per_condition(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_root(root)
            self.assertEqual(_metadata_root(root, "telemetry"), root)
            df = pd.read_csv(root / "sleepedf-telemetry-pyhealth.csv")
            self.assertEqual(sorted(df["condition"]), ["placebo", "temazepam"])
            placebo = df[df["condition"] == "placebo"].iloc[0]
            self.assertEqual(placebo["night"], 1)
            self.assertEqual(placebo["sex"], "M")
            self.assertTrue(placebo["signal_file"].endswith("ST7011J0-PSG.edf"))

    def test_nights_without_files_are_left_out_with_a_warning(self, _):
        # A partial download: the sheet lists two nights, only one is present.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "sleep-cassette" / "SC4001E0-PSG.edf")
            _touch(root / "sleep-cassette" / "SC4001EC-Hypnogram.edf")
            with self.assertLogs("pyhealth.datasets.sleepedf", "WARNING") as logs:
                _metadata_root(root, "cassette")
            df = pd.read_csv(root / "sleepedf-cassette-pyhealth.csv")
            self.assertEqual(df["night"].tolist(), [1])
            self.assertIn("1 of 2 cassette nights", logs.output[0])

    def test_read_only_root_falls_back_to_home_cache(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            home, root = tmp / "home", tmp / "sleep-edfx"
            _make_root(root)
            with patch.object(Path, "home", return_value=home), _read_only(root):
                meta_root = _metadata_root(root, "cassette")

            self.assertTrue(meta_root.is_relative_to(home))
            self.assertFalse((root / "sleepedf-cassette-pyhealth.csv").exists())
            df = pd.read_csv(meta_root / "sleepedf-cassette-pyhealth.csv")
            self.assertEqual(len(df), 2)
            # Paths still point to the raw files, not to the cache.
            self.assertTrue(df["signal_file"].iloc[0].startswith(str(root)))

    def test_existing_cached_csv_is_reused(self, read_excel):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            home, root = tmp / "home", tmp / "sleep-edfx"
            _make_root(root)
            with patch.object(Path, "home", return_value=home), _read_only(root):
                first = _metadata_root(root, "cassette")
                calls = read_excel.call_count
                second = _metadata_root(root, "cassette")
            self.assertEqual(first, second)
            self.assertEqual(read_excel.call_count, calls)


CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "Event marker"]
TYPES = ["eeg", "eeg", "eog", "stim"]
SFREQ = 100


def _run_task(descriptions, chunk_duration=30.0, task=None, n_epochs=None):
    """Run a task on one synthetic night scored with ``descriptions``.

    Each description covers one 30 s epoch, in order. Every sample of epoch
    ``i`` holds the value ``i``, so a test can tell which epoch a signal came
    from. ``n_epochs`` shortens the recording below the scored epochs.
    """
    n_epochs = len(descriptions) if n_epochs is None else n_epochs
    values = np.repeat(np.arange(n_epochs, dtype=float), SFREQ * 30)
    raw = mne.io.RawArray(
        np.tile(values, (len(CHANNELS), 1)),
        mne.create_info(CHANNELS, SFREQ, TYPES),
        verbose="error",
    )
    annotations = mne.Annotations(
        onset=[30.0 * i for i in range(len(descriptions))],
        duration=[30.0] * len(descriptions),
        description=list(descriptions),
    )
    event = SimpleNamespace(
        signal_file="SC4001E0-PSG.edf",
        label_file="SC4001EC-Hypnogram.edf",
        night=1,
        age=33,
        sex=1,
    )
    patient = SimpleNamespace(patient_id="0", get_events=lambda *a, **k: [event])
    with (
        patch.object(sleep_staging_v2.mne.io, "read_raw_edf", return_value=raw),
        patch.object(
            sleep_staging_v2.mne, "read_annotations", return_value=annotations
        ),
    ):
        if task is None:
            task = SleepStagingSleepEDF(chunk_duration=chunk_duration)
        return task(patient)


class TestSleepStagingSleepEDF(unittest.TestCase):
    def test_all_stages_get_fixed_codes(self):
        samples = _run_task(
            [
                "Sleep stage W",
                "Sleep stage 1",
                "Sleep stage 2",
                "Sleep stage 3",
                "Sleep stage 4",
                "Sleep stage R",
            ]
        )
        self.assertEqual([s["label"] for s in samples], [0, 1, 2, 3, 4, 5])
        self.assertEqual(samples[0]["signal"].shape, (len(CHANNELS), 30 * SFREQ))
        self.assertEqual(samples[0]["patient_id"], "0")
        self.assertEqual(samples[0]["night"], 1)

    def test_night_without_stage_3_and_4_does_not_raise(self):
        # Many real Sleep-EDF nights have no stage 4 (69 of 153 cassette
        # hypnograms); mne.Epochs raised on the missing event id.
        samples = _run_task(
            ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage R"]
        )
        self.assertEqual([s["label"] for s in samples], [0, 1, 2, 5])

    def test_unscored_and_movement_epochs_are_dropped(self):
        samples = _run_task(
            ["Sleep stage W", "Sleep stage ?", "Movement time", "Sleep stage 2"]
        )
        self.assertEqual([s["label"] for s in samples], [0, 2])

    def test_night_without_any_scored_stage_returns_no_sample(self):
        self.assertEqual(_run_task(["Sleep stage ?", "Movement time"]), [])

    def test_annotations_are_cut_into_chunk_duration_epochs(self):
        samples = _run_task(["Sleep stage 2", "Sleep stage R"], chunk_duration=15.0)
        self.assertEqual([s["label"] for s in samples], [2, 2, 5, 5])
        self.assertEqual(samples[0]["signal"].shape, (len(CHANNELS), 15 * SFREQ))


W, N1, N2, N3, N4, R = (f"Sleep stage {s}" for s in "W1234R")


def _run_sequence(descriptions, n_epochs=None, **kwargs):
    kwargs.setdefault("wake_edge_minutes", None)
    task = SleepStagingSequenceSleepEDF(**kwargs)
    return _run_task(descriptions, task=task, n_epochs=n_epochs)


class TestSleepStagingSequenceSleepEDF(unittest.TestCase):
    def test_windows_hold_aasm_codes_and_their_epochs(self):
        samples = _run_sequence([W, N1, N2, N3, N4, R], seq_len=3)
        self.assertEqual([s["label"].tolist() for s in samples], [[0, 1, 2], [3, 3, 4]])
        first = samples[0]
        self.assertEqual(first["signal"].shape, (3, 1, 30 * SFREQ))
        self.assertEqual(first["signal"].dtype, np.float32)
        self.assertEqual(first["label"].dtype, np.int64)
        # Epoch i holds i volts, read out in microvolts: the second window is
        # epochs 3-5.
        self.assertEqual(samples[1]["signal"][:, 0, 0].tolist(), [3e6, 4e6, 5e6])
        self.assertEqual([s["start"] for s in samples], [0.0, 90.0])

    def test_windows_do_not_span_unscored_epochs(self):
        samples = _run_sequence(
            [N2, N2, "Sleep stage ?", N2, N2, "Movement time", N2, N2, N2],
            seq_len=2,
        )
        starts = [(s["signal"][:, 0, 0] / 1e6).tolist() for s in samples]
        # The run 6-8 gives one window; its last epoch is left over.
        self.assertEqual(starts, [[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])

    def test_stride_gives_overlapping_windows(self):
        samples = _run_sequence([N2] * 4, seq_len=2, stride=1)
        self.assertEqual(len(samples), 3)

    def test_wake_is_trimmed_around_sleep(self):
        # 2 min of wake on each side is kept: four epochs before N1, four after R.
        night = [W] * 10 + [N1, N2, R] + [W] * 10
        samples = _run_sequence(night, seq_len=1, wake_edge_minutes=2)
        epochs = [int(s["signal"][0, 0, 0] / 1e6) for s in samples]
        self.assertEqual(epochs, list(range(6, 17)))

    def test_night_without_sleep_is_skipped_when_trimming(self):
        self.assertEqual(_run_sequence([W] * 3, seq_len=1, wake_edge_minutes=30), [])

    def test_epoch_past_the_end_of_the_recording_is_dropped(self):
        samples = _run_sequence([N2, N2, N2], n_epochs=2, seq_len=1)
        self.assertEqual(len(samples), 2)

    def test_samples_load_with_long_labels(self):
        samples = _run_sequence([W, N1, N2, N3, R, W], seq_len=3)
        task = SleepStagingSequenceSleepEDF
        dataset = create_sample_dataset(
            samples=[
                {k: s[k] for k in ("patient_id", "signal", "label")} for s in samples
            ],
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )
        item = dataset[0]
        self.assertEqual(item["label"].dtype, torch.long)
        self.assertEqual(tuple(item["signal"].shape), (3, 1, 30 * SFREQ))

    def test_rejects_bad_window_sizes(self):
        with self.assertRaises(ValueError):
            SleepStagingSequenceSleepEDF(seq_len=0)
        with self.assertRaises(ValueError):
            SleepStagingSequenceSleepEDF(stride=0)


if __name__ == "__main__":
    unittest.main()
