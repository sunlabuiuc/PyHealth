"""HMCDataset and SleepStagingHMC, using small EDF files written here."""

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.datasets import HMCDataset
from pyhealth.datasets.metadata_csv import resolve_metadata_root
from pyhealth.tasks import SleepStagingHMC
from pyhealth.tasks.sleep_staging_hmc import (
    HMC_EEG_CHANNELS,
    HMC_PSG_CHANNELS,
    SLEEP_STAGES,
    read_hmc_scoring,
)

SFREQ = 10
# The channel order of every HMC v1.1 recording.
CHANNELS = [
    "EEG F4-M1",
    "EEG C4-M1",
    "EEG O2-M1",
    "EEG C3-M2",
    "EMG chin",
    "EOG E1-M2",
    "EOG E2-M2",
    "ECG",
]
HEADER = "Date, Time, Recording onset, Duration, Annotation, Linked channel\n"


def _write_edf(path: Path, channels: list[str], data: np.ndarray) -> None:
    """Write ``data`` (microvolts, one row per channel) as a minimal EDF file.

    Each data record is one second. Physical and digital ranges match, so
    a stored integer is the value in microvolts.
    """
    n_channels = len(channels)
    n_records = data.shape[1] // SFREQ

    def field(value, width):
        return f"{value:<{width}}"[:width]

    header = (
        field("0", 8)
        + field("X X X X", 80)
        + field("Startdate 01-JAN-2001 X X X", 80)
        + "01.01.01"
        + "23.59.30"
        + field(256 * (n_channels + 1), 8)
        + field("", 44)
        + field(n_records, 8)
        + field(1, 8)
        + field(n_channels, 4)
    )
    for width, values in [
        (16, channels),
        (80, [""] * n_channels),
        (8, ["uV"] * n_channels),
        (8, ["-32768"] * n_channels),
        (8, ["32767"] * n_channels),
        (8, ["-32768"] * n_channels),
        (8, ["32767"] * n_channels),
        (80, [""] * n_channels),
        (8, [SFREQ] * n_channels),
        (32, [""] * n_channels),
    ]:
        header += "".join(field(v, width) for v in values)
    records = data[:, : n_records * SFREQ].astype("<i2")
    records = records.reshape(n_channels, n_records, SFREQ).transpose(1, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + records.tobytes())


def _signal(seconds: int, channels: list[str]) -> np.ndarray:
    """Channel ``i`` holds ``100 * i`` plus the second the sample is in."""
    second = np.arange(seconds * SFREQ) // SFREQ
    return np.stack([100 * i + second for i in range(len(channels))])


# Default onset of ``Lights on``: inside the last epoch.
_LAST_EPOCH = object()


def _scoring_text(
    stages: Sequence[str],
    lights_off: float | None = 33.43,
    lights_on: float | None | object = _LAST_EPOCH,
) -> str:
    """An HMC ``_sleepscoring.txt`` body with the markers where HMC has them.

    By default ``Lights off`` falls inside the second epoch and ``Lights on``
    inside the last one, as in most HMC v1.1 recordings. A marker set to
    None is left out of the file.
    """
    if lights_on is _LAST_EPOCH:
        lights_on = 30 * (len(stages) - 1) + 28.74
    rows = [(30.0 * i, 30, f"Sleep stage {s}", "") for i, s in enumerate(stages)]
    if lights_off is not None:
        rows.append((lights_off, 0, "Lights off", "EEG F4-A1"))
    if lights_on is not None:
        rows.append((lights_on, 0, "Lights on", "EEG Fpz-Cz"))
    rows.sort(key=lambda row: row[0])
    lines = [f"01.01.01, 00.00.00, {o:g}, {d}, {a}, {c}" for o, d, a, c in rows]
    return HEADER + "\n".join(lines)  # HMC files end without a newline


def _write_recording(
    folder: Path,
    stem: str,
    stages: Sequence[str],
    seconds: int | None = None,
    channels: list[str] = CHANNELS,
    **markers,
) -> None:
    """Write an HMC style recording and its text scoring in ``folder``."""
    seconds = 30 * len(stages) if seconds is None else seconds
    _write_edf(folder / f"{stem}.edf", channels, _signal(seconds, channels))
    (folder / f"{stem}_sleepscoring.txt").write_text(
        _scoring_text(stages, **markers), encoding="utf-8"
    )


def _patient(folder: Path, stem: str) -> SimpleNamespace:
    """A patient with one ``recordings`` event, like a row of the CSV."""
    event = SimpleNamespace(
        signal_file=str(folder / f"{stem}.edf"),
        label_file=str(folder / f"{stem}_sleepscoring.txt"),
    )
    return SimpleNamespace(patient_id=stem, get_events=lambda event_type=None: [event])


def _prepare(root: Path) -> pd.DataFrame:
    ds = HMCDataset.__new__(HMCDataset)
    ds.root = str(root)
    ds.prepare_metadata()
    return pd.read_csv(root / "hmc-pyhealth.csv", dtype=str)


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class TestReadHMCScoring(_TempDir):
    def test_stages_and_lights_markers(self):
        path = self.root / "SN001_sleepscoring.txt"
        path.write_text(_scoring_text(["W", "W", "N1", "R"]), encoding="utf-8")

        scoring = read_hmc_scoring(str(path))

        self.assertEqual(scoring.onsets, [0.0, 30.0, 60.0, 90.0])
        self.assertEqual(
            scoring.stages,
            ["Sleep stage W", "Sleep stage W", "Sleep stage N1", "Sleep stage R"],
        )
        self.assertEqual(scoring.lights_off, 33.43)
        self.assertEqual(scoring.lights_on, 118.74)

    def test_missing_markers_are_none(self):
        path = self.root / "SN001_sleepscoring.txt"
        path.write_text(HEADER + "01.01.01, 00.00.00, 0, 30, Sleep stage W, ")

        scoring = read_hmc_scoring(str(path))

        self.assertEqual(scoring.stages, ["Sleep stage W"])
        self.assertIsNone(scoring.lights_off)
        self.assertIsNone(scoring.lights_on)


class TestHMCMetadata(_TempDir):
    def test_recordings_folder_and_scoring_edf_ignored(self):
        recordings = self.root / "recordings"
        for stem in ("SN002", "SN001"):
            _write_recording(recordings, stem, ["W"])
        (recordings / "SN001_sleepscoring.edf").write_bytes(b"")

        df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), ["SN001", "SN002"])
        self.assertEqual(Path(df.iloc[0]["signal_file"]).name, "SN001.edf")
        self.assertEqual(Path(df.iloc[0]["label_file"]).name, "SN001_sleepscoring.txt")

    def test_recording_without_text_scoring_is_left_out(self):
        _write_recording(self.root, "SN001", ["W"])
        _write_edf(self.root / "SN002.edf", CHANNELS, _signal(30, CHANNELS))

        with self.assertLogs("pyhealth.datasets.hmc", level="WARNING") as logs:
            df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), ["SN001"])
        self.assertIn("SN002", logs.output[0])

    def test_metadata_goes_to_the_home_cache_when_root_is_read_only(self):
        for stem in ("SN002", "SN001"):
            _write_recording(self.root, stem, ["W"])
        home_dir = tempfile.TemporaryDirectory()
        self.addCleanup(home_dir.cleanup)
        home = Path(home_dir.name)
        real_to_csv = pd.DataFrame.to_csv

        def to_csv(df, path_or_buf=None, *args, **kwargs):
            # Stands in for a read-only root: chmod has no effect on Windows.
            if str(path_or_buf).startswith(str(self.root)):
                raise PermissionError(f"read-only: {path_or_buf}")
            return real_to_csv(df, path_or_buf, *args, **kwargs)

        with (
            patch.object(pd.DataFrame, "to_csv", to_csv),
            patch("pyhealth.datasets.metadata_csv.Path.home", return_value=home),
        ):
            ds = HMCDataset.__new__(HMCDataset)
            ds.root = str(self.root)
            ds.prepare_metadata()
            metadata_root = Path(
                resolve_metadata_root(self.root, "hmc", "hmc-pyhealth.csv")
            )

        self.assertFalse((self.root / "hmc-pyhealth.csv").exists())
        self.assertTrue(metadata_root.is_relative_to(home))
        cached = pd.read_csv(metadata_root / "hmc-pyhealth.csv", dtype=str)
        self.assertEqual(list(cached["patient_id"]), ["SN001", "SN002"])
        self.assertEqual(
            Path(cached.iloc[0]["signal_file"]), (self.root / "SN001.edf").absolute()
        )

    def test_default_task(self):
        task = HMCDataset.__new__(HMCDataset).default_task

        self.assertIsInstance(task, SleepStagingHMC)
        self.assertEqual(task.channels, HMC_EEG_CHANNELS)
        self.assertEqual(task.epoch_seconds, 30)
        self.assertTrue(task.lights_period_only)

    def test_empty_root_writes_nothing(self):
        with self.assertLogs("pyhealth.datasets.hmc", level="WARNING"):
            ds = HMCDataset.__new__(HMCDataset)
            ds.root = str(self.root)
            ds.prepare_metadata()
        self.assertFalse((self.root / "hmc-pyhealth.csv").exists())


class TestSleepStagingHMC(_TempDir):
    STAGES = ("W", "W", "N1", "N2", "N3", "R", "W")

    def test_labels_shapes_and_lights_period(self):
        _write_recording(self.root, "SN001", self.STAGES)

        samples = SleepStagingHMC()(_patient(self.root, "SN001"))

        # Epoch 0 starts and epoch 1 starts before lights off; the last
        # epoch ends after lights on.
        self.assertEqual([s["epoch_index"] for s in samples], [2, 3, 4, 5])
        self.assertEqual([s["label"] for s in samples], [1, 2, 3, 4])
        self.assertEqual(
            [SLEEP_STAGES[s["label"]] for s in samples], ["N1", "N2", "N3", "REM"]
        )
        first = samples[0]
        self.assertEqual(first["patient_id"], "SN001")
        self.assertEqual(first["signal"].shape, (4, 30 * SFREQ))
        self.assertEqual(first["signal"].dtype, np.float32)
        # Channel i holds 100 * i plus the second: epoch 2 covers 60-89 s.
        np.testing.assert_allclose(first["signal"][1, :SFREQ], 160.0)
        np.testing.assert_allclose(first["signal"][3, -SFREQ:], 389.0)

    def test_every_scored_epoch_without_lights_period(self):
        _write_recording(self.root, "SN001", self.STAGES)

        task = SleepStagingHMC(lights_period_only=False)
        samples = task(_patient(self.root, "SN001"))

        self.assertEqual([s["label"] for s in samples], [0, 0, 1, 2, 3, 4, 0])

    def test_missing_markers_keep_every_epoch_with_a_warning(self):
        for markers in (
            {"lights_off": None},
            {"lights_on": None},
            {"lights_off": None, "lights_on": None},
        ):
            with self.subTest(**markers):
                _write_recording(self.root, "SN001", ["W", "N1", "W"], **markers)
                text = (self.root / "SN001_sleepscoring.txt").read_text()
                self.assertEqual("Lights off" in text, "lights_off" not in markers)
                self.assertEqual("Lights on" in text, "lights_on" not in markers)

                with self.assertLogs(
                    "pyhealth.tasks.sleep_staging_hmc", level="WARNING"
                ) as logs:
                    samples = SleepStagingHMC()(_patient(self.root, "SN001"))

                self.assertEqual(len(samples), 3)
                self.assertIn("lights", logs.output[0])

    def test_shorter_epochs_split_each_score(self):
        _write_recording(self.root, "SN001", self.STAGES)

        samples = SleepStagingHMC(epoch_seconds=10)(_patient(self.root, "SN001"))

        self.assertEqual(len(samples), 12)
        self.assertEqual([s["epoch_index"] for s in samples[:4]], [6, 7, 8, 9])
        self.assertEqual(samples[0]["signal"].shape, (4, 10 * SFREQ))
        np.testing.assert_allclose(samples[1]["signal"][0, :SFREQ], 70.0)

    def test_psg_channels(self):
        _write_recording(self.root, "SN001", self.STAGES)

        task = SleepStagingHMC(channels=HMC_PSG_CHANNELS)
        samples = task(_patient(self.root, "SN001"))

        self.assertEqual(samples[0]["signal"].shape, (7, 30 * SFREQ))
        # EOG E1-M2 is the sixth channel of the file (index 5).
        np.testing.assert_allclose(samples[0]["signal"][4, 0], 500 + 60)

    def test_missing_channel_skips_the_recording(self):
        _write_recording(self.root, "SN001", self.STAGES, channels=CHANNELS[1:])

        with self.assertLogs("pyhealth.tasks.sleep_staging_hmc", level="WARNING"):
            samples = SleepStagingHMC()(_patient(self.root, "SN001"))

        self.assertEqual(samples, [])

    def test_unknown_stage_is_dropped_with_a_warning(self):
        _write_recording(self.root, "SN001", ["W", "W", "N2", "?", "N2", "W"])

        with self.assertLogs(
            "pyhealth.tasks.sleep_staging_hmc", level="WARNING"
        ) as logs:
            samples = SleepStagingHMC()(_patient(self.root, "SN001"))

        self.assertEqual([s["epoch_index"] for s in samples], [2, 4])
        self.assertIn("Dropped 1 epochs", logs.output[0])

    def test_scored_epoch_past_the_signal_end_is_dropped(self):
        _write_recording(
            self.root,
            "SN001",
            self.STAGES,
            seconds=30 * 5 + 7,
            lights_on=30 * 7.0,
        )

        with self.assertLogs("pyhealth.tasks.sleep_staging_hmc", level="WARNING"):
            samples = SleepStagingHMC()(_patient(self.root, "SN001"))

        self.assertEqual([s["epoch_index"] for s in samples], [2, 3, 4])

    def test_invalid_arguments(self):
        for kwargs in ({"epoch_seconds": 7}, {"epoch_seconds": 0}, {"channels": ()}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SleepStagingHMC(**kwargs)

    def test_default_channels_are_the_four_eeg_derivations(self):
        self.assertEqual(SleepStagingHMC().channels, HMC_EEG_CHANNELS)
        self.assertEqual(len(HMC_EEG_CHANNELS), 4)


class TestHMCEndToEnd(_TempDir):
    def test_set_task_on_two_recordings(self):
        recordings = self.root / "recordings"
        _write_recording(recordings, "SN001", TestSleepStagingHMC.STAGES)
        _write_recording(recordings, "SN002", ["W", "W", "N2", "N2", "W"])

        dataset = HMCDataset(root=str(self.root), cache_dir=str(self.root / "c"))
        samples = dataset.set_task()

        self.assertEqual(sorted(dataset.unique_patient_ids), ["SN001", "SN002"])
        self.assertEqual(len(samples), 4 + 2)
        self.assertEqual(tuple(samples[0]["signal"].shape), (4, 30 * SFREQ))


if __name__ == "__main__":
    unittest.main()
