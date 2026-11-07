"""ISRUCDataset and SleepStagingISRUC, using small EDF files written here."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.datasets import ISRUCDataset
from pyhealth.tasks import SleepStagingISRUC
from pyhealth.tasks.sleep_staging_isruc import (
    SLEEP_STAGES,
    base_channel_name,
    read_isruc_scores,
)

SFREQ = 10
# Subgroup I style names, with both A and M references and non EEG channels.
CHANNELS = ["LOC-A2", "F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-M1", "O2-A1", "X1"]


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
        + field("Startdate 01-JAN-2000 X X X", 80)
        + "01.01.00"
        + "00.00.00"
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


def _signal(seconds: int, channels: list[str] = CHANNELS) -> np.ndarray:
    """Channel ``i`` holds ``100 * i`` plus the second the sample is in."""
    second = np.arange(seconds * SFREQ) // SFREQ
    return np.stack([100 * i + second for i in range(len(channels))])


def _write_recording(
    folder: Path,
    stem: str,
    scores: list[int],
    seconds: int | None = None,
    suffix: str = ".rec",
    channels: list[str] = CHANNELS,
    scorings: tuple[int, ...] = (1, 2),
) -> None:
    """Write an ISRUC style recording and its scoring files in ``folder``."""
    seconds = 30 * len(scores) if seconds is None else seconds
    _write_edf(folder / f"{stem}{suffix}", channels, _signal(seconds, channels))
    for annotator in scorings:
        (folder / f"{stem}_{annotator}.txt").write_text(
            "".join(f"{s}\n" for s in scores)
        )


def _event(folder: Path, stem: str, suffix: str = ".edf", **overrides):
    """A ``recordings`` event like the rows of ``isruc-pyhealth.csv``."""
    fields = {
        "session_id": "1",
        "signal_file": str(folder / f"{stem}{suffix}"),
        "label_file_1": str(folder / f"{stem}_1.txt"),
        "label_file_2": str(folder / f"{stem}_2.txt"),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _patient(patient_id: str, events: list) -> SimpleNamespace:
    return SimpleNamespace(
        patient_id=patient_id,
        get_events=lambda event_type=None: events,
    )


def _prepare(root: Path) -> pd.DataFrame:
    ds = ISRUCDataset.__new__(ISRUCDataset)
    ds.root = str(root)
    ds.prepare_metadata()
    return pd.read_csv(root / "isruc-pyhealth.csv", dtype=str, keep_default_na=False)


class TestISRUCMetadata(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "ISRUC-Sleep-I"
        self.root.mkdir()

    def test_subgroup_one_layout(self):
        for subject in ("2", "10", "1"):
            _write_recording(self.root / subject, subject, [0])
        (self.root / "1" / "1_1.xlsx").write_bytes(b"")

        df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), ["1", "2", "10"])
        self.assertEqual(set(df["session_id"]), {"1"})
        row = df.iloc[0]
        self.assertEqual(Path(row["signal_file"]).name, "1.edf")
        self.assertEqual(Path(row["label_file_1"]).name, "1_1.txt")
        self.assertEqual(Path(row["label_file_2"]).name, "1_2.txt")

    def test_rec_gets_a_hard_link_with_the_edf_extension(self):
        _write_recording(self.root / "1", "1", [0, 5])

        row = _prepare(self.root).iloc[0]

        rec = self.root / "1" / "1.rec"
        edf = Path(row["signal_file"])
        self.assertEqual(edf, (self.root / "1" / "1.edf").absolute())
        self.assertTrue(rec.exists())
        self.assertTrue(os.path.samefile(rec, edf))

    def test_subgroup_two_layout_has_sessions(self):
        for subject in ("1", "2"):
            for session in ("1", "2"):
                folder = self.root / subject / session
                _write_recording(folder, session, [0])

        df = _prepare(self.root)

        self.assertEqual(
            list(zip(df["patient_id"], df["session_id"])),
            [("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")],
        )
        self.assertEqual(Path(df.iloc[1]["signal_file"]).parent.name, "2")

    def test_existing_edf_is_used(self):
        _write_recording(self.root / "1", "1", [0], suffix=".edf")
        _write_recording(self.root / "2", "2", [0], suffix=".rec")
        _write_edf(self.root / "2" / "2.edf", CHANNELS, _signal(30))

        df = _prepare(self.root)

        self.assertEqual([Path(f).name for f in df["signal_file"]], ["1.edf", "2.edf"])
        self.assertFalse(
            os.path.samefile(self.root / "2" / "2.rec", df.iloc[1]["signal_file"])
        )

    def test_missing_scoring_is_left_empty(self):
        _write_recording(self.root / "1", "1", [0], scorings=(1,))

        row = _prepare(self.root).iloc[0]

        self.assertTrue(row["label_file_1"].endswith("1_1.txt"))
        self.assertEqual(row["label_file_2"], "")

    def test_files_not_named_after_their_folder_are_ignored(self):
        _write_recording(self.root / "1", "1", [0])
        _write_recording(self.root / "1", "backup", [0], suffix=".edf")
        _write_recording(self.root, "stray", [0], suffix=".edf")

        df = _prepare(self.root)

        self.assertEqual(len(df), 1)
        self.assertEqual(Path(df.iloc[0]["signal_file"]).name, "1.edf")

    def test_symlink_in_cache_when_hard_link_fails(self):
        _write_recording(self.root / "1", "1", [0])
        home = self.root.parent / "home"
        made = []

        def fake_symlink(link, target):
            made.append((link, Path(target)))
            Path(link).write_bytes(b"")

        with (
            patch("pyhealth.datasets.isruc.os.link", side_effect=PermissionError),
            patch("pyhealth.datasets.metadata_csv.Path.home", return_value=home),
            patch(
                "pyhealth.datasets.isruc.Path.symlink_to",
                lambda link, target: fake_symlink(link, target),
            ),
        ):
            row = _prepare(self.root).iloc[0]

        cached = Path(row["signal_file"])
        self.assertEqual(cached.name, "1_1.edf")
        self.assertTrue(str(cached).startswith(str(home)))
        self.assertEqual(made, [(cached, (self.root / "1" / "1.rec").resolve())])
        self.assertFalse((self.root / "1" / "1.edf").exists())

    def test_recording_is_skipped_when_no_link_can_be_made(self):
        _write_recording(self.root / "1", "1", [0])
        _write_recording(self.root / "2", "2", [0], suffix=".edf")
        with (
            patch("pyhealth.datasets.isruc.os.link", side_effect=PermissionError),
            patch("pyhealth.datasets.isruc.Path.symlink_to", side_effect=OSError),
            patch(
                "pyhealth.datasets.metadata_csv.Path.home",
                return_value=self.root.parent / "home",
            ),
            self.assertLogs("pyhealth.datasets.isruc", "WARNING") as logs,
        ):
            df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), ["2"])
        self.assertIn("rename it to 1.edf", logs.output[0])

    def test_empty_root_writes_nothing(self):
        ds = ISRUCDataset.__new__(ISRUCDataset)
        ds.root = str(self.root)
        ds.prepare_metadata()
        self.assertFalse((self.root / "isruc-pyhealth.csv").exists())

    def test_existing_csv_is_left_untouched(self):
        _write_recording(self.root / "1", "1", [0])
        _prepare(self.root)
        csv_path = self.root / "isruc-pyhealth.csv"
        first = csv_path.read_bytes()
        _write_recording(self.root / "2", "2", [0])

        _prepare(self.root)

        self.assertEqual(csv_path.read_bytes(), first)

    def test_default_task(self):
        ds = ISRUCDataset.__new__(ISRUCDataset)
        self.assertIsInstance(ds.default_task, SleepStagingISRUC)


class TestHelpers(unittest.TestCase):
    def test_base_channel_name(self):
        self.assertEqual(
            [base_channel_name(c) for c in ("F3-A2", "c4-m1", "O2-M2 ", "X1")],
            ["F3", "c4", "O2", "X1"],
        )

    def test_read_isruc_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1_1.txt"
            path.write_text("0\n5\n\n2\n")
            self.assertEqual(read_isruc_scores(str(path)).tolist(), [0, 5, 2])
            path.write_text("3\n")
            self.assertEqual(read_isruc_scores(str(path)).tolist(), [3])

    def test_sleep_stages(self):
        self.assertEqual(SLEEP_STAGES, ("W", "N1", "N2", "N3", "REM"))


class TestSleepStagingISRUC(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name) / "1"

    def test_epochs_channels_and_labels(self):
        _write_recording(self.folder, "1", [0, 1, 2, 3, 5], suffix=".edf")

        samples = SleepStagingISRUC()(_patient("1", [_event(self.folder, "1")]))

        self.assertEqual([s["label"] for s in samples], [0, 1, 2, 3, 4])
        self.assertEqual([s["epoch_index"] for s in samples], [0, 1, 2, 3, 4])
        self.assertEqual(samples[0]["patient_id"], "1")
        self.assertEqual(samples[0]["session_id"], "1")
        signal = samples[0]["signal"]
        self.assertEqual(signal.shape, (6, 30 * SFREQ))
        self.assertEqual(signal.dtype, np.float32)
        # Rows follow F3, F4, C3, C4, O1, O2. The file order is
        # F3, C3, O1, F4, C4, O2 at indices 1 to 6.
        np.testing.assert_array_equal(signal[:, 0], [100, 400, 200, 500, 300, 600])
        # The second epoch covers seconds 30 to 59.
        np.testing.assert_array_equal(
            samples[1]["signal"][0, ::SFREQ], 100 + np.arange(30, 60)
        )

    def test_shorter_epochs_repeat_the_score(self):
        _write_recording(self.folder, "1", [0, 5], suffix=".edf")
        task = SleepStagingISRUC(epoch_seconds=10, channels=("C4", "LOC", "X1"))

        samples = task(_patient("1", [_event(self.folder, "1")]))

        self.assertEqual([s["label"] for s in samples], [0, 0, 0, 4, 4, 4])
        self.assertEqual(samples[4]["epoch_index"], 4)
        self.assertEqual(samples[4]["signal"].shape, (3, 10 * SFREQ))
        np.testing.assert_array_equal(samples[4]["signal"][:, 0], [540, 40, 740])

    def test_second_annotator(self):
        _write_recording(self.folder, "1", [0, 0], suffix=".edf")
        (self.folder / "1_2.txt").write_text("2\n3\n")

        samples = SleepStagingISRUC(annotator=2)(
            _patient("1", [_event(self.folder, "1")])
        )

        self.assertEqual([s["label"] for s in samples], [2, 3])

    def test_scores_past_the_signal_are_dropped(self):
        _write_recording(self.folder, "1", [0, 1, 2, 3], seconds=75, suffix=".edf")

        with self.assertLogs("pyhealth.tasks.sleep_staging_isruc", "WARNING") as logs:
            samples = SleepStagingISRUC()(_patient("1", [_event(self.folder, "1")]))

        self.assertEqual([s["label"] for s in samples], [0, 1])
        self.assertIn("4 scored epochs but 2 full epochs", logs.output[0])

    def test_unknown_scores_are_dropped(self):
        _write_recording(self.folder, "1", [0, 4, 7, 2], suffix=".edf")

        with self.assertLogs("pyhealth.tasks.sleep_staging_isruc", "WARNING") as logs:
            samples = SleepStagingISRUC()(_patient("1", [_event(self.folder, "1")]))

        self.assertEqual([s["epoch_index"] for s in samples], [0, 3])
        self.assertIn("Dropped 2 epochs", logs.output[0])

    def test_missing_channel_skips_the_recording(self):
        channels = [c for c in CHANNELS if not c.startswith("O2")]
        _write_recording(self.folder, "1", [0], suffix=".edf", channels=channels)

        with self.assertLogs("pyhealth.tasks.sleep_staging_isruc", "WARNING") as logs:
            samples = SleepStagingISRUC()(_patient("1", [_event(self.folder, "1")]))

        self.assertEqual(samples, [])
        self.assertIn("['O2']", logs.output[0])

    def test_missing_scoring_skips_the_recording(self):
        _write_recording(self.folder, "1", [0], suffix=".edf", scorings=(1,))
        event = _event(self.folder, "1", label_file_2="")

        with self.assertLogs("pyhealth.tasks.sleep_staging_isruc", "WARNING"):
            samples = SleepStagingISRUC(annotator=2)(_patient("1", [event]))

        self.assertEqual(samples, [])

    def test_every_recording_of_a_subject(self):
        for session in ("1", "2"):
            _write_recording(self.folder / session, session, [0, 1], suffix=".edf")
        events = [_event(self.folder / s, s, session_id=s) for s in ("1", "2")]

        samples = SleepStagingISRUC()(_patient("1", events))

        self.assertEqual(
            [(s["session_id"], s["epoch_index"]) for s in samples],
            [("1", 0), ("1", 1), ("2", 0), ("2", 1)],
        )

    def test_invalid_arguments(self):
        for kwargs in (
            {"epoch_seconds": 7},
            {"epoch_seconds": 0},
            {"annotator": 3},
            {"channels": ()},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SleepStagingISRUC(**kwargs)


class TestISRUCSetTask(unittest.TestCase):
    def test_set_task_from_rec_files(self):
        # litdata can keep the cache open after the test, which blocks the
        # cleanup on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp) / "ISRUC-Sleep-III"
            _write_recording(root / "1", "1", [0, 2, 5])
            _write_recording(root / "2", "2", [3, 1])

            dataset = ISRUCDataset(root=str(root), cache_dir=str(Path(tmp) / "cache"))
            samples = dataset.set_task()

            self.assertEqual(len(samples), 5)
            by_patient = {}
            for i in range(len(samples)):
                sample = samples[i]
                by_patient.setdefault(sample["patient_id"], []).append(sample)
                self.assertEqual(tuple(sample["signal"].shape), (6, 30 * SFREQ))
            self.assertEqual(sorted(by_patient), ["1", "2"])
            self.assertEqual(len(by_patient["1"]), 3)


if __name__ == "__main__":
    unittest.main()
