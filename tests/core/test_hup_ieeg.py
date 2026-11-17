"""HUPiEEGDataset and SOZLocalizationHUP, using small EDF files written here."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pyhealth.datasets import HUPiEEGDataset
from pyhealth.tasks import SOZLocalizationHUP
from pyhealth.tasks.soz_localization_hup import (
    read_hup_channels,
    read_hup_electrodes,
)

SFREQ = 200
SECONDS = 6
# name, status, status_description: four good contacts, a bad contact marked
# as seizure onset, and an ECG channel.
CHANNELS = (
    ("LA1", "good", "soz"),
    ("LA2", "good", "resect,soz"),
    ("LA3", "good", "resect"),
    ("LB1", "good", "n/a"),
    ("LB2", "bad", "soz"),
    ("EKG1", "bad", "n/a"),
)
PARTICIPANTS = (
    # id, age, sex, hand, outcome, engel, therapy, implant, target, lesion,
    # age at onset
    ("sub-HUP060", "42", "F", "R", "F", "3A", "ABLATION", "SEEG", "FRONTAL")
    + ("n/a", "12"),
    ("sub-HUP064", "21", "M", "L", "S", "1D", "RESECTION", "ECOG", "FRONTAL")
    + ("LESIONAL", "3"),
    # The real file writes this id with a trailing space.
    ("sub-HUP151 ", "33", "M", "R", "F", "2A", "ABLATION", "SEEG", "MFL")
    + ("NON-LESIONAL", "6"),
)
PARTICIPANT_COLUMNS = (
    "participant_id", "age", "sex", "hand", "outcome", "engel", "therapy",
    "implant", "target", "lesion_status", "age_onset",
)  # fmt: skip
# The task settings that leave the signal of the file unchanged.
RAW = {
    "window_seconds": 2,
    "stride_seconds": 2,
    "resample_rate": SFREQ,
    "bandpass": None,
    "notch": None,
    "reference": "stored",
}


def _write_edf(path: Path, names: list[str], data: np.ndarray, sfreq: int) -> None:
    """Write ``data`` (microvolts, one row per channel) as a minimal EDF file.

    Each data record is one second. Physical and digital ranges match, so
    a stored integer is the value in microvolts.
    """
    n_channels = len(names)
    n_records = data.shape[1] // sfreq

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
        (16, names),
        (80, [""] * n_channels),
        (8, ["uV"] * n_channels),
        (8, ["-32768"] * n_channels),
        (8, ["32767"] * n_channels),
        (8, ["-32768"] * n_channels),
        (8, ["32767"] * n_channels),
        (80, [""] * n_channels),
        (8, [sfreq] * n_channels),
        (32, [""] * n_channels),
    ]:
        header += "".join(field(v, width) for v in values)
    records = data[:, : n_records * sfreq].astype("<i2")
    records = records.reshape(n_channels, n_records, sfreq).transpose(1, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + records.tobytes())


def _signal(n_channels: int, seconds: int = SECONDS, sfreq: int = SFREQ) -> np.ndarray:
    """Channel ``i`` holds ``100 * i`` plus the second the sample is in."""
    second = np.arange(seconds * sfreq) // sfreq
    return np.stack([100 * i + second for i in range(n_channels)])


def _write_channels(path: Path, channels=CHANNELS) -> None:
    """Write a ``channels.tsv`` like the real ones, with a byte order mark."""
    header = (
        "name\ttype\tunits\tlow_cutoff\thigh_cutoff\tdescription"
        "\tsampling_frequency\tstatus\tstatus_description"
    )
    lines = [header]
    lines += [
        f"{name}\tSEEG\tuV\t0.0\t100.0\tStereoEEG\t{SFREQ}.0\t{status}\t{marks}"
        for name, status, marks in channels
    ]
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _stem(subject: str, kind: str, run: str, acquisition: str = "seeg") -> str:
    return f"{subject}_ses-presurgery_task-{kind}_acq-{acquisition}_run-{run}"


def _electrodes_path(root: Path, subject: str, acquisition: str = "seeg") -> Path:
    name = f"{subject}_ses-presurgery_acq-{acquisition}_space-fsaverage_electrodes.tsv"
    return root / subject / "ses-presurgery" / "ieeg" / name


def _position(index: int) -> tuple[float, float, float]:
    """The position written for the channel at ``index`` of ``channels.tsv``."""
    return (-10.5 - index, 20.0 + index, 3.25 * index)


def _write_electrodes(path: Path, channels=CHANNELS) -> None:
    """Write an ``electrodes.tsv`` like the real ones.

    Channel ``i`` is at ``_position(i)``; ECG channels have no position, as
    in the real files.
    """
    lines = ["name\tx\ty\tz\tsize"]
    for index, (name, _, _) in enumerate(channels):
        if name.strip().startswith("EKG"):
            lines.append(f"{name}\tn/a\tn/a\tn/a\tn/a")
        else:
            x, y, z = _position(index)
            lines.append(f"{name}\t{x}\t{y}\t{z}\tn/a")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _write_recording(
    root: Path,
    subject: str,
    kind: str = "interictal",
    run: str = "01",
    acquisition: str = "seeg",
    channels=CHANNELS,
    seconds: int = SECONDS,
    sfreq: int = SFREQ,
    edf_names: list[str] | None = None,
) -> Path:
    """Write one recording with its sidecar files; return the EDF path."""
    folder = root / subject / "ses-presurgery" / "ieeg"
    stem = _stem(subject, kind, run, acquisition)
    names = edf_names or [name for name, _, _ in channels]
    edf = folder / f"{stem}_ieeg.edf"
    _write_edf(edf, names, _signal(len(names), seconds, sfreq), sfreq)
    _write_channels(folder / f"{stem}_channels.tsv", channels)
    _write_electrodes(_electrodes_path(root, subject, acquisition), channels)
    if kind == "ictal":
        (folder / f"{stem}_events.tsv").write_text(
            "onset\tduration\ttrial_type\tvalue\tsample\n"
            "2.0\t0.0\tsz onset\t2\t400\n"
            "4.5\t0.0\tsz offset\t1\t900",
            encoding="utf-8-sig",
        )
    return edf


def _write_participants(root: Path, participants=PARTICIPANTS) -> None:
    lines = ["\t".join(PARTICIPANT_COLUMNS)]
    lines += ["\t".join(row) for row in participants]
    root.mkdir(parents=True, exist_ok=True)
    (root / "participants.tsv").write_text("\n".join(lines) + "\n")


def _event(edf: Path, **overrides) -> SimpleNamespace:
    """A ``recordings`` event like the rows of ``hup_ieeg-pyhealth.csv``."""
    fields = {
        "signal_file": str(edf),
        "channels_file": str(edf).replace("_ieeg.edf", "_channels.tsv"),
        "electrodes_file": str(next(edf.parent.glob("*_electrodes.tsv"), "")),
        "recording_type": "interictal",
        "run": "01",
        "outcome": "S",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _patient(patient_id: str, events: list) -> SimpleNamespace:
    return SimpleNamespace(
        patient_id=patient_id, get_events=lambda event_type=None: events
    )


def _prepare(root: Path) -> pd.DataFrame:
    dataset = HUPiEEGDataset.__new__(HUPiEEGDataset)
    dataset.root = str(root)
    dataset.prepare_metadata()
    return pd.read_csv(root / "hup_ieeg-pyhealth.csv", dtype=str, keep_default_na=False)


class _TempRoot(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "ds004100"
        self.root.mkdir()


class TestHUPMetadata(_TempRoot):
    def test_one_row_per_recording(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060", "ictal", "02")
        _write_recording(self.root, "sub-HUP060", "interictal", "01")
        _write_recording(self.root, "sub-HUP064", "interictal", "02", "ecog")
        table = _prepare(self.root)

        self.assertEqual(
            list(table["patient_id"]), ["sub-HUP060", "sub-HUP060", "sub-HUP064"]
        )
        self.assertEqual(
            list(table["recording_type"]), ["ictal", "interictal", "interictal"]
        )
        self.assertEqual(list(table["run"]), ["02", "01", "02"])
        self.assertEqual(list(table["acquisition"]), ["seeg", "seeg", "ecog"])
        self.assertEqual(list(table["seizure_onset_second"]), ["2.0", "", ""])
        self.assertEqual(list(table["seizure_offset_second"]), ["4.5", "", ""])
        for column in ("signal_file", "channels_file"):
            self.assertTrue(all(Path(path).is_file() for path in table[column]))
        self.assertTrue(table["signal_file"][0].endswith("run-02_ieeg.edf"))
        self.assertTrue(table["channels_file"][0].endswith("run-02_channels.tsv"))
        first = table.iloc[0]
        expected = dict(zip(PARTICIPANT_COLUMNS[1:], PARTICIPANTS[0][1:]))
        self.assertEqual({c: first[c] for c in expected}, expected)
        self.assertEqual(table["outcome"][2], "S")
        self.assertEqual(table["implant"][2], "ECOG")

    def test_participant_id_with_trailing_space(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP151")
        table = _prepare(self.root)
        self.assertEqual(table["patient_id"][0], "sub-HUP151")
        self.assertEqual(table["engel"][0], "2A")

    def test_unlisted_patient_has_empty_columns(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP999")
        with self.assertLogs("pyhealth.datasets.hup_ieeg", "WARNING") as logs:
            table = _prepare(self.root)
        self.assertIn("sub-HUP999", logs.output[0])
        self.assertEqual(table["patient_id"][0], "sub-HUP999")
        self.assertEqual(table["outcome"][0], "")

    def test_listed_patient_without_recording_has_no_row(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060")
        table = _prepare(self.root)
        self.assertEqual(list(table["patient_id"]), ["sub-HUP060"])

    def test_missing_participants_file(self):
        _write_recording(self.root, "sub-HUP060")
        with self.assertLogs("pyhealth.datasets.hup_ieeg", "WARNING"):
            table = _prepare(self.root)
        self.assertEqual(len(table), 1)
        self.assertEqual(table["outcome"][0], "")

    def test_participants_without_id_column(self):
        (self.root / "participants.tsv").write_text("subject\tage\nsub-HUP060\t42\n")
        _write_recording(self.root, "sub-HUP060")
        with self.assertRaisesRegex(ValueError, "participant_id"):
            _prepare(self.root)

    def test_missing_channels_file_is_left_empty(self):
        _write_participants(self.root)
        edf = _write_recording(self.root, "sub-HUP060")
        Path(str(edf).replace("_ieeg.edf", "_channels.tsv")).unlink()
        table = _prepare(self.root)
        self.assertEqual(table["channels_file"][0], "")

    def test_electrodes_file_of_the_implant(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060", "ictal", "02")
        _write_recording(self.root, "sub-HUP060", "interictal", "01")
        _write_recording(self.root, "sub-HUP064", acquisition="ecog")
        table = _prepare(self.root)
        self.assertEqual(
            [Path(path).name for path in table["electrodes_file"]],
            [
                "sub-HUP060_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv",
                "sub-HUP060_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv",
                "sub-HUP064_ses-presurgery_acq-ecog_space-fsaverage_electrodes.tsv",
            ],
        )
        self.assertTrue(all(Path(path).is_file() for path in table["electrodes_file"]))

    def test_electrodes_file_matches_the_implant_of_the_recording(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060", run="01", acquisition="seeg")
        _write_recording(self.root, "sub-HUP060", run="02", acquisition="ecog")
        folder = self.root / "sub-HUP060" / "ses-presurgery" / "ieeg"
        other = "sub-HUP060_ses-presurgery_acq-seeg_run-01_notelectrodes.tsv"
        (folder / other).write_text("name\n")
        table = _prepare(self.root).sort_values("acquisition")
        self.assertEqual(
            [Path(path).name for path in table["electrodes_file"]],
            [
                "sub-HUP060_ses-presurgery_acq-ecog_space-fsaverage_electrodes.tsv",
                "sub-HUP060_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv",
            ],
        )

    def test_electrodes_file_without_acquisition_is_the_fallback(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060")
        path = _electrodes_path(self.root, "sub-HUP060")
        shared = path.with_name("sub-HUP060_ses-presurgery_electrodes.tsv")
        shared.write_bytes(path.read_bytes())
        self.assertEqual(Path(_prepare(self.root)["electrodes_file"][0]), path)
        (self.root / "hup_ieeg-pyhealth.csv").unlink()
        path.unlink()
        self.assertEqual(Path(_prepare(self.root)["electrodes_file"][0]), shared)

    def test_several_electrodes_files_warn(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060")
        path = _electrodes_path(self.root, "sub-HUP060")
        first = path.with_name(path.name.replace("fsaverage", "MNI152NLin6Asym"))
        first.write_bytes(path.read_bytes())
        with self.assertLogs("pyhealth.datasets.hup_ieeg", "WARNING") as logs:
            table = _prepare(self.root)
        self.assertIn("several electrodes files", logs.output[0])
        self.assertEqual(Path(table["electrodes_file"][0]), first)

    def test_missing_electrodes_file_is_left_empty(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060")
        _electrodes_path(self.root, "sub-HUP060").unlink()
        table = _prepare(self.root)
        self.assertEqual(table["electrodes_file"][0], "")

    def test_unexpected_events_leave_the_seizure_times_empty(self):
        _write_participants(self.root)
        edf = _write_recording(self.root, "sub-HUP060", "ictal")
        events = Path(str(edf).replace("_ieeg.edf", "_events.tsv"))
        events.write_text(
            "onset\tduration\ttrial_type\n1.0\t0.0\tsz onset\n2.0\t0.0\tsz onset\n"
        )
        with self.assertLogs("pyhealth.datasets.hup_ieeg", "WARNING"):
            table = _prepare(self.root)
        self.assertEqual(table["seizure_onset_second"][0], "")
        self.assertEqual(table["seizure_offset_second"][0], "")

    def test_ictal_recording_without_events_file(self):
        _write_participants(self.root)
        edf = _write_recording(self.root, "sub-HUP060", "ictal")
        Path(str(edf).replace("_ieeg.edf", "_events.tsv")).unlink()
        table = _prepare(self.root)
        self.assertEqual(table["seizure_onset_second"][0], "")

    def test_file_with_another_name_is_skipped(self):
        _write_participants(self.root)
        edf = _write_recording(self.root, "sub-HUP060")
        edf.with_name("sub-HUP060_notes_ieeg.edf").write_bytes(edf.read_bytes())
        with self.assertLogs("pyhealth.datasets.hup_ieeg", "WARNING"):
            table = _prepare(self.root)
        self.assertEqual(len(table), 1)

    def test_empty_root_writes_nothing(self):
        dataset = HUPiEEGDataset.__new__(HUPiEEGDataset)
        dataset.root = str(self.root)
        with self.assertLogs("pyhealth.datasets.hup_ieeg", "WARNING"):
            dataset.prepare_metadata()
        self.assertFalse((self.root / "hup_ieeg-pyhealth.csv").exists())

    def test_existing_csv_is_left_untouched(self):
        _write_participants(self.root)
        _write_recording(self.root, "sub-HUP060")
        csv = self.root / "hup_ieeg-pyhealth.csv"
        csv.write_text("patient_id\nkept\n")
        _prepare(self.root)
        self.assertEqual(csv.read_text(), "patient_id\nkept\n")

    def test_default_task(self):
        dataset = HUPiEEGDataset.__new__(HUPiEEGDataset)
        task = dataset.default_task
        self.assertIsInstance(task, SOZLocalizationHUP)
        self.assertEqual(task.target, "soz")


class TestReadChannels(_TempRoot):
    def test_marks_and_status(self):
        path = self.root / "channels.tsv"
        _write_channels(path)
        channels = read_hup_channels(path)
        self.assertEqual([c["name"] for c in channels], [c[0] for c in CHANNELS])
        self.assertEqual([c["good"] for c in channels], [True] * 4 + [False] * 2)
        self.assertEqual(
            [c["soz"] for c in channels], [True, True, False, False, True, False]
        )
        self.assertEqual(
            [c["resect"] for c in channels], [False, True, True, False, False, False]
        )
        self.assertEqual({c["type"] for c in channels}, {"SEEG"})

    def test_spaces_and_case(self):
        path = self.root / "channels.tsv"
        _write_channels(path, [(" LA1 ", "Good ", " Resect , SOZ")])
        channel = read_hup_channels(path)[0]
        self.assertEqual(channel["name"], "LA1")
        self.assertTrue(channel["good"] and channel["soz"] and channel["resect"])

    def test_missing_column(self):
        path = self.root / "channels.tsv"
        path.write_text("name\tstatus\nLA1\tgood\n")
        with self.assertRaisesRegex(ValueError, "status_description"):
            read_hup_channels(path)


class TestReadElectrodes(_TempRoot):
    def test_positions(self):
        path = self.root / "electrodes.tsv"
        _write_electrodes(path)
        positions = read_hup_electrodes(path)
        self.assertEqual(list(positions), [name for name, _, _ in CHANNELS])
        self.assertEqual(positions["LA1"], (-10.5, 20.0, 0.0))
        self.assertEqual(positions["LB1"], (-13.5, 23.0, 9.75))
        self.assertTrue(all(math.isnan(value) for value in positions["EKG1"]))

    def test_incomplete_or_invalid_position_is_nan(self):
        path = self.root / "electrodes.tsv"
        path.write_text(
            "name\tx\ty\tz\n"
            "LA1\t1.0\tn/a\t3.0\n"
            "LA2\tleft\t2.0\t3.0\n"
            "LA3\t\t\t\n"
            "LA4\tinf\t2.0\t3.0\n"
            " LA5 \t1.5\t2.5\t3.5\n"
            "LA5\t9.0\t9.0\t9.0\n"
        )
        positions = read_hup_electrodes(path)
        for name in ("LA1", "LA2", "LA3", "LA4"):
            self.assertTrue(all(math.isnan(value) for value in positions[name]), name)
        # The name is stripped, and a repeated name keeps its first row.
        self.assertEqual(positions["LA5"], (1.5, 2.5, 3.5))

    def test_missing_column(self):
        path = self.root / "electrodes.tsv"
        path.write_text("name\tx\ty\nLA1\t1\t2\n")
        with self.assertRaisesRegex(ValueError, "'z'"):
            read_hup_electrodes(path)


class TestSOZLocalizationHUP(_TempRoot):
    def setUp(self):
        super().setUp()
        self.edf = _write_recording(self.root, "sub-HUP060")
        self.patient = _patient("sub-HUP060", [_event(self.edf)])

    def test_one_sample_per_good_channel_and_window(self):
        samples = SOZLocalizationHUP(**RAW)(self.patient)

        # 4 good channels, 3 windows of 2 s in 6 s.
        self.assertEqual(len(samples), 12)
        self.assertEqual(
            [s["channel"] for s in samples],
            ["LA1"] * 3 + ["LA2"] * 3 + ["LA3"] * 3 + ["LB1"] * 3,
        )
        self.assertEqual([s["label"] for s in samples], [1] * 6 + [0] * 6)
        self.assertEqual([s["start_second"] for s in samples], [0.0, 2.0, 4.0] * 4)
        first, last = samples[0], samples[-1]
        self.assertEqual(first["patient_id"], "sub-HUP060")
        self.assertEqual(first["channel_type"], "SEEG")
        self.assertEqual(first["run"], "01")
        self.assertEqual(first["signal"].shape, (1, 2 * SFREQ))
        self.assertEqual(first["signal"].dtype, np.float32)
        # Channel i holds 100 * i plus the second: LA1 is channel 0, LB1 is 3.
        seconds = np.repeat([0, 1], SFREQ)
        np.testing.assert_allclose(first["signal"][0], seconds, atol=1e-3)
        np.testing.assert_allclose(last["signal"][0], 300 + 4 + seconds, atol=1e-3)

    def test_position_of_each_contact(self):
        samples = SOZLocalizationHUP(**RAW)(self.patient)
        for sample in samples:
            index = [name for name, _, _ in CHANNELS].index(sample["channel"])
            self.assertEqual((sample["x"], sample["y"], sample["z"]), _position(index))
        self.assertEqual(samples[-1]["x"], -13.5)

    def test_contact_without_position_is_nan(self):
        # A contact with n/a, one absent from electrodes.tsv, and no file.
        channels = [("LA1", "good", "soz"), ("EKG9", "good", "n/a")]
        edf = _write_recording(self.root, "sub-HUP064", channels=channels)
        path = _electrodes_path(self.root, "sub-HUP064")
        path.write_text("name\tx\ty\tz\nLA1\tn/a\tn/a\tn/a\n")
        for overrides in ({}, {"electrodes_file": ""}, {"electrodes_file": None}):
            patient = _patient("sub-HUP064", [_event(edf, **overrides)])
            samples = SOZLocalizationHUP(**RAW)(patient)
            self.assertEqual(len(samples), 6)
            for sample in samples:
                position = (sample["x"], sample["y"], sample["z"])
                self.assertTrue(all(math.isnan(value) for value in position))

    def test_event_without_electrodes_column(self):
        event = _event(self.edf)
        del event.electrodes_file
        samples = SOZLocalizationHUP(**RAW)(_patient("sub-HUP060", [event]))
        self.assertEqual(len(samples), 12)
        self.assertTrue(math.isnan(samples[0]["x"]))

    def test_electrodes_file_that_no_longer_exists(self):
        event = _event(self.edf, electrodes_file=str(self.root / "gone.tsv"))
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING") as logs:
            samples = SOZLocalizationHUP(**RAW)(_patient("sub-HUP060", [event]))
        self.assertIn("No electrodes file", logs.output[0])
        self.assertEqual(len(samples), 12)
        self.assertTrue(all(math.isnan(s["y"]) for s in samples))

    def test_resect_target(self):
        samples = SOZLocalizationHUP(target="resect", **RAW)(self.patient)
        self.assertEqual([s["label"] for s in samples], [0] * 3 + [1] * 6 + [0] * 3)

    def test_average_reference_uses_the_good_channels(self):
        options = {**RAW, "reference": "average"}
        samples = SOZLocalizationHUP(**options)(self.patient)
        # The good channels hold 0, 100, 200 and 300 plus the second: their
        # mean is 150 plus the second. With the bad ones it would be 250.
        np.testing.assert_allclose(samples[0]["signal"][0], -150, atol=1e-3)
        np.testing.assert_allclose(samples[-1]["signal"][0], 150, atol=1e-3)

    def test_overlapping_windows_and_dropped_end(self):
        options = {**RAW, "window_seconds": 4, "stride_seconds": 1.5}
        samples = SOZLocalizationHUP(**options)(self.patient)
        # Starts at 0 and 1.5 s; a window starting at 3 s would end at 7 s.
        self.assertEqual([s["start_second"] for s in samples[:2]], [0.0, 1.5])
        self.assertEqual(len(samples), 8)

    def test_outcome_filter(self):
        task = SOZLocalizationHUP(outcomes=("F",), **RAW)
        self.assertEqual(task(self.patient), [])
        for outcomes in (("S",), "s", ("F", "S")):
            task = SOZLocalizationHUP(outcomes=outcomes, **RAW)
            self.assertEqual(len(task(self.patient)), 12)

    def test_ictal_recordings_are_not_used(self):
        ictal = _write_recording(self.root, "sub-HUP060", "ictal", "02")
        patient = _patient(
            "sub-HUP060",
            [_event(ictal, recording_type="ictal", run="02"), _event(self.edf)],
        )
        samples = SOZLocalizationHUP(**RAW)(patient)
        self.assertEqual({s["run"] for s in samples}, {"01"})

    def test_every_interictal_recording_of_a_patient(self):
        second = _write_recording(self.root, "sub-HUP060", run="02")
        patient = _patient("sub-HUP060", [_event(self.edf), _event(second, run="02")])
        samples = SOZLocalizationHUP(**RAW)(patient)
        self.assertEqual([s["run"] for s in samples], ["01"] * 12 + ["02"] * 12)

    def test_recording_without_the_mark_is_skipped(self):
        channels = [(name, status, "n/a") for name, status, _ in CHANNELS]
        edf = _write_recording(self.root, "sub-HUP064", channels=channels)
        patient = _patient("sub-HUP064", [_event(edf)])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            self.assertEqual(SOZLocalizationHUP(**RAW)(patient), [])

    def test_mark_on_a_bad_channel_does_not_count(self):
        channels = [("LA1", "good", "n/a"), ("LA2", "bad", "soz")]
        edf = _write_recording(self.root, "sub-HUP064", channels=channels)
        patient = _patient("sub-HUP064", [_event(edf)])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            self.assertEqual(SOZLocalizationHUP(**RAW)(patient), [])

    def test_channel_absent_from_the_edf_is_left_out(self):
        edf = _write_recording(
            self.root, "sub-HUP064", edf_names=["LA1", "LA2", "LB1", "EKG1"]
        )
        patient = _patient("sub-HUP064", [_event(edf)])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING") as logs:
            samples = SOZLocalizationHUP(**RAW)(patient)
        self.assertIn("LA3", logs.output[0])
        self.assertEqual(
            [s["channel"] for s in samples], ["LA1"] * 3 + ["LA2"] * 3 + ["LB1"] * 3
        )
        # LB1 is the third channel of this file.
        np.testing.assert_allclose(samples[-1]["signal"][0, 0], 204, atol=1e-3)

    def test_recording_shorter_than_a_window_is_skipped(self):
        task = SOZLocalizationHUP(**{**RAW, "window_seconds": 7})
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            self.assertEqual(task(self.patient), [])

    def test_missing_file_columns_skip_the_recording(self):
        for overrides in ({"signal_file": ""}, {"channels_file": None}):
            patient = _patient("sub-HUP060", [_event(self.edf, **overrides)])
            with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
                self.assertEqual(SOZLocalizationHUP(**RAW)(patient), [])

    def test_mark_differing_between_recordings(self):
        # The second recording marks LA1 only, as one file of sub-HUP180.
        stale = [
            ("LA1", "good", "soz"),
            ("LA2", "good", "resect"),
            ("LA3", "good", "resect"),
            ("LB1", "good", "n/a"),
        ]
        second = _write_recording(self.root, "sub-HUP060", run="02", channels=stale)
        patient = _patient("sub-HUP060", [_event(self.edf), _event(second, run="02")])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING") as logs:
            samples = SOZLocalizationHUP(**RAW)(patient)
        self.assertIn("LA2", logs.output[0])
        self.assertNotIn("LA1", logs.output[0])
        labels = {(s["run"], s["channel"]): s["label"] for s in samples}
        for run in ("01", "02"):
            self.assertEqual(
                [labels[(run, name)] for name in ("LA1", "LA2", "LA3", "LB1")],
                [1, 1, 0, 0],
            )

    def test_mark_in_one_recording_labels_the_patient(self):
        unmarked = [(name, status, "n/a") for name, status, _ in CHANNELS]
        second = _write_recording(self.root, "sub-HUP060", run="02", channels=unmarked)
        patient = _patient("sub-HUP060", [_event(second, run="02"), _event(self.edf)])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            samples = SOZLocalizationHUP(**RAW)(patient)
        self.assertEqual(sum(s["label"] for s in samples), 12)

    def test_average_reference_needs_two_good_channels(self):
        channels = [("LA1", "good", "soz"), ("LA2", "bad", "n/a")]
        edf = _write_recording(self.root, "sub-HUP064", channels=channels)
        patient = _patient("sub-HUP064", [_event(edf)])
        task = SOZLocalizationHUP(**{**RAW, "reference": "average"})
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            self.assertEqual(task(patient), [])
        self.assertEqual(len(SOZLocalizationHUP(**RAW)(patient)), 3)

    def test_repeated_channel_name_is_used_once(self):
        channels = [("LA1", "good", "soz"), ("LA1", "good", "n/a"), *CHANNELS[1:]]
        edf = _write_recording(
            self.root,
            "sub-HUP064",
            channels=channels,
            edf_names=[name for name, _, _ in CHANNELS],
        )
        patient = _patient("sub-HUP064", [_event(edf)])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            samples = SOZLocalizationHUP(**RAW)(patient)
        self.assertEqual(len(samples), 12)
        self.assertEqual([s["label"] for s in samples[:3]], [1, 1, 1])

    def test_short_recording_is_skipped_before_filtering(self):
        edf = _write_recording(self.root, "sub-HUP064", seconds=1, sfreq=256)
        patient = _patient("sub-HUP064", [_event(edf)])
        with self.assertLogs("pyhealth.tasks.soz_localization_hup", "WARNING"):
            self.assertEqual(SOZLocalizationHUP()(patient), [])

    def test_default_chain_keeps_a_rhythm_and_removes_the_shared_signal(self):
        # 10 Hz on LA1 only, plus 60 Hz and an offset shared by all channels.
        sfreq, seconds = 256, 62
        times = np.arange(seconds * sfreq) / sfreq
        shared = 400 * np.sin(2 * np.pi * 60 * times) + 1000
        data = np.stack([shared] * 6)
        data[0] += 2000 * np.sin(2 * np.pi * 10 * times)
        folder = self.root / "sub-HUP064" / "ses-presurgery" / "ieeg"
        stem = _stem("sub-HUP064", "interictal", "01")
        edf = folder / f"{stem}_ieeg.edf"
        _write_edf(edf, [name for name, _, _ in CHANNELS], data, sfreq)
        _write_channels(folder / f"{stem}_channels.tsv")
        samples = SOZLocalizationHUP()(_patient("sub-HUP064", [_event(edf)]))

        self.assertEqual(len(samples), 8)
        # LA1 keeps 3/4 of the rhythm after the average over 4 good channels,
        # sampled at 200 Hz; the others hold the remaining quarter, inverted.
        expected = 1500 * np.sin(2 * np.pi * 10 * np.arange(6000) / 200)
        middle = slice(1000, 5000)
        np.testing.assert_allclose(
            samples[0]["signal"][0, middle], expected[middle], atol=15
        )
        np.testing.assert_allclose(
            samples[2]["signal"][0, middle], -expected[middle] / 3, atol=15
        )

    def test_window_shorter_than_one_sample(self):
        task = SOZLocalizationHUP(**{**RAW, "window_seconds": 0.001})
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            task(self.patient)

    def test_default_settings(self):
        edf = _write_recording(self.root, "sub-HUP064", seconds=65, sfreq=256)
        samples = SOZLocalizationHUP()(_patient("sub-HUP064", [_event(edf)]))
        # 65 s hold two windows of 30 s, for each of the 4 good channels.
        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0]["signal"].shape, (1, 6000))
        self.assertTrue(all(np.isfinite(s["signal"]).all() for s in samples))

    def test_window_below_one_sample(self):
        task = SOZLocalizationHUP(**{**RAW, "stride_seconds": 0.001})
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            task(self.patient)

    def test_invalid_arguments(self):
        for options in (
            {"target": "spikes"},
            {"reference": "bipolar"},
            {"window_seconds": 0},
            {"stride_seconds": -1},
            {"resample_rate": float("nan")},
            {"notch": 0},
            {"bandpass": (80, 0.5)},
            {"bandpass": (0, 80)},
            {"bandpass": (0.5,)},
            {"outcomes": ()},
            {"outcomes": ("X",)},
            {"outcomes": 1},
            {"window_seconds": "30"},
            {"window_seconds": True},
            {"bandpass": 5},
            {"bandpass": ("a", "b")},
        ):
            with self.subTest(options), self.assertRaises(ValueError):
                SOZLocalizationHUP(**options)


class TestPreprocess(unittest.TestCase):
    """Filters and resampling, on signals with a known content."""

    def setUp(self):
        self.times = np.arange(10 * SFREQ) / SFREQ
        self.middle = slice(2 * SFREQ, 8 * SFREQ)

    def _task(self, **options) -> SOZLocalizationHUP:
        return SOZLocalizationHUP(**{**RAW, **options})

    def test_notch_removes_the_line_frequency_only(self):
        line = np.sin(2 * np.pi * 60 * self.times)
        alpha = np.sin(2 * np.pi * 10 * self.times)
        out = self._task(notch=60).preprocess(np.stack([line, alpha]), SFREQ)
        self.assertLess(np.abs(out[0, self.middle]).max(), 0.01)
        np.testing.assert_allclose(out[1, self.middle], alpha[self.middle], atol=0.01)

    def test_bandpass_removes_the_offset_and_fast_activity(self):
        alpha = np.sin(2 * np.pi * 10 * self.times)
        fast = np.sin(2 * np.pi * 95 * self.times)
        data = np.stack([alpha + 500, fast])
        out = self._task(bandpass=(0.5, 80)).preprocess(data, SFREQ)
        # First order: 10 Hz passes almost unchanged, 95 Hz is attenuated.
        np.testing.assert_allclose(out[0, self.middle], alpha[self.middle], atol=0.02)
        self.assertLess(np.abs(out[1, self.middle]).max(), 0.2)

    def test_resampling_keeps_the_signal(self):
        alpha = 50 * np.sin(2 * np.pi * 10 * self.times) + 300
        out = self._task(resample_rate=100).preprocess(alpha[None], SFREQ)
        self.assertEqual(out.shape, (1, 1000))
        expected = 50 * np.sin(2 * np.pi * 10 * np.arange(1000) / 100) + 300
        np.testing.assert_allclose(out[0, 100:-100], expected[100:-100], atol=0.5)
        # No drop towards zero at the two ends.
        self.assertGreater(out[0, :5].min(), 240)
        self.assertGreater(out[0, -5:].min(), 240)

    def test_filter_edges_must_be_below_half_the_rate(self):
        data = np.zeros((1, 400))
        with self.assertRaisesRegex(ValueError, "notch"):
            self._task(notch=60).preprocess(data, 100)
        with self.assertRaisesRegex(ValueError, "bandpass"):
            self._task(bandpass=(0.5, 80)).preprocess(data, 160)

    def test_unreachable_rate(self):
        for rate in (np.pi, 100000):
            with self.assertRaisesRegex(ValueError, "cannot be reached"):
                self._task(resample_rate=rate).preprocess(np.zeros((1, 400)), 256.0)


class TestHUPSetTask(unittest.TestCase):
    def test_set_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds004100"
            _write_participants(root)
            for subject in ("sub-HUP060", "sub-HUP064", "sub-HUP151"):
                _write_recording(root, subject)
                _write_recording(root, subject, "ictal", "02")
            _electrodes_path(root, "sub-HUP151").unlink()
            dataset = HUPiEEGDataset(root=str(root), cache_dir=str(Path(tmp) / "cache"))
            samples = dataset.set_task(SOZLocalizationHUP(**RAW), num_workers=1)
            # 3 patients, 4 good channels, 3 windows; ictal recordings unused.
            self.assertEqual(len(samples), 36)
            self.assertEqual(tuple(samples[0]["signal"].shape), (1, 2 * SFREQ))
            self.assertEqual(
                {sample["patient_id"] for sample in samples},
                {"sub-HUP060", "sub-HUP064", "sub-HUP151"},
            )
            self.assertEqual({sample["run"] for sample in samples}, {"01"})
            self.assertEqual(
                {sample["channel"] for sample in samples},
                {"LA1", "LA2", "LA3", "LB1"},
            )
            positions = {
                (s["patient_id"], s["channel"]): (s["x"], s["y"], s["z"])
                for s in samples
            }
            self.assertEqual(len(positions), 12)
            for (patient_id, channel), position in positions.items():
                self.assertTrue(all(type(value) is float for value in position))
                if patient_id == "sub-HUP151":  # its electrodes.tsv is missing
                    self.assertTrue(all(math.isnan(value) for value in position))
                else:
                    index = ["LA1", "LA2", "LA3", "LB1"].index(channel)
                    self.assertEqual(position, _position(index))
            labels = sorted(int(sample["label"]) for sample in samples)
            self.assertEqual(labels, [0] * 18 + [1] * 18)
            samples.close()


if __name__ == "__main__":
    unittest.main()
