"""HelsinkiNeonatalDataset and SeizureDetectionHelsinki, on small EDF files."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import sosfreqz

from pyhealth.datasets import HelsinkiNeonatalDataset
from pyhealth.tasks import SeizureDetectionHelsinki
from pyhealth.tasks.seizure_detection_helsinki import (
    HELSINKI_BIPOLAR_MONTAGE,
    LABEL_MODES,
    bandpass_sections,
    combine_annotations,
    electrode_name,
    read_helsinki_annotations,
)

SFREQ = 16
ELECTRODES = (
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz",
    "T3", "T5", "T4", "T6", "P3", "P4", "Pz", "O1", "O2",
)  # fmt: skip
# The two label styles found in the dataset, plus the non-EEG channels.
CHANNELS_REF = [f"EEG {e}-Ref" for e in ELECTRODES] + ["ECG EKG", "Resp Effort"]
CHANNELS_UPPER = [f"EEG {e}-REF" for e in reversed(ELECTRODES)] + [
    "ECG EKG-REF",
    "Resp Effort-REF",
]
CLINICAL_HEADER = (
    "ID,EEG file,Gender,BW (g),GA (weeks),EEG to PMA (weeks),Diagnosis,"
    "Neuroimaging Findings,PNA of Imaging (days),"
    "Number of Reviewers Annotating Seizure,Primary Localisation,Other\n"
)


def _write_edf(
    path: Path, channels: list[str], data: np.ndarray, sfreq: int = SFREQ
) -> None:
    """Write ``data`` (microvolts, one row per channel) as a minimal EDF file.

    Each data record is one second. Physical and digital ranges match, so
    a stored integer is the value in microvolts.
    """
    n_channels = len(channels)
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
        (16, channels),
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


def _electrode_value(electrode: str) -> int:
    """A distinct constant per electrode, so each difference is known."""
    return 100 * (ELECTRODES.index(electrode) + 1)


def _signal(seconds: int, channels: list[str]) -> np.ndarray:
    """An EEG channel holds its electrode's constant plus the second."""
    second = np.arange(seconds * SFREQ) // SFREQ
    rows = []
    for name in channels:
        electrode = electrode_name(name)
        base = _electrode_value(electrode) if electrode in ELECTRODES else 5000
        rows.append(base + second)
    return np.stack(rows)


def _write_annotations(root: Path, columns: dict[str, dict]) -> None:
    """Write the three annotation CSVs: ``columns[expert][neonate]``."""
    names = {"A": "annotations_2017_A_fixed.csv", "B": "annotations_2017_B.csv"}
    names["C"] = "annotations_2017_C.csv"
    for expert, per_neonate in columns.items():
        frame = pd.DataFrame(
            {neonate: pd.Series(values) for neonate, values in per_neonate.items()}
        )
        frame.to_csv(root / names[expert], index=False)


def _votes(a, b, c, neonate: str = "1") -> dict:
    return {"A": {neonate: a}, "B": {neonate: b}, "C": {neonate: c}}


def _patient(root: Path, neonate: str = "1", **overrides) -> SimpleNamespace:
    """A patient with one ``recordings`` event, like a row of the CSV."""
    fields = {
        "signal_file": str(root / f"eeg{neonate}.edf"),
        "annotation_file_a": str(root / "annotations_2017_A_fixed.csv"),
        "annotation_file_b": str(root / "annotations_2017_B.csv"),
        "annotation_file_c": str(root / "annotations_2017_C.csv"),
    }
    fields.update(overrides)
    event = SimpleNamespace(**fields)
    return SimpleNamespace(
        patient_id=neonate, get_events=lambda event_type=None: [event]
    )


def _raw_task(**kwargs) -> SeizureDetectionHelsinki:
    """A task on the stored signal, with 4 second windows shifted by 2."""
    options = {
        "window_seconds": 4,
        "stride_seconds": 2,
        "resample_rate": None,
        "bandpass": None,
    }
    options.update(kwargs)
    return SeizureDetectionHelsinki(**options)


def _prepare(root: Path) -> pd.DataFrame:
    ds = HelsinkiNeonatalDataset.__new__(HelsinkiNeonatalDataset)
    ds.root = str(root)
    ds.prepare_metadata()
    return pd.read_csv(
        root / "helsinki_neonatal-pyhealth.csv", dtype=str, keep_default_na=False
    )


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class TestHelpers(_TempDir):
    def test_electrode_name_handles_both_label_styles(self):
        self.assertEqual(electrode_name("EEG Fp1-Ref"), "Fp1")
        self.assertEqual(electrode_name("EEG Cz-REF"), "Cz")
        self.assertEqual(electrode_name("Resp Effort-REF"), "Resp Effort")

    def test_default_filter_passband(self):
        # The docstring gives the band that passes at -3 dB for the default
        # stopband edges (0.5, 32) Hz on a 256 Hz recording.
        sos = bandpass_sections(SeizureDetectionHelsinki().bandpass, 256.0)
        freqs, response = sosfreqz(sos, worN=2**16, fs=256.0)
        gain_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
        passing = freqs[gain_db >= -3]
        self.assertAlmostEqual(float(passing[0]), 1.25, delta=0.02)
        self.assertAlmostEqual(float(passing[-1]), 13.4, delta=0.05)
        # 80 dB of attenuation at and beyond the stopband edges.
        self.assertLess(float(gain_db[freqs <= 0.5].max()), -79.9)
        self.assertLess(float(gain_db[freqs >= 32].max()), -79.9)

    def test_read_annotations_drops_rows_after_the_recording(self):
        _write_annotations(
            self.root,
            {"A": {"1": [0, 1, 1], "2": [0, 0, 0, 1, 0]}, "B": {"1": [0]}, "C": {}},
        )
        path = str(self.root / "annotations_2017_A_fixed.csv")

        self.assertEqual(read_helsinki_annotations(path, "1").tolist(), [0, 1, 1])
        self.assertEqual(read_helsinki_annotations(path, "2").tolist(), [0, 0, 0, 1, 0])

    def test_combine_annotations(self):
        votes = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
        expected = {
            "consensus": [0, -1, -1, 1],
            "majority": [0, 0, 1, 1],
            "any": [0, 1, 1, 1],
            "A": [0, 1, 1, 1],
            "B": [0, 0, 1, 1],
            "C": [0, 0, 0, 1],
        }
        self.assertEqual(set(expected), set(LABEL_MODES))
        for mode, labels in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(combine_annotations(votes, mode).tolist(), labels)
        with self.assertRaises(ValueError):
            combine_annotations(votes, "D")


class TestHelsinkiMetadata(_TempDir):
    def _recordings(self, *numbers: int) -> None:
        for number in numbers:
            _write_edf(
                self.root / f"eeg{number}.edf", CHANNELS_REF, _signal(4, CHANNELS_REF)
            )

    def test_rows_are_sorted_by_recording_number(self):
        self._recordings(10, 2, 1)
        _write_annotations(self.root, _votes([0], [0], [0]))
        (self.root / "clinical_information.csv").write_text(
            CLINICAL_HEADER
            + "1,eeg1,f,,37 to 38,37 to 38,mild/moderate asphyxia,,,3,,\n"
            + "2,eeg2,m,,40 to 41,41 to 42,,,,0,,\n"
        )
        (self.root / "notes.edf").write_bytes(b"")

        df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), ["1", "2", "10"])
        first = df.iloc[0]
        self.assertEqual(Path(first["signal_file"]).name, "eeg1.edf")
        self.assertEqual(
            Path(first["annotation_file_a"]).name, "annotations_2017_A_fixed.csv"
        )
        self.assertEqual(
            Path(first["annotation_file_c"]).name, "annotations_2017_C.csv"
        )
        self.assertEqual(first["gender"], "f")
        self.assertEqual(first["gestational_age"], "37 to 38")
        self.assertEqual(first["diagnosis"], "mild/moderate asphyxia")
        self.assertEqual(first["n_reviewers_seizure"], "3")
        # No diagnosis for neonate 2, and no clinical row for neonate 10.
        self.assertEqual(df.iloc[1]["diagnosis"], "")
        self.assertEqual(df.iloc[2]["gender"], "")

    def test_missing_annotation_and_clinical_files_are_left_empty(self):
        self._recordings(1)
        (self.root / "annotations_2017_B.csv").write_text("1\n0\n")

        with self.assertLogs(
            "pyhealth.datasets.helsinki_neonatal", level="WARNING"
        ) as logs:
            df = _prepare(self.root)

        row = df.iloc[0]
        self.assertEqual(row["annotation_file_a"], "")
        self.assertTrue(row["annotation_file_b"].endswith("annotations_2017_B.csv"))
        self.assertEqual(row["annotation_file_c"], "")
        self.assertEqual(row["gender"], "")
        self.assertEqual(len(logs.output), 3)

    def test_original_expert_a_file_is_used_with_a_warning(self):
        self._recordings(1)
        _write_annotations(self.root, _votes([0], [0], [0]))
        (self.root / "annotations_2017_A_fixed.csv").rename(
            self.root / "annotations_2017_A.csv"
        )

        with self.assertLogs(
            "pyhealth.datasets.helsinki_neonatal", level="WARNING"
        ) as logs:
            df = _prepare(self.root)

        self.assertTrue(
            df.iloc[0]["annotation_file_a"].endswith("annotations_2017_A.csv")
        )
        self.assertTrue(any("version 4" in line for line in logs.output))

    def test_empty_root_writes_nothing(self):
        with self.assertLogs("pyhealth.datasets.helsinki_neonatal", level="WARNING"):
            ds = HelsinkiNeonatalDataset.__new__(HelsinkiNeonatalDataset)
            ds.root = str(self.root)
            ds.prepare_metadata()
        self.assertFalse((self.root / "helsinki_neonatal-pyhealth.csv").exists())


class TestSeizureDetectionHelsinki(_TempDir):
    #           second: 0  1  2  3  4  5  6  7  8  9 10 11
    A = (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)
    B = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0)
    C = (0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0)

    def _write(self, channels: list[str] = CHANNELS_REF, seconds: int = 12) -> None:
        _write_edf(self.root / "eeg1.edf", channels, _signal(seconds, channels))
        _write_annotations(self.root, _votes(self.A, self.B, self.C))

    def _run(self, **kwargs) -> list[dict]:
        return _raw_task(**kwargs)(_patient(self.root))

    def test_consensus_drops_windows_with_disagreement(self):
        self._write()

        samples = self._run()

        # Seconds 4 and 9 are disputed. Windows start every 2 s and last 4 s:
        # only 0-3 (no seizure) and 5-8 would be clean, and 5 is not a start.
        self.assertEqual([s["start_second"] for s in samples], [0])
        self.assertEqual(samples[0]["label"], 0)
        self.assertEqual(samples[0]["seizure_fraction"], 0.0)

        # With a 1 s shift the window on seconds 5 to 8 is a start: it is the
        # only consensus seizure window.
        samples = self._run(stride_seconds=1)
        self.assertEqual([s["start_second"] for s in samples], [0, 5])
        self.assertEqual([s["label"] for s in samples], [0, 1])

    def test_partly_seizure_windows_are_dropped_by_default(self):
        self._write()
        # (starts, labels) of the windows that are all seizure or all not.
        expected = {
            "majority": ([0, 6], [0, 1]),  # seizure seconds 5 to 9
            "any": ([0, 4, 6], [0, 1, 1]),  # seizure seconds 4 to 9
            "A": ([0, 4, 6], [0, 1, 1]),
            "B": ([0, 6], [0, 1]),  # seizure seconds 5 to 9
            "C": ([0], [0]),  # seizure seconds 5 to 8, never a whole window
        }
        for mode, (starts, labels) in expected.items():
            with self.subTest(mode=mode):
                samples = self._run(label_mode=mode)
                self.assertEqual([s["start_second"] for s in samples], starts)
                self.assertEqual([s["label"] for s in samples], labels)
                self.assertTrue(
                    all(s["seizure_fraction"] in (0.0, 1.0) for s in samples)
                )

    def test_min_seizure_fraction_keeps_and_labels_every_window(self):
        self._write()
        expected = {
            "majority": [0, 0, 1, 1, 1],
            "any": [0, 1, 1, 1, 1],
            "A": [0, 1, 1, 1, 1],
            "B": [0, 0, 1, 1, 1],
            "C": [0, 0, 1, 1, 0],
        }
        for mode, labels in expected.items():
            with self.subTest(mode=mode):
                samples = self._run(label_mode=mode, min_seizure_fraction=0.5)
                self.assertEqual([s["start_second"] for s in samples], [0, 2, 4, 6, 8])
                self.assertEqual([s["label"] for s in samples], labels)

        strict = self._run(label_mode="C", min_seizure_fraction=1.0)
        loose = self._run(label_mode="C", min_seizure_fraction=0.25)

        # Expert C marks seconds 5 to 8: fractions 0, .25, .75, .75, .25.
        self.assertEqual([s["label"] for s in strict], [0, 0, 0, 0, 0])
        self.assertEqual([s["label"] for s in loose], [0, 1, 1, 1, 1])
        self.assertEqual(
            [s["seizure_fraction"] for s in loose], [0.0, 0.25, 0.75, 0.75, 0.25]
        )
        # Consensus still drops the windows with a disputed second.
        consensus = self._run(min_seizure_fraction=0.5)
        self.assertEqual([s["start_second"] for s in consensus], [0])

    def test_bipolar_montage_values_for_both_label_styles(self):
        for channels in (CHANNELS_REF, CHANNELS_UPPER):
            with self.subTest(first=channels[0]):
                self._write(channels)
                sample = self._run(label_mode="A")[1]

                self.assertEqual(sample["signal"].shape, (18, 4 * SFREQ))
                self.assertEqual(sample["signal"].dtype, np.float32)
                for i, (a, b) in enumerate(HELSINKI_BIPOLAR_MONTAGE):
                    # The per-second ramp cancels in the difference.
                    np.testing.assert_allclose(
                        sample["signal"][i],
                        _electrode_value(a) - _electrode_value(b),
                    )

    def test_custom_montage_keeps_the_time_course(self):
        self._write()

        sample = self._run(label_mode="A", montage=[("Cz", "Pz")])[1]

        self.assertEqual(sample["signal"].shape, (1, 4 * SFREQ))
        self.assertEqual(sample["start_second"], 4)

    def test_bandpass_and_resampling(self):
        seconds, rate = 40, 64
        self._write_tone(seconds, rate)

        task = SeizureDetectionHelsinki(
            label_mode="A",
            window_seconds=8,
            stride_seconds=8,
            resample_rate=16,
            bandpass=(0.5, 12.0),
            montage=[("Fp2", "F4")],
        )
        samples = task(_patient(self.root))

        self.assertEqual(len(samples), seconds // 8)
        window = samples[2]["signal"][0]
        self.assertEqual(window.shape, (8 * 16,))
        # Fp2 holds a 3 Hz tone plus a constant and a 20 Hz tone; F4 is zero.
        # 0.5 and 12 Hz are the stopband edges: the band-pass keeps the 3 Hz
        # tone and removes the other two.
        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(window.size, d=1 / 16)
        self.assertEqual(freqs[int(spectrum.argmax())], 3.0)
        self.assertLess(abs(float(window.mean())), 1.0)
        self.assertAlmostEqual(float(np.abs(window).max()), 100.0, delta=5.0)

    def test_band_edge_above_the_recording_nyquist_raises(self):
        self._write()  # a 16 Hz recording

        with self.assertRaises(ValueError):
            self._run(label_mode="A", bandpass=(0.5, 8.0))

    def _write_tone(self, seconds: int, rate: int) -> None:
        """Write a ``rate`` Hz recording whose Fp2 channel carries tones."""
        times = np.arange(seconds * rate) / rate
        data = np.zeros((len(CHANNELS_REF), times.size))
        fp2 = CHANNELS_REF.index("EEG Fp2-Ref")
        data[fp2] = (
            1000
            + 100 * np.sin(2 * np.pi * 3 * times)
            + 80 * np.sin(2 * np.pi * 20 * times)
        )
        _write_edf(self.root / "eeg1.edf", CHANNELS_REF, np.round(data), sfreq=rate)
        _write_annotations(self.root, _votes([0] * seconds, [0], [0]))

    def test_annotations_longer_than_the_signal_are_truncated(self):
        self._write(seconds=7)

        with self.assertLogs(
            "pyhealth.tasks.seizure_detection_helsinki", level="WARNING"
        ):
            samples = self._run(label_mode="A", min_seizure_fraction=0.5)

        self.assertEqual([s["start_second"] for s in samples], [0, 2])

    def test_missing_electrode_skips_the_recording(self):
        channels = [c for c in CHANNELS_REF if "Cz" not in c]
        self._write(channels)

        with self.assertLogs(
            "pyhealth.tasks.seizure_detection_helsinki", level="WARNING"
        ) as logs:
            samples = self._run()

        self.assertEqual(samples, [])
        self.assertIn("CZ", logs.output[0])

    def test_missing_annotations_skip_the_recording(self):
        self._write()
        log = "pyhealth.tasks.seizure_detection_helsinki"

        with self.assertLogs(log, level="WARNING"):
            no_file = _raw_task()(_patient(self.root, annotation_file_b=""))
        with self.assertLogs(log, level="WARNING"):
            no_column = _raw_task()(_patient(self.root, neonate="2"))
        one_expert = _raw_task(label_mode="A")(
            _patient(self.root, annotation_file_b="", annotation_file_c="")
        )

        self.assertEqual(no_file, [])
        self.assertEqual(no_column, [])
        self.assertEqual([s["start_second"] for s in one_expert], [0, 4, 6])

    def test_invalid_arguments(self):
        cases = (
            {"label_mode": "D"},
            {"window_seconds": 0},
            {"window_seconds": 2.5},
            {"stride_seconds": -1},
            {"resample_rate": 0},
            {"bandpass": (16.0, 0.5)},
            {"min_seizure_fraction": 0.0},
            {"min_seizure_fraction": 1.5},
            {"montage": []},
        )
        for kwargs in cases:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SeizureDetectionHelsinki(**kwargs)

    def test_defaults(self):
        task = SeizureDetectionHelsinki()
        self.assertEqual(task.label_mode, "consensus")
        self.assertEqual((task.window_seconds, task.stride_seconds), (16, 4))
        self.assertEqual(task.resample_rate, 32.0)
        self.assertEqual(task.bandpass, (0.5, 32.0))
        self.assertIsNone(task.min_seizure_fraction)
        self.assertEqual(task.montage, HELSINKI_BIPOLAR_MONTAGE)
        self.assertEqual(len(HELSINKI_BIPOLAR_MONTAGE), 18)


class TestHelsinkiEndToEnd(_TempDir):
    def test_set_task_on_two_recordings(self):
        for neonate in ("1", "2"):
            _write_edf(
                self.root / f"eeg{neonate}.edf", CHANNELS_REF, _signal(12, CHANNELS_REF)
            )
        seizure = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        quiet = [0] * 12
        _write_annotations(
            self.root,
            {expert: {"1": seizure, "2": quiet} for expert in ("A", "B", "C")},
        )

        dataset = HelsinkiNeonatalDataset(
            root=str(self.root), cache_dir=str(self.root / "c")
        )
        samples = dataset.set_task(_raw_task())

        self.assertEqual(sorted(dataset.unique_patient_ids), ["1", "2"])
        # Neonate 1 has a seizure from second 4: the window starting at 2 is
        # half seizure and dropped; those starting at 4, 6 and 8 are seizures.
        self.assertEqual(len(samples), 4 + 5)
        self.assertEqual(tuple(samples[0]["signal"].shape), (18, 4 * SFREQ))
        self.assertEqual(sum(int(s["label"]) for s in samples), 3)


if __name__ == "__main__":
    unittest.main()
