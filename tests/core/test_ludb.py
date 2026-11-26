"""LUDBDataset tests on synthetic WFDB records (no real LUDB data)."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import LUDBDataset
from pyhealth.datasets.ludb import (
    LUDB_LEADS,
    METADATA_COLUMNS,
    load_ludb_record,
    parse_wave_annotations,
)

FS = 500
N_SAMPLES = 500  # 1 s per record keeps the fixture small and fast.

# One beat: P (50-80), QRS (120-150), T (220-280).
BEAT_SYMBOLS = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]
BEAT_SAMPLES = [50, 65, 80, 120, 135, 150, 220, 250, 280]
# A record with no P wave (e.g. atrial fibrillation) and a T wave cut off
# at the end of the recording (no offset marker, so it ends at its peak).
NO_P_SYMBOLS = ["(", "N", ")", "(", "t"]
NO_P_SAMPLES = [120, 135, 150, 400, 450]


HAS_WFDB = importlib.util.find_spec("wfdb") is not None


def write_synthetic_record(data_dir: Path, record_id: str, has_p: bool) -> None:
    """Write a 12-lead WFDB record plus one annotation file per lead.

    ``wfdb.wrann`` only accepts letters in the extension, so each lead is
    written as ``.tmpann`` and renamed to its LUDB extension (``.v1``, ...).
    """
    import wfdb

    t = np.arange(N_SAMPLES) / FS
    signal = np.stack(
        [np.sin(2 * np.pi * (k + 1) * t) for k in range(len(LUDB_LEADS))], axis=1
    )
    wfdb.wrsamp(
        record_id,
        fs=FS,
        units=["mV"] * len(LUDB_LEADS),
        sig_name=list(LUDB_LEADS),
        p_signal=signal,
        fmt=["16"] * len(LUDB_LEADS),
        write_dir=str(data_dir),
    )
    symbols, samples = (
        (BEAT_SYMBOLS, BEAT_SAMPLES) if has_p else (NO_P_SYMBOLS, NO_P_SAMPLES)
    )
    for lead in LUDB_LEADS:
        wfdb.wrann(
            record_id,
            "tmpann",
            sample=np.asarray(samples),
            symbol=list(symbols),
            write_dir=str(data_dir),
        )
        (data_dir / f"{record_id}.tmpann").replace(data_dir / f"{record_id}.{lead}")


class TestParseWaveAnnotations(unittest.TestCase):
    def test_full_beat(self):
        self.assertEqual(
            parse_wave_annotations(BEAT_SYMBOLS, BEAT_SAMPLES),
            [("P", 50, 80), ("QRS", 120, 150), ("T", 220, 280)],
        )

    def test_missing_bracket_uses_peak(self):
        # QRS without onset (onset at its peak) and P without offset.
        symbols = ["N", ")", "(", "t", ")", "(", "p"]
        samples = [5, 10, 20, 30, 40, 90, 95]
        self.assertEqual(
            parse_wave_annotations(symbols, samples),
            [("QRS", 5, 10), ("T", 20, 40), ("P", 90, 95)],
        )

    def test_wave_without_brackets_is_skipped(self):
        symbols = ["(", "p", ")", "N", "(", "t", ")"]
        samples = [1, 2, 3, 10, 20, 25, 30]
        self.assertEqual(
            parse_wave_annotations(symbols, samples), [("P", 1, 3), ("T", 20, 30)]
        )

    def test_unknown_symbols_and_empty_input(self):
        self.assertEqual(parse_wave_annotations(["(", "x", ")"], [1, 2, 3]), [])
        self.assertEqual(parse_wave_annotations([], []), [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            parse_wave_annotations(["(", "N"], [1])


@unittest.skipUnless(HAS_WFDB, "wfdb optional extra not installed")
class TestLUDBDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "ludb"
        data_dir = cls.root / "data"
        data_dir.mkdir(parents=True)
        write_synthetic_record(data_dir, "1", has_p=True)
        write_synthetic_record(data_dir, "2", has_p=False)
        write_synthetic_record(data_dir, "10", has_p=True)
        pd.DataFrame(
            {
                "ID": [1, 2, 10, 11],  # record 11 has no WFDB files
                "Sex": ["F\n", "M\n", "M\n", "F\n"],
                "Age": ["51\n", ">89", "64\n", "30\n"],
                "Rhythms": [
                    "Sinus rhythm\nWandering atrial pacemaker",
                    "Atrial fibrillation",
                    "Sinus bradycardia",
                    "Sinus rhythm",
                ],
                "Electric axis of the heart": [
                    "Electric axis of the heart: normal",
                    "Electric axis of the heart: left axis deviation",
                    np.nan,
                    "Electric axis of the heart: normal",
                ],
            }
        ).to_csv(cls.root / "ludb.csv", index=False)
        cls.dataset = LUDBDataset(root=str(cls.root), cache_dir=str(cls.tmp / "cache"))
        cls.meta = pd.read_csv(cls.root / "ludb-pyhealth.csv", dtype=str)
        # Build the lazily loaded event table once here, not inside a test.
        cls.dataset.get_patient("1")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_metadata_rows_sorted_numerically(self):
        self.assertEqual(self.meta["record_id"].tolist(), ["1", "2", "10"])
        self.assertTrue((self.meta["patient_id"] == self.meta["record_id"]).all())

    def test_metadata_parses_demographics_and_labels(self):
        rows = self.meta.set_index("record_id")
        self.assertEqual(rows.loc["1", "rhythm"], "Sinus rhythm")
        self.assertEqual(rows.loc["1", "sex"], "F")
        self.assertEqual(rows.loc["2", "age"], "90")
        self.assertEqual(rows.loc["2", "age_is_censored"], "1")
        self.assertEqual(rows.loc["1", "age_is_censored"], "0")
        self.assertEqual(rows.loc["2", "electric_axis"], "left axis deviation")
        self.assertEqual(rows.loc["10", "electric_axis"], "none")

    def test_metadata_has_p_wave(self):
        rows = self.meta.set_index("record_id")
        self.assertEqual(rows.loc["1", "has_p_wave"], "1")
        self.assertEqual(rows.loc["2", "has_p_wave"], "0")

    def test_signal_file_points_to_record(self):
        for path in self.meta["signal_file"]:
            self.assertTrue(Path(f"{path}.hea").is_file())

    def test_patients_and_events(self):
        self.assertEqual(
            sorted(self.dataset.unique_patient_ids, key=int), ["1", "2", "10"]
        )
        events = self.dataset.get_patient("2").get_events(event_type="recordings")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].rhythm, "Atrial fibrillation")
        self.assertEqual(str(events[0].has_p_wave), "0")

    def test_load_ludb_record(self):
        signal, annotations = load_ludb_record(self.root / "data" / "1")
        self.assertEqual(signal.shape, (len(LUDB_LEADS), N_SAMPLES))
        self.assertEqual(signal.dtype, np.float32)
        self.assertEqual(set(annotations), set(LUDB_LEADS))
        self.assertEqual(
            annotations["ii"], [("P", 50, 80), ("QRS", 120, 150), ("T", 220, 280)]
        )

    def test_load_ludb_record_lead_subset_and_extension(self):
        signal, annotations = load_ludb_record(
            self.root / "data" / "1.hea", leads=["v1", "i"]
        )
        self.assertEqual(signal.shape, (2, N_SAMPLES))
        self.assertEqual(list(annotations), ["v1", "i"])
        full, _ = load_ludb_record(self.root / "data" / "1")
        np.testing.assert_allclose(signal[1], full[0])

    def test_load_ludb_record_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_ludb_record(self.root / "data" / "999")
        with self.assertRaises(ValueError):
            load_ludb_record(self.root / "data" / "1", leads=["x"])


@unittest.skipUnless(HAS_WFDB, "wfdb optional extra not installed")
class TestLUDBMetadataReuse(unittest.TestCase):
    """``prepare_metadata`` reuses a current CSV and rebuilds a stale one.

    These tests call ``prepare_metadata`` without building the full
    ``BaseDataset`` so they stay fast.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "ludb"
        data_dir = self.root / "data"
        data_dir.mkdir(parents=True)
        write_synthetic_record(data_dir, "1", has_p=True)
        pd.DataFrame(
            {
                "ID": [1],
                "Sex": ["F"],
                "Age": ["40"],
                "Rhythms": ["Sinus rhythm"],
                "Electric axis of the heart": ["Electric axis of the heart: normal"],
            }
        ).to_csv(self.root / "ludb.csv", index=False)
        self.csv = self.root / "ludb-pyhealth.csv"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _prepare(root: Path) -> None:
        dataset = LUDBDataset.__new__(LUDBDataset)
        dataset.root = str(root)
        dataset.prepare_metadata()

    def test_current_csv_is_reused(self):
        self._prepare(self.root)
        before = self.csv.stat().st_mtime_ns
        self._prepare(self.root)
        self.assertEqual(self.csv.stat().st_mtime_ns, before)

    def test_moved_root_rebuilds_signal_paths(self):
        self._prepare(self.root)
        moved = self.tmp / "moved"
        shutil.move(str(self.root), str(moved))
        self._prepare(moved)
        meta = pd.read_csv(moved / "ludb-pyhealth.csv", dtype=str)
        self.assertEqual(
            meta["signal_file"].tolist(), [str((moved / "data" / "1").resolve())]
        )

    def test_incomplete_csv_is_rebuilt(self):
        pd.DataFrame({"patient_id": ["1"]}).to_csv(self.csv, index=False)
        self._prepare(self.root)
        meta = pd.read_csv(self.csv, dtype=str)
        self.assertEqual(list(meta.columns), list(METADATA_COLUMNS))
        self.assertEqual(meta["has_p_wave"].tolist(), ["1"])


class TestPaperSplit(unittest.TestCase):
    def test_split_covers_all_records_once(self):
        train, val, test = LUDBDataset.default_subject_split().subjects
        self.assertEqual((len(train), len(val), len(test)), (160, 20, 20))
        everything = set(train) | set(val) | set(test)
        self.assertEqual(everything, {str(i) for i in range(1, 201)})
        self.assertIn("12", test)
        self.assertIn("1", val)


@unittest.skipUnless(HAS_WFDB, "wfdb optional extra not installed")
class TestLUDBDatasetErrors(unittest.TestCase):
    def test_missing_root(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaises(FileNotFoundError),
        ):
            LUDBDataset(root=str(Path(tmp) / "nope"))

    def test_missing_source_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            with self.assertRaises(FileNotFoundError):
                LUDBDataset(root=tmp)

    def test_no_records_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            pd.DataFrame(
                {
                    "ID": [1],
                    "Sex": ["F"],
                    "Age": ["40"],
                    "Rhythms": ["Sinus rhythm"],
                    "Electric axis of the heart": [np.nan],
                }
            ).to_csv(Path(tmp) / "ludb.csv", index=False)
            with self.assertRaises(ValueError):
                LUDBDataset(root=tmp)


if __name__ == "__main__":
    unittest.main()
