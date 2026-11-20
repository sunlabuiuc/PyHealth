"""Tests for LUDBDataset and ECGDelineationLUDB on small synthetic records.

Authors: Shray Srivastava (ssriv5@illinois.edu), Sid Wanjara (swanj2@illinois.edu)
"""

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import LUDBDataset
from pyhealth.tasks import ECGDelineationLUDB
from pyhealth.tasks.ecg_delineation_ludb import (
    IGNORE_INDEX,
    LUDB_LEADS,
    annotations_to_mask,
)

try:
    import wfdb
except ImportError:  # wfdb is an optional dependency
    wfdb = None

N_SAMPLES = 500  # 1 s at 500 Hz keeps the files tiny
SYMBOLS = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]
SAMPLES = [100, 110, 120, 200, 210, 220, 300, 320, 340]

# Same quirks as the real ludb.csv: cells end with a newline, some cells
# have several lines, one age is ">89" and paced records have no axis.
LUDB_CSV = (
    "ID,Sex,Age,Rhythms,Electric axis of the heart\n"
    '1,"F\n","51\n",Sinus rhythm,Electric axis of the heart: normal\n'
    '2,"M\n",">89\n","Atrial fibrillation\nAberrant conduction",'
    "Electric axis of the heart: left axis deviation\n"
    '10,"M\n","40\n",Sinus bradycardia,\n'
)


def write_record(
    folder: Path, record_id: str, leads: Sequence[str] = LUDB_LEADS, fs: int = 500
) -> None:
    """Write a fake 12-lead WFDB record plus one annotation file per lead."""
    t = np.arange(N_SAMPLES) / fs
    signal = np.tile(0.5 * np.sin(2 * np.pi * 5 * t), (12, 1)).T
    wfdb.wrsamp(
        record_id,
        fs=fs,
        units=["mV"] * 12,
        sig_name=list(LUDB_LEADS),
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(folder),
    )
    for lead in leads:
        # wrann only accepts letters in the extension, so write and rename.
        wfdb.wrann(record_id, "tmp", np.array(SAMPLES), SYMBOLS, write_dir=str(folder))
        (folder / f"{record_id}.tmp").rename(folder / f"{record_id}.{lead}")


def fake_patient(folder: Path, record_id: str = "1") -> SimpleNamespace:
    """A patient with one records event, so the task runs without a dataset."""
    event = SimpleNamespace(signal_file=str(folder / record_id))
    return SimpleNamespace(patient_id=record_id, get_events=lambda table=None: [event])


def prepare_metadata(root: Path) -> pd.DataFrame:
    """Run only prepare_metadata (skipping the slow BaseDataset build)."""
    dataset = LUDBDataset.__new__(LUDBDataset)
    dataset.root = str(root)
    dataset.prepare_metadata()
    return pd.read_csv(root / "ludb-pyhealth.csv", dtype=str, keep_default_na=False)


class TempDirTest(unittest.TestCase):
    """Gives each test a fresh temporary folder with an empty data/ inside."""

    def setUp(self) -> None:
        """Make the folder and delete it again after the test."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()


class TestAnnotationsToMask(unittest.TestCase):
    """Label generation: annotation symbols to a point-wise mask."""

    def test_waves_background_and_ignored_edges(self) -> None:
        """A clean P, QRS, T triplet gives the right label in every region."""
        mask = annotations_to_mask(SYMBOLS, SAMPLES, N_SAMPLES)
        self.assertTrue((mask[:100] == IGNORE_INDEX).all())
        self.assertTrue((mask[100:121] == 1).all())
        self.assertTrue((mask[121:200] == 0).all())
        self.assertTrue((mask[200:221] == 2).all())
        self.assertTrue((mask[300:341] == 3).all())
        self.assertTrue((mask[341:] == IGNORE_INDEX).all())

    def test_missing_onset_uses_the_peak(self) -> None:
        """A QRS written as "N )" with no "(" starts at its peak."""
        mask = annotations_to_mask(["N", ")", "(", "t", ")"], [5, 8, 10, 12, 14], 20)
        self.assertEqual(mask[4], IGNORE_INDEX)
        self.assertEqual(mask[5:9].tolist(), [2, 2, 2, 2])
        self.assertEqual(mask[10:15].tolist(), [3, 3, 3, 3, 3])

    def test_no_waves_is_all_ignored(self) -> None:
        """With no annotations, every point is 255."""
        self.assertTrue((annotations_to_mask([], [], 10) == IGNORE_INDEX).all())

    def test_length_mismatch_raises(self) -> None:
        """Two symbols but one sample index raises ValueError."""
        with self.assertRaises(ValueError):
            annotations_to_mask(["(", "p"], [1], 10)


@unittest.skipUnless(wfdb is not None, "wfdb is not installed")
class TestMetadata(TempDirTest):
    """Data loading and patient/event parsing in prepare_metadata."""

    def test_one_row_per_record_with_clean_metadata(self) -> None:
        """Records become rows in numeric order, with cleaned ludb.csv values."""
        for record_id in ("10", "2", "1"):
            write_record(self.data, record_id)
        (self.root / "ludb.csv").write_text(LUDB_CSV)

        df = prepare_metadata(self.root)

        self.assertEqual(list(df["patient_id"]), ["1", "2", "10"])  # numeric order
        self.assertEqual(df.loc[0, "signal_file"], str((self.data / "1").absolute()))
        self.assertEqual(
            list(df.loc[0, ["sex", "age", "rhythm"]]), ["F", "51", "Sinus rhythm"]
        )
        self.assertEqual(df.loc[0, "electric_axis"], "normal")
        self.assertEqual(df.loc[1, "age"], ">89")
        self.assertEqual(
            df.loc[1, "rhythm"], "Atrial fibrillation; Aberrant conduction"
        )
        self.assertEqual(df.loc[2, "electric_axis"], "")

    def test_data_folder_works_as_root(self) -> None:
        """Passing the data/ folder as root still finds ludb.csv one level up."""
        write_record(self.data, "1")
        (self.root / "ludb.csv").write_text(LUDB_CSV)
        df = prepare_metadata(self.data)
        self.assertEqual(df.loc[0, "rhythm"], "Sinus rhythm")

    def test_missing_ludb_csv_leaves_metadata_empty(self) -> None:
        """Without ludb.csv the rows are still written, with a warning."""
        write_record(self.data, "1")
        with self.assertLogs("pyhealth.datasets.ludb", level="WARNING"):
            df = prepare_metadata(self.root)
        self.assertEqual(df.loc[0, "rhythm"], "")

    def test_bad_inputs_raise(self) -> None:
        """A missing folder, no records, or ludb.csv without ID all raise."""
        with self.assertRaises(FileNotFoundError):
            prepare_metadata(self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            prepare_metadata(self.root)  # no records yet
        write_record(self.data, "1")
        (self.root / "ludb.csv").write_text("Sex,Age\nF,51\n")
        with self.assertRaises(ValueError):
            prepare_metadata(self.root)


@unittest.skipUnless(wfdb is not None, "wfdb is not installed")
class TestTask(TempDirTest):
    """Sample processing, feature extraction and edge cases of the task."""

    def test_one_sample_per_lead(self) -> None:
        """One record gives 12 samples with the right shapes and labels."""
        write_record(self.data, "1")
        samples = ECGDelineationLUDB()(fake_patient(self.data))

        self.assertEqual([s["lead"] for s in samples], list(LUDB_LEADS))
        self.assertEqual(samples[0]["patient_id"], "1")
        self.assertEqual(samples[0]["signal"].shape, (1, 1, N_SAMPLES))
        self.assertEqual(samples[0]["signal"].dtype, np.float32)
        np.testing.assert_array_equal(
            samples[0]["mask"][0], annotations_to_mask(SYMBOLS, SAMPLES, N_SAMPLES)
        )

    def test_unfiltered_signal_is_the_raw_lead(self) -> None:
        """With the filter off, the signal is exactly the stored lead."""
        write_record(self.data, "1")
        raw, _ = wfdb.rdsamp(str(self.data / "1"))
        task = ECGDelineationLUDB(leads=["II"], apply_filter=False)
        samples = task(fake_patient(self.data))
        self.assertEqual(len(samples), 1)
        np.testing.assert_allclose(samples[0]["signal"][0, 0], raw[:, 1], rtol=1e-5)

    def test_filter_keeps_ecg_band_and_removes_the_rest(self) -> None:
        """The band-pass keeps 10 Hz and removes a constant and 200 Hz."""
        task = ECGDelineationLUDB()
        t = np.arange(5000) / 500.0
        for frequency, keeps in ((0.0, False), (10.0, True), (200.0, False)):
            wave = np.cos(2 * np.pi * frequency * t)
            peak = np.abs(task.filter_signal(wave, 500.0)[1000:4000]).max()
            if keeps:
                # The 0.05 Hz high-pass is slow, so a 10 s record keeps a
                # small drift; without the mirror padding this was ~1.6.
                self.assertAlmostEqual(peak, 1.0, delta=0.15)
            else:
                self.assertLess(peak, 0.1)

    def test_noise_level_and_reproducibility(self) -> None:
        """Noise comes out at the requested SNR and repeats exactly."""
        task = ECGDelineationLUDB(noise_snr_db=10.0)
        clean = np.sin(np.linspace(0, 100, 5000))
        noise = task.add_noise(clean, "1", "ii") - clean
        snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
        self.assertAlmostEqual(snr, 10.0, delta=0.3)
        np.testing.assert_array_equal(task.add_noise(clean, "1", "ii"), clean + noise)
        self.assertFalse(np.allclose(task.add_noise(clean, "1", "iii"), clean + noise))

    def test_missing_annotation_file_skips_the_lead(self) -> None:
        """A lead without an annotation file is skipped with a warning."""
        write_record(self.data, "1", leads=LUDB_LEADS[:-1])
        with self.assertLogs("pyhealth.tasks.ecg_delineation_ludb", level="WARNING"):
            samples = ECGDelineationLUDB()(fake_patient(self.data))
        self.assertEqual(len(samples), 11)

    def test_low_sampling_rate_raises(self) -> None:
        """A 250 Hz record can't take the 150 Hz low-pass, so ValueError."""
        write_record(self.data, "1", leads=["i"], fs=250)
        with self.assertRaises(ValueError):
            ECGDelineationLUDB(leads=["i"])(fake_patient(self.data))

    def test_bad_leads_raise(self) -> None:
        """An empty list, a duplicate ("ii" and "II") or "x9" all raise."""
        for leads in ([], ["ii", "II"], ["x9"]):
            with self.assertRaises(ValueError):
                ECGDelineationLUDB(leads=leads)


@unittest.skipUnless(wfdb is not None, "wfdb is not installed")
class TestEndToEnd(unittest.TestCase):
    """Builds the dataset and runs set_task once for all tests (the slow part)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Write 3 records, build the dataset and run the task once."""
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "data").mkdir()
        for record_id in ("1", "2", "10"):
            write_record(root / "data", record_id)
        (root / "ludb.csv").write_text(LUDB_CSV)
        cls.dataset = LUDBDataset(root=str(root), cache_dir=str(root / "cache"))
        cls.samples = cls.dataset.set_task(ECGDelineationLUDB(leads=["i", "ii"]))

    def test_patients_and_events(self) -> None:
        """The dataset has the 3 patients and their events carry metadata."""
        self.assertEqual(
            sorted(self.dataset.unique_patient_ids, key=int), ["1", "2", "10"]
        )
        event = self.dataset.get_patient("2").get_events("records")[0]
        self.assertEqual(event.rhythm, "Atrial fibrillation; Aberrant conduction")
        self.assertIsInstance(self.dataset.default_task, ECGDelineationLUDB)

    def test_samples_are_tensors(self) -> None:
        """After set_task the samples are tensors of the right shape and type."""
        self.assertEqual(len(self.samples), 3 * 2)
        sample = self.samples[0]
        self.assertEqual(tuple(sample["signal"].shape), (1, 1, N_SAMPLES))
        self.assertEqual(sample["mask"].dtype, torch.int64)
        self.assertEqual(set(torch.unique(sample["mask"]).tolist()), {0, 1, 2, 3, 255})


if __name__ == "__main__":
    unittest.main()
