"""Tests for the HeartLang CPSC2018 task on CardiologyDataset, with synthetic ECGs."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import signal as sps
from scipy.io import savemat

from pyhealth.datasets import CardiologyDataset, create_sample_dataset
from pyhealth.models import ecg_to_heartbeat_sentence
from pyhealth.tasks import CPSC2018HeartLangClassification
from pyhealth.tasks.cpsc2018_heartlang import (
    HEARTLANG_CPSC2018_CODES,
    HEARTLANG_CPSC2018_LABELS,
    heartlang_cpsc2018_labels,
    load_heartlang_cpsc2018_split,
    preprocess_heartlang_cpsc2018,
    split_by_heartlang_cpsc2018,
)

HAS_WFDB = importlib.util.find_spec("wfdb") is not None
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PEAKS_100HZ = np.arange(50, 480, 80)  # beats in the kept first 5 s only


def reference_preprocess(ecg: np.ndarray) -> np.ndarray:
    """HeartLang's make_cpsc_dataset.py steps, restated lead by lead."""
    x = ecg[:, 0:2500]
    x = np.pad(x, ((0, 0), (0, 2500)), "constant", constant_values=0)
    x = 6 * (x - x.min()) / (x.max() - x.min()) - 3
    return np.stack([sps.resample(lead, 1000) for lead in x])


def synthetic_ecg(n_samples: int = 7500) -> np.ndarray:
    """R-peak-like spikes every 0.8 s on a slow wave, in µV as int16."""
    t = np.arange(n_samples) / 500
    x = 200 * np.sin(2 * np.pi * 0.3 * t) + 100
    for p in range(250, n_samples - 100, 400):
        x[p - 5 : p + 6] += np.hanning(11) * 1500
    lead_gain = np.linspace(0.5, 1.5, 12)[:, None]
    return np.round(x[None, :] * lead_gain).astype(np.int16)


def write_record(folder: Path, name: str, dx: str, ecg: np.ndarray) -> None:
    """Writes a Challenge 2020 record: ``.mat`` (``val``) plus ``.hea``."""
    folder.mkdir(parents=True, exist_ok=True)
    savemat(folder / f"{name}.mat", {"val": ecg})
    lines = [f"{name} 12 500 {ecg.shape[1]}"]
    lines += [f"{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in LEADS]
    lines += ["# Age: 60", "# Sex: Male", f"# Dx: {dx}"]
    lines += ["# Rx: Unknown", "# Hx: Unknown", "# Sx: Unknown"]
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")


def fixed_qrs(signal, **kwargs):
    return ecg_to_heartbeat_sentence(signal, qrs_inds=PEAKS_100HZ, **kwargs)


class TestLabels(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(len(HEARTLANG_CPSC2018_LABELS), 9)
        self.assertEqual(
            set(HEARTLANG_CPSC2018_CODES.values()), set(HEARTLANG_CPSC2018_LABELS)
        )
        self.assertEqual(len(set(HEARTLANG_CPSC2018_CODES)), 9)

    def test_codes_map_to_labels_in_vocabulary_order(self):
        # RBBB and AF, given in the opposite order to the vocabulary.
        self.assertEqual(
            heartlang_cpsc2018_labels("59118001,164889003"), ["AFIB", "CRBBB"]
        )
        self.assertEqual(heartlang_cpsc2018_labels("164884008"), ["VPC"])

    def test_other_codes_and_empty_values(self):
        # Complete RBBB is not a CPSC 2018 code; the source uses plain RBBB.
        self.assertEqual(heartlang_cpsc2018_labels("713427006"), [])
        self.assertEqual(heartlang_cpsc2018_labels(""), [])
        self.assertEqual(heartlang_cpsc2018_labels(None), [])
        self.assertEqual(
            heartlang_cpsc2018_labels(" 426783006 ; 429622005 "), ["NORM", "STD"]
        )


class TestPreprocess(unittest.TestCase):
    def test_matches_reference_steps(self):
        ecg = synthetic_ecg().astype(np.float64)
        np.testing.assert_allclose(
            preprocess_heartlang_cpsc2018(ecg), reference_preprocess(ecg), atol=1e-5
        )

    def test_shape_and_padding(self):
        out = preprocess_heartlang_cpsc2018(synthetic_ecg())
        self.assertEqual(out.shape, (12, 1000))
        self.assertEqual(out.dtype, np.float32)
        # The padded half is a constant that is not 0 after the min-max.
        tail = out[:, 600:950]
        self.assertLess(float(np.ptp(tail)), 0.05)
        self.assertGreater(abs(float(tail.mean())), 0.5)

    def test_unit_does_not_matter(self):
        ecg = synthetic_ecg().astype(np.float64)
        np.testing.assert_allclose(
            preprocess_heartlang_cpsc2018(ecg),
            preprocess_heartlang_cpsc2018(ecg / 1000.0),
            atol=1e-5,
        )

    def test_short_record_is_padded_to_ten_seconds(self):
        out = preprocess_heartlang_cpsc2018(synthetic_ecg(2000))
        self.assertEqual(out.shape, (12, 1000))

    def test_flat_record_becomes_zeros(self):
        out = preprocess_heartlang_cpsc2018(np.zeros((12, 5000)))
        np.testing.assert_array_equal(out, np.zeros((12, 1000), dtype=np.float32))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            preprocess_heartlang_cpsc2018(np.zeros(5000))
        with self.assertRaises(ValueError):
            preprocess_heartlang_cpsc2018(
                np.zeros((12, 5000)), keep_seconds=11, total_seconds=10
            )

    def test_non_finite_input_is_an_error(self):
        for bad in (np.nan, np.inf):
            signal = np.zeros((12, 5000))
            signal[3, 100] = bad
            with self.assertRaises(ValueError):
                preprocess_heartlang_cpsc2018(signal)


@unittest.skipUnless(HAS_WFDB, "wfdb is needed for QRS detection")
class TestQRSDetection(unittest.TestCase):
    def test_beats_are_found_in_the_kept_part_only(self):
        signal = preprocess_heartlang_cpsc2018(synthetic_ecg())
        _, lead_ids, _ = ecg_to_heartbeat_sentence(signal, sampling_rate=100)
        n_beats = int((lead_ids == 1).sum())
        self.assertGreaterEqual(n_beats, 5)
        self.assertLessEqual(n_beats, 7)


class TestTaskEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "training"
        cls.ecg = synthetic_ecg()
        write_record(root / "cpsc_2018" / "g1", "A0001", "164889003,59118001", cls.ecg)
        write_record(root / "cpsc_2018" / "g1", "A0002", "426783006", cls.ecg)
        write_record(root / "cpsc_2018" / "g1", "A0003", "713427006", cls.ecg)
        write_record(root / "georgia" / "g1", "E00001", "164889003", cls.ecg)
        cls.dataset = CardiologyDataset(
            root=str(root),
            sources=["cpsc_2018", "georgia"],
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, **kwargs):
        with mock.patch(
            "pyhealth.models.heartlang.ecg_to_heartbeat_sentence",
            side_effect=fixed_qrs,
        ):
            return self.dataset.set_task(
                CPSC2018HeartLangClassification(**kwargs), num_workers=1
            )

    def test_reads_the_mat_waveform(self):
        path = self.tmp / "training" / "cpsc_2018" / "g1" / "A0001.mat"
        raw = CPSC2018HeartLangClassification._read_mat(str(path))
        np.testing.assert_array_equal(raw, self.ecg)

    def test_samples(self):
        samples = self.build(leads=[0, 1])
        by_id = {samples[i]["record_id"]: samples[i] for i in range(len(samples))}
        # Georgia records are ignored: only the cpsc_2018 table is read.
        self.assertEqual(set(by_id), {"A0001", "A0002", "A0003"})
        first = by_id["A0001"]
        self.assertEqual(tuple(first["ecg_sentence"].shape), (256, 96))
        self.assertEqual(int((first["lead_ids"] == 2).sum()), len(PEAKS_100HZ))
        positive = {
            HEARTLANG_CPSC2018_LABELS[i]
            for i, v in enumerate(first["labels"].tolist())
            if v
        }
        self.assertEqual(positive, {"AFIB", "CRBBB"})
        self.assertEqual(int(by_id["A0003"]["labels"].sum()), 0)

    def test_drop_empty_labels(self):
        samples = self.build(drop_empty_labels=True)
        ids = {samples[i]["record_id"] for i in range(len(samples))}
        self.assertEqual(ids, {"A0001", "A0002"})


def write_split_csvs(folder: Path, splits: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    header = "patient_id,ecg_id,filename,validation,age,sex,scp_codes,"
    header += ",".join(HEARTLANG_CPSC2018_LABELS)
    for split, ids in splits.items():
        rows = [header]
        rows += [f"1,1,{i},False,60.0,1,{{}}," + ",".join("0" * 9) for i in ids]
        (folder / f"icbeb_{split}.csv").write_text("\n".join(rows) + "\n")


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        write_split_csvs(
            self.tmp,
            {"train": ["A0001", "A0002"], "val": ["A0003"], "test": ["A0004"]},
        )

    def make_samples(self, ids):
        rng = np.random.default_rng(0)
        return create_sample_dataset(
            [
                {
                    "patient_id": i,
                    "record_id": i,
                    "ecg_sentence": rng.standard_normal((8, 16)).astype(np.float32),
                    "lead_ids": np.arange(8) % 12 + 1,
                    "time_ids": np.arange(8) % 10 + 1,
                    "labels": ["NORM"],
                }
                for i in ids
            ],
            input_schema={
                "ecg_sentence": "tensor",
                "lead_ids": "tensor",
                "time_ids": "tensor",
            },
            output_schema={"labels": "multilabel"},
        )

    def test_load(self):
        ids = load_heartlang_cpsc2018_split(self.tmp)
        self.assertEqual(ids["train"], {"A0001", "A0002"})
        self.assertEqual(ids["val"], {"A0003"})
        self.assertEqual(ids["test"], {"A0004"})

    def test_overlap_is_an_error(self):
        write_split_csvs(self.tmp, {"test": ["A0001"]})
        with self.assertRaises(ValueError):
            load_heartlang_cpsc2018_split(self.tmp)

    def test_split_drops_unlisted_records(self):
        samples = self.make_samples(["A0001", "A0002", "A0003", "A0009"])
        with self.assertLogs(
            "pyhealth.tasks.cpsc2018_heartlang", level="WARNING"
        ) as logs:
            train, val, test = split_by_heartlang_cpsc2018(samples, self.tmp)
        self.assertEqual((len(train), len(val), len(test)), (2, 1, 0))
        self.assertIn("test split is empty", logs.output[0])
        self.assertEqual(val[0]["record_id"], "A0003")
        with self.assertRaises(ValueError):
            split_by_heartlang_cpsc2018(samples, self.tmp, record_ids=["A0001"])


if __name__ == "__main__":
    unittest.main()
