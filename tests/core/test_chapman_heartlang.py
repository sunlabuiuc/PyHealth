"""Tests for the HeartLang CSN task on Chapman-Shaoxing, with synthetic ECGs."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import signal as sps

from pyhealth.datasets import ChapmanShaoxingDataset, create_sample_dataset
from pyhealth.models import ecg_to_heartbeat_sentence
from pyhealth.tasks import ChapmanShaoxingHeartLangClassification
from pyhealth.tasks.chapman_heartlang import (
    HEARTLANG_CSN_LABELS,
    heartlang_csn_labels,
    load_heartlang_csn_split,
    preprocess_heartlang_csn,
    split_by_heartlang_csn,
)

HAS_WFDB = importlib.util.find_spec("wfdb") is not None
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PEAKS_100HZ = np.arange(50, 975, 80)


def reference_preprocess(ecg: np.ndarray) -> np.ndarray:
    """HeartLang's make_csn_datasets.py steps, restated lead by lead."""
    x = 6 * (ecg - ecg.min()) / (ecg.max() - ecg.min()) - 3
    n = int(x.shape[1] * 100 / 500)
    x = np.stack([sps.resample(lead, n) for lead in x])
    b, a = sps.butter(5, [0.67 / 50, 40 / 50], btype="band")
    return np.stack([sps.filtfilt(b, a, lead) for lead in x])


def synthetic_adc(n_samples: int = 5000) -> np.ndarray:
    """R-peak-like spikes every 0.8 s on a slow wave, as int16 ADC values."""
    t = np.arange(n_samples) / 500
    x = 200 * np.sin(2 * np.pi * 0.3 * t)
    for p in range(250, n_samples - 100, 400):
        x[p - 5 : p + 6] += np.hanning(11) * 1500
    lead_gain = np.linspace(0.5, 1.5, 12)[:, None]
    return np.round(x[None, :] * lead_gain).astype("<i2")


def write_record(folder: Path, name: str, dx: str, adc: np.ndarray) -> None:
    """Write a WFDB header and ``.mat`` (format 16, 24-byte offset)."""
    folder.mkdir(parents=True, exist_ok=True)
    n_samples = adc.shape[1]
    lines = [f"{name} 12 500 {n_samples}"]
    lines += [f"{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in LEADS]
    lines += ["#Age: 50", "#Sex: Male", f"#Dx: {dx}"]
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")
    (folder / f"{name}.mat").write_bytes(b"\0" * 24 + adc.T.tobytes())


def fixed_qrs(signal, **kwargs):
    return ecg_to_heartbeat_sentence(signal, qrs_inds=PEAKS_100HZ, **kwargs)


class TestLabels(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(len(HEARTLANG_CSN_LABELS), 38)
        self.assertEqual(len(set(HEARTLANG_CSN_LABELS)), 38)
        self.assertNotIn("SB", HEARTLANG_CSN_LABELS)

    def test_heartlang_matching_is_by_substring(self):
        cases = {
            "164889003": ["AF", "AFIB"],  # AFIB
            "270492004": ["VB", "AVB", "1AVB"],  # 1AVB
            "251146004": ["SA"],  # LVQRSAL (low QRS voltage)
            "251199005": ["CCR", "CR"],  # CCR
            "428750005": ["STTC", "ST"],  # STTC
            "54016002": ["2AVB", "VB", "2AVB1", "AVB"],  # 2AVB1
        }
        for dx, expected in cases.items():
            with self.subTest(dx=dx):
                self.assertEqual(
                    sorted(heartlang_csn_labels(dx)), sorted(expected)
                )

    def test_exact_matching_and_order(self):
        self.assertEqual(heartlang_csn_labels("164889003", match="exact"), ["AFIB"])
        self.assertEqual(
            heartlang_csn_labels("59118001,426783006;164889003", match="exact"),
            ["SR", "RBBB", "AFIB"],
        )

    def test_unlabelled_and_unknown_codes(self):
        self.assertEqual(heartlang_csn_labels("426177001"), [])  # SB
        self.assertEqual(heartlang_csn_labels("713427006"), [])  # unlisted code
        self.assertEqual(heartlang_csn_labels(float("nan")), [])
        with self.assertRaises(ValueError):
            heartlang_csn_labels("164889003", match="fuzzy")


class TestPreprocess(unittest.TestCase):
    def test_matches_the_reference_steps(self):
        ecg = synthetic_adc().astype(np.float64)
        out = preprocess_heartlang_csn(ecg)
        self.assertEqual(out.shape, (12, 1000))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, reference_preprocess(ecg), atol=1e-5)

    def test_scaling_does_not_depend_on_units(self):
        ecg = synthetic_adc().astype(np.float64)
        np.testing.assert_allclose(
            preprocess_heartlang_csn(ecg),
            preprocess_heartlang_csn(ecg / 1000),
            atol=1e-5,
        )

    def test_constant_and_bad_inputs(self):
        np.testing.assert_array_equal(
            preprocess_heartlang_csn(np.ones((12, 5000))), np.zeros((12, 1000))
        )
        with self.assertRaises(ValueError):
            preprocess_heartlang_csn(np.zeros(5000))
        with self.assertRaises(ValueError):
            preprocess_heartlang_csn(np.random.default_rng(0).normal(size=(12, 100)))


class TestTask(unittest.TestCase):
    def test_schema_and_options(self):
        task = ChapmanShaoxingHeartLangClassification()
        self.assertEqual(
            task.output_schema["labels"],
            ("multilabel", {"labels": list(HEARTLANG_CSN_LABELS)}),
        )
        self.assertEqual(
            set(task.input_schema), {"ecg_sentence", "lead_ids", "time_ids"}
        )
        with self.assertRaises(ValueError):
            ChapmanShaoxingHeartLangClassification(label_match="fuzzy")


@unittest.skipUnless(HAS_WFDB, "wfdb is not installed")
class TestTaskEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "ecg"
        sub = root / "WFDBRecords" / "01" / "010"
        cls.adc = synthetic_adc()
        write_record(sub, "JS00001", "164889003,59118001", cls.adc)  # AFIB, RBBB
        write_record(sub, "JS00002", "426783006", cls.adc)  # SR
        write_record(sub, "JS00003", "426177001", cls.adc)  # SB: no label
        cls.dataset = ChapmanShaoxingDataset(
            root=str(root),
            metadata_cache_dir=str(cls.tmp / "meta"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reads_raw_adc_values(self):
        record = self.tmp / "ecg" / "WFDBRecords" / "01" / "010" / "JS00001"
        adc, rate = ChapmanShaoxingHeartLangClassification._read_adc(str(record))
        self.assertEqual(rate, 500)
        np.testing.assert_array_equal(adc, self.adc)

    def test_samples(self):
        with mock.patch(
            "pyhealth.models.heartlang.ecg_to_heartbeat_sentence",
            side_effect=fixed_qrs,
        ):
            samples = self.dataset.set_task(
                ChapmanShaoxingHeartLangClassification(leads=[0, 1]),
                num_workers=1,
            )
        self.assertEqual(len(samples), 3)
        by_id = {samples[i]["record_id"]: samples[i] for i in range(len(samples))}
        first = by_id["JS00001"]
        self.assertEqual(tuple(first["ecg_sentence"].shape), (256, 96))
        self.assertEqual(int((first["lead_ids"] == 2).sum()), len(PEAKS_100HZ))
        positive = {
            HEARTLANG_CSN_LABELS[i]
            for i, v in enumerate(first["labels"].tolist())
            if v
        }
        self.assertEqual(positive, {"AF", "AFIB", "RBBB"})
        self.assertEqual(int(by_id["JS00003"]["labels"].sum()), 0)

    def test_samples_with_default_qrs_detection(self):
        """Runs set_task without mocking the QRS detector."""
        samples = self.dataset.set_task(
            ChapmanShaoxingHeartLangClassification(leads=[0, 1]),
            num_workers=1,
        )
        by_id = {samples[i]["record_id"]: samples[i] for i in range(len(samples))}
        first = by_id["JS00001"]
        self.assertEqual(tuple(first["ecg_sentence"].shape), (256, 96))
        lead_ids = first["lead_ids"].numpy()
        time_ids = first["time_ids"].numpy()
        words = lead_ids > 0
        # The synthetic beats are found, and on both leads alike.
        self.assertGreater(int((lead_ids == 1).sum()), 0)
        self.assertEqual(int((lead_ids == 1).sum()), int((lead_ids == 2).sum()))
        self.assertTrue(set(np.unique(lead_ids[words])) <= {1, 2})
        # time_ids are the second of the beat plus one: 1..10 for 10 s.
        self.assertTrue(((time_ids[words] >= 1) & (time_ids[words] <= 10)).all())
        self.assertTrue((time_ids[~words] == 0).all())
        self.assertTrue(np.abs(first["ecg_sentence"].numpy()[words]).sum() > 0)
        positive = {
            HEARTLANG_CSN_LABELS[i]
            for i, v in enumerate(first["labels"].tolist())
            if v
        }
        self.assertEqual(positive, {"AF", "AFIB", "RBBB"})

    def test_drop_empty_labels(self):
        with mock.patch(
            "pyhealth.models.heartlang.ecg_to_heartbeat_sentence",
            side_effect=fixed_qrs,
        ):
            samples = self.dataset.set_task(
                ChapmanShaoxingHeartLangClassification(drop_empty_labels=True),
                num_workers=1,
            )
        ids = {samples[i]["record_id"] for i in range(len(samples))}
        self.assertEqual(ids, {"JS00001", "JS00002"})


def write_split_csvs(folder: Path, splits: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        rows = ["ecg_path,age,diagnose,SR"]
        rows += [f"/chapman/WFDBRecords/01/010/{i}.mat,12,SR,1" for i in ids]
        (folder / f"chapman_{split}.csv").write_text("\n".join(rows) + "\n")


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        write_split_csvs(
            self.tmp,
            {"train": ["JS00001", "JS00002"], "val": ["JS00003"], "test": ["JS00004"]},
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
                    "labels": ["SR"],
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
        ids = load_heartlang_csn_split(self.tmp)
        self.assertEqual(ids["train"], {"JS00001", "JS00002"})
        self.assertEqual(ids["test"], {"JS00004"})

    def test_overlap_is_an_error(self):
        write_split_csvs(self.tmp, {"test": ["JS00001"]})
        with self.assertRaises(ValueError):
            load_heartlang_csn_split(self.tmp)

    def test_split_drops_unlisted_records(self):
        samples = self.make_samples(["JS00001", "JS00002", "JS00003", "JS00009"])
        train, val, test = split_by_heartlang_csn(samples, self.tmp)
        self.assertEqual((len(train), len(val), len(test)), (2, 1, 0))
        self.assertEqual(val[0]["record_id"], "JS00003")
        with self.assertRaises(ValueError):
            split_by_heartlang_csn(samples, self.tmp, record_ids=["JS00001"])


if __name__ == "__main__":
    unittest.main()
