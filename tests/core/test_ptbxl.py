"""
Unit tests for PTBXLDataset, PTBXLSuperclassClassification, and
split_by_strat_fold.

Uses small synthetic WFDB fixtures under test-resources/core/ptbxl/ (not real
PhysioNet records). Waveforms are generated at test time. Covers censored
age (300), missing age, and empty diagnostic-superclass labels.

Author:
    AxelNoun (GitHub: @AxelNoun) — external contributor, no NetID
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets.ptbxl import (
    AGE_CENSOR_SENTINEL,
    PTBXLDataset,
    format_patient_id,
    is_age_censored,
    is_age_missing,
    load_ptbxl_record,
    metadata_filename,
    parse_scp_codes,
)
from pyhealth.datasets.splitter import split_by_strat_fold
from pyhealth.tasks.ptbxl import (
    PTBXLSuperclassClassification,
    _to_bool,
    aggregate_diagnostic_superclasses,
    load_diagnostic_class_map,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "test-resources" / "core" / "ptbxl"
)


def _write_dummy_wfdb(
    record_base: Path,
    n_leads: int = 12,
    n_samples: int = 50,
    fs: int = 100,
) -> None:
    """Write a minimal WFDB record (header + int16 dat) for tests."""
    record_base.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((n_samples, n_leads), dtype=np.int16)
    for lead in range(n_leads):
        data[:, lead] = np.arange(n_samples, dtype=np.int16) + lead
    Path(str(record_base) + ".dat").write_bytes(data.tobytes())
    name = record_base.name
    lines = [f"{name} {n_leads} {fs} {n_samples}"]
    for i in range(n_leads):
        lines.append(f"{name}.dat 16 1000.0(0)/uV 16 0 0 0 0 {i}")
    Path(str(record_base) + ".hea").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_fixture(dest: Path) -> Path:
    """Copy committed CSVs and synthesize WFDB records under ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_ROOT / "ptbxl_database.csv", dest / "ptbxl_database.csv")
    shutil.copy(FIXTURE_ROOT / "scp_statements.csv", dest / "scp_statements.csv")
    db = pd.read_csv(dest / "ptbxl_database.csv")
    for col, fs, n_samples in (
        ("filename_lr", 100, 50),
        ("filename_hr", 500, 250),
    ):
        for rel in db[col]:
            _write_dummy_wfdb(dest / str(rel), n_samples=n_samples, fs=fs)
    return dest


class TestPTBXLHelpers(unittest.TestCase):
    def test_format_patient_id_strips_float(self):
        self.assertEqual(format_patient_id(15709.0), "15709")
        self.assertEqual(format_patient_id("15709.0"), "15709")

    def test_format_patient_id_includes_ecg_id_in_errors(self):
        with self.assertRaisesRegex(ValueError, r"ecg_id=99"):
            format_patient_id(None, ecg_id=99)
        with self.assertRaisesRegex(ValueError, r"ecg_id=7"):
            format_patient_id("not-a-number", ecg_id=7)

    def test_parse_scp_codes_keeps_likelihood_zero(self):
        codes = parse_scp_codes("{'IMI': 80.0, 'SR': 0.0}")
        self.assertEqual(codes["SR"], 0.0)
        self.assertIn("IMI", codes)

    def test_parse_scp_codes_empty(self):
        self.assertEqual(parse_scp_codes("{}"), {})
        self.assertEqual(parse_scp_codes(None), {})

    def test_age_missing_vs_censored(self):
        self.assertTrue(is_age_censored(AGE_CENSOR_SENTINEL))
        self.assertFalse(is_age_missing(AGE_CENSOR_SENTINEL))
        self.assertTrue(is_age_missing(float("nan")))
        self.assertTrue(is_age_missing(""))
        self.assertFalse(is_age_censored(float("nan")))
        self.assertFalse(is_age_censored(65))

    def test_to_bool_parses_flag_cells(self):
        self.assertTrue(_to_bool(1))
        self.assertTrue(_to_bool("True"))
        self.assertTrue(_to_bool("true"))
        self.assertTrue(_to_bool(True))
        self.assertFalse(_to_bool(0))
        self.assertFalse(_to_bool("0"))
        self.assertFalse(_to_bool(False))

    def test_metadata_filename_includes_rate_and_root(self):
        name_a = metadata_filename(100, "/data/ptb-xl/a")
        name_b = metadata_filename(100, "/data/ptb-xl/b")
        self.assertTrue(name_a.startswith("ptbxl-pyhealth-100hz-"))
        self.assertTrue(name_a.endswith(".csv"))
        self.assertNotEqual(name_a, name_b)
        self.assertNotEqual(
            metadata_filename(100, "/tmp/x"), metadata_filename(500, "/tmp/x")
        )
        keyed_a = metadata_filename(100, "/tmp/x", "aaaaaaaaaa")
        keyed_b = metadata_filename(100, "/tmp/x", "bbbbbbbbbb")
        self.assertIn("aaaaaaaaaa", keyed_a)
        self.assertNotEqual(keyed_a, keyed_b)


class TestPTBXLAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_map = load_diagnostic_class_map(FIXTURE_ROOT / "scp_statements.csv")

    def test_aggregate_superclasses(self):
        labels = aggregate_diagnostic_superclasses(
            "{'NORM': 100.0}", self.class_map
        )
        self.assertEqual(labels, ["NORM"])

        labels = aggregate_diagnostic_superclasses(
            "{'IMI': 80.0, 'SR': 0.0}", self.class_map
        )
        self.assertEqual(labels, ["MI"])

        # Non-empty dict with no diagnostic statements (PACE is rhythm-only).
        labels = aggregate_diagnostic_superclasses(
            "{'PACE': 100.0}", self.class_map
        )
        self.assertEqual(labels, [])

        # Empty dict — distinct path from PACE-only.
        labels = aggregate_diagnostic_superclasses("{}", self.class_map)
        self.assertEqual(labels, [])

        # True multi-label: two distinct diagnostic superclasses.
        labels = aggregate_diagnostic_superclasses(
            "{'IMI': 100.0, 'LVH': 80.0}", self.class_map
        )
        self.assertEqual(labels, ["HYP", "MI"])


class TestPTBXLDatasetMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_root = _materialize_fixture(self.tmp / "data")
        self.cache_dir = self.tmp / "meta_cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_prepare_metadata_writes_cache_not_root(self):
        ds = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=100,
            cache_dir=self.tmp / "pyhealth_cache",
        )
        csv_path = self.cache_dir / ds.metadata_file_name
        self.assertTrue(csv_path.is_file())
        self.assertFalse((self.data_root / ds.metadata_file_name).exists())
        meta = pd.read_csv(csv_path)
        self.assertEqual(
            sorted(meta["patient_id"].astype(str).unique()),
            ["15709", "42", "7", "99"],
        )
        self.assertNotIn("15709.0", set(meta["patient_id"].astype(str)))
        self.assertEqual(len(meta), 5)
        # Age flags: record 2 censored, record 3 missing
        row2 = meta[meta["record_id"].astype(str) == "2"].iloc[0]
        row3 = meta[meta["record_id"].astype(str) == "3"].iloc[0]
        self.assertEqual(int(row2["age_is_censored"]), 1)
        self.assertEqual(int(row2["age_is_missing"]), 0)
        self.assertEqual(int(row2["age"]), 300)
        self.assertEqual(int(row3["age_is_missing"]), 1)
        self.assertEqual(int(row3["age_is_censored"]), 0)
        # Signal paths are extension-free record bases
        for path in meta["signal_file"]:
            self.assertFalse(str(path).endswith(".hea"))
            self.assertFalse(str(path).endswith(".dat"))
        self.assertEqual(ds.sampling_rate, 100)

    def test_100_and_500_hz_coexist(self):
        ds = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=100,
            cache_dir=self.tmp / "c100",
        )
        self.assertTrue((self.cache_dir / ds.metadata_file_name).is_file())
        # Second rate gets a distinct filename (rate + root hash).
        ds500 = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=500,
            cache_dir=self.tmp / "c500",
        )
        self.assertTrue((self.cache_dir / ds500.metadata_file_name).is_file())
        self.assertNotEqual(ds.metadata_file_name, ds500.metadata_file_name)

    def test_end_to_end_reads_event_from_fixture(self):
        """Instantiate PTBXLDataset and read a real event (not just helpers)."""
        ds = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=100,
            cache_dir=self.tmp / "pyhealth_cache_e2e",
        )
        # Config must already point at the derived CSV before any load.
        self.assertEqual(
            ds.config.tables["records"].file_path, ds.metadata_file_name
        )
        self.assertEqual(Path(ds.root).resolve(), self.data_root.resolve())
        self.assertTrue(
            (ds.metadata_cache_dir / ds.config.tables["records"].file_path).is_file()
        )

        patient_ids = ds.unique_patient_ids
        self.assertGreaterEqual(len(patient_ids), 1)
        patient = ds.get_patient(patient_ids[0])
        events = patient.get_events(event_type="records")
        self.assertGreaterEqual(len(events), 1)
        event = events[0]
        self.assertTrue(hasattr(event, "signal_file"))
        self.assertTrue(hasattr(event, "strat_fold"))
        self.assertTrue(hasattr(event, "scp_codes"))
        self.assertTrue(str(event.signal_file))
        # Absolute waveform path must live under the fixture data root.
        Path(str(event.signal_file)).resolve().relative_to(self.data_root.resolve())
        # load_table must restore self.root after scanning the cache CSV.
        self.assertEqual(Path(ds.root).resolve(), self.data_root.resolve())


@unittest.skipUnless(
    __import__("importlib").util.find_spec("wfdb") is not None,
    "wfdb optional extra not installed",
)
class TestPTBXLSignalIO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_root = _materialize_fixture(self.tmp / "data")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_fixture_shape_channels_time(self):
        record = self.data_root / "records100" / "00000" / "00001_lr"
        self.assertTrue(Path(str(record) + ".hea").is_file())
        self.assertTrue(Path(str(record) + ".dat").is_file())
        signal = load_ptbxl_record(record)
        # 12 leads != 50 samples: catches a missing transpose.
        self.assertEqual(signal.shape, (12, 50))
        self.assertNotEqual(signal.shape[0], signal.shape[1])

    def test_load_strips_extension_never_appends(self):
        record = self.data_root / "records100" / "00000" / "00001_lr"
        signal = load_ptbxl_record(Path(str(record) + ".hea"))
        self.assertEqual(signal.shape, (12, 50))
        self.assertNotEqual(signal.shape[0], signal.shape[1])


class TestPTBXLTaskAndSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_root = _materialize_fixture(self.tmp / "data")
        self.cache_dir = self.tmp / "meta_cache"
        self.dataset = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=100,
            cache_dir=self.tmp / "pyhealth_cache",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_task_drops_empty_labels_by_default(self):
        fake_signal = np.zeros((12, 50), dtype=np.float32)

        def _fake_load(_path):
            return fake_signal

        task = PTBXLSuperclassClassification(
            scp_statements_path=self.data_root / "scp_statements.csv",
            drop_empty_labels=True,
        )
        with patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record", side_effect=_fake_load
        ):
            samples = []
            for patient in self.dataset.iter_patients():
                samples.extend(task(patient))
        # Keep: 1 NORM, 2 MI, 4 HYP+MI. Drop: 3 PACE-only and 5 empty {}.
        self.assertEqual(len(samples), 3)
        labels = {tuple(s["labels"]) for s in samples}
        self.assertEqual(labels, {("NORM",), ("MI",), ("HYP", "MI")})
        multi = next(s for s in samples if len(s["labels"]) > 1)
        self.assertEqual(multi["labels"], ["HYP", "MI"])
        for s in samples:
            self.assertIn("strat_fold", s)
            self.assertIn("site", s)
            self.assertIn("device", s)
            self.assertIn("sex", s)

    def test_task_keeps_empty_when_disabled(self):
        fake_signal = np.zeros((12, 50), dtype=np.float32)
        task = PTBXLSuperclassClassification(
            scp_statements_path=self.data_root / "scp_statements.csv",
            drop_empty_labels=False,
        )
        with patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record", return_value=fake_signal
        ):
            samples = []
            for patient in self.dataset.iter_patients():
                samples.extend(task(patient))
        self.assertEqual(len(samples), 5)
        empty = [s for s in samples if s["labels"] == []]
        # Both PACE-only and empty-dict {} yield empty superclass sets.
        self.assertEqual(len(empty), 2)

    def test_split_by_strat_fold(self):
        # Lightweight fake SampleDataset-like object
        class _FakeDS:
            def __init__(self, samples):
                self._samples = samples

            def __len__(self):
                return len(self._samples)

            def __getitem__(self, i):
                return self._samples[i]

            def subset(self, indices):
                return _FakeDS([self._samples[i] for i in indices])

        samples = [
            {"strat_fold": 1, "id": "a"},
            {"strat_fold": 9, "id": "b"},
            {"strat_fold": 10, "id": "c"},
            {"strat_fold": 3, "id": "d"},
        ]
        train, val, test = split_by_strat_fold(_FakeDS(samples))
        self.assertEqual([s["id"] for s in train._samples], ["a", "d"])
        self.assertEqual([s["id"] for s in val._samples], ["b"])
        self.assertEqual([s["id"] for s in test._samples], ["c"])

    def test_split_by_strat_fold_check_patient_disjoint(self):
        class _FakeDS:
            def __init__(self, samples):
                self._samples = samples
                self.patient_to_index = {}
                for i, sample in enumerate(samples):
                    self.patient_to_index.setdefault(sample["patient_id"], []).append(
                        i
                    )

            def __len__(self):
                return len(self._samples)

            def __getitem__(self, i):
                return self._samples[i]

            def subset(self, indices):
                return _FakeDS([self._samples[i] for i in indices])

        leaked = [
            {"strat_fold": 1, "patient_id": "15709", "id": "a"},
            {"strat_fold": 9, "patient_id": "15709", "id": "b"},
            {"strat_fold": 10, "patient_id": "99", "id": "c"},
            {"strat_fold": 3, "patient_id": "42", "id": "d"},
        ]
        train, val, test = split_by_strat_fold(_FakeDS(leaked))
        self.assertEqual([s["id"] for s in train._samples], ["a", "d"])
        self.assertEqual([s["id"] for s in val._samples], ["b"])
        self.assertEqual([s["id"] for s in test._samples], ["c"])
        with self.assertRaisesRegex(ValueError, r"15709"):
            split_by_strat_fold(_FakeDS(leaked), check_patient_disjoint=True)

        class _NoMap:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {"strat_fold": 1}

            def subset(self, indices):
                return indices

        with self.assertRaises(TypeError):
            split_by_strat_fold(
                _NoMap(),
                train_folds=(1,),
                val_folds=(9,),
                test_folds=(10,),
                check_patient_disjoint=True,
            )

    def test_split_by_strat_fold_uses_precomputed_folds(self):
        class _BoomDS:
            def __init__(self, n):
                self._n = n

            def __len__(self):
                return self._n

            def __getitem__(self, i):
                raise AssertionError(
                    "dataset[i] must not be called when folds= is provided"
                )

            def subset(self, indices):
                return list(indices)

        train, val, test = split_by_strat_fold(
            _BoomDS(4),
            train_folds=(1, 3),
            val_folds=(9,),
            test_folds=(10,),
            folds=[1, 9, 10, 3],
        )
        self.assertEqual(train, [0, 3])
        self.assertEqual(val, [1])
        self.assertEqual(test, [2])


@unittest.skipUnless(
    __import__("importlib").util.find_spec("wfdb") is not None,
    "wfdb optional extra not installed",
)
class TestPTBXLSetTaskE2E(unittest.TestCase):
    """Exercise the real set_task / litdata / SampleDataset path (no mocks)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_root = _materialize_fixture(self.tmp / "data")
        self.cache_dir = self.tmp / "meta_cache"
        self.dataset = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=100,
            cache_dir=self.tmp / "pyhealth_cache",
        )
        self.task = PTBXLSuperclassClassification(
            scp_statements_path=self.data_root / "scp_statements.csv",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_set_task_builds_samples_without_mocks(self):
        samples = self.dataset.set_task(self.task, num_workers=1)
        # Keep: ecg 1 NORM, 2 MI, 4 HYP+MI. Drop: 3 PACE-only and 5 empty {}.
        self.assertEqual(len(samples), 3)

        sample = samples[0]
        self.assertEqual(tuple(sample["signal"].shape), (12, 50))
        self.assertNotEqual(sample["signal"].shape[0], sample["signal"].shape[1])
        self.assertIn("strat_fold", sample)
        self.assertIn("labels", sample)
        # Multi-hot from MultiLabelProcessor, not a list of strings.
        self.assertEqual(tuple(sample["labels"].shape), (3,))
        self.assertTrue(torch.is_tensor(sample["labels"]))

    def test_multilabel_vocab_order_is_alphabetical_observed(self):
        samples = self.dataset.set_task(self.task, num_workers=1)
        vocab = samples.output_processors["labels"].label_vocab
        # Fixture only observes HYP, MI, NORM — not CD/STTC. Processor sorts.
        self.assertEqual(list(vocab.keys()), ["HYP", "MI", "NORM"])
        self.assertEqual(vocab, {"HYP": 0, "MI": 1, "NORM": 2})

        by_record = {str(s["record_id"]): s for s in samples}
        # ecg 1 = NORM → [0, 0, 1]; ecg 2 = MI → [0, 1, 0]; ecg 4 = HYP+MI → [1, 1, 0]
        self.assertEqual(by_record["1"]["labels"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(by_record["2"]["labels"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(by_record["4"]["labels"].tolist(), [1.0, 1.0, 0.0])

    def test_split_by_strat_fold_on_sample_dataset(self):
        samples = self.dataset.set_task(self.task, num_workers=1)
        # Kept folds are 1 (ecg 1), 2 (ecg 4), 9 (ecg 2). Fold 10 was dropped.
        train, val, test = split_by_strat_fold(
            samples,
            train_folds=(1,),
            val_folds=(2,),
            test_folds=(9,),
        )
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)
        self.assertEqual(int(train[0]["strat_fold"]), 1)
        self.assertEqual(int(val[0]["strat_fold"]), 2)
        self.assertEqual(int(test[0]["strat_fold"]), 9)
        self.assertEqual(tuple(train[0]["signal"].shape), (12, 50))

    def test_split_by_strat_fold_detects_fixture_patient_leak(self):
        samples = self.dataset.set_task(self.task, num_workers=1)
        # Fixture: ecg 1 (fold 1) and ecg 2 (fold 9) share patient_id 15709.
        with self.assertRaisesRegex(ValueError, r"15709"):
            split_by_strat_fold(
                samples,
                train_folds=(1,),
                val_folds=(2,),
                test_folds=(9,),
                check_patient_disjoint=True,
            )

    def test_age_sentinel_through_set_task(self):
        """Missing age must be an int sentinel; censored age is clipped to 90."""
        task = PTBXLSuperclassClassification(
            scp_statements_path=self.data_root / "scp_statements.csv",
            drop_empty_labels=False,
        )
        samples = self.dataset.set_task(task, num_workers=1)
        by_record = {str(s["record_id"]): s for s in samples}

        missing = by_record["3"]
        self.assertTrue(missing["age_is_missing"])
        self.assertIsInstance(missing["age"], (int, np.integer))
        self.assertEqual(int(missing["age"]), -1)
        self.assertIsNotNone(missing["age"])

        censored = by_record["2"]
        self.assertTrue(censored["age_is_censored"])
        self.assertFalse(censored["age_is_missing"])
        self.assertEqual(int(censored["age"]), 90)


class TestPTBXLCacheIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_distinct_data_roots_get_distinct_cache_dirs(self):
        root_a = _materialize_fixture(self.tmp / "a")
        root_b = _materialize_fixture(self.tmp / "b")
        db_b = pd.read_csv(root_b / "ptbxl_database.csv")
        db_b.loc[0, "scp_codes"] = "{'LVH': 100.0}"
        db_b.to_csv(root_b / "ptbxl_database.csv", index=False)

        shared_meta = self.tmp / "meta"
        shared_cache = self.tmp / "pyhealth_cache"
        ds_a = PTBXLDataset(
            root=str(root_a),
            metadata_cache_dir=shared_meta,
            sampling_rate=100,
            cache_dir=shared_cache,
        )
        ds_b = PTBXLDataset(
            root=str(root_b),
            metadata_cache_dir=shared_meta,
            sampling_rate=100,
            cache_dir=shared_cache,
        )
        self.assertNotEqual(ds_a.cache_dir, ds_b.cache_dir)
        self.assertNotEqual(ds_a.dataset_name, ds_b.dataset_name)

    def test_source_csv_change_invalidates_derived_metadata(self):
        data_root = _materialize_fixture(self.tmp / "data")
        meta = self.tmp / "meta"
        ds1 = PTBXLDataset(
            root=str(data_root),
            metadata_cache_dir=meta,
            sampling_rate=100,
            cache_dir=self.tmp / "c1",
        )
        name1 = ds1.metadata_file_name
        derived1 = pd.read_csv(meta / name1)

        db = pd.read_csv(data_root / "ptbxl_database.csv")
        db.loc[0, "scp_codes"] = "{'LVH': 100.0}"
        db.to_csv(data_root / "ptbxl_database.csv", index=False)

        ds2 = PTBXLDataset(
            root=str(data_root),
            metadata_cache_dir=meta,
            sampling_rate=100,
            cache_dir=self.tmp / "c2",
        )
        self.assertNotEqual(ds2.metadata_file_name, name1)
        derived2 = pd.read_csv(meta / ds2.metadata_file_name)
        codes1 = str(
            derived1.loc[derived1["record_id"].astype(str) == "1", "scp_codes"].iloc[0]
        )
        codes2 = str(
            derived2.loc[derived2["record_id"].astype(str) == "1", "scp_codes"].iloc[0]
        )
        self.assertNotEqual(codes1, codes2)


class TestPTBXLTaskStability(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_root = _materialize_fixture(self.tmp / "data")
        self.dataset = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.tmp / "meta_cache",
            sampling_rate=100,
            cache_dir=self.tmp / "pyhealth_cache",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vars_task_stable_after_call(self):
        task = PTBXLSuperclassClassification(
            scp_statements_path=self.data_root / "scp_statements.csv",
        )
        before = dict(vars(task))
        fake_signal = np.zeros((12, 50), dtype=np.float32)
        with patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record", return_value=fake_signal
        ):
            patient = self.dataset.get_patient(self.dataset.unique_patient_ids[0])
            task(patient)
        after = dict(vars(task))
        self.assertEqual(before, after)

    def test_age_sentinel_from_task_call(self):
        task = PTBXLSuperclassClassification(
            scp_statements_path=self.data_root / "scp_statements.csv",
            drop_empty_labels=False,
        )
        fake_signal = np.zeros((12, 50), dtype=np.float32)
        with patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record", return_value=fake_signal
        ):
            samples = []
            for patient in self.dataset.iter_patients():
                samples.extend(task(patient))
        by_record = {str(s["record_id"]): s for s in samples}

        missing = by_record["3"]
        self.assertTrue(missing["age_is_missing"])
        self.assertIsInstance(missing["age"], int)
        self.assertEqual(missing["age"], -1)

        censored = by_record["2"]
        self.assertTrue(censored["age_is_censored"])
        self.assertEqual(censored["age"], 90)

        present = by_record["1"]
        self.assertFalse(present["age_is_missing"])
        self.assertFalse(present["age_is_censored"])
        self.assertEqual(present["age"], 65)


class TestWFDBImportError(unittest.TestCase):
    def test_import_error_points_at_extra(self):
        import builtins

        real_import = builtins.__import__

        def _guard(name, *args, **kwargs):
            if name == "wfdb" or name.startswith("wfdb."):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_guard):
            with self.assertRaises(ImportError) as ctx:
                load_ptbxl_record("/tmp/does-not-matter")
            self.assertIn("pyhealth[ptbxl]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
