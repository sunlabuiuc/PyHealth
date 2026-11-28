"""
Unit tests for PTBXLDataset, PTBXLSuperclassClassification, and
split_by_strat_fold.

Uses small synthetic WFDB fixtures under test-resources/core/ptbxl/ (not real
PhysioNet records). Waveforms are generated at test time. Covers censored
age (300), missing age, and empty diagnostic-superclass labels.

Speed: building processed samples through ``set_task`` is the expensive part
(litdata spawns workers), so the fixture and the dataset objects are built
once per *variant* and shared by every test that needs them, and each class
prepares its sample sets in ``setUpClass``. Tests then assert on cached
objects and run in milliseconds; only the handful of distinct fixture/task
combinations pay the build cost, once.

Author:
    AxelNoun (GitHub: @AxelNoun) — external contributor, no NetID
"""

from __future__ import annotations

import atexit
import functools
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
    compute_ptbxl_lead_stats,
    format_patient_id,
    is_age_censored,
    is_age_missing,
    load_ptbxl_lead_stats,
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
from pyhealth.tasks.ptbxl_multiclass_classification import (
    PTBXLMulticlassClassification,
    compute_multiclass_class_weights,
)
from pyhealth.tasks.ptbxl_multilabel_classification import (
    PTBXLMultilabelClassification,
    compute_multilabel_pos_weight,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "test-resources" / "core" / "ptbxl"
)


def _write_dummy_wfdb(
    record_base: Path,
    n_leads: int = 12,
    n_samples: int = 50,
    fs: int = 100,
    offset: int = 0,
) -> None:
    """Write a minimal WFDB record (header + int16 dat) for tests."""
    record_base.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((n_samples, n_leads), dtype=np.int16)
    for lead in range(n_leads):
        data[:, lead] = np.arange(n_samples, dtype=np.int16) + lead + offset
    Path(str(record_base) + ".dat").write_bytes(data.tobytes())
    name = record_base.name
    lines = [f"{name} {n_leads} {fs} {n_samples}"]
    for i in range(n_leads):
        lines.append(f"{name}.dat 16 1000.0(0)/uV 16 0 0 0 0 {i}")
    Path(str(record_base) + ".hea").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


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


def _materialize_long_fixture(dest: Path) -> Path:
    """Same as ``_materialize_fixture`` with 10 s records (1000 / 5000 samples)."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_ROOT / "ptbxl_database.csv", dest / "ptbxl_database.csv")
    shutil.copy(FIXTURE_ROOT / "scp_statements.csv", dest / "scp_statements.csv")
    db = pd.read_csv(dest / "ptbxl_database.csv")
    for col, fs, n_samples in (("filename_lr", 100, 1000), ("filename_hr", 500, 5000)):
        for rel in db[col]:
            _write_dummy_wfdb(dest / str(rel), n_samples=n_samples, fs=fs)
    return dest


def _set_scp_codes(root: Path, ecg_id: int, codes: str) -> None:
    db = pd.read_csv(root / "ptbxl_database.csv")
    db.loc[db["ecg_id"] == ecg_id, "scp_codes"] = codes
    db.to_csv(root / "ptbxl_database.csv", index=False)


def _apply_variant(root: Path, variant: str) -> None:
    """Extra edits that make a fixture variant exercise a specific code path."""
    if variant == "afib":
        # A target-diagnosis record, and one record that carries NORM *and* the
        # target, so the 3-way task sees every class plus the precedence rule.
        _set_scp_codes(root, 4, "{'IMI': 100.0, 'LVH': 80.0, 'AFIB': 60.0}")
        _set_scp_codes(root, 2, "{'NORM': 100.0, 'AFIB': 60.0}")
    elif variant == "changed":  # source CSV differs: derived metadata must re-key
        _set_scp_codes(root, 1, "{'LVH': 100.0}")
    elif variant == "offset":  # fold 9/10 waveforms shifted, so train fits differ
        for rec in ("00002_lr", "00003_lr"):
            _write_dummy_wfdb(
                root / "records100" / "00000" / rec, n_samples=50, fs=100, offset=400
            )


_FIXTURES: dict[str, Path] = {}
_SHARED_TMP = Path(tempfile.mkdtemp(prefix="ptbxl-fixture-"))
atexit.register(shutil.rmtree, _SHARED_TMP, ignore_errors=True)


def fixture(variant: str = "base") -> Path:
    """Materialize a fixture variant once per run and reuse it everywhere."""
    if variant not in _FIXTURES:
        dest = _SHARED_TMP / variant
        if variant == "long":
            _materialize_long_fixture(dest)
        else:
            _materialize_fixture(dest)
        _apply_variant(dest, variant)
        _FIXTURES[variant] = dest
    return _FIXTURES[variant]


@functools.lru_cache(maxsize=None)
def dataset_for(variant: str = "base", sampling_rate: int = 100) -> PTBXLDataset:
    """One dataset per (variant, rate), so its processed-sample cache is shared."""
    return PTBXLDataset(
        root=str(fixture(variant)),
        metadata_cache_dir=_SHARED_TMP / f"meta-{variant}-{sampling_rate}",
        sampling_rate=sampling_rate,
        cache_dir=_SHARED_TMP / f"ds-{variant}-{sampling_rate}",
    )


def class_tmp(name: str) -> Path:
    """A class's scratch directory, reused across that class's tests."""
    path = _SHARED_TMP / f"class-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def multiclass_task(target_dx: str) -> PTBXLMulticlassClassification:
    """One task spelling, so every class asking for it shares one sample cache."""
    return PTBXLMulticlassClassification(target_dx=target_dx, standardize=None)


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
    """Derived-metadata naming and content; no event table is built here."""

    @classmethod
    def setUpClass(cls):
        cls.data_root = fixture("base")
        cls.dataset = dataset_for("base")
        cls.tmp = class_tmp(cls.__name__)
        cls.cache_dir = cls.tmp / "meta_cache"
        # Reading the property publishes the dataset's event table (a Dask
        # cluster spin-up) once for every class that shares this dataset.
        _ = cls.dataset.global_event_df

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

    def test_500_hz_signal_file_points_at_records500(self):
        ds = PTBXLDataset(
            root=str(self.data_root),
            metadata_cache_dir=self.cache_dir,
            sampling_rate=500,
            cache_dir=self.tmp / "c500",
        )
        meta = pd.read_csv(self.cache_dir / ds.metadata_file_name)
        self.assertEqual(len(meta), 5)
        self.assertEqual(set(meta["sampling_rate"]), {500})
        # 500 Hz must read filename_hr (records500/…_hr), never filename_lr.
        records500 = (self.data_root / "records500").resolve()
        for path in meta["signal_file"]:
            Path(str(path)).relative_to(records500)
            self.assertTrue(Path(str(path)).name.endswith("_hr"))

    def test_end_to_end_reads_event_from_fixture(self):
        """Instantiate PTBXLDataset and read a real event (not just helpers)."""
        # The shared base dataset, so the event table is built once per run.
        ds = self.dataset
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
    @classmethod
    def setUpClass(cls):
        cls.data_root = fixture("base")

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


class TestPTBXLTaskOnFixture(unittest.TestCase):
    """Runs the superclass task over fixture patients (waveforms mocked)."""

    @classmethod
    def setUpClass(cls):
        cls.data_root = fixture("base")
        cls.dataset = dataset_for("base")
        cls.scp_path = cls.data_root / "scp_statements.csv"

    def test_task_drops_empty_labels_by_default(self):
        fake_signal = np.zeros((12, 50), dtype=np.float32)

        def _fake_load(_path):
            return fake_signal

        task = PTBXLSuperclassClassification(
            scp_statements_path=self.scp_path,
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

    @classmethod
    def setUpClass(cls):
        """Build each sample set once; the tests below only assert on them."""
        cls.data_root = fixture("base")
        cls.dataset = dataset_for("base")
        cls.task = PTBXLSuperclassClassification(
            scp_statements_path=cls.data_root / "scp_statements.csv",
        )
        cls.samples = cls.dataset.set_task(cls.task, num_workers=1)

        cls.dataset_500 = dataset_for("base", 500)
        cls.samples_500 = cls.dataset_500.set_task(cls.task, num_workers=1)

        cls.samples_kept = cls.dataset.set_task(
            PTBXLSuperclassClassification(
                scp_statements_path=cls.data_root / "scp_statements.csv",
                drop_empty_labels=False,
            ),
            num_workers=1,
        )

    def test_set_task_builds_samples_without_mocks(self):
        samples = self.samples
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

    def test_set_task_at_500_hz_loads_records500(self):
        samples = self.samples_500
        self.assertEqual(len(samples), 3)
        # Fixture records are 12 x 250 at 500 Hz but 12 x 50 at 100 Hz, so a
        # signal_file pointing at records100 or a missing transpose fails here.
        for sample in samples:
            self.assertEqual(tuple(sample["signal"].shape), (12, 250))

    def test_multilabel_vocab_order_is_alphabetical_observed(self):
        samples = self.samples
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
        # Kept folds are 1 (ecg 1), 2 (ecg 4), 9 (ecg 2). Fold 10 was dropped.
        train, val, test = split_by_strat_fold(
            self.samples,
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
        # Fixture: ecg 1 (fold 1) and ecg 2 (fold 9) share patient_id 15709.
        with self.assertRaisesRegex(ValueError, r"15709"):
            split_by_strat_fold(
                self.samples,
                train_folds=(1,),
                val_folds=(2,),
                test_folds=(9,),
                check_patient_disjoint=True,
            )

    def test_age_sentinel_through_set_task(self):
        """Missing age must be an int sentinel; censored age is clipped to 90."""
        by_record = {str(s["record_id"]): s for s in self.samples_kept}

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
    """Derived metadata is keyed by data root *and* source-CSV contents."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = class_tmp(cls.__name__)

    def test_distinct_data_roots_get_distinct_cache_dirs(self):
        # "changed" is the same fixture with one record's scp_codes rewritten.
        root_a = fixture("base")
        root_b = fixture("changed")

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
        # Its own copy: this test edits the source CSV in place, which would
        # otherwise change the shared "base" fixture for other classes.
        data_root = _materialize_fixture(self.tmp / "data")
        meta = self.tmp / "meta-change"
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
    @classmethod
    def setUpClass(cls):
        cls.data_root = fixture("base")
        cls.dataset = dataset_for("base")
        cls.scp_path = cls.data_root / "scp_statements.csv"

    def test_vars_task_stable_after_call(self):
        task = PTBXLSuperclassClassification(
            scp_statements_path=self.scp_path,
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


try:
    import wfdb as _wfdb  # noqa: F401
    _WFDB_AVAILABLE = True
except ImportError:
    _WFDB_AVAILABLE = False


@unittest.skipUnless(
    _WFDB_AVAILABLE,
    "wfdb not installed; install the 'pyhealth[ptbxl]' extra to run these tests",
)
class TestPTBXLMultilabelClassification(unittest.TestCase):
    """Tests for PTBXLMultilabelClassification wired to the master dataset."""

    @classmethod
    def setUpClass(cls):
        cls.root = fixture("long")
        cls.dataset = dataset_for("long")
        cls.scp_path = cls.dataset.scp_statements_path

        def _multi(label_set: str):
            return PTBXLMultilabelClassification(
                scp_statements_path=cls.scp_path, label_set=label_set
            )

        # One litdata build per label set; the tests below only assert.
        cls.samples_superdiagnostic = cls.dataset.set_task(_multi("superdiagnostic"))
        cls.samples_rhythm = cls.dataset.set_task(_multi("rhythm"))
        cls.samples_diagnostic = cls.dataset.set_task(_multi("diagnostic"))
        cls.samples_superclass = cls.dataset.set_task(
            PTBXLSuperclassClassification(
                scp_statements_path=cls.scp_path, drop_empty_labels=True
            )
        )

    def test_superdiagnostic_samples_shape(self):
        samples = self.samples_superdiagnostic
        self.assertGreater(len(samples), 0)
        item = samples[0]
        self.assertEqual(tuple(item["signal"].shape), (12, 1000))

    def test_rhythm_skips_non_rhythm_ecgs(self):
        samples = self.samples_rhythm
        self.assertGreater(len(samples), 0)
        for item in samples:
            self.assertTrue(len(item["labels"]) > 0)

    def test_invalid_label_set_raises(self):
        from pyhealth.tasks import PTBXLMultilabelClassification
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(
                scp_statements_path=self.scp_path, label_set="nope"
            )

    def test_invalid_standardize_raises(self):
        from pyhealth.tasks import PTBXLMultilabelClassification
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(
                scp_statements_path=self.scp_path, standardize="dataset"
            )

    def test_prepare_signal_shape_and_dtype(self):
        from pyhealth.tasks import PTBXLMultilabelClassification
        task = PTBXLMultilabelClassification(
            scp_statements_path=self.scp_path, remove_baseline=True
        )
        rec = str(self.root / "records100" / "00000" / "00001_lr")
        out = task._prepare_signal(rec, 100)
        self.assertEqual(out.shape, (12, 1000))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

    def test_prepare_signal_500hz(self):
        from pyhealth.tasks import PTBXLMultilabelClassification
        task = PTBXLMultilabelClassification(scp_statements_path=self.scp_path)
        rec = str(self.root / "records500" / "00000" / "00001_hr")
        out = task._prepare_signal(rec, 500)
        self.assertEqual(out.shape, (12, 5000))

    def test_aggregate_labels_likelihood_filter(self):
        from pyhealth.tasks.ptbxl_multilabel_classification import (
            _aggregate_labels,
            _load_scp_statements,
        )
        stmts = _load_scp_statements(str(self.scp_path))
        labels = _aggregate_labels(
            {"IMI": 80.0, "NORM": 100.0}, stmts, "superdiagnostic", 0.0
        )
        self.assertIn("MI", labels)
        self.assertIn("NORM", labels)
        labels_hi = _aggregate_labels(
            {"IMI": 80.0, "NORM": 50.0}, stmts, "superdiagnostic", 75.0
        )
        self.assertIn("MI", labels_hi)
        self.assertNotIn("NORM", labels_hi)

    def test_all_label_set_includes_rhythm(self):
        from pyhealth.tasks.ptbxl_multilabel_classification import (
            _aggregate_labels,
            _load_scp_statements,
        )
        stmts = _load_scp_statements(str(self.scp_path))
        # IMI is diagnostic; SR is rhythm — both appear under "all"
        labels = _aggregate_labels({"IMI": 80.0, "SR": 100.0}, stmts, "all", 0.0)
        self.assertIn("IMI", labels)
        self.assertIn("SR", labels)

    def test_diagnostic_label_set_excludes_rhythm(self):
        from pyhealth.tasks.ptbxl_multilabel_classification import (
            _aggregate_labels,
            _load_scp_statements,
        )
        stmts = _load_scp_statements(str(self.scp_path))
        labels = _aggregate_labels({"IMI": 80.0, "SR": 100.0}, stmts, "diagnostic", 0.0)
        self.assertIn("IMI", labels)
        self.assertNotIn("SR", labels)

    def test_subdiagnostic_label_set_maps_to_subclass(self):
        from pyhealth.tasks.ptbxl_multilabel_classification import (
            _aggregate_labels,
            _load_scp_statements,
        )
        stmts = _load_scp_statements(str(self.scp_path))
        # IMI → diagnostic_subclass=IMI, LVH → diagnostic_subclass=LVH
        labels = _aggregate_labels(
            {"IMI": 100.0, "LVH": 80.0}, stmts, "subdiagnostic", 0.0
        )
        self.assertIn("IMI", labels)
        self.assertIn("LVH", labels)
        # SR has no diagnostic==1, so should not appear
        labels_rhythm = _aggregate_labels({"SR": 100.0}, stmts, "subdiagnostic", 0.0)
        self.assertEqual(labels_rhythm, [])

    def test_form_label_set_empty_in_fixture(self):
        from pyhealth.tasks.ptbxl_multilabel_classification import (
            _aggregate_labels,
            _load_scp_statements,
        )
        stmts = _load_scp_statements(str(self.scp_path))
        # Fixture scp_statements has no form==1 codes
        labels = _aggregate_labels({"NORM": 100.0, "IMI": 80.0}, stmts, "form", 0.0)
        self.assertEqual(labels, [])

    def test_superdiagnostic_matches_superclass_task(self):
        """When label_set='superdiagnostic' and min_likelihood=0, labels match
        PTBXLSuperclassClassification record-for-record.

        Guards the Blocker-1 refactor that delegates the overlapping path to
        aggregate_diagnostic_superclasses.
        """
        multi_samples = {
            (s["patient_id"], s["strat_fold"]): sorted(s["labels"])
            for s in self.samples_superdiagnostic
        }
        super_samples = {
            (s["patient_id"], s["strat_fold"]): sorted(s["labels"])
            for s in self.samples_superclass
        }
        self.assertEqual(set(multi_samples.keys()), set(super_samples.keys()))
        for key, labels in multi_samples.items():
            self.assertEqual(labels, super_samples[key])

    def test_diagnostic_samples_labels_nonempty(self):
        self.assertGreater(len(self.samples_diagnostic), 0)
        for item in self.samples_diagnostic:
            self.assertGreater(len(item["labels"]), 0)
            self.assertIn("strat_fold", item)

    def test_standardize_record_versus_none(self):
        """'record' forces per-lead mean≈0 and std≈1; None does not."""
        from pyhealth.tasks import PTBXLMultilabelClassification
        rec = str(self.root / "records100" / "00000" / "00004_lr")

        task_std = PTBXLMultilabelClassification(scp_statements_path=self.scp_path)
        out_std = task_std._prepare_signal(rec, 100)
        np.testing.assert_allclose(out_std.mean(axis=1), 0.0, atol=1e-5)
        np.testing.assert_allclose(out_std.std(axis=1), 1.0, atol=1e-3)

        task_raw = PTBXLMultilabelClassification(
            scp_statements_path=self.scp_path, standardize=None
        )
        out_raw = task_raw._prepare_signal(rec, 100)
        self.assertEqual(out_raw.shape, (12, 1000))
        # The fixture signals are not centered, so standardize=None keeps a
        # per-lead mean clearly away from zero for at least one lead.
        self.assertTrue((np.abs(out_raw.mean(axis=1)) > 0.01).any())

    def test_remove_baseline_changes_signal(self):
        """remove_baseline=True must perturb the output vs. remove_baseline=False."""
        from pyhealth.tasks import PTBXLMultilabelClassification
        rec = str(self.root / "records100" / "00000" / "00004_lr")

        task_no_bl = PTBXLMultilabelClassification(
            scp_statements_path=self.scp_path,
            standardize=None,
            remove_baseline=False,
        )
        task_bl = PTBXLMultilabelClassification(
            scp_statements_path=self.scp_path,
            standardize=None,
            remove_baseline=True,
        )
        out_no_bl = task_no_bl._prepare_signal(rec, 100)
        out_bl = task_bl._prepare_signal(rec, 100)
        self.assertEqual(out_bl.shape, out_no_bl.shape)
        self.assertFalse(np.allclose(out_no_bl, out_bl))
        # A high-pass filter removes DC; so the per-lead mean of the filtered
        # output is closer to zero than the unfiltered one.
        self.assertLess(
            float(np.abs(out_bl.mean(axis=1)).max()),
            float(np.abs(out_no_bl.mean(axis=1)).max()),
        )

    def test_load_signal_matches_superclass_task_signal(self):
        """Both tasks must load the exact same signal tensor from the same record.

        The multi-label task now uses load_ptbxl_record directly (same as the
        superclass task), so there is one decoding path and the signals match.
        """
        from pyhealth.datasets.ptbxl import load_ptbxl_record
        from pyhealth.tasks import PTBXLMultilabelClassification

        task = PTBXLMultilabelClassification(
            scp_statements_path=self.scp_path, standardize=None,
        )
        rec = str(self.root / "records100" / "00000" / "00001_lr")
        prepped = task._prepare_signal(rec, 100)
        canonical = load_ptbxl_record(rec)[:, :1000]
        np.testing.assert_allclose(prepped, canonical, atol=1e-5)


@unittest.skipUnless(
    _WFDB_AVAILABLE,
    "wfdb not installed; install the 'pyhealth[ptbxl]' extra to run these tests",
)
class TestPTBXLTrainLeadStats(unittest.TestCase):
    """compute_ptbxl_lead_stats / load_ptbxl_lead_stats artifacts."""

    @classmethod
    def setUpClass(cls):
        # The "offset" variant shifts the fold-9/10 waveforms, so train-fold
        # subsets produce genuinely different statistics (the default dummies
        # are identical).
        cls.root = fixture("offset")
        cls.dataset = dataset_for("offset")
        cls.tmp = class_tmp(cls.__name__)

    def _manual_stats(self, folds):
        meta_path = (
            self.dataset.metadata_cache_dir / self.dataset.metadata_file_name
        )
        meta = pd.read_csv(meta_path)
        rows = meta[meta["strat_fold"].astype(int).isin(folds)]
        signals = [
            load_ptbxl_record(str(p)).astype(np.float64)
            for p in rows["signal_file"]
        ]
        return signals

    def test_stats_match_manual_per_lead_calculation(self):
        path = compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 2, 3))
        mean, std = load_ptbxl_lead_stats(path)
        signals = self._manual_stats((1, 2, 3))
        # Fixture folds 1/2/3 are ecg 1, 4, 5 (three records).
        self.assertEqual(len(signals), 3)
        cat = np.concatenate(signals, axis=1)
        np.testing.assert_allclose(mean, cat.mean(axis=1), rtol=1e-5)
        np.testing.assert_allclose(std, cat.std(axis=1), rtol=1e-5)

    def test_stats_exclude_non_train_folds(self):
        all_mean, _ = load_ptbxl_lead_stats(
            compute_ptbxl_lead_stats(
                self.dataset, train_folds=(1, 2, 3, 9, 10)
            )
        )
        partial_mean, _ = load_ptbxl_lead_stats(
            compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 2, 3))
        )
        # Fold 9/10 records differ, so the two fits must differ.
        self.assertFalse(np.allclose(all_mean, partial_mean))

    def test_stats_cached_and_fold_keyed(self):
        path_a = compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 2, 3))
        again = compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 2, 3))
        self.assertEqual(path_a, again)
        self.assertTrue(path_a.is_file())
        path_b = compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 9, 10))
        self.assertNotEqual(path_a, path_b)
        compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 2, 3), force=True)
        self.assertTrue(path_a.is_file())

    def test_stats_empty_or_unknown_folds_raise(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compute_ptbxl_lead_stats(self.dataset, train_folds=())
        with self.assertRaisesRegex(ValueError, "No records"):
            compute_ptbxl_lead_stats(self.dataset, train_folds=(99,))

    def test_load_stats_missing_file_raises(self):
        missing = Path(self.tmp) / "nope.npz"
        with self.assertRaises(FileNotFoundError):
            load_ptbxl_lead_stats(missing)

    def test_multilabel_train_stats_matches_manual(self):
        path = compute_ptbxl_lead_stats(self.dataset, train_folds=(1, 2, 3))
        mean, std = load_ptbxl_lead_stats(path)
        task = PTBXLMultilabelClassification(
            scp_statements_path=self.root / "scp_statements.csv",
            standardize="train_stats",
            stats_path=path,
        )
        rec = str(self.root / "records100" / "00000" / "00002_lr")
        raw = load_ptbxl_record(rec).astype(np.float64)
        # _prepare_signal pads to 10 * rate before normalizing.
        padded = np.pad(raw, ((0, 0), (0, 1000 - raw.shape[1])))
        out = task._prepare_signal(rec, 100)
        expected = (padded - mean[:, None]) / (std[:, None] + 1e-6)
        np.testing.assert_allclose(out, expected, atol=1e-5)
        # Train-fit stats do NOT force this record's own mean to zero.
        self.assertGreater(float(np.abs(out.mean(axis=1)).max()), 1e-3)

    def test_multilabel_train_stats_requires_stats_path(self):
        with self.assertRaisesRegex(ValueError, "stats_path is required"):
            PTBXLMultilabelClassification(
                scp_statements_path=self.root / "scp_statements.csv",
                standardize="train_stats",
            )
        with self.assertRaises(FileNotFoundError):
            PTBXLMultilabelClassification(
                scp_statements_path=self.root / "scp_statements.csv",
                standardize="train_stats",
                stats_path=Path(self.tmp) / "missing.npz",
            )


@unittest.skipUnless(
    _WFDB_AVAILABLE,
    "wfdb not installed; install the 'pyhealth[ptbxl]' extra to run these tests",
)
class TestPTBXLMulticlassClassification(unittest.TestCase):
    """PTBXLMulticlassClassification over fixture variants with AFIB."""

    @classmethod
    def setUpClass(cls):
        # The "afib" variant: ecg 4 carries the target diagnosis and ecg 2
        # carries NORM *and* the target, so one build covers all three classes
        # plus the precedence rule.
        cls.root = fixture("afib")
        cls.dataset = dataset_for("afib")
        cls.samples_af = cls.dataset.set_task(multiclass_task("AF"), num_workers=1)
        cls.samples_lbbb = cls.dataset.set_task(
            multiclass_task("LBBB"), num_workers=1
        )

    def test_labels_and_empty_drop(self):
        samples = self.samples_af
        # ecg 5 (empty scp_codes) is dropped; the other four are kept.
        self.assertEqual(len(samples), 4)
        by_record = {str(s["record_id"]): int(s["label"]) for s in samples}
        self.assertEqual(by_record["1"], 0)  # NORM -> normal
        self.assertEqual(by_record["2"], 1)  # NORM + AFIB -> target
        self.assertEqual(by_record["3"], 2)  # PACE -> other
        self.assertEqual(by_record["4"], 1)  # AFIB -> target
        for s in samples:
            self.assertEqual(tuple(s["signal"].shape), (12, 1000))
            self.assertIn("strat_fold", s)
        vocab = samples.output_processors["label"].label_vocab
        # Fixed class order, matching the benchmark loss layout.
        self.assertEqual(vocab, {"normal": 0, "target": 1, "other": 2})

    def test_target_takes_precedence_over_normal(self):
        # ecg 2 is labelled NORM *and* AFIB: the target class wins.
        by_record = {str(s["record_id"]): int(s["label"]) for s in self.samples_af}
        self.assertEqual(by_record["2"], 1)

    def test_target_dx_for_other_diagnosis(self):
        by_record = {str(s["record_id"]): int(s["label"]) for s in self.samples_lbbb}
        # No LBBB in the fixture: only normal (1) and other (2).
        self.assertEqual(by_record["1"], 0)
        self.assertEqual(by_record["4"], 2)

    def test_invalid_target_dx_raises(self):
        with self.assertRaisesRegex(ValueError, "target_dx must be one of"):
            PTBXLMulticlassClassification(target_dx="STE")

    def test_paper_fold_split_constant_shape(self):
        from pyhealth.tasks.ptbxl_multiclass_classification import (
            PTBXL_PAPER_FOLD_SPLITS,
        )
        self.assertEqual(sorted(PTBXL_PAPER_FOLD_SPLITS), [1, 2, 3, 4, 5])
        for val_fold, test_fold in PTBXL_PAPER_FOLD_SPLITS.values():
            self.assertNotEqual(val_fold, test_fold)
            self.assertIn(val_fold, range(1, 11))
            self.assertIn(test_fold, range(1, 11))


class TestComputeMulticlassClassWeights(unittest.TestCase):
    """compute_multiclass_class_weights from string and processed labels."""

    def test_string_labels(self):
        samples = [
            {"label": "normal"}, {"label": "normal"},
            {"label": "target"},
            {"label": "other"}, {"label": "other"},
            {"label": "other"},
        ]
        self.assertEqual(
            compute_multiclass_class_weights(samples), [1.0, 0.5, 1.5]
        )

    def test_processed_index_labels_from_fixture(self):
        # Same dataset and task spelling as TestPTBXLMulticlassClassification,
        # so this reads that class's cached samples instead of rebuilding.
        samples = dataset_for("afib").set_task(multiclass_task("AF"), num_workers=1)
        # ecg 1 normal; ecgs 2 and 4 target; ecg 3 other (ecg 5 dropped).
        weights = compute_multiclass_class_weights(samples)
        self.assertEqual(weights, [1.0, 2.0, 1.0])

    def test_missing_normal_raises(self):
        samples = [{"label": "target"}, {"label": "other"}]
        with self.assertRaisesRegex(ValueError, "no 'normal' samples"):
            compute_multiclass_class_weights(samples)

    def test_unknown_label_raises(self):
        with self.assertRaisesRegex(ValueError, "not in"):
            compute_multiclass_class_weights([{"label": "NORM"}])
        with self.assertRaisesRegex(ValueError, "outside"):
            compute_multiclass_class_weights([{"label": 7}])


class TestComputeMultilabelPosWeight(unittest.TestCase):
    """compute_multilabel_pos_weight from label-name lists and vectors."""

    def test_raw_label_lists_sorted_vocab(self):
        samples = [
            {"labels": ["NORM", "ST_"]}, {"labels": ["NORM"]},
            {"labels": ["NORM"]}, {"labels": ["ST_"]},
            {"labels": ["CD_"]}, {"labels": ["CD_", "NORM"]},
        ]
        # Sorted vocab [CD_, NORM, ST_]; n=6: (3/3, 2/4, 3/3).
        self.assertEqual(
            compute_multilabel_pos_weight(samples), [2.0, 0.5, 2.0]
        )

    def test_multi_hot_vectors(self):
        samples = [
            {"labels": [1, 0]}, {"labels": [1, 1]},
            {"labels": [0, 0]}, {"labels": [1, 0]},
        ]
        # n=4: column 0 has 3 positives -> 1/3; column 1 has 1 -> 3/1.
        weights = compute_multilabel_pos_weight(samples)
        self.assertAlmostEqual(weights[0], 1 / 3)
        self.assertEqual(weights[1], 3.0)

    def test_always_positive_and_never_positive_columns(self):
        samples = [
            {"labels": [1, 0, 1]}, {"labels": [1, 0, 1]},
            {"labels": [1, 0, 1]},
        ]
        # Column 1 has no positives (weight 1.0, cannot be learned from
        # these samples); column 2 has no negatives (weight 0.0).
        self.assertEqual(
            compute_multilabel_pos_weight(samples), [0.0, 1.0, 0.0]
        )

    def test_empty_or_unlabelled_samples_raise(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compute_multilabel_pos_weight([])
        with self.assertRaisesRegex(ValueError, "no labels found"):
            compute_multilabel_pos_weight([{"labels": []}])

    def test_custom_label_key(self):
        samples = [{"y": ["a"]}, {"y": ["a", "b"]}, {"y": ["b"]}]
        weights = compute_multilabel_pos_weight(samples, label_key="y")
        self.assertEqual(weights, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
