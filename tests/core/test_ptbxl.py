"""
Unit tests for PTBXLDataset, PTBXLSuperclassClassification, and
split_by_strat_fold.

Uses small synthetic WFDB fixtures under test-resources/ptbxl/ (not real
PhysioNet records). Covers censored age (300), missing age, and empty
diagnostic-superclass labels.

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
    PTBXL_EMPTY_SUPERCLASS_COUNT,
    PTBXLSuperclassClassification,
    aggregate_diagnostic_superclasses,
    load_diagnostic_class_map,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / ".." / "test-resources" / "ptbxl"
FIXTURE_ROOT = FIXTURE_ROOT.resolve()


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
    """Copy committed CSVs and WFDB records under ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_ROOT / "ptbxl_database.csv", dest / "ptbxl_database.csv")
    shutil.copy(FIXTURE_ROOT / "scp_statements.csv", dest / "scp_statements.csv")
    for records_dir in ("records100", "records500"):
        src = FIXTURE_ROOT / records_dir
        if src.is_dir():
            shutil.copytree(src, dest / records_dir, dirs_exist_ok=True)
    # Fallback: synthesize WFDB if committed waveforms are missing.
    db = pd.read_csv(dest / "ptbxl_database.csv")
    for col, fs, n_samples in (
        ("filename_lr", 100, 50),
        ("filename_hr", 500, 250),
    ):
        for rel in db[col]:
            base = dest / str(rel)
            if not Path(str(base) + ".hea").is_file():
                _write_dummy_wfdb(base, n_samples=n_samples, fs=fs)
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

    def test_metadata_filename_includes_rate_and_root(self):
        name_a = metadata_filename(100, "/data/ptb-xl/a")
        name_b = metadata_filename(100, "/data/ptb-xl/b")
        self.assertTrue(name_a.startswith("ptbxl-pyhealth-100hz-"))
        self.assertTrue(name_a.endswith(".csv"))
        self.assertNotEqual(name_a, name_b)
        self.assertNotEqual(
            metadata_filename(100, "/tmp/x"), metadata_filename(500, "/tmp/x")
        )

    def test_empty_superclass_count_documented(self):
        self.assertEqual(PTBXL_EMPTY_SUPERCLASS_COUNT, 407)


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
        self.assertTrue(
            (Path(ds.root) / ds.config.tables["records"].file_path).is_file()
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


@unittest.skipUnless(
    __import__("importlib").util.find_spec("wfdb") is not None,
    "wfdb optional extra not installed",
)
class TestPTBXLSignalIO(unittest.TestCase):
    def test_load_committed_fixture_shape_channels_time(self):
        record = FIXTURE_ROOT / "records100" / "00000" / "00001_lr"
        self.assertTrue(Path(str(record) + ".hea").is_file())
        self.assertTrue(Path(str(record) + ".dat").is_file())
        signal = load_ptbxl_record(record)
        # 12 leads != 50 samples: catches a missing transpose.
        self.assertEqual(signal.shape, (12, 50))
        self.assertNotEqual(signal.shape[0], signal.shape[1])

    def test_load_strips_extension_never_appends(self):
        record = FIXTURE_ROOT / "records100" / "00000" / "00001_lr"
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
