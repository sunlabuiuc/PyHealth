"""Fast synthetic tests for LUDB dataset, ECG delineation task, and ECGCODE model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from pyhealth.data.data import Event
from pyhealth.datasets.ludb import (
    FS as LUDB_FS,
)
from pyhealth.datasets.ludb import (
    LEADS,
    N_SAMPLES,
    LUDBDataset,
    get_stratified_ludb_split,
)
from pyhealth.tasks.ecg_delineation import (
    ECGDelineationLUDB,
    _build_segmentation_mask,
    _extract_extension,
    _has_any_wave,
    _parse_annotations,
    _safe_str,
    ecg_delineation_ludb_fn,
    get_ecg_delineation_ludb_task,
)

SIGNAL_LEN = LUDB_FS * 10  # 10 seconds


def _write_synthetic_record(data_dir: Path, record_id: int) -> None:
    """Write one synthetic 12-lead WFDB record."""
    pid = str(record_id)
    n_leads = len(LEADS)
    t = np.linspace(0, 2 * np.pi, SIGNAL_LEN)
    signal = (0.01 * np.sin(t)[:, np.newaxis] * np.ones((1, n_leads))).astype(
        np.float64
    )

    wfdb.wrsamp(
        record_name=pid,
        fs=LUDB_FS,
        units=["mV"] * n_leads,
        sig_name=list(LEADS),
        p_signal=signal,
        write_dir=str(data_dir),
    )


def _write_triplet_annotation(data_dir: Path, record_id: int, lead: str) -> None:
    """Write one P-QRS-T annotation triplet."""
    pid = str(record_id)
    ann_samples = np.array(
        [
            90,
            100,
            110,  # P
            240,
            250,
            260,  # QRS
            390,
            400,
            410,  # T
        ],
        dtype=np.int32,
    )
    ann_symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]

    wfdb.wrann(
        record_name=pid,
        extension=lead,
        sample=ann_samples,
        symbol=ann_symbols,
        write_dir=str(data_dir),
    )


def _write_qrs_only_annotation(data_dir: Path, record_id: int, lead: str) -> None:
    """Write QRS-only annotations (for incomplete pulse edge-case tests)."""
    pid = str(record_id)
    ann_samples = np.array([240, 250, 260], dtype=np.int32)
    ann_symbols = ["(", "N", ")"]

    wfdb.wrann(
        record_name=pid,
        extension=lead,
        sample=ann_samples,
        symbol=ann_symbols,
        write_dir=str(data_dir),
    )


@dataclass
class _MockEvent:
    visit_id: str
    signal_file: str
    lead_i: str | None = None
    lead_ii: str | None = None
    lead_iii: str | None = None
    lead_avr: str | None = None
    lead_avl: str | None = None
    lead_avf: str | None = None
    lead_v1: str | None = None
    lead_v2: str | None = None
    lead_v3: str | None = None
    lead_v4: str | None = None
    lead_v5: str | None = None
    lead_v6: str | None = None


class _MockPatient:
    def __init__(self, patient_id: str, event: _MockEvent):
        self.patient_id = patient_id
        self._event = event

    def get_events(self, event_type=None, *args, **kwargs):
        if event_type is not None:
            assert event_type == "ludb"
        return [self._event]


@pytest.fixture(scope="module")
def synthetic_ludb_root(tmp_path_factory) -> Path:
    """Create a tiny LUDB-like root with 3 records and minimal annotations."""
    root = tmp_path_factory.mktemp("ludb_synth")
    for rid in (1, 2, 3):
        _write_synthetic_record(root, rid)

    # Record 1: complete on i/ii
    _write_triplet_annotation(root, 1, "i")
    _write_triplet_annotation(root, 1, "ii")

    # Record 2: incomplete on i (QRS-only), no ii
    _write_qrs_only_annotation(root, 2, "i")

    # Record 3: complete on i only
    _write_triplet_annotation(root, 3, "i")

    # Optional LUDB CSV for metadata enrichment & split tests
    pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Rhythms": ["Sinus rhythm", "Atrial fibrillation", "Sinus rhythm"],
            "Electric axis of the heart": [
                "Electric axis of the heart: normal",
                "Electric axis of the heart: normal",
                "Electric axis of the heart: left",
            ],
        }
    ).to_csv(root / "ludb.csv", index=False)

    return root


@pytest.fixture()
def strat_csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "ludb.csv"
    # Two groups of 10 each -> deterministic 8/1/1 per group
    df = pd.DataFrame(
        {
            "ID": list(range(1, 21)),
            "Rhythms": ["Sinus rhythm"] * 10 + ["Atrial fibrillation"] * 10,
            "Electric axis of the heart": ["Electric axis of the heart: normal"] * 20,
        }
    )
    df.to_csv(path, index=False)
    return path


class TestHelpersAndParsing:
    def test_safe_str_and_extract_extension(self):
        assert _safe_str(None) is None
        assert _safe_str("  nan ") is None
        assert _safe_str(" abc ") == "abc"

        assert _extract_extension("i") == "i"
        assert _extract_extension("/tmp/1.i") == "i"
        assert _extract_extension("") is None

    def test_has_any_wave(self):
        assert _has_any_wave({"P": [], "QRS": [], "T": []}) is False
        assert (
            _has_any_wave(
                {"P": [{"onset": 1, "peak": 2, "offset": 3}], "QRS": [], "T": []}
            )
            is True
        )

    def test_parse_annotations_returns_expected_keys(self, synthetic_ludb_root: Path):
        waves = _parse_annotations(str(synthetic_ludb_root / "1"), "i")
        assert set(waves.keys()) == {"P", "QRS", "T"}

    def test_parse_annotations_triplets(self, synthetic_ludb_root: Path):
        waves = _parse_annotations(str(synthetic_ludb_root / "1"), "i")
        for wave_name in ("P", "QRS", "T"):
            assert len(waves[wave_name]) == 1
            w = waves[wave_name][0]
            assert set(w.keys()) == {"onset", "peak", "offset"}
            assert w["onset"] < w["peak"] < w["offset"]

    def test_missing_annotation_returns_empty(self, synthetic_ludb_root: Path):
        waves = _parse_annotations(str(synthetic_ludb_root / "1"), "v6")
        assert waves == {"P": [], "QRS": [], "T": []}


class TestSegmentationMask:
    def test_shape_dtype_and_values(self):
        waves = {
            "P": [{"onset": 2, "peak": 3, "offset": 4}],
            "QRS": [{"onset": 7, "peak": 8, "offset": 9}],
            "T": [{"onset": 12, "peak": 13, "offset": 14}],
        }
        mask = _build_segmentation_mask(20, waves)
        assert mask.shape == (20,)
        assert mask.dtype == np.int64
        assert np.all(mask[2:5] == 1)
        assert np.all(mask[7:10] == 2)
        assert np.all(mask[12:15] == 3)
        assert mask[0] == 0

    def test_clamps_out_of_range_boundaries(self):
        waves = {
            "P": [{"onset": -5, "peak": 1, "offset": 2}],
            "QRS": [{"onset": 8, "peak": 9, "offset": 30}],
            "T": [],
        }
        mask = _build_segmentation_mask(10, waves)
        assert np.all(mask[0:3] == 1)
        assert np.all(mask[8:10] == 2)


class TestECGDelineationLUDBTask:
    def _make_patient(self, root: Path, pid: str = "1") -> _MockPatient:
        base = root / pid
        event = _MockEvent(
            visit_id="ecg",
            signal_file=str(base),
            lead_i=str(base.with_suffix(".i")),
            lead_ii=str(base.with_suffix(".ii")),
        )
        return _MockPatient(patient_id=pid, event=event)

    def test_full_record_mode_returns_samples(self, synthetic_ludb_root: Path):
        patient = self._make_patient(synthetic_ludb_root, pid="1")
        task = ECGDelineationLUDB(split_by_pulse=False)
        samples = task(patient)

        # Record 1 has annotations on i and ii
        assert len(samples) == 2
        for s in samples:
            assert {
                "patient_id",
                "visit_id",
                "record_id",
                "lead",
                "signal",
                "mask",
                "label",
            } <= set(s.keys())
            assert s["signal"].shape == (1, SIGNAL_LEN)
            assert s["mask"].shape == (SIGNAL_LEN,)
            assert s["label"] in (0, 1, 2, 3)

    def test_pulse_mode_returns_fixed_windows(self, synthetic_ludb_root: Path):
        patient = self._make_patient(synthetic_ludb_root, pid="1")
        task = ECGDelineationLUDB(split_by_pulse=True, pulse_window=250)
        samples = task(patient)
        assert len(samples) > 0
        for s in samples:
            assert s["signal"].shape == (1, 500)
            assert s["mask"].shape == (500,)

    def test_incomplete_pulse_filtering_edge_case(self, synthetic_ludb_root: Path):
        patient = self._make_patient(synthetic_ludb_root, pid="2")  # QRS-only on lead i
        loose = ECGDelineationLUDB(
            split_by_pulse=True, pulse_window=250, filter_incomplete_pulses=False
        )(patient)
        strict = ECGDelineationLUDB(
            split_by_pulse=True, pulse_window=250, filter_incomplete_pulses=True
        )(patient)
        assert len(loose) >= 1
        assert len(strict) == 0

    def test_bad_record_returns_empty(self):
        patient = _MockPatient(
            patient_id="999",
            event=_MockEvent(
                visit_id="ecg",
                signal_file="/non/existent/record",
                lead_i="/non/existent/record.i",
            ),
        )
        task = ECGDelineationLUDB(split_by_pulse=False)
        assert task(patient) == []

    def test_backward_compat_alias_and_factory(self, synthetic_ludb_root: Path):
        patient = self._make_patient(synthetic_ludb_root, pid="1")
        assert len(ecg_delineation_ludb_fn(patient)) == 2

        task = get_ecg_delineation_ludb_task(split_by_pulse=True, pulse_window=125)
        assert isinstance(task, ECGDelineationLUDB)
        assert task.split_by_pulse is True
        assert task.pulse_window == 125


class TestLUDBDatasetModern:
    def test_metadata_generation_and_columns(self, synthetic_ludb_root: Path):
        ds = LUDBDataset(root=str(synthetic_ludb_root), dev=False, num_workers=1)
        metadata_path = synthetic_ludb_root / "ludb-pyhealth.csv"
        assert metadata_path.exists()

        df = pd.read_csv(metadata_path)
        assert set(
            ["patient_id", "visit_id", "record_id", "signal_file", "fs", "n_samples"]
        ).issubset(df.columns)
        for lead in LEADS:
            assert f"lead_{lead}" in df.columns

        first = df.iloc[0]
        assert Path(str(first["signal_file"])).is_absolute()
        assert int(first["fs"]) == LUDB_FS
        assert int(first["n_samples"]) == N_SAMPLES

        # event parsing/data integrity
        patient = ds.get_patient("1")
        events = cast(list[Event], patient.get_events("ludb", return_df=False))
        assert len(events) == 1
        ev = events[0]
        assert ev.visit_id == "ecg"
        assert ev.record_id == "1"
        assert Path(ev.signal_file).is_absolute()
        assert ev.lead_i.endswith(".i")

    def test_task_integration_with_set_task(
        self, synthetic_ludb_root: Path, tmp_path: Path
    ):
        ds = LUDBDataset(
            root=str(synthetic_ludb_root),
            dev=False,
            num_workers=1,
            cache_dir=str(tmp_path / "cache"),
        )
        task = ECGDelineationLUDB(split_by_pulse=False)
        sample_ds = ds.set_task(task=task, num_workers=1)
        assert len(sample_ds) > 0

        sample = sample_ds[0]
        assert "signal" in sample and "mask" in sample and "label" in sample
        assert isinstance(sample["signal"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["signal"].ndim == 2  # [C, T]
        assert sample["mask"].ndim == 1

    def test_strict_root_resolution_raises_on_empty_root(self, tmp_path: Path):
        empty_root = tmp_path / "empty_ludb"
        empty_root.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError):
            LUDBDataset(root=str(empty_root), dev=True, num_workers=1)

        with pytest.raises(FileNotFoundError):
            LUDBDataset(root=str(empty_root), dev=False, num_workers=1)

    def test_stats_and_info_smoke(self, synthetic_ludb_root: Path):
        ds = LUDBDataset(root=str(synthetic_ludb_root), dev=True, num_workers=1)
        ds.stats()
        ds.info()


class TestStratifiedSplit:
    def test_split_sizes_disjoint_complete_and_deterministic(
        self, strat_csv_path: Path
    ):
        train_ids, val_ids, test_ids = get_stratified_ludb_split(
            str(strat_csv_path), train_ratio=0.8, val_ratio=0.1, seed=42
        )
        assert len(train_ids) == 16
        assert len(val_ids) == 2
        assert len(test_ids) == 2

        all_ids = set(train_ids) | set(val_ids) | set(test_ids)
        assert all_ids == set(range(1, 21))
        assert set(train_ids).isdisjoint(val_ids)
        assert set(train_ids).isdisjoint(test_ids)
        assert set(val_ids).isdisjoint(test_ids)

        split1 = get_stratified_ludb_split(str(strat_csv_path), seed=123)
        split2 = get_stratified_ludb_split(str(strat_csv_path), seed=123)
        assert split1 == split2
