"""Focused fast tests for ECG delineation tasks on synthetic WFDB records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import wfdb

from pyhealth.tasks.ecg_delineation import (
    ECGDelineationLUDB,
    ECGDelineationQTDB,
    ECGDelineationTask,
    _build_segmentation_mask,
    _extract_extension,
    _has_any_wave,
    _parse_annotations,
    _resolve_lead_index,
    _safe_str,
)

FS = 250
SIGNAL_LEN = 1000


def _write_record(base_dir: Path, record_name: str, sig_names: list[str]) -> Path:
    t = np.linspace(0, 4 * np.pi, SIGNAL_LEN, dtype=np.float64)
    sig = np.stack([0.01 * np.sin(t + i) for i in range(len(sig_names))], axis=1)
    wfdb.wrsamp(
        record_name=record_name,
        fs=FS,
        units=["mV"] * len(sig_names),
        sig_name=sig_names,
        p_signal=sig,
        write_dir=str(base_dir),
    )
    return base_dir / record_name


def _write_triplet_annotation(base_dir: Path, record_name: str, ext: str) -> None:
    samples = np.array([90, 100, 110, 240, 250, 260, 390, 400, 410], dtype=np.int32)
    symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]
    wfdb.wrann(
        record_name=record_name,
        extension=ext,
        sample=samples,
        symbol=symbols,
        write_dir=str(base_dir),
    )


def _write_qrs_only_annotation(base_dir: Path, record_name: str, ext: str) -> None:
    samples = np.array([240, 250, 260], dtype=np.int32)
    symbols = ["(", "N", ")"]
    wfdb.wrann(
        record_name=record_name,
        extension=ext,
        sample=samples,
        symbol=symbols,
        write_dir=str(base_dir),
    )


@dataclass
class _Event:
    visit_id: str
    signal_file: str
    lead_i: str | None = None
    lead_ii: str | None = None
    lead_0: str | None = None
    lead_1: str | None = None
    ann_pu0: str | None = None
    ann_pu1: str | None = None
    ann_q1c: str | None = None
    ann_qt1: str | None = None
    ann_man: str | None = None
    ann_atr: str | None = None


class _Patient:
    def __init__(self, patient_id: str, event: _Event):
        self.patient_id = patient_id
        self._event = event

    def get_events(self, event_type: str | None = None, *args: Any, **kwargs: Any):
        if event_type is not None:
            assert event_type in {"ludb", "qtdb", "toy"}
        return [self._event]


@pytest.fixture()
def ludb_record(tmp_path: Path):
    base = _write_record(tmp_path, "1", ["i", "ii"])
    _write_triplet_annotation(tmp_path, "1", "i")
    _write_triplet_annotation(tmp_path, "1", "ii")
    return tmp_path, base


@pytest.fixture()
def qtdb_record(tmp_path: Path):
    base = _write_record(tmp_path, "sel100", ["0", "1"])
    # wfdb annotation extensions must be alphabetic-only; use custom
    # extensions to verify field-based fallback in ECGDelineationQTDB.
    _write_triplet_annotation(tmp_path, "sel100", "pua")
    _write_triplet_annotation(tmp_path, "sel100", "pub")
    return tmp_path, base


class TestHelpers:
    def test_safe_str_and_extract_extension(self):
        assert _safe_str(None) is None
        assert _safe_str("  nan ") is None
        assert _safe_str("  ok ") == "ok"

        assert _extract_extension("i") == "i"
        assert _extract_extension("/tmp/a/b/1.pu0") == "pu0"
        assert _extract_extension("") is None

    def test_has_any_wave(self):
        assert _has_any_wave({"P": [], "QRS": [], "T": []}) is False
        assert (
            _has_any_wave(
                {"P": [{"onset": 1, "peak": 2, "offset": 3}], "QRS": [], "T": []}
            )
            is True
        )

    def test_build_segmentation_mask_clamps_and_labels(self):
        waves = {
            "P": [{"onset": -10, "peak": 2, "offset": 3}],
            "QRS": [{"onset": 5, "peak": 6, "offset": 1000}],
            "T": [],
        }
        mask = _build_segmentation_mask(10, waves)
        assert mask.shape == (10,)
        assert mask.dtype == np.int64
        assert np.all(mask[0:4] == 1)
        assert np.all(mask[5:10] == 2)

    def test_parse_annotations_triplets(self, ludb_record):
        _, base = ludb_record
        waves = _parse_annotations(str(base), "i")
        assert set(waves.keys()) == {"P", "QRS", "T"}
        assert len(waves["P"]) == len(waves["QRS"]) == len(waves["T"]) == 1
        assert (
            waves["QRS"][0]["onset"]
            < waves["QRS"][0]["peak"]
            < waves["QRS"][0]["offset"]
        )

    def test_resolve_lead_index_with_fallback(self, ludb_record):
        _, base = ludb_record
        record = wfdb.rdrecord(str(base))
        assert _resolve_lead_index(record, "ii", fallback_idx=0) == 1
        assert _resolve_lead_index(record, "v6", fallback_idx=0) == 0


class TestECGDelineationTask:
    def test_invalid_pulse_window_raises(self):
        with pytest.raises(ValueError, match="pulse_window must be a positive integer"):
            ECGDelineationTask(event_type="toy", leads=["i"], pulse_window=0)

    def test_candidate_extension_priority(self, ludb_record):
        _, base = ludb_record
        event = _Event(
            visit_id="ecg",
            signal_file=str(base),
            lead_i=str(base.with_suffix(".i")),
        )
        task = ECGDelineationTask(
            event_type="toy",
            leads=["i"],
            lead_field_map={"i": "lead_i"},
            annotation_extension_map={"i": ["i", "man", "i"]},
        )
        # should deduplicate while preserving order
        exts = task._candidate_extensions(event, "i")
        assert exts[0] == "i"
        assert exts.count("i") == 1


class TestLUDBTask:
    def test_full_record_returns_expected_samples(self, ludb_record):
        _, base = ludb_record
        event = _Event(
            visit_id="ecg",
            signal_file=str(base),
            lead_i=str(base.with_suffix(".i")),
            lead_ii=str(base.with_suffix(".ii")),
        )
        patient = _Patient("1", event)
        task = ECGDelineationLUDB(split_by_pulse=False)

        samples = task(patient)
        assert len(samples) == 2  # i + ii
        for s in samples:
            assert s["signal"].shape == (1, SIGNAL_LEN)
            assert s["mask"].shape == (SIGNAL_LEN,)
            assert s["label"] in (0, 1, 2, 3)

    def test_pulse_mode_shape_and_filtering(self, tmp_path: Path):
        base = _write_record(tmp_path, "2", ["i", "ii"])
        _write_qrs_only_annotation(tmp_path, "2", "i")

        event = _Event(
            visit_id="ecg",
            signal_file=str(base),
            lead_i=str(base.with_suffix(".i")),
        )
        patient = _Patient("2", event)

        loose = ECGDelineationLUDB(
            split_by_pulse=True, pulse_window=120, filter_incomplete_pulses=False
        )(patient)
        strict = ECGDelineationLUDB(
            split_by_pulse=True, pulse_window=120, filter_incomplete_pulses=True
        )(patient)

        assert len(loose) >= 1
        assert len(strict) == 0
        assert loose[0]["signal"].shape == (1, 240)
        assert loose[0]["mask"].shape == (240,)

    def test_bad_signal_path_returns_empty(self):
        patient = _Patient(
            "999",
            _Event(
                visit_id="ecg",
                signal_file="/does/not/exist/record",
                lead_i="/does/not/exist/record.i",
            ),
        )
        assert ECGDelineationLUDB(split_by_pulse=False)(patient) == []


class TestQTDBTask:
    def test_qtdb_uses_annotation_field_extensions_when_present(self, qtdb_record):
        _, base = qtdb_record
        event = _Event(
            visit_id="ecg",
            signal_file=str(base),
            lead_0="0",
            lead_1="1",
            ann_pu0=str(base.with_suffix(".pua")),
            ann_pu1=str(base.with_suffix(".pub")),
        )
        patient = _Patient("sel100", event)
        task = ECGDelineationQTDB(split_by_pulse=False)

        samples = task(patient)
        assert len(samples) == 2
        leads = {s["lead"] for s in samples}
        assert leads == {"0", "1"}
        for s in samples:
            assert s["signal"].shape == (1, SIGNAL_LEN)
            assert s["mask"].shape == (SIGNAL_LEN,)
            assert s["label"] in (0, 1, 2, 3)
