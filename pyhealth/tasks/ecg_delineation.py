"""ECG wave delineation tasks for LUDB and QTDB.

This module provides task classes compatible with the BaseDataset + Patient/Event
pipeline for ECG waveform delineation.

Label encoding
--------------
0: background
1: P wave
2: QRS complex
3: T wave
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np

from .base_task import BaseTask

try:
    wfdb_module = importlib.import_module("wfdb")
except Exception:  # pragma: no cover
    wfdb_module = None


LUDB_LEADS: List[str] = [
    "i",
    "ii",
    "iii",
    "avr",
    "avl",
    "avf",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
]

QTDB_LEADS: List[str] = ["0", "1"]

DEFAULT_PULSE_WINDOW: int = 250


def _safe_str(value: Any) -> Optional[str]:
    """Convert value to stripped string and normalize missing-like tokens."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    return text


def _extract_extension(value: Any) -> Optional[str]:
    """Extract WFDB annotation extension from a path or extension-like value."""
    text = _safe_str(value)
    if text is None:
        return None

    # Path-like input: take suffix as extension
    suffix = Path(text).suffix
    if suffix:
        return suffix.lstrip(".")

    # Extension-like input (e.g., "i", "pu0")
    return text


def _parse_annotations(record_base_path: str, extension: str) -> Dict[str, List[dict]]:
    """Read WFDB delineation annotations for one record/extension pair.

    Parser expects triplets:
      "(" -> onset
      wave symbol -> peak
      ")" -> offset

    Supported wave peak symbols:
      - P-wave: p / P
      - QRS:    N / n
      - T-wave: t / T
    """
    try:
        if wfdb_module is None:
            return {"P": [], "QRS": [], "T": []}
        ann = wfdb_module.rdann(record_base_path, extension)
    except Exception:
        return {"P": [], "QRS": [], "T": []}

    symbols = ann.symbol
    samples = ann.sample
    waves: Dict[str, List[dict]] = {"P": [], "QRS": [], "T": []}
    wave_map = {
        "p": "P",
        "P": "P",
        "N": "QRS",
        "n": "QRS",
        "t": "T",
        "T": "T",
    }

    i = 0
    while i < len(symbols):
        if symbols[i] == "(" and i + 2 < len(symbols) and symbols[i + 2] == ")":
            peak_sym = symbols[i + 1]
            if peak_sym in wave_map:
                waves[wave_map[peak_sym]].append(
                    {
                        "onset": int(samples[i]),
                        "peak": int(samples[i + 1]),
                        "offset": int(samples[i + 2]),
                    }
                )
            i += 3
        else:
            i += 1

    return waves


def _build_segmentation_mask(
    signal_length: int, waves: Dict[str, List[dict]]
) -> np.ndarray:
    """Build point-wise segmentation mask with values in {0, 1, 2, 3}."""
    mask = np.zeros(signal_length, dtype=np.int64)
    label_map = {"P": 1, "QRS": 2, "T": 3}

    for wave_name, label in label_map.items():
        for w in waves[wave_name]:
            onset = max(0, int(w["onset"]))
            offset = min(signal_length - 1, int(w["offset"]))
            if onset <= offset:
                mask[onset : offset + 1] = label

    return mask


def _resolve_lead_index(wfdb_record: Any, lead: str, fallback_idx: int) -> int:
    """Resolve lead index robustly from WFDB signal names."""
    sig_names = [str(name).strip().lower() for name in wfdb_record.sig_name]
    lead_norm = str(lead).strip().lower()
    if lead_norm in sig_names:
        return sig_names.index(lead_norm)
    return fallback_idx


def _has_any_wave(waves: Dict[str, List[dict]]) -> bool:
    """Return whether parsed waves contain at least one delineated segment."""
    return any(len(waves[k]) > 0 for k in ("P", "QRS", "T"))


class ECGDelineationTask(BaseTask):
    """Generic ECG delineation task for event-table ECG datasets.

    This task expects one event row per patient that includes:
      - `signal_file`: absolute WFDB base record path (without extension)
      - lead-related metadata used to resolve annotation extensions
    """

    task_name: str = "ecg_delineation"
    input_schema: Dict[str, Union[str, Type]] = {"signal": "tensor"}
    output_schema: Dict[str, Union[str, Type]] = {
        "mask": "tensor",
        "label": "multiclass",
    }

    def __init__(
        self,
        event_type: str,
        leads: Sequence[str],
        lead_field_map: Optional[Mapping[str, str]] = None,
        annotation_field_map: Optional[Mapping[str, Sequence[str]]] = None,
        annotation_extension_map: Optional[Mapping[str, Sequence[str]]] = None,
        split_by_pulse: bool = False,
        pulse_window: int = DEFAULT_PULSE_WINDOW,
        filter_incomplete_pulses: bool = False,
    ) -> None:
        super().__init__()
        if pulse_window <= 0:
            raise ValueError("pulse_window must be a positive integer.")
        self.event_type = event_type
        self.leads = list(leads)
        self.lead_field_map = dict(lead_field_map or {})
        self.annotation_field_map = {
            k: list(v) for k, v in (annotation_field_map or {}).items()
        }
        self.annotation_extension_map = {
            k: list(v) for k, v in (annotation_extension_map or {}).items()
        }
        self.split_by_pulse = bool(split_by_pulse)
        self.pulse_window = int(pulse_window)
        self.filter_incomplete_pulses = bool(filter_incomplete_pulses)

    @staticmethod
    def _is_complete_pulse_annotation(pulse_mask: np.ndarray) -> bool:
        """Return True if pulse window contains P, QRS, and T labels."""
        if pulse_mask.size == 0:
            return False
        labels = set(np.unique(pulse_mask).tolist())
        return {1, 2, 3}.issubset(labels)

    def _candidate_extensions(self, event: Any, lead: str) -> List[str]:
        """Build candidate annotation extensions for a given lead."""
        candidates: List[str] = []

        # 1) Explicit annotation field list for this lead
        for field in self.annotation_field_map.get(lead, []):
            ext = _extract_extension(getattr(event, field, None))
            if ext:
                candidates.append(ext)

        # 2) Explicit extension preference list for this lead
        candidates.extend(self.annotation_extension_map.get(lead, []))

        # 3) Lead field (can be extension or path)
        lead_field = self.lead_field_map.get(lead)
        if lead_field is not None:
            ext = _extract_extension(getattr(event, lead_field, None))
            if ext:
                candidates.append(ext)

        # 4) Fallback: use lead itself as extension
        candidates.append(lead)

        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for ext in candidates:
            if ext not in seen:
                seen.add(ext)
                unique.append(ext)
        return unique

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events = patient.get_events(self.event_type)
        if len(events) == 0:
            return []

        event = events[0]
        pid = str(patient.patient_id)
        visit_id = getattr(event, "visit_id", "ecg")
        record_base_path = _safe_str(getattr(event, "signal_file", None))
        if record_base_path is None:
            return []

        try:
            if wfdb_module is None:
                return []
            wfdb_record = wfdb_module.rdrecord(record_base_path)
        except Exception:
            return []

        samples: List[Dict[str, Any]] = []

        for lead_fallback_idx, lead in enumerate(self.leads):
            lead_idx = _resolve_lead_index(wfdb_record, lead, lead_fallback_idx)
            if lead_idx >= wfdb_record.p_signal.shape[1]:
                continue

            signal = wfdb_record.p_signal[:, lead_idx].astype(np.float32)

            # choose first candidate extension that yields at least one wave
            waves = {"P": [], "QRS": [], "T": []}
            for ext in self._candidate_extensions(event, lead):
                parsed = _parse_annotations(record_base_path, ext)
                if _has_any_wave(parsed):
                    waves = parsed
                    break

            # Skip if no usable annotation exists
            if not _has_any_wave(waves):
                continue

            mask = _build_segmentation_mask(len(signal), waves)

            if self.split_by_pulse:
                for pulse_idx, qrs in enumerate(waves["QRS"]):
                    r_peak = int(qrs["peak"])
                    start = r_peak - self.pulse_window
                    end = r_peak + self.pulse_window
                    if start < 0 or end > len(signal):
                        continue

                    pulse_signal = signal[start:end]
                    pulse_mask = mask[start:end]
                    if pulse_signal.size == 0:
                        continue

                    if (
                        self.filter_incomplete_pulses
                        and not self._is_complete_pulse_annotation(pulse_mask)
                    ):
                        continue

                    samples.append(
                        {
                            "patient_id": pid,
                            "visit_id": visit_id,
                            "record_id": f"{pid}_{lead}_{pulse_idx}",
                            "lead": lead,
                            "signal": pulse_signal[np.newaxis, :],
                            "mask": pulse_mask,
                            "label": int(np.bincount(pulse_mask).argmax()),
                        }
                    )
            else:
                samples.append(
                    {
                        "patient_id": pid,
                        "visit_id": visit_id,
                        "record_id": f"{pid}_{lead}",
                        "lead": lead,
                        "signal": signal[np.newaxis, :],
                        "mask": mask,
                        "label": int(np.bincount(mask).argmax()),
                    }
                )

        return samples


class ECGDelineationLUDB(ECGDelineationTask):
    """LUDB-specific ECG delineation task."""

    task_name: str = "ecg_delineation_ludb"

    def __init__(
        self,
        split_by_pulse: bool = False,
        pulse_window: int = DEFAULT_PULSE_WINDOW,
        filter_incomplete_pulses: bool = False,
    ) -> None:
        super().__init__(
            event_type="ludb",
            leads=LUDB_LEADS,
            lead_field_map={lead: f"lead_{lead}" for lead in LUDB_LEADS},
            split_by_pulse=split_by_pulse,
            pulse_window=pulse_window,
            filter_incomplete_pulses=filter_incomplete_pulses,
        )


class ECGDelineationQTDB(ECGDelineationTask):
    """QTDB-specific ECG delineation task (two leads)."""

    task_name: str = "ecg_delineation_qtdb"

    def __init__(
        self,
        split_by_pulse: bool = False,
        pulse_window: int = DEFAULT_PULSE_WINDOW,
        filter_incomplete_pulses: bool = False,
    ) -> None:
        super().__init__(
            event_type="qtdb",
            leads=QTDB_LEADS,
            lead_field_map={"0": "lead_0", "1": "lead_1"},
            # Prefer QTDB lead-specific automatic delineations first.
            annotation_field_map={
                "0": ["ann_pu0", "ann_q1c", "ann_qt1", "ann_man", "ann_atr"],
                "1": ["ann_pu1", "ann_q1c", "ann_qt1", "ann_man", "ann_atr"],
            },
            annotation_extension_map={
                "0": ["pu0", "q1c", "qt1", "man", "atr"],
                "1": ["pu1", "q1c", "qt1", "man", "atr"],
            },
            split_by_pulse=split_by_pulse,
            pulse_window=pulse_window,
            filter_incomplete_pulses=filter_incomplete_pulses,
        )


def get_ecg_delineation_ludb_task(
    split_by_pulse: bool = False,
    pulse_window: int = DEFAULT_PULSE_WINDOW,
    filter_incomplete_pulses: bool = False,
) -> ECGDelineationLUDB:
    """Factory helper for configurable LUDB delineation task."""
    return ECGDelineationLUDB(
        split_by_pulse=split_by_pulse,
        pulse_window=pulse_window,
        filter_incomplete_pulses=filter_incomplete_pulses,
    )


def get_ecg_delineation_qtdb_task(
    split_by_pulse: bool = False,
    pulse_window: int = DEFAULT_PULSE_WINDOW,
    filter_incomplete_pulses: bool = False,
) -> ECGDelineationQTDB:
    """Factory helper for configurable QTDB delineation task."""
    return ECGDelineationQTDB(
        split_by_pulse=split_by_pulse,
        pulse_window=pulse_window,
        filter_incomplete_pulses=filter_incomplete_pulses,
    )


# Backward-compatible symbol kept for existing imports/usages.
ecg_delineation_ludb_fn = ECGDelineationLUDB(split_by_pulse=False)
