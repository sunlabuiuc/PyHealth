"""Observation window default 0 = off, on synthetic two-stay patients.

Commit scope: the shared window lives on the base class and applies through
the single ``_admission_window_end`` helper to labs and radiology notes.
``0`` means no cap (full stay through discharge). A positive value caps each
admission at ``min(admit + N h, discharge)``. Negative or ``None`` raises.

Discharge notes stay unbounded by design: the collected sections are
admission context stamped at admit, so windowing them would drop context
without removing any post-admit information.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl


T0 = datetime(2180, 5, 6, 8, 0, 0)
STAY2 = T0 + timedelta(days=400)


def _lab_df(rows):
    """Rows: (timestamp, itemid, value). Storetime == charttime here."""
    return pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "labevents/itemid": [r[1] for r in rows],
            "labevents/valuenum": [float(r[2]) for r in rows],
            "labevents/storetime": [r[0].strftime("%Y-%m-%d %H:%M:%S") for r in rows],
        }
    )


def _admission(admits, days=9, hadm_id=1, died=0):
    return SimpleNamespace(
        timestamp=admits,
        dischtime=(admits + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S"),
        hadm_id=hadm_id,
        hospital_expire_flag=died,
    )


class _FakePatient:
    patient_id = "P0"

    def __init__(self, admissions, lab_rows, notes):
        self._admissions = admissions
        self._lab_rows = lab_rows
        self._notes = notes

    def get_events(
        self, event_type, start=None, end=None, filters=None, return_df=None
    ):
        if event_type == "patients":
            return [object()]
        if event_type == "admissions":
            return self._admissions
        if event_type == "labevents":
            df = _lab_df(self._lab_rows)
            if start is not None:
                df = df.filter(pl.col("timestamp") >= start)
            if end is not None:
                df = df.filter(pl.col("timestamp") <= end)
            return df
        if event_type in ("discharge", "radiology"):
            out = list(self._notes.get(event_type, []))
            if start is not None:
                out = [n for n in out if n.timestamp >= start]
            if end is not None:
                out = [n for n in out if n.timestamp <= end]
            if filters:
                for key, op, value in filters:
                    assert op == "=="
                    out = [n for n in out if getattr(n, key) == value]
            return out
        return []


def _two_stay_patient():
    admissions = [_admission(T0, hadm_id=1), _admission(STAY2, hadm_id=2)]
    lab_rows = [
        (T0 + timedelta(hours=6), "50824", 140.0),
        (T0 + timedelta(hours=30), "50824", 141.0),
        (STAY2 + timedelta(hours=6), "50824", 139.0),
        (STAY2 + timedelta(hours=30), "50824", 138.0),
    ]
    notes = {
        "discharge": [
            SimpleNamespace(
                text="Chief Complaint:\n\nshortness of breath\n\n",
                timestamp=T0 + timedelta(hours=30),
                hadm_id=1,
            ),
            SimpleNamespace(
                text="Chief Complaint:\n\nchest pain\n\n",
                timestamp=STAY2 + timedelta(hours=30),
                hadm_id=2,
            ),
        ],
        "radiology": [
            SimpleNamespace(
                text="Indication:\n\nearly film\n\n",
                timestamp=T0 + timedelta(hours=6),
                hadm_id=1,
            ),
            SimpleNamespace(
                text="Indication:\n\nlate film\n\n",
                timestamp=T0 + timedelta(hours=30),
                hadm_id=1,
            ),
            SimpleNamespace(
                text="Indication:\n\nsecond stay film\n\n",
                timestamp=STAY2 + timedelta(hours=6),
                hadm_id=2,
            ),
        ],
    }
    return _FakePatient(admissions, lab_rows, notes)


class TestObservationWindowDefaultZero(unittest.TestCase):
    def test_zero_equals_full_stay_exactly(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        patient = _two_stay_patient()
        default_record = LabsMIMIC4()(patient)[0]
        explicit_record = LabsMIMIC4(window_hours=0)(patient)[0]
        huge_record = LabsMIMIC4(window_hours=10**6)(patient)[0]
        for key in ("labs", "labs_mask"):
            self.assertEqual(default_record[key], explicit_record[key])
            self.assertEqual(default_record[key], huge_record[key])
        # All four lab events, both stays.
        self.assertEqual(len(default_record["labs"][0]), 4)

    def test_positive_window_drops_late_events_per_admission(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        record = LabsMIMIC4(window_hours=24)(_two_stay_patient())[0]
        times, _values = record["labs"]
        # +6h of each stay kept; +30h of each stay dropped. The second
        # stay's +6h is ~9606h on the sample clock, so a first-admission
        # clock would have dropped it.
        self.assertEqual(times, [6.0, 400 * 24.0 + 6.0])

    def test_window_bounds_radiology_but_not_discharge_sections(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        record = NotesLabsMIMIC4(window_hours=24)(_two_stay_patient())[0]
        texts, times = record["admission_note_times"]
        joined = " [SEP] ".join(texts)
        # Radiology at +6h kept (both stays); radiology at +30h dropped.
        self.assertIn("early film", joined)
        self.assertIn("second stay film", joined)
        self.assertNotIn("late film", joined)
        # Discharge sections are stamped at admit and unbounded: both stays
        # present even though their charttimes (+30h) are past the window.
        self.assertIn("shortness of breath", joined)
        self.assertIn("chest pain", joined)
        self.assertEqual(
            times[texts.index(next(t for t in texts if "shortness of breath" in t))],
            0.0,
        )

    def test_negative_and_none_raise(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4, NotesLabsMIMIC4

        for cls in (LabsMIMIC4, NotesLabsMIMIC4):
            with self.assertRaises(ValueError):
                cls(window_hours=-1)
            with self.assertRaises(ValueError):
                cls(window_hours=None)


if __name__ == "__main__":
    unittest.main()
