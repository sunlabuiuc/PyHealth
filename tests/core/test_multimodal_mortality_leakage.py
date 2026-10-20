import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl

from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4


class DummyPatient:
    def __init__(self, events):
        self.patient_id = "patient"
        self.events = events

    def get_events(
        self,
        event_type=None,
        start=None,
        end=None,
        filters=None,
        return_df=False,
    ):
        events = list(self.events.get(event_type, []))
        if start is not None:
            events = [event for event in events if event.timestamp >= start]
        if end is not None:
            events = [event for event in events if event.timestamp <= end]
        for field, operator, value in filters or []:
            if operator == "==":
                events = [
                    event for event in events if getattr(event, field, None) == value
                ]

        if return_df:
            return pl.DataFrame(
                {
                    "timestamp": [event.timestamp for event in events],
                    "labevents/itemid": [event.itemid for event in events],
                    "labevents/valuenum": [event.valuenum for event in events],
                    "labevents/storetime": [event.storetime for event in events],
                }
            )
        return events


def make_event(timestamp, **attributes):
    return SimpleNamespace(timestamp=timestamp, **attributes)


def make_patient(*, include_available_xray):
    admission_time = datetime(2025, 1, 1, 8)  # noqa: DTZ001
    cutoff = admission_time + timedelta(days=1)
    death_admission_time = cutoff + timedelta(days=3)

    metadata = [
        make_event(
            death_admission_time,
            image_path="future.jpg",
        )
    ]
    negbio = [
        make_event(
            death_admission_time,
            edema=1,
        )
    ]
    if include_available_xray:
        metadata.insert(0, make_event(cutoff, image_path="available.jpg"))
        negbio.insert(0, make_event(cutoff, cardiomegaly=1))

    return DummyPatient(
        {
            "patients": [make_event(admission_time, anchor_age=50)],
            "admissions": [
                make_event(
                    admission_time,
                    hadm_id="history",
                    dischtime=cutoff.strftime("%Y-%m-%d %H:%M:%S"),
                    hospital_expire_flag=0,
                ),
                make_event(
                    death_admission_time,
                    hadm_id="outcome",
                    dischtime=(death_admission_time + timedelta(days=1)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    hospital_expire_flag=1,
                ),
            ],
            "diagnoses_icd": [
                make_event(admission_time, hadm_id="history", icd_code="I10")
            ],
            "procedures_icd": [
                make_event(admission_time, hadm_id="history", icd_code="0W3P0ZZ")
            ],
            "prescriptions": [
                make_event(admission_time, hadm_id="history", ndc="0001")
            ],
            "discharge": [
                make_event(admission_time, hadm_id="history", text="Discharged")
            ],
            "radiology": [],
            "labevents": [
                make_event(
                    admission_time + timedelta(hours=1),
                    itemid="50824",
                    valuenum=140.0,
                    storetime=(admission_time + timedelta(hours=1)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
            ],
            "metadata": metadata,
            "negbio": negbio,
        }
    )


class TestMultimodalMortalityLeakage(unittest.TestCase):
    def test_excludes_future_xrays(self):
        sample = MultimodalMortalityPredictionMIMIC4()(
            make_patient(include_available_xray=True)
        )[0]

        self.assertEqual(sample["image_path"], "available.jpg")
        self.assertEqual(sample["negbio_findings"], ["cardiomegaly"])

    def test_requires_xray_by_prediction_time(self):
        samples = MultimodalMortalityPredictionMIMIC4()(
            make_patient(include_available_xray=False)
        )

        self.assertEqual(samples, [])
