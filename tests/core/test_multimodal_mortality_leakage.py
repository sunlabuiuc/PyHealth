import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
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


class ColumnsPatient(DummyPatient):
    """A DummyPatient whose dataset loaded only ``tables``, like ``Patient``."""

    def __init__(self, events, tables):
        super().__init__({k: v for k, v in events.items() if k in tables})
        self.data_source = pl.DataFrame(
            schema={f"{table}/hadm_id": pl.Utf8 for table in tables}
        )

    def get_events(self, event_type=None, filters=None, **kwargs):
        if filters and f"{event_type}/hadm_id" not in self.data_source.columns:
            raise pl.exceptions.ColumnNotFoundError(f"{event_type}/hadm_id")
        return super().get_events(event_type, filters=filters, **kwargs)


class TestMultimodalMortalityTables(unittest.TestCase):
    def setUp(self):
        self.events = make_patient(include_available_xray=True).events

    def test_runs_without_the_radiology_table(self):
        tables = [k for k in self.events if k != "radiology"]
        samples = MultimodalMortalityPredictionMIMIC4()(
            ColumnsPatient(self.events, tables)
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["discharge"], ["Discharged"])
        self.assertEqual(samples[0]["radiology"], [])

    def test_runs_without_the_discharge_table(self):
        events = dict(self.events)
        events["radiology"] = [
            make_event(e.timestamp, hadm_id="history", text="Chest clear")
            for e in events["discharge"]
        ]
        tables = [k for k in events if k != "discharge"]
        samples = MultimodalMortalityPredictionMIMIC4()(ColumnsPatient(events, tables))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["discharge"], [])
        self.assertEqual(samples[0]["radiology"], ["Chest clear"])

    def test_names_the_missing_sources(self):
        tables = [k for k in self.events if k not in ("discharge", "radiology")]
        with self.assertRaisesRegex(ValueError, "note table"):
            MultimodalMortalityPredictionMIMIC4()(ColumnsPatient(self.events, tables))
        tables = [k for k in self.events if k != "metadata"]
        with self.assertRaisesRegex(ValueError, "CXR metadata"):
            MultimodalMortalityPredictionMIMIC4()(ColumnsPatient(self.events, tables))

    def test_ehr_only_dataset_gets_a_clear_error(self):
        from pyhealth.datasets import MIMIC4Dataset

        root = Path(__file__).parents[2] / "test-resources/core/mimic4demo"
        with tempfile.TemporaryDirectory() as cache:
            dataset = MIMIC4Dataset(
                ehr_root=str(root),
                ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                cache_dir=cache,
            )
            patient = next(iter(dataset.iter_patients()))
            with self.assertRaisesRegex(ValueError, "note table.*CXR metadata"):
                MultimodalMortalityPredictionMIMIC4()(patient)
