from datetime import datetime
import unittest


class _DummyEvent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyPatient:
    def __init__(self) -> None:
        self.patient_id = "p-1"
        self._admissions = [
            _DummyEvent(
                timestamp=datetime(2020, 1, 1, 0, 0, 0),
                dischtime="malformed-dischtime",
                hadm_id=101,
                hospital_expire_flag=0,
            )
        ]
        self._patients = [_DummyEvent(anchor_age=55)]

    def get_events(self, event_type, start=None, end=None, filters=None, return_df=False):
        if event_type == "patients":
            return self._patients
        if event_type == "admissions":
            return self._admissions
        if event_type in {"diagnoses_icd", "procedures_icd", "discharge", "radiology"}:
            return []
        if event_type == "labevents" and return_df:
            import polars as pl

            return pl.DataFrame(
                {
                    "timestamp": [],
                    "labevents/itemid": [],
                    "labevents/storetime": [],
                    "labevents/valuenum": [],
                }
            )
        return []


class TestClinicalNotesICDLabsMIMIC4EdgeCases(unittest.TestCase):
    def test_malformed_dischtime_keeps_temporal_fields_non_empty(self):
        from pyhealth.processors.stagenet_processor import (
            StageNetProcessor,
            StageNetTensorProcessor,
        )
        from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesICDLabsMIMIC4

        task = ClinicalNotesICDLabsMIMIC4(window_hours=24)
        samples = task(_DummyPatient())

        self.assertEqual(len(samples), 1)
        sample = samples[0]

        self.assertGreater(len(sample["icd_codes"][0]), 0)
        self.assertGreater(len(sample["labs"][0]), 0)
        self.assertGreater(len(sample["labs_mask"][0]), 0)

        icd_proc = StageNetProcessor()
        icd_proc.fit([{"icd_codes": sample["icd_codes"]}], "icd_codes")
        icd_time, _ = icd_proc.process(sample["icd_codes"])
        self.assertIsNotNone(icd_time)

        labs_proc = StageNetTensorProcessor()
        labs_proc.fit([{"labs": sample["labs"]}], "labs")
        labs_time, _ = labs_proc.process(sample["labs"])
        self.assertIsNotNone(labs_time)


if __name__ == "__main__":
    unittest.main()
