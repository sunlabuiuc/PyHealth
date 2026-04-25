import unittest
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

import pyhealth.tasks.drug_recommendation as drug_rec
from pyhealth.tasks import (
    DrugRecommendationMIMIC3,
    DrugRecommendationMIMIC4,
    drug_recommendation_mimic3_fn,
    drug_recommendation_mimic4_fn,
)


class FakeNDCToATC3Map:
    def __init__(self):
        self.calls = []
        self.mapping = {
            "11111111111": ["A10B"],
            "22222222222": ["C03C", "C03C"],
            "33333333333": ["N02B"],
        }

    def map(self, ndc, target_kwargs=None):
        self.calls.append((ndc, target_kwargs))
        return self.mapping.get(ndc, [])


class FakePatient:
    patient_id = "patient-1"

    def __init__(self, tables):
        self.tables = tables

    def get_events(self, event_type, filters=None, return_df=False):
        if event_type == "admissions":
            return [SimpleNamespace(hadm_id="visit-1"), SimpleNamespace(hadm_id="visit-2")]

        hadm_id = filters[0][2]
        rows = self.tables[event_type][hadm_id]
        if return_df:
            return pl.DataFrame(rows)
        return rows


class FakeVisit:
    def __init__(self, visit_id, table_codes):
        self.visit_id = visit_id
        self.table_codes = table_codes

    def get_code_list(self, table):
        return self.table_codes[table]


class FakeLegacyPatient:
    patient_id = "patient-legacy"

    def __init__(self, visits):
        self.visits = visits

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, index):
        return self.visits[index]


class TestDrugRecommendationATC3(unittest.TestCase):
    def setUp(self):
        drug_rec._NDC_TO_ATC3_MAPPER = None
        drug_rec._NDC_TO_ATC3_CACHE.clear()
        self.mapper = FakeNDCToATC3Map()
        patcher = patch(
            "pyhealth.tasks.drug_recommendation.CrossMap.load",
            return_value=self.mapper,
        )
        self.addCleanup(patcher.stop)
        self.crossmap_load = patcher.start()

    def tearDown(self):
        drug_rec._NDC_TO_ATC3_MAPPER = None
        drug_rec._NDC_TO_ATC3_CACHE.clear()

    def test_mimic3_drug_recommendation_maps_ndc_to_atc3(self):
        patient = FakePatient(
            {
                "diagnoses_icd": {
                    "visit-1": {"diagnoses_icd/icd9_code": ["25000"]},
                    "visit-2": {"diagnoses_icd/icd9_code": ["4019"]},
                },
                "procedures_icd": {
                    "visit-1": {"procedures_icd/icd9_code": ["3893"]},
                    "visit-2": {"procedures_icd/icd9_code": ["9904"]},
                },
                "prescriptions": {
                    "visit-1": {
                        "prescriptions/ndc": [
                            "11111111111",
                            "22222222222",
                            "11111111111",
                            "0",
                            None,
                            "99999999999",
                        ]
                    },
                    "visit-2": {"prescriptions/ndc": ["33333333333"]},
                },
            }
        )

        samples = DrugRecommendationMIMIC3()(patient)

        self.crossmap_load.assert_called_once_with("NDC", "ATC")
        self.assertEqual(samples[0]["drugs"], ["A10B", "C03C"])
        self.assertEqual(samples[0]["drugs_hist"], [[]])
        self.assertEqual(samples[1]["drugs"], ["N02B"])
        self.assertEqual(samples[1]["drugs_hist"], [["A10B", "C03C"], []])
        self.assertNotIn("1111", samples[0]["drugs"])
        self.assertNotIn("0", samples[0]["drugs"])
        self.assertTrue(
            all(kwargs == {"level": 3} for _, kwargs in self.mapper.calls)
        )

    def test_mimic4_drug_recommendation_maps_ndc_to_atc3(self):
        patient = FakePatient(
            {
                "diagnoses_icd": {
                    "visit-1": {
                        "diagnoses_icd/icd_version": ["9"],
                        "diagnoses_icd/icd_code": ["25000"],
                    },
                    "visit-2": {
                        "diagnoses_icd/icd_version": ["10"],
                        "diagnoses_icd/icd_code": ["I10"],
                    },
                },
                "procedures_icd": {
                    "visit-1": {
                        "procedures_icd/icd_version": ["9"],
                        "procedures_icd/icd_code": ["3893"],
                    },
                    "visit-2": {
                        "procedures_icd/icd_version": ["10"],
                        "procedures_icd/icd_code": ["5A1D70Z"],
                    },
                },
                "prescriptions": {
                    "visit-1": {"prescriptions/ndc": ["11111111111", "22222222222"]},
                    "visit-2": {
                        "prescriptions/ndc": ["33333333333", "", "<NA>"]
                    },
                },
            }
        )

        samples = DrugRecommendationMIMIC4()(patient)

        self.crossmap_load.assert_called_once_with("NDC", "ATC")
        self.assertEqual(samples[0]["drugs"], ["A10B", "C03C"])
        self.assertEqual(samples[1]["drugs"], ["N02B"])
        self.assertNotIn("3333", samples[1]["drugs"])
        self.assertTrue(
            all(kwargs == {"level": 3} for _, kwargs in self.mapper.calls)
        )

    def test_legacy_drug_recommendation_functions_map_ndc_to_atc3(self):
        mimic3_patient = FakeLegacyPatient(
            [
                FakeVisit(
                    "visit-1",
                    {
                        "DIAGNOSES_ICD": ["25000"],
                        "PROCEDURES_ICD": ["3893"],
                        "PRESCRIPTIONS": ["11111111111", "22222222222"],
                    },
                ),
                FakeVisit(
                    "visit-2",
                    {
                        "DIAGNOSES_ICD": ["4019"],
                        "PROCEDURES_ICD": ["9904"],
                        "PRESCRIPTIONS": ["33333333333"],
                    },
                ),
            ]
        )
        mimic4_patient = FakeLegacyPatient(
            [
                FakeVisit(
                    "visit-1",
                    {
                        "diagnoses_icd": ["9_25000"],
                        "procedures_icd": ["9_3893"],
                        "prescriptions": ["11111111111", "22222222222"],
                    },
                ),
                FakeVisit(
                    "visit-2",
                    {
                        "diagnoses_icd": ["10_I10"],
                        "procedures_icd": ["10_5A1D70Z"],
                        "prescriptions": ["33333333333"],
                    },
                ),
            ]
        )

        mimic3_samples = drug_recommendation_mimic3_fn(mimic3_patient)
        mimic4_samples = drug_recommendation_mimic4_fn(mimic4_patient)

        self.assertEqual(mimic3_samples[0]["drugs"], ["A10B", "C03C"])
        self.assertEqual(mimic3_samples[1]["drugs"], ["N02B"])
        self.assertEqual(mimic4_samples[0]["drugs"], ["A10B", "C03C"])
        self.assertEqual(mimic4_samples[1]["drugs"], ["N02B"])
        self.assertNotIn("1111", mimic3_samples[0]["drugs"])
        self.assertNotIn("1111", mimic4_samples[0]["drugs"])


if __name__ == "__main__":
    unittest.main()
