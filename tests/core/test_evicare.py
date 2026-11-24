"""Tests for EviCare: the diagnosis prediction tasks (and, later, the model).

Contributor: Kavish Shah (GitHub: thekavishshah)
Paper: Zhang et al., "EviCare: Enhancing Diagnosis Prediction with Deep
    Model-Guided Evidence for In-Context Reasoning", KDD 2026,
    https://arxiv.org/abs/2604.10455

All tests use tiny synthetic patients and a 13-row CCS fixture: no MIMIC
data, downloads or LLM calls.
"""

import os
import pickle
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    CCSMapping,
    EviCareDiagnosisPredictionMIMIC3,
    EviCareDiagnosisPredictionMIMIC4,
)
from pyhealth.tasks.evicare_diagnosis import normalize_icd

MAPPING_CSV = str(
    Path(__file__).resolve().parents[2]
    / "test-resources"
    / "core"
    / "evicare"
    / "ccs_mapping_tiny.csv"
)

T1 = datetime(2100, 1, 1, tzinfo=UTC)
T2 = datetime(2100, 1, 11, tzinfo=UTC)
T3 = datetime(2100, 1, 19, tzinfo=UTC)


def make_patient(patient_id, visits, mimic4=False):
    """Builds a Patient from ``(hadm_id, admittime, codes)`` visits.

    For MIMIC-IV, codes are ``(icd_code, icd_version)`` pairs.
    """
    rows = []
    for hadm_id, admittime, codes in visits:
        rows.append(
            {
                "event_type": "admissions",
                "timestamp": admittime,
                "admissions/hadm_id": hadm_id,
            }
        )
        for code in codes:
            row = {
                "event_type": "diagnoses_icd",
                "timestamp": admittime,
                "diagnoses_icd/hadm_id": hadm_id,
            }
            if mimic4:
                row["diagnoses_icd/icd_code"] = code[0]
                row["diagnoses_icd/icd_version"] = code[1]
            else:
                row["diagnoses_icd/icd9_code"] = code
            rows.append(row)
    return Patient(patient_id=patient_id, data_source=pl.DataFrame(rows))


class TestCCSMapping(unittest.TestCase):
    def setUp(self):
        self.mapping = CCSMapping.from_csv(MAPPING_CSV)

    def test_normalize_icd(self):
        self.assertEqual(normalize_icd(" 250.00 "), "25000")
        self.assertEqual(normalize_icd("i50.9"), "I509")
        self.assertIsNone(normalize_icd(None))
        self.assertIsNone(normalize_icd(float("nan")))
        self.assertIsNone(normalize_icd(" "))

    def test_map_by_version(self):
        self.assertEqual(self.mapping.map("4280"), "108")
        self.assertEqual(self.mapping.map("428.0"), "108")
        # Dotted keys in the CSV match dot-less MIMIC codes.
        self.assertEqual(self.mapping.map("25000"), "49")
        self.assertEqual(self.mapping.map("I509", version=10), "108")
        self.assertIsNone(self.mapping.map("I509", version=9))
        self.assertIsNone(self.mapping.map("99999"))
        self.assertIsNone(self.mapping.map(None))

    def test_names_and_fallbacks(self):
        self.assertEqual(self.mapping.ccs_name("106"), "Cardiac dysrhythmias")
        self.assertEqual(self.mapping.icd_name("42731"), "Atrial fibrillation")
        self.assertEqual(
            self.mapping.icd_name("I480", version=10), "Paroxysmal atrial fibrillation"
        )
        self.assertEqual(self.mapping.ccs_name("999"), "999")
        self.assertEqual(self.mapping.icd_name("V1582"), "V1582")

    def test_codes_len_and_names_copy(self):
        self.assertEqual(
            self.mapping.ccs_codes,
            ["101", "103", "105", "106", "108", "157", "49", "98", "99"],
        )
        self.assertEqual(len(self.mapping), 13)
        names = self.mapping.ccs_names
        names["108"] = "changed"
        self.assertNotEqual(self.mapping.ccs_name("108"), "changed")

    def test_repr_is_deterministic_and_content_keyed(self):
        again = CCSMapping.from_csv(MAPPING_CSV)
        self.assertEqual(repr(self.mapping), repr(again))
        self.assertIn("icd9=9", repr(self.mapping))
        self.assertIn("icd10=4", repr(self.mapping))
        other = CCSMapping(icd9_to_ccs={"4280": "108"})
        self.assertNotEqual(repr(self.mapping), repr(other))

    def test_pickles(self):
        clone = pickle.loads(pickle.dumps(self.mapping))
        self.assertEqual(clone.map("4019"), "98")
        self.assertEqual(repr(clone), repr(self.mapping))

    def test_csv_missing_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w") as handle:
                handle.write("icd_code,name\n4280,x\n")
            with self.assertRaises(ValueError):
                CCSMapping.from_csv(path)

    def test_csv_optional_columns_and_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ccsr.csv")
            with open(path, "w") as handle:
                handle.write(
                    "icd_code,ccs_code,icd_version\n"
                    "I50.9,CIR019,10\n"
                    "4280,CIR019,\n"  # no version: ICD-9
                    "X1,CIR001,7\n"  # unknown version: skipped
                    ",CIR001,10\n"  # no code: skipped
                    "I10,,10\n"  # no CCS: skipped
                )
            mapping = CCSMapping.from_csv(path)
        self.assertEqual(mapping.map("I509", version=10), "CIR019")
        self.assertEqual(mapping.map("4280", version=9), "CIR019")
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping.ccs_name("CIR019"), "CIR019")


class _FakeGraph:
    """Stands in for the networkx graph of an InnerMap."""

    def __init__(self, names):
        self.nodes = _FakeNodes(names)


class _FakeNodes(dict):
    def __init__(self, names):
        super().__init__({code: {"name": name} for code, name in names.items()})

    def __call__(self, data=False):
        return list(self.items())


class TestCCSMappingFromCrossMap(unittest.TestCase):
    """from_crossmap with stubbed medcode tables (no download)."""

    def setUp(self):
        cross = {
            "ICD9CM": {"428.0": ["108"], "401.9": ["98"], "999.9": []},
            "ICD10CM": {"I50.9": ["108"]},
        }
        names = {
            "ICD9CM": {"428.0": "CHF, unspecified", "401.9": ""},
            "ICD10CM": {"I50.9": "Heart failure, unspecified"},
            "CCSCM": {"108": "Congestive heart failure; nonhypertensive"},
        }
        self.cross_patch = mock.patch(
            "pyhealth.medcode.CrossMap.load",
            side_effect=lambda s, t: SimpleNamespace(mapping=cross[s]),
        )
        self.inner_patch = mock.patch(
            "pyhealth.medcode.InnerMap.load",
            side_effect=lambda v: SimpleNamespace(graph=_FakeGraph(names[v])),
        )
        self.cross_load = self.cross_patch.start()
        self.inner_patch.start()

    def tearDown(self):
        mock.patch.stopall()

    def test_builds_tables_and_names(self):
        mapping = CCSMapping.from_crossmap()
        self.assertEqual(mapping.map("4280"), "108")
        self.assertEqual(mapping.map("I509", version=10), "108")
        self.assertIsNone(mapping.map("9999"))
        self.assertEqual(mapping.icd_name("4280"), "CHF, unspecified")
        self.assertEqual(mapping.icd_name("4019"), "4019")  # empty name
        self.assertEqual(
            mapping.ccs_name("108"), "Congestive heart failure; nonhypertensive"
        )
        self.assertIn("crossmap:CCSCM", repr(mapping))

    def test_task_default_uses_crossmap_for_its_versions(self):
        EviCareDiagnosisPredictionMIMIC3()
        self.assertEqual(
            [c.args for c in self.cross_load.call_args_list], [("ICD9CM", "CCSCM")]
        )
        self.cross_load.reset_mock()
        EviCareDiagnosisPredictionMIMIC4()
        self.assertEqual(
            [c.args[0] for c in self.cross_load.call_args_list], ["ICD9CM", "ICD10CM"]
        )


class TestEviCareTaskMIMIC3(unittest.TestCase):
    def setUp(self):
        # Visit 1: hypertension (98) and diabetes (49).
        # Visit 2: AF (106), hypertension again, an unmapped code, a duplicate.
        # Target: CHF (108), AF (106, seen), AKI (157).
        self.patient = make_patient(
            "p1",
            [
                ("h1", T1, ["4019", "25000"]),
                ("h2", T2, ["42731", "4019", "99999", "42731"]),
                ("h3", T3, ["4280", "42731", "5849"]),
            ],
        )

    def task(self, **kwargs):
        return EviCareDiagnosisPredictionMIMIC3(ccs_mapping=MAPPING_CSV, **kwargs)

    def test_schema(self):
        task = self.task()
        self.assertEqual(task.input_schema, {"conditions": "nested_sequence"})
        self.assertEqual(task.output_schema, {"ccs_targets": "multilabel"})
        self.assertEqual(task.task_name, "EviCareDiagnosisPredictionMIMIC3")

    def test_overall_sample(self):
        samples = self.task()(self.patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["visit_id"], "h3")
        self.assertEqual(sample["conditions"], [["98", "49"], ["106", "98"]])
        self.assertEqual(sample["ccs_targets"], ["108", "106", "157"])

    def test_evidence_fields_are_aligned(self):
        sample = self.task()(self.patient)[0]
        self.assertEqual(sample["history_icd"], [["4019", "25000"], ["42731", "4019"]])
        self.assertEqual(sample["history_ccs"], [["98", "49"], ["106", "98"]])
        self.assertEqual(
            sample["history_icd_names"][1],
            ["Atrial fibrillation", "Unspecified essential hypertension"],
        )
        for icd, ccs, names in zip(
            sample["history_icd"],
            sample["history_ccs"],
            sample["history_icd_names"],
        ):
            self.assertEqual(len(icd), len(ccs))
            self.assertEqual(len(icd), len(names))

    def test_days_ago(self):
        sample = self.task()(self.patient)[0]
        self.assertEqual(sample["days_ago"], [18.0, 8.0])

    def test_novel_drops_seen_codes(self):
        sample = self.task(setting="novel")(self.patient)[0]
        self.assertEqual(sample["ccs_targets"], ["108", "157"])
        # The history is unchanged: the model decides what the prompt shows.
        self.assertEqual(sample["conditions"], [["98", "49"], ["106", "98"]])

    def test_novel_without_new_codes_gives_no_sample(self):
        patient = make_patient(
            "p2", [("a", T1, ["4019", "4280"]), ("b", T2, ["I10", "4280"])]
        )
        # "I10" is an ICD-10 code and is unmapped as ICD-9; the target is
        # then only CHF, already seen.
        self.assertEqual(self.task(setting="novel")(patient), [])
        self.assertEqual(len(self.task()(patient)), 1)

    def test_too_few_visits(self):
        patient = make_patient("p3", [("a", T1, ["4019"])])
        self.assertEqual(self.task()(patient), [])
        patient = make_patient(
            "p4",
            [("a", T1, ["4019"]), ("b", T2, ["4280"]), ("c", T3, ["5849"])],
        )
        self.assertEqual(self.task(min_visits=4)(patient), [])
        self.assertEqual(len(self.task(min_visits=3)(patient)), 1)

    def test_visits_without_mapped_codes_are_dropped(self):
        patient = make_patient(
            "p5",
            [
                ("a", T1, ["4019"]),
                ("b", T2, ["99999"]),  # no mapped code: dropped
                ("c", T3, ["4280"]),
            ],
        )
        sample = self.task()(patient)[0]
        self.assertEqual(sample["conditions"], [["98"]])
        self.assertEqual(sample["days_ago"], [18.0])

    def test_visits_sorted_by_time(self):
        patient = make_patient(
            "p6", [("late", T3, ["4280"]), ("early", T1, ["4019"])]
        )
        sample = self.task()(patient)[0]
        self.assertEqual(sample["visit_id"], "late")
        self.assertEqual(sample["conditions"], [["98"]])

    def test_missing_hadm_ids_are_skipped(self):
        rows = [
            {"event_type": "admissions", "timestamp": T1, "admissions/hadm_id": 1.0},
            {"event_type": "admissions", "timestamp": T2, "admissions/hadm_id": 2.0},
            {"event_type": "admissions", "timestamp": T3, "admissions/hadm_id": None},
            {
                "event_type": "diagnoses_icd",
                "timestamp": T1,
                "diagnoses_icd/hadm_id": 1.0,
                "diagnoses_icd/icd9_code": "4019",
            },
            {
                "event_type": "diagnoses_icd",
                "timestamp": T2,
                "diagnoses_icd/hadm_id": 2.0,
                "diagnoses_icd/icd9_code": "4280",
            },
            {
                "event_type": "diagnoses_icd",
                "timestamp": T3,
                "diagnoses_icd/hadm_id": float("nan"),
                "diagnoses_icd/icd9_code": "5849",
            },
        ]
        patient = Patient(patient_id="p9", data_source=pl.DataFrame(rows))
        sample = self.task()(patient)[0]
        self.assertEqual(sample["visit_id"], "2")
        self.assertEqual(sample["ccs_targets"], ["108"])

    def test_base_class_needs_event_code(self):
        from pyhealth.tasks.evicare_diagnosis import _EviCareDiagnosisPrediction

        class Incomplete(_EviCareDiagnosisPrediction):
            task_name = "incomplete"

        with self.assertRaises(NotImplementedError):
            Incomplete(ccs_mapping=MAPPING_CSV)(self.patient)

    def test_numeric_hadm_ids_match(self):
        rows = [
            {"event_type": "admissions", "timestamp": T1, "admissions/hadm_id": 1.0},
            {"event_type": "admissions", "timestamp": T2, "admissions/hadm_id": 2.0},
            {
                "event_type": "diagnoses_icd",
                "timestamp": T1,
                "diagnoses_icd/hadm_id": 1.0,
                "diagnoses_icd/icd9_code": "4019",
            },
            {
                "event_type": "diagnoses_icd",
                "timestamp": T2,
                "diagnoses_icd/hadm_id": 2.0,
                "diagnoses_icd/icd9_code": "4280",
            },
        ]
        patient = Patient(patient_id="p7", data_source=pl.DataFrame(rows))
        sample = self.task()(patient)[0]
        self.assertEqual(sample["visit_id"], "2")

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.task(setting="rare")
        with self.assertRaises(ValueError):
            self.task(min_visits=1)
        with self.assertRaises(TypeError):
            EviCareDiagnosisPredictionMIMIC3(ccs_mapping=42)

    def test_accepts_mapping_instance(self):
        mapping = CCSMapping(icd9_to_ccs={"4019": "98", "4280": "108"})
        task = EviCareDiagnosisPredictionMIMIC3(ccs_mapping=mapping)
        self.assertIs(task.ccs_mapping, mapping)
        sample = task(self.patient)[0]
        self.assertEqual(sample["conditions"], [["98"], ["98"]])
        self.assertEqual(sample["history_icd_names"], [["4019"], ["4019"]])

    def test_evidence_fields_survive_sample_dataset(self):
        from pyhealth.datasets import create_sample_dataset, get_dataloader

        task = self.task()
        other = make_patient("p8", [("x", T1, ["4168"]), ("y", T2, ["4269"])])
        samples = task(self.patient) + task(other)
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="evicare_test",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertEqual(batch["history_ccs"][0], [["98", "49"], ["106", "98"]])
        self.assertEqual(batch["days_ago"][1], [10.0])
        self.assertEqual(tuple(batch["ccs_targets"].shape), (2, 4))

    def test_task_params_are_stable_for_cache(self):
        # The dataset cache hashes str() of every task attribute.
        first = {k: str(v) for k, v in vars(self.task()).items()}
        second = {k: str(v) for k, v in vars(self.task()).items()}
        self.assertEqual(first, second)
        novel = {k: str(v) for k, v in vars(self.task(setting="novel")).items()}
        self.assertNotEqual(first, novel)


class TestEviCareTaskMIMIC4(unittest.TestCase):
    def test_mixed_icd_versions(self):
        patient = make_patient(
            "p1",
            [
                ("a", T1, [("4019", 9), ("E119", "10")]),
                ("b", T2, [("I509", 10.0), ("4280", "9")]),
                ("c", T3, [("I480", 10), ("4168", 9), ("4269", None)]),
            ],
            mimic4=True,
        )
        task = EviCareDiagnosisPredictionMIMIC4(ccs_mapping=MAPPING_CSV)
        sample = task(patient)[0]
        self.assertEqual(sample["conditions"], [["98", "49"], ["108"]])
        self.assertEqual(sample["history_icd"][1], ["I509", "4280"])
        self.assertEqual(
            sample["history_icd_names"][1],
            ["Heart failure, unspecified", "Congestive heart failure, unspecified"],
        )
        # The code without a version is dropped.
        self.assertEqual(sample["ccs_targets"], ["106", "103"])

    def test_novel(self):
        patient = make_patient(
            "p2",
            [("a", T1, [("I10", 10)]), ("b", T2, [("4019", 9), ("4168", 9)])],
            mimic4=True,
        )
        task = EviCareDiagnosisPredictionMIMIC4(
            setting="novel", ccs_mapping=MAPPING_CSV
        )
        self.assertEqual(task(patient)[0]["ccs_targets"], ["103"])


if __name__ == "__main__":
    unittest.main()
