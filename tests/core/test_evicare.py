"""Tests for EviCare: the diagnosis prediction tasks and the model.

Contributor: Kavish Shah (GitHub: thekavishshah)
Paper: Zhang et al., "EviCare: Enhancing Diagnosis Prediction with Deep
    Model-Guided Evidence for In-Context Reasoning", KDD 2026,
    https://arxiv.org/abs/2604.10455

All tests use tiny synthetic patients and a 13-row CCS fixture: no MIMIC
data, downloads or network calls (LLMs are plain Python functions).
"""

import os
import pickle
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN, CooccurrenceGraph, EviCare, EviCarePrediction
from pyhealth.models.evicare import (
    SYSTEM_PROMPT,
    accuracy_at_k_code,
    build_prompt,
    eligible_codes,
    format_time_gap,
    merge_ranking,
    parse_answer,
    precision_at_k_visit,
    prioritize_history,
    rank_codes,
    relational_evidence,
    select_candidates,
)
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
        kwargs.setdefault("sample_mode", "last")
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
        with self.assertRaises(ValueError):
            self.task(sample_mode="every")
        with self.assertRaises(TypeError):
            EviCareDiagnosisPredictionMIMIC3(ccs_mapping=42)

    def test_accepts_mapping_instance(self):
        mapping = CCSMapping(icd9_to_ccs={"4019": "98", "4280": "108"})
        task = EviCareDiagnosisPredictionMIMIC3(ccs_mapping=mapping, sample_mode="last")
        self.assertIs(task.ccs_mapping, mapping)
        sample = task(self.patient)[0]
        self.assertEqual(sample["conditions"], [["98"], ["98"]])
        self.assertEqual(sample["history_icd_names"], [["4019"], ["4019"]])

    def test_evidence_fields_survive_sample_dataset(self):
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


class TestEviCareSampleModes(unittest.TestCase):
    def setUp(self):
        t4 = datetime(2100, 2, 1, tzinfo=UTC)
        self.patient = make_patient(
            "p1",
            [
                ("h1", T1, ["4019"]),
                ("h2", T2, ["4019", "42731"]),
                ("h3", T3, ["4280", "42731"]),
                ("h4", t4, ["4019", "4280"]),
            ],
        )

    def task(self, **kwargs):
        return EviCareDiagnosisPredictionMIMIC3(ccs_mapping=MAPPING_CSV, **kwargs)

    def test_default_is_all_prefixes(self):
        self.assertEqual(self.task().sample_mode, "all_prefixes")

    def test_all_prefixes_overall(self):
        samples = self.task()(self.patient)
        self.assertEqual([s["visit_id"] for s in samples], ["h2", "h3", "h4"])
        self.assertEqual(samples[0]["conditions"], [["98"]])
        self.assertEqual(samples[1]["conditions"], [["98"], ["98", "106"]])
        self.assertEqual(len(samples[2]["conditions"]), 3)
        self.assertEqual(samples[1]["ccs_targets"], ["108", "106"])
        # Time gaps are measured to each sample's own target.
        self.assertEqual(samples[0]["days_ago"], [10.0])
        self.assertEqual(samples[1]["days_ago"], [18.0, 8.0])
        self.assertEqual(samples[2]["days_ago"], [31.0, 21.0, 13.0])

    def test_all_prefixes_novel_skips_targets_without_new_codes(self):
        samples = self.task(setting="novel")(self.patient)
        # h2 adds AF (106), h3 adds CHF (108), h4 adds nothing new.
        self.assertEqual([s["visit_id"] for s in samples], ["h2", "h3"])
        self.assertEqual(samples[0]["ccs_targets"], ["106"])
        self.assertEqual(samples[1]["ccs_targets"], ["108"])

    def test_last_matches_last_prefix(self):
        last = self.task(sample_mode="last")(self.patient)
        prefixes = self.task()(self.patient)
        self.assertEqual(len(last), 1)
        self.assertEqual(last[0], prefixes[-1])

    def test_min_visits_applies_to_patient(self):
        self.assertEqual(self.task(min_visits=5)(self.patient), [])
        self.assertEqual(len(self.task(min_visits=4)(self.patient)), 3)

    def test_sample_mode_changes_cache_key(self):
        last = {k: str(v) for k, v in vars(self.task(sample_mode="last")).items()}
        every = {k: str(v) for k, v in vars(self.task()).items()}
        self.assertNotEqual(last, every)


class TestPatientCodeSets(unittest.TestCase):
    """``G`` must count every training patient, whatever the task setting."""

    def setUp(self):
        t4 = datetime(2100, 2, 1, tzinfo=UTC)
        # p_repeat: both admissions coded {98, 106}; no novel target.
        # p_rare: 157 (renal failure) appears only in this patient, whose
        #   admissions repeat {157}: no novel sample either.
        # p_late: the last admission adds nothing new, an earlier one does,
        #   so novel + sample_mode="last" gives no sample.
        # p_new: every admission adds a code.
        # p_single: one admission, below min_visits.
        self.patients = [
            make_patient(
                "p_repeat",
                [("r1", T1, ["4019", "42731"]), ("r2", T2, ["42731", "4019"])],
            ),
            make_patient("p_rare", [("x1", T1, ["5849"]), ("x2", T3, ["5849"])]),
            make_patient(
                "p_late",
                [
                    ("l1", T1, ["4019"]),
                    ("l2", T2, ["4280", "4019"]),
                    ("l3", T3, ["4280"]),
                ],
            ),
            make_patient(
                "p_new",
                [("n1", T1, ["41401"]), ("n2", T2, ["4019"]), ("n3", t4, ["4269"])],
            ),
            make_patient("p_single", [("s1", T1, ["4168", "4019"])]),
        ]
        # Hand-written rows of A (all CCS codes of all admissions).
        self.expected = {
            "p_late": ["108", "98"],
            "p_new": ["101", "105", "98"],
            "p_rare": ["157"],
            "p_repeat": ["106", "98"],
        }

    def task(self, **kwargs):
        return EviCareDiagnosisPredictionMIMIC3(ccs_mapping=MAPPING_CSV, **kwargs)

    def samples(self, task):
        return [s for p in self.patients for s in task(p)]

    def test_novel_repeat_patient_has_no_sample_but_is_counted(self):
        task = self.task(setting="novel")
        self.assertEqual(task(self.patients[0]), [])
        codes = task.patient_code_sets(self.patients)
        self.assertEqual(codes["p_repeat"], ["106", "98"])
        graph = CooccurrenceGraph.fit(codes.values())
        self.assertEqual(graph.count("98", "106"), 1)  # p_repeat only
        self.assertEqual(graph.count("106", "106"), 1)
        self.assertEqual(graph.count("98", "98"), 3)  # p_repeat, p_late, p_new

    def test_code_only_in_no_sample_patients_is_in_vocabulary(self):
        task = self.task(setting="novel")
        samples = self.samples(task)
        sampled_codes = {c for s in samples for c in s["ccs_targets"]}
        sampled_codes.update(c for s in samples for v in s["history_ccs"] for c in v)
        self.assertNotIn("157", sampled_codes)
        self.assertNotIn("p_rare", {s["patient_id"] for s in samples})
        model = EviCare(make_dataset(samples), setting="novel")
        self.assertNotIn("157", model.label_codes)
        model.fit_cooccurrence(task.patient_code_sets(self.patients))
        self.assertIn("157", model.cooccurrence.codes)
        self.assertEqual(model.cooccurrence.count("157", "157"), 1)
        self.assertTrue(set(model.label_codes) <= set(model.cooccurrence.codes))

    def test_sample_mode_last(self):
        task = self.task(setting="novel", sample_mode="last")
        self.assertEqual(task(self.patients[2]), [])  # p_late gives no sample
        codes = task.patient_code_sets(self.patients)
        self.assertEqual(codes["p_late"], ["108", "98"])
        self.assertEqual(codes, self.expected)

    def test_independent_of_setting_and_sample_mode(self):
        for setting in ("overall", "novel"):
            for mode in ("all_prefixes", "last"):
                with self.subTest(setting=setting, sample_mode=mode):
                    task = self.task(setting=setting, sample_mode=mode)
                    self.assertEqual(
                        task.patient_code_sets(self.patients), self.expected
                    )

    def test_graph_equals_hand_computed_a_transpose_a(self):
        vocab = ["101", "105", "106", "108", "157", "98"]
        # Rows: p_late, p_new, p_rare, p_repeat (patient ID order).
        a = np.array(
            [
                [0, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
            ]
        )
        for setting in ("overall", "novel"):
            with self.subTest(setting=setting):
                task = self.task(setting=setting)
                graph = CooccurrenceGraph.fit(
                    task.patient_code_sets(self.patients).values()
                )
                self.assertEqual(graph.codes, vocab)
                np.testing.assert_array_equal(graph.counts, a.T @ a)

    def test_patient_ids_and_min_visits(self):
        task = self.task()
        subset = task.patient_code_sets(self.patients, patient_ids=["p_rare"])
        self.assertEqual(subset, {"p_rare": ["157"]})
        self.assertNotIn("p_single", task.patient_code_sets(self.patients))
        self.assertEqual(task.patient_codes(self.patients[4]), [])
        self.assertEqual(
            self.task(min_visits=3).patient_code_sets(self.patients),
            {k: self.expected[k] for k in ("p_late", "p_new")},
        )

    def test_reads_a_base_dataset_in_batches(self):
        events = pl.concat(
            [
                p.data_source.with_columns(pl.lit(p.patient_id).alias("patient_id"))
                for p in self.patients
            ],
            how="diagonal",
        )
        dataset = SimpleNamespace(
            global_event_df=events.lazy(),
            unique_patient_ids=[p.patient_id for p in self.patients],
        )
        task = self.task(setting="novel")
        self.assertEqual(task.patient_code_sets(dataset, batch_size=2), self.expected)
        self.assertEqual(
            task.patient_code_sets(dataset, patient_ids=["p_new", "p_single"]),
            {"p_new": self.expected["p_new"]},
        )


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
        sample = task(patient)[-1]
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


# ---------------------------------------------------------------------------
# Model: helpers
# ---------------------------------------------------------------------------

NAMES = {
    "98": "Essential hypertension",
    "99": "Hypertension with complications and secondary hypertension",
    "103": "Pulmonary heart disease",
    "105": "Conduction disorders",
    "106": "Cardiac dysrhythmias",
    "108": "Congestive heart failure; nonhypertensive",
    "157": "Acute and unspecified renal failure",
}


class TestHelpers(unittest.TestCase):
    def test_format_time_gap(self):
        self.assertEqual(format_time_gap(0), "today")
        self.assertEqual(format_time_gap(1), "1 day ago")
        self.assertEqual(format_time_gap(31), "1 month 1 day ago")
        self.assertEqual(format_time_gap(60), "2 months ago")
        self.assertEqual(format_time_gap(365 * 2 + 3), "2 years 3 days ago")

    def test_rank_codes(self):
        codes = ["a", "b", "c", "d"]
        self.assertEqual(rank_codes([0.1, 2.0, -1.0, 2.0], codes), ["b", "d", "a", "c"])
        tensor = torch.tensor([[3.0, 1.0, 2.0, 0.0]], requires_grad=True)
        self.assertEqual(rank_codes(tensor, codes), ["a", "c", "b", "d"])
        with self.assertRaises(ValueError):
            rank_codes([1.0], codes)

    def test_candidates_overall_keep_recurring_codes(self):
        ranking = ["98", "108", "106", "157", "105"]
        self.assertEqual(select_candidates(ranking, ["98", "106"], k=3), ranking[:3])

    def test_candidates_novel_filter_before_top_k(self):
        ranking = ["98", "108", "106", "157", "105"]
        self.assertEqual(
            select_candidates(ranking, ["98", "106"], k=3, setting="novel"),
            ["108", "157", "105"],
        )
        # Fewer eligible codes than k: all of them.
        self.assertEqual(
            select_candidates(ranking, ranking[:4], k=3, setting="novel"), ["105"]
        )
        with self.assertRaises(ValueError):
            eligible_codes(ranking, [], setting="rare")

    def test_prioritize_history(self):
        groups = prioritize_history(
            ["AF", "HTN", "AF", "Flutter", "Unknown thing"],
            ["106", "98", "106", "106", "999"],
            ranking=["98", "108", "106"],
        )
        # By guide rank; the code missing from the ranking goes last; the
        # repeated ICD name is listed once.
        self.assertEqual(
            groups,
            [("98", ["HTN"]), ("106", ["AF", "Flutter"]), ("999", ["Unknown thing"])],
        )
        with self.assertRaises(ValueError):
            prioritize_history(["a"], [], ranking=[])

    def test_merge_ranking(self):
        self.assertEqual(merge_ranking([], ["a", "b"]), ["a", "b"])
        self.assertEqual(merge_ranking(["b", "b"], ["a", "b", "c"]), ["b", "a", "c"])


class TestCooccurrenceGraph(unittest.TestCase):
    def setUp(self):
        self.patients = [["98", "108", "98"], ["98", "106"], ["106", "105"], ["98"]]
        self.graph = CooccurrenceGraph.fit(self.patients)

    def test_matches_a_transpose_a(self):
        codes = self.graph.codes
        a = np.array(
            [[int(code in set(p)) for code in codes] for p in self.patients]
        )
        np.testing.assert_array_equal(self.graph.counts, a.T @ a)

    def test_hand_counts(self):
        self.assertEqual(self.graph.count("98", "98"), 3)
        self.assertEqual(self.graph.count("98", "108"), 1)
        self.assertEqual(self.graph.count("108", "98"), 1)
        self.assertEqual(self.graph.count("98", "105"), 0)
        self.assertEqual(self.graph.count("98", "unknown"), 0)

    def test_best_partner(self):
        # 106 co-occurs with 98 (1 patient) and 105 (1 patient): tie goes to
        # the first code of the history.
        self.assertEqual(self.graph.best_partner("106", ["105", "98"]), "105")
        self.assertEqual(self.graph.best_partner("106", ["98", "105"]), "98")
        self.assertIsNone(self.graph.best_partner("105", ["98", "108"]))
        self.assertIsNone(self.graph.best_partner("98", ["98"]))

    def test_vocab_adds_zero_rows(self):
        graph = CooccurrenceGraph.fit([["98", "108"]], vocab=["157", "98"])
        self.assertEqual(graph.codes, ["108", "157", "98"])
        self.assertEqual(graph.count("157", "157"), 0)
        self.assertEqual(graph.count("98", "108"), 1)
        self.assertEqual(CooccurrenceGraph.fit([], vocab=["a"]).codes, ["a"])

    def test_empty_and_bad_shape(self):
        empty = CooccurrenceGraph.fit([])
        self.assertEqual(empty.codes, [])
        self.assertEqual(empty.count("a", "a"), 0)
        with self.assertRaises(ValueError):
            CooccurrenceGraph(["a", "b"], np.zeros((1, 1)))

    def test_relational_evidence(self):
        links = relational_evidence(
            ["98", "108", "105", "157"], history=["98", "106"], graph=self.graph
        )
        # 98 is historical (skipped), 157 never co-occurs (skipped).
        self.assertEqual(links, [("98", "108"), ("106", "105")])


class TestBuildPrompt(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "history_icd_names": [
                ["Unspecified essential hypertension"],
                ["Atrial fibrillation", "Hypertension NOS", "Flutter"],
            ],
            "history_ccs": [["98"], ["106", "98", "106"]],
            "days_ago": [400.0, 8.0],
            "candidates": ["108", "105", "98"],
            "ranking": ["108", "98", "105", "106"],
            "ccs_names": NAMES,
        }

    def test_novel_shows_only_last_visit(self):
        prompt = build_prompt(setting="novel", **self.kwargs)
        self.assertIn("Last Diagnostic Visit (8 days ago):", prompt)
        self.assertNotIn("Unspecified essential hypertension", prompt)
        self.assertNotIn("Visit 1", prompt)
        self.assertIn("Candidate CCS Codes (Novel Only):", prompt)
        self.assertIn("NEW", prompt)

    def test_prioritized_groups(self):
        prompt = build_prompt(setting="novel", **self.kwargs)
        htn = '- {"Hypertension NOS"} BELONG TO "Essential hypertension"'
        af = '- {"Atrial fibrillation", "Flutter"} BELONG TO "Cardiac dysrhythmias"'
        self.assertIn(htn, prompt)
        self.assertIn(af, prompt)
        # 98 ranks above 106 in the guide ranking.
        self.assertLess(prompt.index(htn), prompt.index(af))

    def test_overall_shows_all_visits(self):
        prompt = build_prompt(setting="overall", **self.kwargs)
        self.assertIn("Visit 1 (1 year 1 month 5 days ago):", prompt)
        self.assertIn("Unspecified essential hypertension", prompt)
        self.assertIn("Last Diagnostic Visit (8 days ago):", prompt)
        self.assertIn("Candidate CCS Codes:\n", prompt)
        self.assertIn("may repeat earlier diagnoses", prompt)
        limited = build_prompt(setting="overall", max_visits=1, **self.kwargs)
        self.assertNotIn("Visit 1 (", limited)

    def test_candidates_relations_and_instruction(self):
        prompt = build_prompt(
            setting="novel", relations=[("106", "105")], **self.kwargs
        )
        self.assertIn('"Cardiac dysrhythmias" => "Conduction disorders"', prompt)
        self.assertIn(
            '"Congestive heart failure; nonhypertensive", '
            '"Conduction disorders", "Essential hypertension"',
            prompt,
        )
        self.assertTrue(
            prompt.endswith("- Output format: Answer: <CCS 1>, <CCS 2>, ...")
        )
        no_relations = build_prompt(setting="novel", **self.kwargs)
        self.assertNotIn("Relational Evidence", no_relations)

    def test_missing_names_and_quotes(self):
        kwargs = dict(self.kwargs, ccs_names=None)
        kwargs["history_icd_names"] = [["x"], ['Say "hi"', "y", "z"]]
        prompt = build_prompt(setting="novel", **kwargs)
        self.assertIn('BELONG TO "106"', prompt)
        self.assertIn("\"Say 'hi'\"", prompt)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            build_prompt(setting="rare", **self.kwargs)
        with self.assertRaises(ValueError):
            build_prompt(**dict(self.kwargs, days_ago=[1.0]))

    def test_system_prompt_asks_for_answer_line(self):
        self.assertIn("Answer: <CCS 1>, <CCS 2>, ...", SYSTEM_PROMPT)


class TestParseAnswer(unittest.TestCase):
    candidates = ("108", "157", "105", "98")

    def parse(self, reply, names=NAMES):
        return parse_answer(reply, self.candidates, names)

    def test_names_in_order(self):
        reply = (
            "Answer: Conduction disorders, Congestive heart failure; "
            "nonhypertensive, Essential hypertension"
        )
        self.assertEqual(self.parse(reply), ["105", "108", "98"])

    def test_junk_gives_nothing(self):
        self.assertEqual(self.parse(None), [])
        self.assertEqual(self.parse(""), [])
        self.assertEqual(self.parse("Sorry, I cannot help with that."), [])
        self.assertEqual(self.parse("Answer:"), [])

    def test_hallucinated_codes_are_dropped(self):
        reply = "Answer: Sepsis, Essential hypertension, Pneumonia, 999, 157"
        self.assertEqual(self.parse(reply), ["98", "157"])
        # A real CCS code that is not a candidate is dropped too.
        self.assertEqual(self.parse("Answer: Cardiac dysrhythmias, 98"), ["98"])

    def test_duplicates_keep_first_position(self):
        reply = "Answer: 157, essential hypertension, 157, ESSENTIAL HYPERTENSION."
        self.assertEqual(self.parse(reply), ["157", "98"])

    def test_formatting_noise(self):
        reply = (
            "Answer:\n1. “Conduction disorders”\n"
            "2) [Essential hypertension]\n- 108: Congestive heart failure\n"
            "* Acute and unspecified renal failure (157)."
        )
        self.assertEqual(self.parse(reply), ["105", "98", "108", "157"])

    def test_uses_last_answer_and_skips_thinking(self):
        reply = (
            "<think>Answer: 98, 108</think>The first guess was "
            "Answer: 157\nOn reflection, answer: 105, 157"
        )
        self.assertEqual(self.parse(reply), ["105", "157"])
        unterminated = "<think>Answer: 98 and then the model ran out of tokens"
        self.assertEqual(self.parse(unterminated), [])

    def test_without_answer_marker_parses_whole_reply(self):
        self.assertEqual(self.parse("Essential hypertension, 157"), ["98", "157"])

    def test_names_with_commas(self):
        names = {"A1": "Fracture, hip", "A2": "Sepsis"}
        self.assertEqual(
            parse_answer("Answer: Sepsis, Fracture, hip", ["A1", "A2"], names),
            ["A2", "A1"],
        )

    def test_codes_only_without_names(self):
        self.assertEqual(self.parse("Answer: 98, 105", names=None), ["98", "105"])


class TestMetrics(unittest.TestCase):
    predictions = (("a", "b", "c", "d"), ("e", "f", "g", "h"), ("i", "j"))
    targets = (("a", "c", "x"), ("f",), ())

    def test_precision_hand_computed(self):
        # Visit 1: top-2 {a, b} hits 1, min(2, 3) = 2 -> 0.5.
        # Visit 2: top-2 {e, f} hits 1, min(2, 1) = 1 -> 1.0.
        # Visit 3: empty truth, skipped.
        self.assertAlmostEqual(
            precision_at_k_visit(self.predictions, self.targets, k=2), 0.75
        )
        # k=4: (2/3 + 1/1) / 2
        self.assertAlmostEqual(
            precision_at_k_visit(self.predictions, self.targets, k=4), 5 / 6
        )

    def test_accuracy_hand_computed(self):
        # Hits (1 + 1 + 0) over truth sizes (3 + 1 + 0).
        self.assertAlmostEqual(
            accuracy_at_k_code(self.predictions, self.targets, k=2), 0.5
        )
        self.assertAlmostEqual(
            accuracy_at_k_code(self.predictions, self.targets, k=4), 0.75
        )

    def test_novel_variants(self):
        histories = (("a",), ("f",), ())
        # Visit 1: truth {c, x}, top-2 {a, b} -> 0/2. Visit 2: truth empty.
        self.assertAlmostEqual(
            precision_at_k_visit(self.predictions, self.targets, 2, histories), 0.0
        )
        # Visit 1 at k=4: {c} of {c, x} -> 1 / min(4, 2).
        self.assertAlmostEqual(
            precision_at_k_visit(self.predictions, self.targets, 4, histories), 0.5
        )
        self.assertAlmostEqual(
            accuracy_at_k_code(self.predictions, self.targets, 4, histories), 0.5
        )

    def test_duplicate_predictions_count_once(self):
        self.assertAlmostEqual(
            precision_at_k_visit([["a", "a", "b"]], [["a", "b"]], k=2), 1.0
        )

    def test_edge_cases(self):
        self.assertEqual(precision_at_k_visit([], [], k=5), 0.0)
        self.assertEqual(accuracy_at_k_code([[]], [[]], k=5), 0.0)
        with self.assertRaises(ValueError):
            precision_at_k_visit([["a"]], [], k=1)
        with self.assertRaises(ValueError):
            accuracy_at_k_code([["a"]], [], k=1)
        with self.assertRaises(ValueError):
            accuracy_at_k_code([["a"]], [["a"]], k=1, histories=[])


# ---------------------------------------------------------------------------
# Model: EviCare
# ---------------------------------------------------------------------------


def evicare_patients():
    """Four synthetic patients for the model tests."""
    t4 = datetime(2100, 2, 1, tzinfo=UTC)
    return [
        make_patient(
            "p1",
            [
                ("a1", T1, ["4019", "25000"]),
                ("a2", T2, ["42731", "4019"]),
                ("a3", T3, ["4280", "42731", "5849"]),
            ],
        ),
        make_patient("p2", [("b1", T1, ["4019"]), ("b2", T3, ["4280", "4168"])]),
        make_patient("p3", [("c1", T2, ["42731"]), ("c2", t4, ["4269", "42731"])]),
        make_patient("p4", [("d1", T1, ["41401"]), ("d2", T2, ["40291", "4019"])]),
    ]


def evicare_samples():
    """Task samples of :func:`evicare_patients` (overall setting)."""
    task = EviCareDiagnosisPredictionMIMIC3(ccs_mapping=MAPPING_CSV)
    return [s for p in evicare_patients() for s in task(p)]


def make_dataset(samples):
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence"},
        output_schema={"ccs_targets": "multilabel"},
        dataset_name="evicare_model_test",
    )


class _ConstantGuide(torch.nn.Module):
    """A guide whose logits rank the label vocabulary in a fixed order."""

    def __init__(self, scores):
        super().__init__()
        self.scores = torch.tensor(scores, dtype=torch.float)

    def forward(self, **kwargs):
        batch = len(kwargs["patient_id"])
        return {"logit": self.scores.repeat(batch, 1)}


class TestEviCareModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = evicare_samples()
        cls.dataset = make_dataset(cls.samples)
        cls.code_sets = EviCareDiagnosisPredictionMIMIC3(
            ccs_mapping=MAPPING_CSV
        ).patient_code_sets(evicare_patients())

    def setUp(self):
        torch.manual_seed(0)

    def fixed_model(self, **kwargs):
        """EviCare whose guide ranks labels in vocabulary order."""
        model = EviCare(self.dataset, ccs_names=NAMES, **kwargs)
        n = len(model.label_codes)
        model.guide = _ConstantGuide([float(n - i) for i in range(n)])
        return model

    def test_default_guide_is_retain_dim_16(self):
        from pyhealth.models import RETAIN

        model = EviCare(self.dataset)
        self.assertIsInstance(model.guide, RETAIN)
        self.assertEqual(model.guide.embedding_dim, 16)
        self.assertEqual(model.k, 50)

    def test_guide_class_and_instance(self):
        model = EviCare(self.dataset, guide_model=RNN, embedding_dim=8, hidden_dim=8)
        self.assertIsInstance(model.guide, RNN)
        guide = RNN(self.dataset, embedding_dim=4, hidden_dim=4)
        self.assertIs(EviCare(self.dataset, guide_model=guide).guide, guide)
        with self.assertRaises(TypeError):
            EviCare(self.dataset, guide_model="retain")

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            EviCare(self.dataset, setting="rare")
        with self.assertRaises(ValueError):
            EviCare(self.dataset, k=0)
        binary = create_sample_dataset(
            samples=[
                {"patient_id": "x", "conditions": [["98"]], "label": 0},
                {"patient_id": "y", "conditions": [["108"]], "label": 1},
            ],
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
            dataset_name="evicare_binary",
        )
        with self.assertRaises(ValueError):
            EviCare(binary)

    def test_forward_trains_the_guide(self):
        model = EviCare(self.dataset)
        batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        out = model(**batch)
        self.assertEqual(
            tuple(out["logit"].shape), (4, len(model.label_codes))
        )
        out["loss"].backward()
        grads = [p.grad for p in model.guide.parameters() if p.grad is not None]
        self.assertTrue(grads)

    def test_fit_cooccurrence_from_patient_code_sets(self):
        model = EviCare(self.dataset)
        model.fit_cooccurrence(self.code_sets)
        expected = CooccurrenceGraph.fit(
            self.code_sets.values(), vocab=model.label_codes
        )
        self.assertEqual(model.cooccurrence.codes, expected.codes)
        np.testing.assert_array_equal(model.cooccurrence.counts, expected.counts)
        # A plain list of per-patient code lists works too.
        model.fit_cooccurrence(list(self.code_sets.values()))
        np.testing.assert_array_equal(model.cooccurrence.counts, expected.counts)

    def test_fit_cooccurrence_vocab_covers_label_space(self):
        model = EviCare(self.dataset)
        model.fit_cooccurrence({"p1": ["98"]})  # far fewer codes than labels
        self.assertTrue(set(model.label_codes) <= set(model.cooccurrence.codes))
        for code in model.label_codes:
            if code != "98":
                self.assertEqual(model.cooccurrence.count(code, code), 0)

    def test_fit_cooccurrence_rejects_samples(self):
        model = EviCare(self.dataset)
        with self.assertRaises(TypeError):
            model.fit_cooccurrence(self.dataset)

    def test_predict_without_llm_is_the_guide(self):
        model = self.fixed_model(k=3)
        predictions = model.predict(self.dataset, batch_size=2)
        self.assertEqual(len(predictions), len(self.samples))
        first = predictions[0]
        self.assertIsInstance(first, EviCarePrediction)
        self.assertEqual(first.ranking, model.label_codes)
        self.assertEqual(first.candidates, model.label_codes[:3])
        self.assertIsNone(first.prompt)
        self.assertEqual(
            [p.visit_id for p in predictions], [s["visit_id"] for s in self.samples]
        )
        self.assertEqual(
            sorted(first.targets), sorted(self.samples[0]["ccs_targets"])
        )

    def test_predict_with_llm_reranks(self):
        model = self.fixed_model(k=4)
        model.fit_cooccurrence(self.code_sets)
        calls = []

        def llm(system, prompt):
            calls.append((system, prompt))
            return "Answer: Acute and unspecified renal failure, Not a code"

        predictions = model.predict(self.dataset, llm=llm)
        self.assertEqual(len(calls), len(self.samples))
        self.assertEqual(calls[0][0], SYSTEM_PROMPT)
        for prediction in predictions:
            self.assertEqual(calls[predictions.index(prediction)][1], prediction.prompt)
            if "157" in prediction.candidates:
                self.assertEqual(prediction.ranking[0], "157")
                self.assertEqual(prediction.n_parsed, 1)
            else:
                self.assertEqual(prediction.ranking, prediction.guide_ranking)
            self.assertEqual(sorted(prediction.ranking), sorted(model.label_codes))

    def test_novel_setting_filters_history(self):
        task = EviCareDiagnosisPredictionMIMIC3(
            ccs_mapping=MAPPING_CSV, setting="novel"
        )
        # Hypertension (98) is history for p1/p3 but a new label for p2/p4,
        # so it is in the label vocabulary and must be filtered per patient.
        histories = {"p1": "4019", "p2": "42731", "p3": "4019", "p4": "42731"}
        samples = []
        for pid, past in histories.items():
            patient = make_patient(
                pid, [("x", T1, [past]), ("y", T2, ["4280", "4019", "42731"])]
            )
            samples += task(patient)
        dataset = make_dataset(samples)
        model = EviCare(dataset, setting="novel", use_relations=False)
        self.assertIn("98", model.label_codes)
        prompts = []
        predictions = model.predict(
            dataset, llm=lambda s, p: prompts.append(p) or "Answer: 98, 106"
        )
        for prediction in predictions:
            self.assertTrue(prediction.history)
            self.assertFalse(set(prediction.ranking) & set(prediction.history))
            self.assertFalse(set(prediction.candidates) & set(prediction.history))
            # The answer's historical code is dropped; the other is kept.
            self.assertEqual(prediction.n_parsed, 1)
        self.assertIn("(Novel Only)", prompts[0])

    def test_llm_failure_falls_back_to_guide(self):
        model = self.fixed_model(k=3)
        model.fit_cooccurrence(self.code_sets)

        def broken(system, prompt):
            raise TimeoutError("endpoint down")

        with self.assertLogs("pyhealth.models.evicare", level="WARNING"):
            predictions = model.predict(self.dataset, llm=broken)
        for prediction in predictions:
            self.assertEqual(prediction.ranking, prediction.guide_ranking)
            self.assertEqual(prediction.error, "TimeoutError: endpoint down")
            self.assertIsNone(prediction.reply)

    def test_junk_reply_falls_back_to_guide(self):
        model = self.fixed_model(k=3)
        model.fit_cooccurrence(self.code_sets)
        predictions = model.predict(self.dataset, llm=lambda s, p: "no idea")
        for prediction in predictions:
            self.assertEqual(prediction.ranking, prediction.guide_ranking)
            self.assertEqual(prediction.n_parsed, 0)
            self.assertIsNone(prediction.error)

    def test_relational_evidence_in_prompt(self):
        n_labels = len(self.dataset.output_processors["ccs_targets"].label_vocab)
        model = self.fixed_model(k=n_labels)
        model.fit_cooccurrence(self.code_sets)
        prompts = []
        model.predict(self.dataset, llm=lambda s, p: prompts.append(p) or "")
        self.assertTrue(any("Relational Evidence" in p for p in prompts))
        self.assertTrue(any(" => " in p for p in prompts))

    def test_requires_cooccurrence_when_relations_are_on(self):
        model = self.fixed_model()
        with self.assertRaises(ValueError):
            model.predict(self.dataset, llm=lambda s, p: "")
        model.use_relations = False
        self.assertEqual(
            len(model.predict(self.dataset, llm=lambda s, p: "")), len(self.samples)
        )

    def test_logit_size_mismatch(self):
        model = EviCare(self.dataset, guide_model=_ConstantGuide([1.0, 0.0]))
        with self.assertRaises(ValueError):
            model.predict(self.dataset)

    def test_evaluate(self):
        predictions = [
            EviCarePrediction("p", "v", ["a", "b", "c"], [], [], targets=["a", "c"]),
            EviCarePrediction("p", "w", ["d", "e"], [], [], targets=["e"]),
        ]
        scores = EviCare.evaluate(predictions, ks=(1, 2))
        self.assertAlmostEqual(scores["P@1"], 0.5)  # (1/1 + 0/1) / 2
        self.assertAlmostEqual(scores["P@2"], 0.75)  # (1/2 + 1/1) / 2
        self.assertAlmostEqual(scores["Acc@2"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
