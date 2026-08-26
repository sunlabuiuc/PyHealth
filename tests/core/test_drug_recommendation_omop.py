import csv
import sys
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import DrugRecommendationOMOP

TABLES = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]

PERSON_COLS = [
    "person_id", "gender_concept_id", "year_of_birth", "month_of_birth",
    "day_of_birth", "race_concept_id", "ethnicity_concept_id",
]
VISIT_COLS = [
    "visit_occurrence_id", "person_id", "visit_concept_id", "visit_start_date",
    "visit_start_datetime", "visit_end_date", "visit_end_datetime",
    "visit_type_concept_id",
]
DEATH_COLS = ["person_id", "death_date", "death_datetime", "death_type_concept_id"]
CONDITION_COLS = [
    "person_id", "visit_occurrence_id", "condition_concept_id",
    "condition_start_date", "condition_start_datetime", "condition_end_date",
    "condition_end_datetime", "condition_type_concept_id",
]
PROCEDURE_COLS = [
    "person_id", "visit_occurrence_id", "procedure_concept_id",
    "procedure_date", "procedure_datetime", "procedure_type_concept_id",
]
DRUG_COLS = [
    "person_id", "visit_occurrence_id", "drug_concept_id",
    "drug_exposure_start_date", "drug_exposure_start_datetime",
    "drug_exposure_end_date", "drug_exposure_end_datetime",
    "drug_type_concept_id",
]

# (person_id, visit_id, "YYYY-MM-DD", conditions, procedures, drugs)
VISITS = [
    # P_ORDER: three fully-coded visits, all codes distinct.
    ("P_ORDER", "V13", "2020-03-01", ["C13"], ["P13"], ["D13"]),
    ("P_ORDER", "V11", "2020-01-01", ["C11"], ["P11"], ["D11"]),
    ("P_ORDER", "V12", "2020-02-01", ["C12"], ["P12"], ["D12"]),
    # P_SKIP: middle visit has no drug -> dropped from the sample list.
    ("P_SKIP", "V21", "2021-01-01", ["C21"], ["P21"], ["D21"]),
    ("P_SKIP", "V22", "2021-02-01", ["C22"], ["P22"], []),
    ("P_SKIP", "V23", "2021-03-01", ["C23"], ["P23"], ["D23"]),
    # P_SINGLE: only one qualifying visit -> patient dropped entirely.
    ("P_SINGLE", "V31", "2022-01-01", ["C31"], ["P31"], ["D31"]),
    ("P_SINGLE", "V32", "2022-02-01", [], [], []),
    # P_DIRTY: blank codes and OMOP's 0 sentinel must be filtered out.
    ("P_DIRTY", "V41", "2023-01-01", ["C41", ""], ["P41"], ["D41", "0"]),
    ("P_DIRTY", "V42", "2023-02-01", ["C42"], ["P42"], ["D42"]),
]


def _write(path: Path, columns, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_omop_fixture(root: Path) -> None:
    """Writes a minimal, fully-coded OMOP CDM v5.3 fixture under `root`."""
    root.mkdir(parents=True, exist_ok=True)

    persons = sorted({person for person, *_ in VISITS})
    _write(root / "person.csv", PERSON_COLS, [
        {
            "person_id": person, "gender_concept_id": "8507",
            "year_of_birth": "1970", "month_of_birth": "01",
            "day_of_birth": "01", "race_concept_id": "0",
            "ethnicity_concept_id": "0",
        }
        for person in persons
    ])
    _write(root / "death.csv", DEATH_COLS, [])

    visits, conditions, procedures, drugs = [], [], [], []
    for person, visit, day, cond_codes, proc_codes, drug_codes in VISITS:
        stamp = f"{day} 12:00:00"
        visits.append({
            "visit_occurrence_id": visit, "person_id": person,
            "visit_concept_id": "9201", "visit_start_date": day,
            "visit_start_datetime": stamp, "visit_end_date": day,
            "visit_end_datetime": stamp, "visit_type_concept_id": "32817",
        })
        for code in cond_codes:
            conditions.append({
                "person_id": person, "visit_occurrence_id": visit,
                "condition_concept_id": code, "condition_start_date": day,
                "condition_start_datetime": stamp, "condition_end_date": day,
                "condition_end_datetime": stamp,
                "condition_type_concept_id": "32020",
            })
        for code in proc_codes:
            procedures.append({
                "person_id": person, "visit_occurrence_id": visit,
                "procedure_concept_id": code, "procedure_date": day,
                "procedure_datetime": stamp,
                "procedure_type_concept_id": "32020",
            })
        for code in drug_codes:
            drugs.append({
                "person_id": person, "visit_occurrence_id": visit,
                "drug_concept_id": code,
                "drug_exposure_start_date": day,
                "drug_exposure_start_datetime": stamp,
                "drug_exposure_end_date": day,
                "drug_exposure_end_datetime": stamp,
                "drug_type_concept_id": "32020",
            })

    # An orphan drug row: no visit_occurrence_id, must be ignored silently.
    drugs.append({
        "person_id": "P_DIRTY", "visit_occurrence_id": "",
        "drug_concept_id": "D_ORPHAN",
        "drug_exposure_start_date": "2023-01-01",
        "drug_exposure_start_datetime": "2023-01-01 12:00:00",
        "drug_exposure_end_date": "2023-01-01",
        "drug_exposure_end_datetime": "2023-01-01 12:00:00",
        "drug_type_concept_id": "32020",
    })

    _write(root / "visit_occurrence.csv", VISIT_COLS, visits)
    _write(root / "condition_occurrence.csv", CONDITION_COLS, conditions)
    _write(root / "procedure_occurrence.csv", PROCEDURE_COLS, procedures)
    _write(root / "drug_exposure.csv", DRUG_COLS, drugs)


class _OMOPFixtureCase(unittest.TestCase):
    """Builds the fixture and the dataset once, in throwaway directories."""

    @classmethod
    def setUpClass(cls):
        # litdata keeps its chunk files memory-mapped; on Windows the handles
        # are still open when the directory is removed. Elsewhere a cleanup
        # failure is a real defect and must surface.
        windows = sys.platform == "win32"
        cls._root_dir = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=windows
        )
        cls._cache_dir = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=windows
        )
        write_omop_fixture(Path(cls._root_dir.name))
        cls.dataset = OMOPDataset(
            root=cls._root_dir.name,
            tables=TABLES,
            cache_dir=cls._cache_dir.name,
        )

    @classmethod
    def tearDownClass(cls):
        cls._cache_dir.cleanup()
        cls._root_dir.cleanup()

    def samples_for(self, person_id: str):
        return DrugRecommendationOMOP()(self.dataset.get_patient(person_id))


class TestDrugRecommendationOMOPUnit(_OMOPFixtureCase):
    """Unit-level behaviour of DrugRecommendationOMOP.__call__."""

    def test_one_sample_per_qualifying_visit(self):
        samples = self.samples_for("P_ORDER")
        self.assertEqual(len(samples), 3)

    def test_visits_are_chronological_not_file_order(self):
        # P_ORDER's rows are written V13, V11, V12 in the CSV.
        samples = self.samples_for("P_ORDER")
        self.assertEqual([s["visit_id"] for s in samples], ["V11", "V12", "V13"])

    def test_history_excludes_current_visit(self):
        for sample_index, sample in enumerate(self.samples_for("P_ORDER")):
            with self.subTest(visit=sample["visit_id"]):
                self.assertEqual(sample["drugs_hist"][sample_index], [])

    def test_history_holds_the_right_visit_in_the_right_slot(self):
        samples = self.samples_for("P_ORDER")
        self.assertEqual(samples[0]["drugs_hist"], [[]])
        self.assertEqual(samples[1]["drugs_hist"], [["D11"], []])
        self.assertEqual(samples[2]["drugs_hist"], [["D11"], ["D12"], []])

    def test_conditions_and_procedures_include_current_visit(self):
        samples = self.samples_for("P_ORDER")
        self.assertEqual(samples[2]["conditions"], [["C11"], ["C12"], ["C13"]])
        self.assertEqual(samples[2]["procedures"], [["P11"], ["P12"], ["P13"]])

    def test_target_is_the_current_visit_drugs(self):
        samples = self.samples_for("P_ORDER")
        self.assertEqual([s["drugs"] for s in samples], [["D11"], ["D12"], ["D13"]])

    def test_incomplete_visit_is_dropped_from_history(self):
        samples = self.samples_for("P_SKIP")
        self.assertEqual([s["visit_id"] for s in samples], ["V21", "V23"])
        # V22 had no drug: it must not occupy a history slot.
        self.assertEqual(samples[1]["drugs_hist"], [["D21"], []])
        self.assertNotIn(["C22"], samples[1]["conditions"])

    def test_patient_with_one_qualifying_visit_is_dropped(self):
        self.assertEqual(self.samples_for("P_SINGLE"), [])

    def test_blank_and_zero_concept_ids_are_filtered(self):
        samples = self.samples_for("P_DIRTY")
        self.assertEqual(samples[0]["conditions"][0], ["C41"])
        self.assertEqual(samples[0]["drugs"], ["D41"])
        for sample in samples:
            for slot in sample["drugs_hist"]:
                self.assertNotIn("0", slot)
                self.assertNotIn("", slot)

    def test_orphan_event_without_visit_id_is_ignored(self):
        for sample in self.samples_for("P_DIRTY"):
            self.assertNotIn("D_ORPHAN", sample["drugs"])
            for slot in sample["drugs_hist"]:
                self.assertNotIn("D_ORPHAN", slot)

    def test_samples_do_not_share_list_objects(self):
        # Regression guard: history slots used to alias the target lists,
        # so mutating one sample silently rewrote another.
        samples = self.samples_for("P_ORDER")
        self.assertIsNot(samples[1]["drugs_hist"][0], samples[0]["drugs"])
        samples[0]["drugs"].append("MUTATED")
        self.assertEqual(samples[1]["drugs_hist"][0], ["D11"])
        self.assertEqual(samples[2]["drugs_hist"][0], ["D11"])

    def test_visit_ids_are_strings(self):
        for sample in self.samples_for("P_ORDER"):
            self.assertIsInstance(sample["visit_id"], str)


class TestDrugRecommendationOMOPSchema(unittest.TestCase):
    """Declared schema — the contract set_task() relies on."""

    def test_schema_is_declared_on_the_class(self):
        self.assertIn("task_name", vars(DrugRecommendationOMOP))
        self.assertIn("input_schema", vars(DrugRecommendationOMOP))
        self.assertIn("output_schema", vars(DrugRecommendationOMOP))
        self.assertEqual(
            "DrugRecommendationOMOP", DrugRecommendationOMOP.task_name
        )

    def test_schema_matches_the_sibling_tasks(self):
        self.assertEqual(
            DrugRecommendationOMOP.input_schema,
            {
                "conditions": "nested_sequence",
                "procedures": "nested_sequence",
                "drugs_hist": "nested_sequence",
            },
        )
        self.assertEqual(
            DrugRecommendationOMOP.output_schema, {"drugs": "multilabel"}
        )

    def test_code_mapping_does_not_mutate_the_class(self):
        task = DrugRecommendationOMOP(
            code_mapping={"conditions": ("ICD9CM", "CCSCM")}
        )
        self.assertIsInstance(task.input_schema["conditions"], tuple)
        self.assertEqual(
            DrugRecommendationOMOP.input_schema["conditions"], "nested_sequence"
        )


class TestDrugRecommendationOMOPIntegration(_OMOPFixtureCase):
    """The claim the PR actually needs to prove: it runs through set_task()."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_dataset = cls.dataset.set_task(DrugRecommendationOMOP())

    @classmethod
    def tearDownClass(cls):
        try:
            cls.sample_dataset.close()
        except OSError:
            if sys.platform != "win32":
                raise
        super().tearDownClass()

    def test_set_task_produces_the_expected_number_of_samples(self):
        expected = sum(
            len(DrugRecommendationOMOP()(self.dataset.get_patient(person)))
            for person in ("P_ORDER", "P_SKIP", "P_SINGLE", "P_DIRTY")
        )
        self.assertEqual(len(self.sample_dataset), expected)
        self.assertGreater(expected, 0)

    def test_processed_samples_expose_the_declared_keys(self):
        for sample in self.sample_dataset:
            self.assertIn("patient_id", sample)
            self.assertIn("visit_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs_hist", sample)
            self.assertIn("drugs", sample)


if __name__ == "__main__":
    unittest.main()
