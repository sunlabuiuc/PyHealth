"""Tests for InHospitalMortalityTemporalMIMIC4 using mimic4demo synthetic data.

The demo dataset has patients 10001-10010 (adults) and patients 1-3 (edge cases).
Patient 1 is a minor (age 17). Admissions 19999/20000 have no procedures.
Admissions 20006 and 20013 have hospital_expire_flag=1 (died).
"""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import InHospitalMortalityTemporalMIMIC4

DEMO_ROOT = str(
    Path(__file__).parent.parent.parent
    / "test-resources" / "core" / "mimic4demo"
)


class TestTemporalMortalityMIMIC4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        ds = MIMIC4Dataset(
            ehr_root=DEMO_ROOT,
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.tmpdir.name,
        )
        cls.samples = ds.set_task(InHospitalMortalityTemporalMIMIC4())

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.tmpdir.cleanup()

    # -- schema checks --

    def test_task_name_and_schemas(self):
        t = InHospitalMortalityTemporalMIMIC4
        self.assertEqual(t.task_name, "InHospitalMortalityTemporalMIMIC4")
        self.assertEqual(t.input_schema, {
            "conditions": "sequence", "procedures": "sequence", "drugs": "sequence"
        })
        self.assertEqual(t.output_schema, {"mortality": "binary"})

    def test_all_samples_have_required_keys(self):
        for s in self.samples:
            for k in ["patient_id", "admission_id", "conditions",
                       "procedures", "drugs", "mortality", "admission_year"]:
                self.assertIn(k, s)

    # -- label checks --

    def test_died_admissions_have_mortality_1(self):
        died = {s["admission_id"] for s in self.samples if s["mortality"].item() == 1}
        self.assertIn("20006", died)
        self.assertIn("20013", died)

    def test_survived_admissions_have_mortality_0(self):
        survived_ids = ["20001", "20002", "20005", "20007", "20012"]
        for s in self.samples:
            if s["admission_id"] in survived_ids:
                self.assertEqual(s["mortality"].item(), 0)

    # -- feature checks --

    def test_admission_year_type_and_values(self):
        year_map = {s["admission_id"]: s["admission_year"] for s in self.samples}
        for yr in year_map.values():
            self.assertIsInstance(yr, int)
        # spot check known years from the csv
        if "20001" in year_map:
            self.assertEqual(year_map["20001"], 2150)
        if "20003" in year_map:
            self.assertEqual(year_map["20003"], 2151)
        if "20005" in year_map:
            self.assertEqual(year_map["20005"], 2152)

    def test_features_not_empty(self):
        for s in self.samples:
            # only check the "real" patients (ids starting with 2xxxx)
            if str(s["admission_id"]).startswith("2"):
                self.assertGreater(len(s["conditions"]), 0)
                self.assertGreater(len(s["procedures"]), 0)
                self.assertGreater(len(s["drugs"]), 0)

    # -- edge cases --

    def test_minor_patient_excluded(self):
        # patient 1 has anchor_age=17
        self.assertNotIn("1", [s["patient_id"] for s in self.samples])

    def test_admissions_missing_procedures_excluded(self):
        # 19999 and 20000 dont have procedures in the test data
        aids = [s["admission_id"] for s in self.samples]
        self.assertNotIn("19999", aids)
        self.assertNotIn("20000", aids)

    def test_patients_with_multiple_admissions(self):
        counts = Counter(s["patient_id"] for s in self.samples)
        # patient 10001: admissions 20001 + 20002 (19999 has no procedures)
        self.assertEqual(counts.get("10001", 0), 2)
        # patient 10003: admissions 20005 + 20006
        self.assertEqual(counts.get("10003", 0), 2)


if __name__ == "__main__":
    unittest.main()
