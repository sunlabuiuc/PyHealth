from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import InHospitalMortalityTemporalMIMIC4


class TestInHospitalMortalityTemporalMIMIC4(unittest.TestCase):
    """Tests for InHospitalMortalityTemporalMIMIC4 using synthetic mimic4demo data.

    The synthetic dataset under test-resources/core/mimic4demo contains:
      - 10 adult patients (10001-10010) plus 3 edge-case patients (1-3).
      - Patient 1 is a minor (anchor_age=17).
      - Admissions 19999 and 20000 lack procedures/prescriptions.
      - Admissions 20006 (patient 10003) and 20013 (patient 10008) have
        hospital_expire_flag=1 (died in hospital).
    """

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()

        dataset = MIMIC4Dataset(
            ehr_root=str(
                Path(__file__).parent.parent.parent
                / "test-resources"
                / "core"
                / "mimic4demo"
            ),
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache_dir.name,
        )

        cls.task = InHospitalMortalityTemporalMIMIC4()
        cls.samples = dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.cache_dir.cleanup()

    # ── Schema tests ──────────────────────────────────────────────

    def test_task_class_attributes(self):
        """Task declares task_name, input_schema, and output_schema."""
        self.assertEqual(
            InHospitalMortalityTemporalMIMIC4.task_name,
            "InHospitalMortalityTemporalMIMIC4",
        )
        self.assertEqual(
            InHospitalMortalityTemporalMIMIC4.input_schema,
            {"conditions": "sequence", "procedures": "sequence", "drugs": "sequence"},
        )
        self.assertEqual(
            InHospitalMortalityTemporalMIMIC4.output_schema,
            {"mortality": "binary"},
        )

    def test_sample_keys(self):
        """Every sample contains all required keys."""
        required = {
            "patient_id",
            "admission_id",
            "conditions",
            "procedures",
            "drugs",
            "mortality",
            "admission_year",
        }
        for sample in self.samples:
            self.assertTrue(
                required.issubset(sample.keys()),
                f"Missing keys: {required - sample.keys()}",
            )

    # ── Label generation ──────────────────────────────────────────

    def test_mortality_positive_labels(self):
        """Admissions 20006 and 20013 (hospital_expire_flag=1) get mortality=1."""
        positive_admissions = {
            s["admission_id"]
            for s in self.samples
            if bool(s["mortality"].item())
        }
        self.assertIn("20006", positive_admissions)
        self.assertIn("20013", positive_admissions)

    def test_mortality_negative_labels(self):
        """Surviving admissions get mortality=0."""
        for s in self.samples:
            aid = s["admission_id"]
            if aid in ("20001", "20002", "20003", "20004", "20005",
                       "20007", "20008", "20009", "20010", "20011",
                       "20012", "20014", "20015"):
                self.assertFalse(
                    bool(s["mortality"].item()),
                    f"Expected mortality=0 for admission {aid}",
                )

    # ── Feature extraction ────────────────────────────────────────

    def test_admission_year_present_and_integer(self):
        """admission_year is an integer for every sample."""
        for s in self.samples:
            self.assertIsInstance(s["admission_year"], int)

    def test_admission_year_values(self):
        """Spot-check admission_year against known admit dates."""
        year_by_adm = {s["admission_id"]: s["admission_year"] for s in self.samples}
        # Patient 10001 admitted in year 2150
        if "20001" in year_by_adm:
            self.assertEqual(year_by_adm["20001"], 2150)
        # Patient 10002 admitted in year 2151
        if "20003" in year_by_adm:
            self.assertEqual(year_by_adm["20003"], 2151)
        # Patient 10003 admitted in year 2152
        if "20005" in year_by_adm:
            self.assertEqual(year_by_adm["20005"], 2152)

    def test_nonempty_features(self):
        """conditions, procedures, and drugs are non-empty for every sample."""
        for s in self.samples:
            aid = s["admission_id"]
            # Only check the well-formed adult admissions
            if str(aid).startswith("2"):
                self.assertTrue(len(s["conditions"]) > 0, f"Empty conditions for {aid}")
                self.assertTrue(len(s["procedures"]) > 0, f"Empty procedures for {aid}")
                self.assertTrue(len(s["drugs"]) > 0, f"Empty drugs for {aid}")

    # ── Edge cases ────────────────────────────────────────────────

    def test_minors_excluded(self):
        """Patient 1 (anchor_age=17) should produce no samples."""
        patient_ids = [s["patient_id"] for s in self.samples]
        self.assertNotIn("1", patient_ids)

    def test_admissions_without_procedures_excluded(self):
        """Admissions 19999 and 20000 have no procedures and should be skipped."""
        admission_ids = [s["admission_id"] for s in self.samples]
        self.assertNotIn("19999", admission_ids)
        self.assertNotIn("20000", admission_ids)

    def test_multiple_admissions_per_patient(self):
        """Patients with multiple valid admissions produce multiple samples."""
        from collections import Counter

        patient_counts = Counter(s["patient_id"] for s in self.samples)
        # Patient 10001 has admissions 20001, 20002 (19999 skipped)
        self.assertEqual(patient_counts.get("10001", 0), 2)
        # Patient 10003 has admissions 20005, 20006
        self.assertEqual(patient_counts.get("10003", 0), 2)


if __name__ == "__main__":
    unittest.main()
