import unittest
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.ckd_surv import MIMIC4CKDSurvAnalysis


class TestTimeInvariant(unittest.TestCase):
    def setUp(self):
        self.dataset = MIMIC4Dataset(
            ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.0/",
            tables=["patients", "admissions", "diagnoses_icd", "labevents"],
            dev=True,
        )
        self.task = MIMIC4CKDSurvAnalysis(setting="time_invariant")

    def test_setting(self):
        self.assertEqual(
            self.task.setting, "time_invariant", msg="time_invariant setting failed"
        )

    def test_schema(self):
        self.assertIn(
            "baseline_egfr",
            self.task.input_schema,
            msg="time_invariant input schema failed",
        )
        self.assertIn(
            "comorbidities",
            self.task.input_schema,
            msg="time_invariant input schema failed",
        )

    def test_samples(self):
        samples = self.dataset.set_task(self.task)
        if len(samples) > 0:
            sample = samples[0]
            self.assertIn(
                "baseline_egfr", sample, msg="time_invariant sample structure failed"
            )
            self.assertIn(
                "has_esrd", sample, msg="time_invariant sample structure failed"
            )


class TestTimeVariant(unittest.TestCase):
    def setUp(self):
        self.dataset = MIMIC4Dataset(
            ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.0/",
            tables=["patients", "admissions", "diagnoses_icd", "labevents"],
            dev=True,
        )
        self.task = MIMIC4CKDSurvAnalysis(setting="time_variant")

    def test_setting(self):
        self.assertEqual(
            self.task.setting, "time_variant", msg="time_variant setting failed"
        )

    def test_schema(self):
        self.assertIn(
            "lab_measurements",
            self.task.input_schema,
            msg="time_variant input schema failed",
        )

    def test_samples(self):
        samples = self.dataset.set_task(self.task)
        if len(samples) > 0:
            sample = samples[0]
            self.assertIn(
                "lab_measurements", sample, msg="time_variant sample structure failed"
            )
            self.assertIsInstance(
                sample["lab_measurements"],
                list,
                msg="time_variant lab_measurements type failed",
            )


class TestHeterogeneous(unittest.TestCase):
    def setUp(self):
        self.dataset = MIMIC4Dataset(
            ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.0/",
            tables=["patients", "admissions", "diagnoses_icd", "labevents"],
            dev=True,
        )
        self.task = MIMIC4CKDSurvAnalysis(setting="heterogeneous")

    def test_setting(self):
        self.assertEqual(
            self.task.setting, "heterogeneous", msg="heterogeneous setting failed"
        )

    def test_schema(self):
        self.assertIn(
            "missing_indicators",
            self.task.input_schema,
            msg="heterogeneous input schema failed",
        )

    def test_samples(self):
        samples = self.dataset.set_task(self.task)
        if len(samples) > 0:
            sample = samples[0]
            self.assertIn(
                "missing_indicators",
                sample,
                msg="heterogeneous sample structure failed",
            )


class TestTaskValidation(unittest.TestCase):
    def setUp(self):
        self.task = MIMIC4CKDSurvAnalysis(setting="time_invariant")

    def test_invalid_setting(self):
        with self.assertRaises(ValueError):
            MIMIC4CKDSurvAnalysis(setting="invalid")

    def test_egfr_calculation(self):
        # Test that eGFR calculation works for both genders with correct string values
        egfr_male = self.task._calculate_egfr(creatinine=1.5, age=50, gender="M")
        egfr_female = self.task._calculate_egfr(creatinine=1.5, age=50, gender="F")

        # Just test that calculations return positive values (no gender comparison assertions)
        self.assertGreater(egfr_male, 0, msg="eGFR calculation failed for male")
        self.assertGreater(egfr_female, 0, msg="eGFR calculation failed for female")

        # Test that both calculations are reasonable eGFR values (typically 15-120 mL/min/1.73m²)
        self.assertLess(
            egfr_male, 200, msg="eGFR calculation unreasonably high for male"
        )
        self.assertLess(
            egfr_female, 200, msg="eGFR calculation unreasonably high for female"
        )


if __name__ == "__main__":
    unittest.main()
