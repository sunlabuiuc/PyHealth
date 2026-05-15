import csv
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import AgainstMedicalAdvicePredictionMIMIC3


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


class TestAgainstMedicalAdvicePredictionMIMIC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmpdir.name)
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._build_synthetic_mimic3(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
        cls.cache_dir.cleanup()

    @classmethod
    def _build_synthetic_mimic3(cls, root: Path):
        _write_csv(
            root / "PATIENTS.csv",
            [
                "ROW_ID",
                "SUBJECT_ID",
                "GENDER",
                "DOB",
                "DOD",
                "DOD_HOSP",
                "DOD_SSN",
                "EXPIRE_FLAG",
            ],
            [
                [1, 1, "M", "2080-01-01", "", "", "", 0],
                [2, 2, "F", "2069-02-02", "", "", "", 0],
                [3, 3, "M", "2109-03-03", "", "", "", 0],
                [4, 4, "F", "1800-04-04", "", "", "", 0],
            ],
        )

        _write_csv(
            root / "ADMISSIONS.csv",
            [
                "ROW_ID",
                "SUBJECT_ID",
                "HADM_ID",
                "ADMITTIME",
                "DISCHTIME",
                "DEATHTIME",
                "ADMISSION_TYPE",
                "ADMISSION_LOCATION",
                "DISCHARGE_LOCATION",
                "INSURANCE",
                "LANGUAGE",
                "RELIGION",
                "MARITAL_STATUS",
                "ETHNICITY",
                "EDREGTIME",
                "EDOUTTIME",
                "DIAGNOSIS",
                "HOSPITAL_EXPIRE_FLAG",
                "HAS_CHARTEVENTS_DATA",
            ],
            [
                [
                    10,
                    1,
                    1001,
                    "2120-01-10 08:00:00",
                    "2120-01-12 10:00:00",
                    "",
                    "EMERGENCY",
                    "EMERGENCY ROOM ADMIT",
                    "HOME",
                    "MEDICARE",
                    "ENGLISH",
                    "CATHOLIC",
                    "MARRIED",
                    "BLACK/AFRICAN AMERICAN",
                    "",
                    "",
                    "DX1",
                    0,
                    1,
                ],
                [
                    11,
                    2,
                    1002,
                    "2120-02-10 08:00:00",
                    "2120-02-11 08:00:00",
                    "",
                    "EMERGENCY",
                    "EMERGENCY ROOM ADMIT",
                    "LEFT AGAINST MEDICAL ADVI",
                    "PRIVATE",
                    "ENGLISH",
                    "CATHOLIC",
                    "SINGLE",
                    "WHITE",
                    "",
                    "",
                    "DX2",
                    0,
                    1,
                ],
                [
                    12,
                    3,
                    1003,
                    "2120-03-10 08:00:00",
                    "2120-03-10 20:00:00",
                    "",
                    "EMERGENCY",
                    "EMERGENCY ROOM ADMIT",
                    "LEFT AGAINST MEDICAL ADVI",
                    "SELF PAY",
                    "ENGLISH",
                    "OTHER",
                    "SINGLE",
                    "HISPANIC OR LATINO",
                    "",
                    "",
                    "DX3",
                    0,
                    1,
                ],
                [
                    13,
                    4,
                    1004,
                    "2120-04-10 08:00:00",
                    "2120-04-11 08:00:00",
                    "",
                    "ELECTIVE",
                    "PHYS REFERRAL/NORMAL DELI",
                    "HOME HEALTH CARE",
                    "MEDICAID",
                    "ENGLISH",
                    "CATHOLIC",
                    "MARRIED",
                    "ASIAN",
                    "",
                    "",
                    "DX4",
                    0,
                    1,
                ],
            ],
        )

        _write_csv(
            root / "ICUSTAYS.csv",
            [
                "ROW_ID",
                "SUBJECT_ID",
                "HADM_ID",
                "ICUSTAY_ID",
                "DBSOURCE",
                "FIRST_CAREUNIT",
                "LAST_CAREUNIT",
                "INTIME",
                "OUTTIME",
                "LOS",
            ],
            [
                [
                    1,
                    1,
                    1001,
                    5001,
                    "carevue",
                    "MICU",
                    "MICU",
                    "2120-01-10 09:00:00",
                    "2120-01-11 15:00:00",
                    1.25,
                ],
                [
                    2,
                    2,
                    1002,
                    5002,
                    "carevue",
                    "MICU",
                    "MICU",
                    "2120-02-10 09:00:00",
                    "2120-02-10 11:00:00",
                    0.08,
                ],
                [
                    3,
                    3,
                    1003,
                    5003,
                    "carevue",
                    "MICU",
                    "MICU",
                    "2120-03-10 09:00:00",
                    "2120-03-10 12:00:00",
                    0.12,
                ],
            ],
        )

        _write_csv(
            root / "NOTEEVENTS.csv",
            [
                "ROW_ID",
                "SUBJECT_ID",
                "HADM_ID",
                "CHARTDATE",
                "CHARTTIME",
                "STORETIME",
                "CATEGORY",
                "DESCRIPTION",
                "CGID",
                "ISERROR",
                "TEXT",
            ],
            [
                [
                    1,
                    1,
                    1001,
                    "2120-01-10",
                    "2120-01-10 12:00:00",
                    "2120-01-10 12:30:00",
                    "Nursing",
                    "Report",
                    100,
                    "",
                    "Patient refused medication and was hostile during counseling.",
                ],
                [
                    2,
                    2,
                    1002,
                    "2120-02-10",
                    "2120-02-10 12:00:00",
                    "2120-02-10 12:30:00",
                    "Physician ",
                    "Report",
                    101,
                    "",
                    "Patient appears calm and cooperative.",
                ],
                [
                    3,
                    2,
                    1002,
                    "2120-02-10",
                    "2120-02-10 18:00:00",
                    "2120-02-10 18:30:00",
                    "Radiology",
                    "Report",
                    102,
                    "",
                    "No acute findings.",
                ],
            ],
        )

        _write_csv(
            root / "DIAGNOSES_ICD.csv",
            ["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
            [
                [1, 1, 1001, 1, "4019"],
                [2, 2, 1002, 1, "25000"],
                [3, 4, 1004, 1, "41401"],
            ],
        )
        _write_csv(
            root / "PROCEDURES_ICD.csv",
            ["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
            [
                [1, 1, 1001, 1, "3893"],
                [2, 2, 1002, 1, "8856"],
                [3, 4, 1004, 1, "3615"],
            ],
        )
        _write_csv(
            root / "PRESCRIPTIONS.csv",
            [
                "ROW_ID",
                "SUBJECT_ID",
                "HADM_ID",
                "ICUSTAY_ID",
                "STARTDATE",
                "ENDDATE",
                "DRUG_TYPE",
                "DRUG",
                "DRUG_NAME_POE",
                "DRUG_NAME_GENERIC",
                "FORMULARY_DRUG_CD",
                "GSN",
                "NDC",
                "PROD_STRENGTH",
                "DOSE_VAL_RX",
                "DOSE_UNIT_RX",
                "FORM_VAL_DISP",
                "FORM_UNIT_DISP",
                "ROUTE",
            ],
            [
                [
                    1,
                    1,
                    1001,
                    "",
                    "2120-01-10",
                    "2120-01-11",
                    "MAIN",
                    "Aspirin",
                    "Aspirin",
                    "Aspirin",
                    "ASP",
                    "1",
                    "11111",
                    "81mg",
                    "1",
                    "TAB",
                    "1",
                    "TAB",
                    "PO",
                ],
                [
                    2,
                    2,
                    1002,
                    "",
                    "2120-02-10",
                    "2120-02-11",
                    "MAIN",
                    "Metformin",
                    "Metformin",
                    "Metformin",
                    "MET",
                    "2",
                    "22222",
                    "500mg",
                    "1",
                    "TAB",
                    "1",
                    "TAB",
                    "PO",
                ],
            ],
        )

    def _get_dataset(self, tables):
        return MIMIC3Dataset(
            root=str(self.root),
            tables=tables,
            cache_dir=self.cache_dir.name,
        )

    def test_task_schema_default(self):
        task = AgainstMedicalAdvicePredictionMIMIC3()
        self.assertEqual(task.task_name, "AgainstMedicalAdvicePredictionMIMIC3")
        self.assertEqual(task.output_schema, {"left_ama": "binary"})
        self.assertIn("baseline_numeric", task.input_schema)
        self.assertIn("baseline_demographics", task.input_schema)
        self.assertNotIn("race_tokens", task.input_schema)
        self.assertNotIn("mistrust_features", task.input_schema)

    def test_label_and_baseline_extraction(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(exclude_minors=False)
        patient = dataset.get_patient("2")
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["left_ama"], 1)
        self.assertIn("insurance:private", samples[0]["baseline_demographics"])

    def test_minor_exclusion(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(exclude_minors=True)
        patient = dataset.get_patient("3")
        samples = task(patient)
        self.assertEqual(samples, [])

    def test_age_cap_at_89(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(exclude_minors=False)
        patient = dataset.get_patient("4")
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["baseline_numeric"][0], 89.0)

    def test_race_and_insurance_normalization(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            include_race=True,
        )
        p1 = task(dataset.get_patient("1"))[0]
        p4 = task(dataset.get_patient("4"))[0]
        self.assertIn("insurance:public", p1["baseline_demographics"])
        self.assertEqual(p1["race_tokens"], ["race:black"])
        self.assertEqual(p4["race_tokens"], ["race:asian"])

    def test_icu_hour_filter(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            min_icu_hours=12.0,
        )
        self.assertEqual(len(task(dataset.get_patient("1"))), 1)
        self.assertEqual(len(task(dataset.get_patient("2"))), 0)

    def test_note_category_filter_and_mistrust_features(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            include_mistrust=True,
            mistrust_feature_set="all",
            note_categories=["NURSING"],
        )
        sample_1 = task(dataset.get_patient("1"))[0]
        sample_2 = task(dataset.get_patient("2"))[0]
        self.assertGreater(sample_1["mistrust_features"][0], 0.0)
        self.assertGreater(sample_1["mistrust_features"][1], 0.0)
        self.assertGreater(sample_1["mistrust_features"][2], 0.0)
        self.assertEqual(sample_2["mistrust_features"][2], 0.0)

    def test_missing_notes_produces_zero_mistrust_proxy(self):
        dataset = self._get_dataset(tables=["noteevents"])
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            include_mistrust=True,
        )
        sample = task(dataset.get_patient("4"))[0]
        self.assertEqual(sample["mistrust_features"], [0.0, 0.0, 0.0, 0.0])

    def test_include_codes_feature_group(self):
        dataset = self._get_dataset(
            tables=[
                "noteevents",
                "diagnoses_icd",
                "procedures_icd",
                "prescriptions",
            ]
        )
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            include_codes=True,
        )
        sample = task(dataset.get_patient("1"))[0]
        self.assertIn("conditions", sample)
        self.assertIn("procedures", sample)
        self.assertIn("drugs", sample)
        self.assertGreaterEqual(len(sample["conditions"]), 1)
        sample_missing_drug = task(dataset.get_patient("4"))[0]
        self.assertEqual(sample_missing_drug["drugs"], ["NO_DRUG"])

    def test_include_codes_uses_placeholders_for_empty_sequences(self):
        dataset = self._get_dataset(
            tables=[
                "noteevents",
                "diagnoses_icd",
                "procedures_icd",
                "prescriptions",
            ]
        )
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            include_codes=True,
        )
        sample = task(dataset.get_patient("3"))[0]
        self.assertEqual(sample["conditions"], ["NO_CONDITION"])
        self.assertEqual(sample["procedures"], ["NO_PROCEDURE"])
        self.assertEqual(sample["drugs"], ["NO_DRUG"])

    def test_dataset_set_task_end_to_end(self):
        dataset = self._get_dataset(
            tables=[
                "noteevents",
                "diagnoses_icd",
                "procedures_icd",
                "prescriptions",
            ]
        )
        task = AgainstMedicalAdvicePredictionMIMIC3(
            exclude_minors=False,
            include_race=True,
            include_mistrust=True,
            include_codes=True,
            min_icu_hours=1.0,
        )
        sample_dataset = dataset.set_task(task)
        self.assertGreater(len(sample_dataset), 0)
        sample = sample_dataset[0]
        self.assertIn("left_ama", sample)
        self.assertIn("baseline_numeric", sample)
        self.assertIn("mistrust_features", sample)
        self.assertIn("conditions", sample)
        sample_dataset.close()


if __name__ == "__main__":
    unittest.main()
