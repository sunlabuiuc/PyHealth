# Contributor: Arun Vishnu (anv5@illinois.edu)
# Paper: Evaluating Model Performance in Medical Datasets Over Time
# https://proceedings.mlr.press/v209/zhou23a.html
# Tests: Synthetic loader, identity, labels, features, and leakage controls.

"""Fast synthetic tests for the CDC COVID case loader and mortality task."""

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from pyhealth.data import Patient
from pyhealth.datasets import CDCCOVIDDataset
from pyhealth.tasks import CDCCOVIDMortalityPrediction
from pyhealth.tasks.cdc_covid_mortality import feature_names

FIELDS = (
    "case_month",
    "res_state",
    "state_fips_code",
    "res_county",
    "county_fips_code",
    "age_group",
    "sex",
    "race",
    "ethnicity",
    "case_positive_specimen",
    "case_onset_interval",
    "process",
    "exposure_yn",
    "current_status",
    "symptom_status",
    "hosp_yn",
    "icu_yn",
    "death_yn",
    "underlying_conditions_yn",
)


def fake_row(**changes: Any) -> dict[str, Any]:
    """Construct one fabricated record with the official 19-column schema."""
    return {
        "case_month": "2021-01",
        "res_state": "ZZ",
        "state_fips_code": "00",
        "res_county": "SYNTHETIC",
        "county_fips_code": "00000",
        "age_group": "18 to 49 years",
        "sex": "Female",
        "race": "White",
        "ethnicity": "Non-Hispanic/Latino",
        "case_positive_specimen": "1.0",
        "case_onset_interval": "0.0",
        "process": "Routine surveillance",
        "exposure_yn": "Yes",
        "current_status": "Laboratory-confirmed case",
        "symptom_status": "Symptomatic",
        "hosp_yn": "No",
        "icu_yn": "No",
        "death_yn": "No",
        "underlying_conditions_yn": "No",
        **changes,
    }


def fake_patient(**changes: Any) -> Patient:
    """Wrap one fabricated source record in PyHealth's event representation."""
    row = fake_row(**changes)
    return Patient(
        "case-1",
        pl.DataFrame(
            [
                {
                    "timestamp": datetime(2021, 1, 1),
                    "event_type": "cases",
                    **{f"cases/{key}": value for key, value in row.items()},
                }
            ]
        ),
    )


class TestCDCCOVID(unittest.TestCase):
    def test_task_labels_feature_sets_and_metadata(self) -> None:
        demographics = CDCCOVIDMortalityPrediction("demographics")
        negative = demographics(fake_patient())[0]
        positive = demographics(fake_patient(death_yn="Yes"))[0]
        self.assertEqual((negative["mortality"], positive["mortality"]), (0, 1))
        self.assertEqual(negative["case_month"], "2021-01")
        names = feature_names("demographics")
        self.assertEqual(len(negative["features"]), len(names))
        self.assertEqual(len(names), len(set(names)))

        official_other = demographics(fake_patient(sex="Other"))[0]["features"]
        unrecognized = demographics(fake_patient(sex="Not listed"))[0]["features"]
        self.assertEqual(official_other[names.index("sex=Other")], 1.0)
        self.assertEqual(official_other[names.index("sex=Unrecognized")], 0.0)
        self.assertEqual(unrecognized[names.index("sex=Other")], 0.0)
        self.assertEqual(unrecognized[names.index("sex=Unrecognized")], 1.0)

        preoutcome = CDCCOVIDMortalityPrediction("preoutcome")(fake_patient())[0]
        self.assertEqual(len(preoutcome["features"]), len(feature_names("preoutcome")))
        self.assertGreater(len(preoutcome["features"]), len(negative["features"]))

    def test_missing_outcomes_and_numeric_values(self) -> None:
        task = CDCCOVIDMortalityPrediction()
        for outcome in ("Unknown", "Missing", "NA", ""):
            with self.subTest(outcome=outcome):
                self.assertEqual(task(fake_patient(death_yn=outcome)), [])
        sample = task(
            fake_patient(case_positive_specimen="", case_onset_interval="not-a-number")
        )[0]
        self.assertEqual(sample["features"][-4:], [0.0, 1.0, 0.0, 1.0])

    def test_hospital_and_icu_values_cannot_change_features(self) -> None:
        task = CDCCOVIDMortalityPrediction()
        baseline = task(fake_patient(hosp_yn="No", icu_yn="No"))[0]["features"]
        severe = task(fake_patient(hosp_yn="Yes", icu_yn="Yes"))[0]["features"]
        self.assertEqual(baseline, severe)

    def test_yaml_csv_loading_and_generated_case_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "n8mc-b4w4.csv"
            rows = [fake_row(), fake_row(death_yn="Yes", case_month="2021-02")]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            dataset = CDCCOVIDDataset(directory, cache_dir=Path(directory) / "cache")
            frame = pl.from_pandas(
                dataset.load_table("cases").compute(scheduler="synchronous")
            )
            self.assertEqual(frame["patient_id"].n_unique(), 2)
            self.assertEqual(frame["timestamp"].dt.month().to_list(), [1, 2])
            samples = []
            for key, partition in frame.partition_by(
                "patient_id", as_dict=True
            ).items():
                samples.extend(
                    CDCCOVIDMortalityPrediction()(Patient(str(key[0]), partition))
                )
            self.assertEqual(sorted(sample["mortality"] for sample in samples), [0, 1])

            config = dataset.config.model_dump()
            config["tables"]["cases"]["patient_id"] = "county_fips_code"
            bad_config = Path(directory) / "bad.yaml"
            bad_config.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "case-level row identity"):
                CDCCOVIDDataset(directory, config_path=str(bad_config))
            path.write_text("case_month\n2021-01\n")
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                CDCCOVIDDataset(directory)

    def test_invalid_configuration_and_multiple_events(self) -> None:
        with self.assertRaises(ValueError):
            feature_names("unsupported")
        with self.assertRaises(ValueError):
            CDCCOVIDMortalityPrediction("unsupported")
        patient = fake_patient()
        duplicate = Patient(
            "case-1",
            pl.concat([patient.data_source, patient.data_source]),
        )
        with self.assertRaisesRegex(ValueError, "one event"):
            CDCCOVIDMortalityPrediction()(duplicate)


if __name__ == "__main__":
    unittest.main()
