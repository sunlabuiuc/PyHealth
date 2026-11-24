"""PBC dataset/task tests using two fabricated patients, never medical data.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from pyhealth.data import Patient
from pyhealth.datasets import PBCDataset
from pyhealth.tasks import CompetingRisksPBC
from pyhealth.tasks.competing_risks_pbc import (
    PBC_CATEGORIES,
    PBC_FEATURES,
    PBC_NUMERIC,
)


def fake_row(**changes: Any) -> dict[str, Any]:
    """Construct one fabricated baseline row with the author's source schema."""
    return {
        **{name: levels[0] for name, levels in PBC_CATEGORIES.items()},
        **dict.fromkeys(PBC_NUMERIC, 1.0),
        "year": 0.0,
        "years": 5.0,
        "status": "alive",
        "age": 40.0,
        **changes,
    }


def fake_patient(*rows: dict[str, Any]) -> Patient:
    """Wrap source-like records in actual PyHealth Patient/Event representations."""
    return Patient(
        "fake",
        pl.DataFrame(
            [
                {
                    "timestamp": None,
                    "event_type": "pbc",
                    **{f"pbc/{name}": value for name, value in row.items()},
                }
                for row in rows
            ]
        ),
    )


class TestPBC(unittest.TestCase):
    def test_labels_and_shapes(self) -> None:
        for status, event in (("alive", 0), ("dead", 1), ("transplanted", 2)):
            sample = CompetingRisksPBC()(fake_patient(fake_row(status=status)))[0]
            self.assertEqual((sample["event"], sample["time"]), (event, 5.0))
            self.assertEqual(len(sample["features"]), 25)
            self.assertEqual(len(sample["missing_mask"]), 25)

    def test_no_future_backfill_or_age_leakage(self) -> None:
        sample = CompetingRisksPBC()(
            fake_patient(
                fake_row(year=1.0, serBilir=20.0),
                fake_row(serBilir=None),
            )
        )[0]
        self.assertTrue(np.isnan(sample["features"][PBC_FEATURES.index("serBilir")]))
        self.assertEqual(sample["features"][PBC_FEATURES.index("age")], 40.0)

    def test_fixed_categories_and_missing_blocks(self) -> None:
        sample = CompetingRisksPBC()(fake_patient(fake_row(ascites="NA", sex="male")))[
            0
        ]
        self.assertEqual(sample["features"][2:4], [0.0, 1.0])
        self.assertTrue(np.isnan(sample["features"][4:6]).all())
        self.assertEqual(sample["missing_mask"][4:6], [1.0, 1.0])

    def test_invalid_baselines_and_endpoints(self) -> None:
        for changes in (
            {"year": 1.0},
            {"years": -1.0},
            {"year": float("nan")},
            {"years": float("inf")},
            {"status": "2"},
            {"drug": "unknown"},
            {"albumin": float("inf")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CompetingRisksPBC()(fake_patient(fake_row(**changes)))
        for other in (
            fake_row(year=6.0),
            fake_row(year=1.0, status="dead"),
            fake_row(age=41.0),
        ):
            with self.assertRaises(ValueError):
                CompetingRisksPBC()(fake_patient(fake_row(), other))

    def test_empty_patient(self) -> None:
        empty = Patient(
            "empty",
            pl.DataFrame(
                {
                    "timestamp": [None],
                    "event_type": ["other"],
                }
            ),
        )
        self.assertEqual(CompetingRisksPBC()(empty), [])

    def test_yaml_csv_loading_and_event_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pbc2.csv"
            with self.assertRaises(FileNotFoundError):
                PBCDataset(directory)
            rows = [
                {"id": "fake-1", **fake_row(status="dead", serChol="")},
                {"id": "fake-2", **fake_row(status="transplanted")},
            ]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            cohort = PBCDataset(directory, cache_dir=Path(directory) / "cache")
            frame = pl.from_pandas(
                cohort.load_table("pbc").compute(scheduler="synchronous")
            )
            samples = []
            for key, partition in frame.partition_by(
                "patient_id", as_dict=True
            ).items():
                samples.extend(CompetingRisksPBC()(Patient(str(key[0]), partition)))
            self.assertEqual(sorted(s["event"] for s in samples), [1, 2])
            self.assertTrue(
                np.isnan(samples[0]["features"][PBC_FEATURES.index("serChol")])
            )
            config = cohort.config.model_dump()
            config["tables"]["pbc"]["attributes"].remove("status")
            bad = Path(directory) / "bad.yaml"
            bad.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                PBCDataset(directory, config_path=str(bad))
            path.write_text("id\n1\n")
            with self.assertRaises(ValueError):
                PBCDataset(directory)
            with self.assertRaises(ValueError):
                PBCDataset(directory, tables=["other"])

    def test_dataset_name_option_is_forwarded_without_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pbc2.csv"
            row = {"id": "fake-1", **fake_row()}
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            for name in ("custom-pbc", None):
                with self.subTest(name=name):
                    cohort = PBCDataset(
                        directory,
                        dataset_name=name,
                        cache_dir=Path(directory) / "cache",
                    )
                    self.assertEqual(cohort.dataset_name, name or "pbc")

    def test_missing_csv_patient_ids_are_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pbc2.csv"
            for identifier in ("", " ", None, "NA", "NaN"):
                rows = [
                    {"id": "fake-1", **fake_row()},
                    {"id": identifier, **fake_row(status="dead")},
                ]
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with (
                    self.subTest(identifier=identifier),
                    self.assertRaisesRegex(ValueError, "patient ID"),
                ):
                    PBCDataset(directory, cache_dir=Path(directory) / "cache")
