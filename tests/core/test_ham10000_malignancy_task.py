"""Tests for HAM10000MalignancyClassification and HAM10000 demographics."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets import HAM10000Dataset
from pyhealth.tasks import HAM10000MalignancyClassification
from tests.core.test_ham10000_dataset import _TRAIN, make_ham10000_root

# image -> (age, sex); ISIC_0000007 has no age, ISIC_0000006 an unknown sex
_DEMOGRAPHICS = {
    "ISIC_0000001": (65.0, "male"),
    "ISIC_0000002": (65.0, "male"),
    "ISIC_0000003": (30.0, "female"),
    "ISIC_0000004": (60.0, "female"),
    "ISIC_0000005": (80.0, "male"),
    "ISIC_0000006": (45.0, "unknown"),
    "ISIC_0000007": (None, "female"),
    "ISIC_0000008": (70.0, "female"),
    "ISIC_0000009": (50.0, "male"),
}


def write_dataverse_metadata(root: Path) -> None:
    pd.DataFrame(
        {
            "lesion_id": [r[2] for r in _TRAIN],
            "image_id": [r[0] for r in _TRAIN],
            "dx": [r[1].lower() for r in _TRAIN],
            "dx_type": "histo",
            "age": [_DEMOGRAPHICS[r[0]][0] for r in _TRAIN],
            "sex": [_DEMOGRAPHICS[r[0]][1] for r in _TRAIN],
            "localization": "back",
        }
    ).to_csv(root / "HAM10000_metadata.csv", index=False)


class TestHAM10000MalignancyClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "isic2018"
        make_ham10000_root(root)
        write_dataverse_metadata(root)
        cls.dataset = HAM10000Dataset(root=str(root), cache_dir=str(cls.tmp / "cache"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _samples(self, **kwargs):
        samples = self.dataset.set_task(
            HAM10000MalignancyClassification(image_size=16, **kwargs)
        )
        rows = {samples[i]["image_id"]: samples[i] for i in range(len(samples))}
        samples.close()
        return rows

    def test_dataset_reads_demographics(self):
        events = self.dataset.get_patient("HAM_1").get_events(event_type="ham10000")
        self.assertEqual(events[0].sex, "M")
        self.assertEqual(float(events[0].age), 65.0)
        self.assertEqual(events[0].localization, "back")

    def test_drops_missing_demographics(self):
        rows = self._samples()
        self.assertEqual(
            sorted(rows),
            [
                "ISIC_0000001",
                "ISIC_0000002",
                "ISIC_0000003",
                "ISIC_0000004",
                "ISIC_0000005",
                "ISIC_0000008",
            ],
        )

    def test_malignant_label_is_mel_or_akiec(self):
        rows = self._samples()
        malignant = {k for k, v in rows.items() if int(v["malignant"]) == 1}
        self.assertEqual(malignant, {"ISIC_0000001", "ISIC_0000002", "ISIC_0000008"})

    def test_sensitive_attributes(self):
        rows = self._samples()
        self.assertEqual(rows["ISIC_0000003"]["sex"], "F")
        self.assertEqual(rows["ISIC_0000001"]["age"], 65.0)
        self.assertEqual(rows["ISIC_0000004"]["age_group"], "<=60")
        self.assertEqual(rows["ISIC_0000001"]["age_group"], ">60")

    def test_custom_age_bins(self):
        rows = self._samples(age_bins=(19, 39, 59, 79))
        self.assertEqual(rows["ISIC_0000003"]["age_group"], "(19, 39]")
        self.assertEqual(rows["ISIC_0000004"]["age_group"], "(59, 79]")
        self.assertEqual(rows["ISIC_0000005"]["age_group"], ">79")

    def test_bad_age_bins_raise(self):
        with self.assertRaises(ValueError):
            HAM10000MalignancyClassification(age_bins=())
        with self.assertRaises(ValueError):
            HAM10000MalignancyClassification(age_bins=(60, 40))


class TestHAM10000DemographicsFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "isic2018"
        make_ham10000_root(self.root)
        write_dataverse_metadata(self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sex(self, metadata_dir):
        dataset = HAM10000Dataset(
            root=str(self.root),
            cache_dir=str(self.tmp / "cache"),
            metadata_dir=str(metadata_dir),
        )
        return dataset.get_patient("HAM_1").get_events(event_type="ham10000")[0].sex

    def test_edited_file_rebuilds_metadata(self):
        metadata_dir = self.tmp / "meta"
        self.assertEqual(self._sex(metadata_dir), "M")
        path = self.root / "HAM10000_metadata.csv"
        frame = pd.read_csv(path).assign(sex="female")
        frame.to_csv(path, index=False)
        self.assertEqual(self._sex(metadata_dir), "F")

    def test_tab_separated_file_with_groupings_leaves_sex_empty(self):
        path = self.root / "HAM10000_metadata.csv"
        pd.read_csv(path).to_csv(path, sep="\t", index=False)
        with self.assertLogs("pyhealth.datasets.ham10000", "WARNING") as logs:
            self.assertIsNone(self._sex(self.tmp / "meta"))
        self.assertIn("format=original", "".join(logs.output))

    def test_tab_separated_file_raises_without_groupings(self):
        path = self.root / "HAM10000_metadata.csv"
        pd.read_csv(path).to_csv(path, sep="\t", index=False)
        (self.root / "ISIC2018_Task3_Training_LesionGroupings.csv").unlink()
        with self.assertRaisesRegex(ValueError, "format=original"):
            HAM10000Dataset(root=str(self.root), cache_dir=str(self.tmp / "cache"))


class TestHAM10000WithoutDemographics(unittest.TestCase):
    def test_columns_empty_and_task_yields_nothing(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            make_ham10000_root(tmp / "isic2018")
            dataset = HAM10000Dataset(
                root=str(tmp / "isic2018"), cache_dir=str(tmp / "cache")
            )
            event = dataset.get_patient("HAM_1").get_events(event_type="ham10000")[0]
            self.assertIsNone(event.sex)
            task = HAM10000MalignancyClassification(image_size=16)
            self.assertEqual(task(dataset.get_patient("HAM_1")), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
