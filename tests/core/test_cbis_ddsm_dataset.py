"""Tests for the CBIS-DDSM mammography dataset loader."""

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd

from pyhealth.datasets import CBISDDSMDataset, SubjectSplit
from tests.core.cbis_ddsm_test_utils import (
    STEMS,
    build_cbis_root,
    pydicom,
    write_dicom,
)


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCBISDDSMDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "cbis"
        self.root.mkdir()
        build_cbis_root(self.root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dataset(self, **kwargs):
        return CBISDDSMDataset(
            root=str(self.root), cache_dir=str(self.temp_dir / "cache"), **kwargs
        )

    def test_metadata_rows(self):
        dataset = self._dataset()
        self.assertFalse((self.root / "cbis-ddsm-metadata-pyhealth.csv").exists())
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(metadata), 7)  # P_00005 has no DICOM
        self.assertEqual(
            sorted(dataset.unique_patient_ids),
            ["P_00001", "P_00002", "P_00003", "P_00004"],
        )
        event = dataset.get_patient("P_00003").get_events("lesions")[0]
        self.assertEqual(event.image_id, "P_00003_LEFT_MLO")
        self.assertEqual(event.split, "test")
        self.assertEqual(event.series_uid, "3.1")
        self.assertTrue(event.image_path.endswith("3.1/00000001.dcm"))
        self.assertEqual(set(metadata["breast_density"]), {2})

    def test_drop_overlap_patients_and_official_split(self):
        with self.assertRaises(ValueError):
            self._dataset().official_split()
        dataset = self._dataset(drop_overlap_patients=True)
        self.assertNotIn("P_00004", dataset.unique_patient_ids)
        split = dataset.official_split()
        self.assertIsInstance(split, SubjectSplit)
        self.assertEqual(
            split.to_dict()["subjects"], [["P_00001", "P_00002"], ["P_00003"]]
        )

    def test_data_retriever_layout(self):
        retriever = self.temp_dir / "retriever"
        retriever.mkdir()
        for stem, _ in STEMS.values():
            shutil.copy(self.root / f"{stem}.csv", retriever / f"{stem}.csv")
        folder = retriever / "CBIS-DDSM" / "Mass-Test_P_00003_LEFT_MLO" / "full"
        write_dicom(folder / "1-1.dcm")
        pd.DataFrame(
            {"Series UID": ["3.1"], "File Location": ["./CBIS-DDSM/Mass-Test_P_00003_LEFT_MLO/full"]}
        ).to_csv(retriever / "metadata.csv", index=False)
        dataset = CBISDDSMDataset(
            root=str(retriever),
            cache_dir=str(self.temp_dir / "cache_retriever"),
            metadata_dir=str(self.temp_dir / "meta_retriever"),
        )
        self.assertEqual(dataset.unique_patient_ids, ["P_00003"])

    def test_missing_csv_raises(self):
        (self.root / "calc_case_description_test_set.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_download_skips_csvs_on_disk(self):
        with mock.patch("pyhealth.datasets.cbis_ddsm.download_file") as fetch:
            self._dataset(download=True)
        fetch.assert_not_called()

    def test_download_fetches_missing_csv(self):
        stem = "calc_case_description_test_set"
        saved = (self.root / f"{stem}.csv").read_bytes()
        (self.root / f"{stem}.csv").unlink()

        def fake_download(url, path):
            Path(path).write_bytes(saved)

        with mock.patch(
            "pyhealth.datasets.cbis_ddsm.download_file", side_effect=fake_download
        ) as fetch:
            dataset = self._dataset(download=True)
        fetch.assert_called_once()
        url, path = fetch.call_args.args
        self.assertEqual(
            url, f"https://www.cancerimagingarchive.net/wp-content/uploads/{stem}.csv"
        )
        self.assertEqual(Path(path), self.root.resolve() / f"{stem}.csv")
        self.assertEqual(len(dataset.unique_patient_ids), 4)

    def test_download_series_extracts_and_removes_archive(self):
        dataset = self._dataset()
        payload = self.temp_dir / "payload.dcm"
        write_dicom(payload)

        def fake_download(url, path):
            with zipfile.ZipFile(path, "w") as handle:
                handle.write(payload, "1-1.dcm")

        with mock.patch(
            "pyhealth.datasets.cbis_ddsm.download_file", side_effect=fake_download
        ) as fetch:
            dataset.download_series(["5.1", "1.1"])  # 1.1 is already on disk
        fetch.assert_called_once()
        self.assertTrue(fetch.call_args.args[0].endswith("SeriesInstanceUID=5.1"))
        dicom_dir = self.root / "dicom"
        self.assertTrue((dicom_dir / "5.1" / "1-1.dcm").is_file())
        self.assertFalse((dicom_dir / "5.1.zip").exists())

        refreshed = self._dataset(refresh_metadata=True)
        self.assertIn("P_00005", refreshed.unique_patient_ids)

    def test_download_series_rejects_unsafe_uid(self):
        dataset = self._dataset()
        with mock.patch("pyhealth.datasets.cbis_ddsm.download_file") as fetch:
            for uid in ["../evil", "1.2/3", "", "1..2"]:
                with self.assertRaises(ValueError):
                    dataset.download_series([uid])
        fetch.assert_not_called()

    def test_default_task(self):
        self.assertEqual(
            self._dataset().default_task.task_name,
            "CBISDDSMPathologyClassification",
        )


if __name__ == "__main__":
    unittest.main()
