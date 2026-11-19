"""Tests for ABIDEDataset on a tiny synthetic PCP layout (no network)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pyhealth.datasets import ABIDEDataset
from pyhealth.datasets.abide import ABIDE_PHENOTYPIC_CSV, abide_roi_relpath

PHENO = pd.DataFrame(
    {
        "SUB_ID": [50003, 50004, 50102, 50103],
        "SITE_ID": ["PITT", "PITT", "OLIN", "OLIN"],
        "FILE_ID": ["Pitt_0050003", "Pitt_0050004", "Olin_0050102", "no_filename"],
        "DX_GROUP": [1, 2, 2, 1],
        "AGE_AT_SCAN": [24.5, 19.0, None, 30.0],
        "SEX": [1, 2, 1, 1],
    }
)


def write_roi_file(path: Path, n_time: int = 20, n_roi: int = 5, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(seed).normal(size=(n_time, n_roi))
    header = "\t".join(f"#{i + 1}" for i in range(n_roi))
    np.savetxt(path, x, delimiter="\t", header=header, comments="")


def build_root(root: Path, file_ids=("Pitt_0050003", "Olin_0050102")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    PHENO.to_csv(root / ABIDE_PHENOTYPIC_CSV, index=False)
    for i, file_id in enumerate(file_ids):
        rel = abide_roi_relpath(file_id, "cpac", "filt_noglobal", "cc200")
        write_roi_file(root / rel, seed=i)


class TestABIDEDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "abide"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        return ABIDEDataset(
            root=str(self.root),
            metadata_cache_dir=str(self.tmp / "meta"),
            cache_dir=str(self.tmp / "cache"),
            **kwargs,
        )

    def test_joins_phenotype_and_roi_files(self):
        build_root(self.root)
        ds = self._dataset()
        # 50004 has no ROI file, 50103 has no functional data
        self.assertEqual(sorted(ds.unique_patient_ids), ["50003", "50102"])
        event = ds.get_patient("50003").get_events(event_type="scans")[0]
        self.assertEqual(str(event.site), "PITT")
        self.assertEqual(int(event.dx_group), 1)
        self.assertEqual(str(event.sex), "M")
        self.assertTrue(str(event.roi_file).endswith("Pitt_0050003_rois_cc200.1D"))
        no_age = ds.get_patient("50102").get_events(event_type="scans")[0]
        self.assertEqual(float(no_age.age), -1.0)
        self.assertEqual(list(self.root.rglob("abide-pyhealth-*")), [])

    def test_flat_layout_and_errors(self):
        self.root.mkdir()
        PHENO.to_csv(self.root / ABIDE_PHENOTYPIC_CSV, index=False)
        write_roi_file(self.root / "rois_cc200" / "Pitt_0050004_rois_cc200.1D")
        self.assertEqual(self._dataset().unique_patient_ids, ["50004"])
        with self.assertRaises(FileNotFoundError):
            self._dataset(atlas="aal")
        with self.assertRaises(ValueError):
            self._dataset(atlas="nope")
        with self.assertRaises(FileNotFoundError):
            ABIDEDataset(root=str(self.tmp / "empty"), metadata_cache_dir=str(self.tmp / "m"))

    def test_selected_strategy_wins_over_other_layouts(self):
        build_root(self.root, file_ids=("Pitt_0050003",))
        # Same scan under another strategy and pipeline: must be ignored.
        for pipeline, strategy in (("cpac", "nofilt_noglobal"), ("ccs", "filt_noglobal")):
            rel = abide_roi_relpath("Pitt_0050003", pipeline, strategy, "cc200")
            write_roi_file(self.root / rel, seed=9)
        # Olin only exists for nofilt_noglobal, so it is skipped for filt_noglobal.
        write_roi_file(
            self.root / abide_roi_relpath("Olin_0050102", "cpac", "nofilt_noglobal", "cc200")
        )
        ds = self._dataset()
        self.assertEqual(ds.unique_patient_ids, ["50003"])
        event = ds.get_patient("50003").get_events(event_type="scans")[0]
        expected = self.root / abide_roi_relpath(
            "Pitt_0050003", "cpac", "filt_noglobal", "cc200"
        )
        self.assertEqual(Path(str(event.roi_file)), expected.resolve())

        other = self._dataset(strategy="nofilt_noglobal")
        self.assertEqual(sorted(other.unique_patient_ids), ["50003", "50102"])
        event = other.get_patient("50003").get_events(event_type="scans")[0]
        self.assertIn("nofilt_noglobal", str(event.roi_file))

    def test_ambiguous_flat_files_raise(self):
        build_root(self.root, file_ids=())
        write_roi_file(self.root / "a" / "Pitt_0050003_rois_cc200.1D")
        write_roi_file(self.root / "b" / "Pitt_0050003_rois_cc200.1D")
        with self.assertRaises(ValueError):
            self._dataset()
        # A file at the selected S3 path resolves the ambiguity.
        build_root(self.root, file_ids=("Pitt_0050003",))
        self.assertEqual(self._dataset().unique_patient_ids, ["50003"])

    def test_download_uses_s3_layout(self):
        calls = []

        def fake_download(url, path, timeout=60.0):
            calls.append(url)
            if url.endswith(ABIDE_PHENOTYPIC_CSV):
                PHENO.to_csv(path, index=False)
            else:
                write_roi_file(Path(path))
            return path

        with mock.patch("pyhealth.utils.download_file", side_effect=fake_download):
            ds = self._dataset(download=True, max_subjects=2)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[1].endswith(
            "Outputs/cpac/filt_noglobal/rois_cc200/Pitt_0050003_rois_cc200.1D"
        ))
        self.assertEqual(sorted(ds.unique_patient_ids), ["50003", "50004"])
        self.assertEqual(ds.default_task.task_name, "ABIDEAutismClassification")


if __name__ == "__main__":
    unittest.main()
