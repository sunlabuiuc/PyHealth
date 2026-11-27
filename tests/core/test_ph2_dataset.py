# Contributor: Aayan Verma (aayanv2)
# Contributor: Bryson Lee (brysonl3)
# Contributor: Xin Zhang (xinz13)
# Paper: A Study of Artifacts on Melanoma Classification under Diffusion-Based
# Perturbations (Jin and Ghassemi, CHIL 2025)
# https://proceedings.mlr.press/v287/jin25b.html
"""Tests for PH2Dataset using a tiny synthetic PH2 archive.

Each test builds a 4-image PH2-style archive in a temporary folder, so no real
data is used and the whole file runs in well under a second.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from pyhealth.datasets import PH2Dataset

# image name, clinical diagnosis code (0 common, 1 atypical, 2 melanoma)
_ROWS = [("IMD001", 0), ("IMD002", 1), ("IMD003", 2), ("IMD004", 2)]
_SIZE = 24


def make_ph2_root(root: Path, with_missing_mask: bool = True) -> None:
    """Write a 4-image PH2 layout with an annotation CSV that has a preamble.

    IMD004 gets no lesion mask (when ``with_missing_mask``), so it is dropped.
    """
    rng = np.random.default_rng(0)
    images = root / "PH2 Dataset images"
    for name, _ in _ROWS:
        dermoscopy = images / name / f"{name}_Dermoscopic_Image"
        lesion = images / name / f"{name}_lesion"
        dermoscopy.mkdir(parents=True)
        lesion.mkdir(parents=True)
        pixels = rng.integers(0, 255, (_SIZE, _SIZE, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(dermoscopy / f"{name}.bmp")
        if with_missing_mask and name == "IMD004":
            continue
        mask = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        mask[8:16, 8:16] = 255
        Image.fromarray(mask).save(lesion / f"{name}_lesion.bmp")
    preamble = [["PH2 database", "", ""], ["", "", ""]]
    header = [["Image Name", "Histological Diagnosis", "Clinical Diagnosis"]]
    body = [[name, "", code] for name, code in _ROWS]
    pd.DataFrame(preamble + header + body).to_csv(
        root / "PH2_dataset.csv", index=False, header=False
    )


class TestPH2Dataset(unittest.TestCase):
    """Tests for PH2Dataset on a synthetic archive."""

    def setUp(self) -> None:
        """Write a synthetic archive to a temporary folder."""
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "PH2Dataset"
        make_ph2_root(self.root)
        self.cache = self.tmp / "cache"

    def tearDown(self) -> None:
        """Remove the temporary folder."""
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs: Any) -> PH2Dataset:
        """Build a PH2Dataset on the synthetic archive."""
        return PH2Dataset(root=str(self.root), cache_dir=str(self.cache), **kwargs)

    def test_metadata_and_patients(self) -> None:
        """Images without a mask are dropped; patients and diagnoses are parsed."""
        dataset = self._dataset()
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 3)  # IMD004 has no mask
        self.assertEqual(
            sorted(dataset.unique_patient_ids), ["IMD001", "IMD002", "IMD003"]
        )
        self.assertEqual(
            dict(zip(meta["image_id"], meta["diagnosis"])),
            {
                "IMD001": "common_nevus",
                "IMD002": "atypical_nevus",
                "IMD003": "melanoma",
            },
        )
        # the source folder is only read, never written to
        self.assertFalse((self.root / "ph2-metadata-pyhealth.csv").exists())

    def test_event_attributes(self) -> None:
        """An event carries the diagnosis and existing image and mask paths."""
        dataset = self._dataset()
        (event,) = dataset.get_patient("IMD003").get_events(event_type="ph2")
        self.assertEqual(event.diagnosis, "melanoma")
        self.assertTrue(Path(event.image_path).is_file())
        self.assertTrue(Path(event.mask_path).name.endswith("_lesion.bmp"))

    def test_reads_the_archives_pipe_separated_txt(self) -> None:
        """The archive's ||/| separated txt, with blank cells, is parsed."""
        (self.root / "PH2_dataset.csv").unlink()
        (self.root / "PH2_dataset.txt").write_text(
            "||   Name || Histological Diagnosis || Clinical Diagnosis || Colors ||\n"
            "|| IMD001 ||                        ||                  0 ||  3  4 ||\n"
            "|| IMD002 ||                        ||                  1 ||    3 ||\n"
            "|| IMD003 ||                        ||                  2 ||    5 ||\n"
        )
        meta = pd.read_csv(self._dataset().metadata_path)
        self.assertEqual(
            dict(zip(meta["image_id"], meta["diagnosis"])),
            {
                "IMD001": "common_nevus",
                "IMD002": "atypical_nevus",
                "IMD003": "melanoma",
            },
        )

    def test_blank_clinical_diagnosis_is_skipped(self) -> None:
        """A row with a blank Clinical Diagnosis is skipped, not an error."""
        (self.root / "PH2_dataset.csv").unlink()
        (self.root / "PH2_dataset.txt").write_text(
            "||   Name || Histological Diagnosis || Clinical Diagnosis || Colors ||\n"
            "|| IMD001 ||                        ||                  0 ||    3 ||\n"
            "|| IMD002 ||                        ||                    ||    3 ||\n"
            "|| IMD003 ||                        ||                  2 ||    5 ||\n"
        )
        meta = pd.read_csv(self._dataset().metadata_path)
        self.assertEqual(sorted(meta["image_id"]), ["IMD001", "IMD003"])

    def test_txt_is_preferred_over_csv(self) -> None:
        """When both files exist, PH2_dataset.txt is read before the .csv."""
        (self.root / "PH2_dataset.txt").write_text(
            "||   Name || Histological Diagnosis || Clinical Diagnosis ||\n"
            "|| IMD001 ||                        ||                  2 ||\n"
            "|| IMD003 ||                        ||                  0 ||\n"
        )
        meta = pd.read_csv(self._dataset().metadata_path)
        self.assertEqual(
            dict(zip(meta["image_id"], meta["diagnosis"])),
            {
                "IMD001": "melanoma",
                "IMD003": "common_nevus",
            },  # the csv says the reverse
        )

    def test_missing_annotation_file_raises(self) -> None:
        """A missing annotation file raises FileNotFoundError."""
        (self.root / "PH2_dataset.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_annotation_without_clinical_column_raises(self) -> None:
        """A file without a Clinical Diagnosis column raises ValueError."""
        pd.DataFrame([["Image Name", "Other"], ["IMD001", 1]]).to_csv(
            self.root / "PH2_dataset.csv", index=False, header=False
        )
        with self.assertRaises(ValueError):
            self._dataset()

    def test_unknown_diagnosis_code_raises(self) -> None:
        """A diagnosis code outside 0, 1, 2 raises ValueError."""
        pd.DataFrame([["Image Name", "Clinical Diagnosis"], ["IMD001", 7]]).to_csv(
            self.root / "PH2_dataset.csv", index=False, header=False
        )
        with self.assertRaises(ValueError):
            self._dataset()

    def test_no_images_raises(self) -> None:
        """A folder with no images raises FileNotFoundError."""
        shutil.rmtree(self.root / "PH2 Dataset images")
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_rebuilds_metadata_when_root_changes(self) -> None:
        """A different source folder rebuilds the metadata, not reuses it."""
        metadata_dir = str(self.tmp / "meta")
        self._dataset(metadata_dir=metadata_dir)
        other = self.tmp / "other"
        make_ph2_root(other, with_missing_mask=False)
        dataset = PH2Dataset(
            root=str(other), cache_dir=str(self.cache), metadata_dir=metadata_dir
        )
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 4)

    def test_default_task(self) -> None:
        """The default task is the whole-image view."""
        self.assertEqual(self._dataset().default_task.view, "whole")
