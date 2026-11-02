"""Synthetic tests for the MSD dataset loaders."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import (
    MSDHippocampusDataset,
    MSDProstateDataset,
    MSDSpleenDataset,
)


def _make_msd_root(
    base: Path,
    name: str,
    modality: dict[str, str],
    labels: dict[str, str],
    image_size: str,
) -> Path:
    root = base / name
    for directory in ("imagesTr", "labelsTr", "imagesTs"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    train_image = root / "imagesTr" / "case_001.nii.gz"
    train_label = root / "labelsTr" / "case_001.nii.gz"
    test_image = root / "imagesTs" / "case_002.nii.gz"
    for path in (train_image, train_label, test_image):
        path.write_bytes(b"")

    specification = {
        "name": name,
        "tensorImageSize": image_size,
        "modality": modality,
        "labels": labels,
        "training": [
            {
                "image": "./imagesTr/case_001.nii.gz",
                "label": "./labelsTr/case_001.nii.gz",
            }
        ],
        "test": ["./imagesTs/case_002.nii.gz"],
    }
    with open(root / "dataset.json", "w") as handle:
        json.dump(specification, handle)
    return root


class TestMSDDatasets(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.cache_dir = self.root / "cache"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _check_dataset(self, dataset_class, modality, labels, channels):
        root = _make_msd_root(
            self.root,
            dataset_class.__name__,
            modality,
            labels,
            "3D",
        )
        metadata_dir = self.root / "metadata" / dataset_class.__name__
        dataset = dataset_class(
            root=str(root),
            cache_dir=str(self.cache_dir),
            metadata_dir=str(metadata_dir),
        )
        self.assertEqual(
            set(dataset.unique_patient_ids), {"case_001", "case_002"}
        )
        event = dataset.get_patient("case_001").get_events()[0]
        self.assertEqual(event.record_id, "case_001")
        self.assertTrue(event.image_path.endswith("case_001.nii.gz"))
        self.assertTrue(event.label_path.endswith("case_001.nii.gz"))
        self.assertEqual(event.split, "train")
        self.assertEqual(int(event.channels), channels)
        self.assertEqual(event.tensor_image_size, "3D")
        test_event = dataset.get_patient("case_002").get_events()[0]
        self.assertFalse(test_event.label_path)
        self.assertEqual(test_event.split, "test")
        self.assertTrue((metadata_dir / "msd-metadata-pyhealth.csv").exists())
        self.assertFalse((root / "msd-metadata-pyhealth.csv").exists())

    def test_hippocampus(self):
        self._check_dataset(
            MSDHippocampusDataset,
            {"0": "MRI"},
            {"0": "background", "1": "Anterior", "2": "Posterior"},
            1,
        )

    def test_prostate(self):
        self._check_dataset(
            MSDProstateDataset,
            {"0": "T2", "1": "ADC"},
            {"0": "background", "1": "PZ", "2": "TZ"},
            2,
        )

    def test_spleen(self):
        self._check_dataset(
            MSDSpleenDataset,
            {"0": "CT"},
            {"0": "background", "1": "spleen"},
            1,
        )


if __name__ == "__main__":
    unittest.main()
