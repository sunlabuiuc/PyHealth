"""Synthetic tests for the MSD dataset loaders."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyhealth.datasets import (
    MSDHippocampusDataset,
    MSDProstateDataset,
    MSDSegmentationDataset,
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

        expected_norm = (
            "ct"
            if any("ct" in str(value).lower() for value in modality.values())
            else "mri"
        )
        self.assertEqual(dataset.default_normalization, expected_norm)
        self.assertEqual(dataset.num_classes, len(labels))
        task = dataset.default_task
        self.assertEqual(
            task.input_schema["image"][1]["normalize"], expected_norm
        )
        self.assertEqual(
            task.output_schema["mask"][1]["num_classes"], len(labels)
        )

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

    def test_missing_dataset_json_raises(self):
        empty_root = self.root / "empty_msd"
        empty_root.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            MSDHippocampusDataset(
                root=str(empty_root),
                cache_dir=str(self.cache_dir),
                metadata_dir=str(self.root / "metadata" / "empty"),
            )

    def test_download_calls_helper_and_base_without_url_raises(self):
        dl_root = self.root / "download_hippo"
        dl_root.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ValueError):
            MSDSegmentationDataset(
                root=str(dl_root),
                cache_dir=str(self.cache_dir),
                metadata_dir=str(self.root / "metadata" / "base_dl"),
                download=True,
            )

        def _populate_on_download(*_args, **_kwargs):
            _make_msd_root(
                self.root,
                "download_hippo",
                {"0": "MRI"},
                {"0": "background", "1": "Anterior"},
                "3D",
            )

        with mock.patch(
            "pyhealth.datasets.msd._download_archives",
            side_effect=_populate_on_download,
        ) as mocked:
            MSDHippocampusDataset(
                root=str(dl_root),
                cache_dir=str(self.cache_dir),
                metadata_dir=str(self.root / "metadata" / "dl"),
                download=True,
            )
            mocked.assert_called_once()

    def test_default_metadata_dir_and_channel_names_and_string_train_item(self):
        root = self.root / "channel_names_msd"
        for directory in ("imagesTr", "labelsTr", "imagesTs"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        for rel in (
            "imagesTr/case_001.nii.gz",
            "labelsTr/case_001.nii.gz",
            "imagesTr/case_003.nii.gz",
            "imagesTs/case_002.nii.gz",
        ):
            (root / rel).write_bytes(b"")

        spec = {
            "name": "CustomMSD",
            "tensorImageSize": "3D",
            "channel_names": ["T2", "ADC"],
            "labels": ["background", "foreground"],
            "training": [
                {
                    "image": "./imagesTr/case_001.nii.gz",
                    "label": "./labelsTr/case_001.nii.gz",
                },
                "./imagesTr/case_003.nii.gz",
                {"image": "", "label": ""},
            ],
            "test": [{"image": "./imagesTs/case_002.nii.gz"}],
        }
        with open(root / "dataset.json", "w") as handle:
            json.dump(spec, handle)

        dataset = MSDSegmentationDataset(
            root=str(root),
            cache_dir=str(self.cache_dir),
        )
        self.assertEqual(dataset.num_classes, 2)
        self.assertEqual(dataset.modality, "T2,ADC")
        self.assertEqual(dataset.default_normalization, "mri")
        self.assertIn("case_003", set(dataset.unique_patient_ids))
        event = dataset.get_patient("case_001").get_events()[0]
        self.assertEqual(event.modality, "T2,ADC")
        self.assertEqual(int(event.channels), 2)


if __name__ == "__main__":
    unittest.main()
