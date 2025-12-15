import unittest
import tempfile
import shutil
import io
import sys
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from pyhealth.datasets import SIIMISICDataset
from pyhealth.tasks import MelanomaClassificationSIIMISIC


class TestSIIMISICDataset(unittest.TestCase):
    """Test SIIMISICDataset with synthetic SIIM-ISIC-like data."""

    def setUp(self):
        """Set up a temporary SIIM-ISIC-style CSV in a temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        siim_data = {
            "image_name": [
                "ISIC_0000001",
                "ISIC_0000002",
                "ISIC_0000003",
            ],
            "patient_id": [
                "IP_0001",
                "IP_0002",
                "IP_0003",
            ],
            "sex": [
                "male",
                "female",
                "female",
            ],
            "age_approx": [
                45,
                60,
                30,
            ],
            "anatom_site_general_challenge": [
                "head/neck",
                "torso",
                "lower extremity",
            ],
            "diagnosis": [
                "unknown",
                "nevus",
                "melanoma",
            ],
            "benign_malignant": [
                "benign",
                "benign",
                "malignant",
            ],
            "target": [
                0,
                0,
                1,
            ],
        }

        df = pd.DataFrame(siim_data)

        for image_name in df["image_name"]:
            self._create_dcm_file(image_name)

        df.to_csv(self.root / "ISIC_2020_Training_GroundTruth.csv", index=False)

    def _create_dcm_file(self, fileName):
        """Creates a random dcm file associated with a patient for use in the test cases"""
        H, W = (256, 256)
        pixel_array = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(fileName, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.Rows = H
        ds.Columns = W
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_array.tobytes()

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.save_as(Path(self.root) / fileName, write_like_original=False)

        return ds


    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test SIIMISICDataset initialization."""
        print("SIIM-ISIC Dataset Tests:")
        dataset = SIIMISICDataset(root=str(self.root), image_dir="")
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "siim_isic")
        self.assertEqual(dataset.root, str(self.root))
        print("Test passed: dataset_initialization\n")

    def test_load_data(self):
        """Test that data loads correctly."""
        dataset = SIIMISICDataset(root=str(self.root), image_dir="")
        self.assertIsNotNone(dataset.global_event_df)
        print("Test passed: load_data\n")

    def test_patient_count(self):
        """Test that the dataset contains the expected number of patients."""
        dataset = SIIMISICDataset(root=str(self.root), image_dir="")
        unique_patients = dataset.unique_patient_ids
        self.assertEqual(len(unique_patients), 3)
        print("Test passed: patient_count\n")

    def test_get_patient(self):
        """Test retrieving a single patient by ID."""
        dataset = SIIMISICDataset(root=str(self.root), image_dir="")
        patient = dataset.get_patient("IP_0001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "IP_0001")
        print("Test passed: get_patient\n")

    def test_stats(self):
        """Test that stats method executes without errors."""
        dataset = SIIMISICDataset(root=str(self.root), image_dir="")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        dataset.stats()
        sys.stdout = old_stdout
        print("Test passed: stats\n")

    def test_melanoma_classification_task(self):
        """Test MelanomaClassificationSIIMISIC task end-to-end."""
        print("\nSIIM-ISIC Task Tests:")
        dataset = SIIMISICDataset(root=str(self.root), image_dir="")

        task = MelanomaClassificationSIIMISIC()
        self.assertEqual(task.task_name, "MelanomaClassificationSIIMISIC")

        self.assertIn("demographics", task.input_schema)
        self.assertIn("lesion_metadata", task.input_schema)
        self.assertIn("target", task.output_schema)
        self.assertIn(task.output_schema["target"], ["binary", "classification"])

        sample_dataset = dataset.set_task(task=task)
        self.assertIsNotNone(sample_dataset)
        self.assertTrue(hasattr(sample_dataset, "samples"))
        self.assertEqual(len(sample_dataset.samples), 3)

        sample = sample_dataset.samples[0]
        required_keys = [
            "patient_id",
            "image_name",
            "demographics",
            "lesion_metadata",
            "target",
        ]
        for key in required_keys:
            self.assertIn(key, sample, f"Sample should contain key: {key}")

        self.assertIsInstance(sample["demographics"], torch.Tensor)
        self.assertIsInstance(sample["lesion_metadata"], torch.Tensor)
        self.assertIsInstance(sample["target"], torch.Tensor)

        label_value = sample["target"].item()
        self.assertIn(label_value, [0, 1])

        self.assertGreater(len(sample["demographics"]), 0)
        self.assertGreater(len(sample["lesion_metadata"]), 0)
        print("Test passed: melanoma_classification_task\n")


if __name__ == "__main__":
    unittest.main()
