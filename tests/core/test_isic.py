import unittest
import shutil
from pathlib import Path
import pandas as pd

from pyhealth.datasets import ISICBiasDataset
from pyhealth.datasets.isic_bias import normalize_image_extensions
from PIL import Image


def create_temp_isic_csv(csv_path: Path):
    df = pd.DataFrame([
        {
            "image": "ISIC_1.png",
            "dark_corner": 0,
            "hair": 0,
            "gel_border": 0,
            "gel_bubble": 1,
            "ruler": 0,
            "ink": 0,
            "patches": 0,
            "label": 1,
            "label_string": "malignant",
        },
        {
            "image": "ISIC_2.png",
            "dark_corner": 1,
            "hair": 0,
            "gel_border": 1,
            "gel_bubble": 0,
            "ruler": 0,
            "ink": 1,
            "patches": 0,
            "label": 0,
            "label_string": "benign",
        },
        {
            "image": "ISIC_3.jpg",
            "dark_corner": 0,
            "hair": 1,
            "gel_border": 0,
            "gel_bubble": 0,
            "ruler": 1,
            "ink": 0,
            "patches": 1,
            "label": 1,
            "label_string": "malignant",
        },
    ])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def read_images(csv_path: Path):
    return pd.read_csv(csv_path)["image"].tolist()


def create_dummy_image(images_dir, img_name, extension):
    # Create a dummy image for testing
    img_path = images_dir / f"{img_name}{extension}"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color="white").save(img_path)


class TestISICBiasDataset(unittest.TestCase):
    """Test ISIC2018BiasDataset with synthetic test data"""

    @classmethod
    def setUpClass(cls):
        """Set up the test resources path."""
        cls.isic_dataset_path = Path(__file__).parent.parent.parent / "test-resources" / "core" / "isic_artifacts"
        cls.raw_isic_dataset_path = cls.isic_dataset_path / "isic_artifacts_raw"
        cls.clean_isic_dataset_path = cls.isic_dataset_path / "isic_artifacts_cleaned"
        cls.temp_generated_folder = cls.isic_dataset_path / "temp_generated_csv"
        cls.test_roots = [
            cls.raw_isic_dataset_path,
            cls.clean_isic_dataset_path
        ]
        cls.backup_csv_file = cls.raw_isic_dataset_path / "isic_bias.csv.bak"
        cls.backup_csv_file.write_text((cls.raw_isic_dataset_path/"isic_bias.csv").read_text())

    @classmethod
    def tearDownClass(cls):
        """Re-create raw csv filed with ; delimiter."""
        if cls.backup_csv_file.exists():
            (cls.raw_isic_dataset_path/"isic_bias.csv").write_text(cls.backup_csv_file.read_text())
            cls.backup_csv_file.unlink()

    def tearDown(self):
        """Remove dummy image files after each test."""
        for root in self.test_roots:
            images_dir = Path(root) / "images"
            if images_dir.exists():
                shutil.rmtree(images_dir, ignore_errors=True)
        temp_generated_folder = Path(self.temp_generated_folder)
        if temp_generated_folder.exists():
            shutil.rmtree(temp_generated_folder, ignore_errors=True)

    def test_raw_artifact_csv(self):
        # Ensure normalized CSV is now comma-separated
        csv_file = self.raw_isic_dataset_path / "isic_bias.csv"
        with open(csv_file, "r") as f:
            line = f.readline()
            self.assertIn(",", line)  # Should now be comma-separated
            self.assertNotIn(";", line)


    def test_clean_artifact_csv(self):
        # Ensure file remains comma-delimited
        csv_file = self.clean_isic_dataset_path / "isic_bias.csv"
        with open(csv_file, "r") as f:
            line = f.readline()
            self.assertIn(",", line)

    def test_patient_count(self):
        """Ensure both raw and cleaned versions load correctly."""

        test_roots = [
            self.raw_isic_dataset_path,  # folder with semicolon CSV
            self.clean_isic_dataset_path,  # folder with comma CSV
        ]

        for root in test_roots:
            images_dir = Path(root) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            create_dummy_image(images_dir, "ISIC_10", ".png")
            with self.subTest(root=root):
                dataset = ISICBiasDataset(root=str(root))
                dataset_size = len(dataset.unique_patient_ids)

                print(f"[{root}] Total Unique patients: {dataset_size}")
                self.assertEqual(dataset_size, 10, "Invalid number of patients")

    def test_stats(self):
        """Test .stats() method execution."""
        test_roots = [
            self.raw_isic_dataset_path,  # folder with semicolon CSV
            self.clean_isic_dataset_path,  # folder with comma CSV
        ]
        for root in test_roots:
            images_dir = Path(root) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            create_dummy_image(images_dir, "ISIC_10", ".png")
            try:
                self.dataset = ISICBiasDataset(root=str(root))
                self.dataset.stats()
                print("dataset.stats() executed successfully")
            except Exception as e:
                print(f"✗ dataset.stats() failed with error: {e}")
                self.fail(f"dataset.stats() failed: {e}")

    def test_get_patient_events_by_id(self):
        test_roots = [
            self.raw_isic_dataset_path,  # folder with semicolon CSV
            self.clean_isic_dataset_path,  # folder with comma CSV
        ]
        for root in test_roots:
            images_dir = Path(root) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            create_dummy_image(images_dir, "ISIC_10", ".png")
            """Test get_patient and get_events methods for image 9."""
            self.dataset = ISICBiasDataset(root=str(root))
            patient = self.dataset.get_patient("9")
            self.assertIsNotNone(patient, msg="ISIC_10 should exist in the dataset")
            print(f"ISIC_10 found: {patient}")

            print("Getting events for patient ISIC_10...")
            events = patient.get_events()
            self.assertEqual(
                len(events), 1, msg="get_events() one sample"
            )
            print(f"Retrieved {len(events)} events")
            self.assertEqual(events[0].event_type, "isic_artifacts_raw")
            self.assertEqual(events[0].__getitem__("dark_corner"), "1")
            self.assertEqual(events[0].__getitem__("hair"), "1")
            self.assertEqual(events[0].__getitem__("gel_border"), "0")
            self.assertEqual(events[0].__getitem__("gel_bubble"), "1")
            self.assertEqual(events[0].__getitem__("ruler"), "0")
            self.assertEqual(events[0].__getitem__("ink"), "1")
            self.assertEqual(events[0].__getitem__("patches"), "0")
            self.assertEqual(events[0].__getitem__("label"), "0")
            self.assertEqual(events[0].__getitem__("label_string"), "malignant")

            image_path = events[0].__getitem__("image")
            self.assertIsNotNone(image_path, msg=f"'image must not be None")
            self.assertTrue(str(image_path).strip(), msg=f" image must not be empty")

            abs_path = Path(root) / "images" / image_path
            print(abs_path)

            self.assertTrue(
                abs_path.exists(),
                msg=f"Image file not found at resolved path: {abs_path}", )

    def test_updates_png_to_jpg(self):
        # folder with temp generated CSV
        root = Path(self.temp_generated_folder)
        images_dir = root / "images"
        csv_path = root / "temp_isic_bias.csv"

        create_dummy_image(images_dir, "ISIC_2", ".jpg")

        create_temp_isic_csv(csv_path)

        # CSV says .png
        csv_path = root / "temp_isic_bias.csv"

        normalize_image_extensions(root_path=root, images_path=images_dir)

        images = read_images(csv_path)
        self.assertEqual(images[1], "ISIC_2.jpg")

    def test_updates_jpg_to_png(self):
        # folder with temp generated CSV
        root = Path(self.temp_generated_folder)
        images_dir = root / "images"
        csv_path = root / "temp_isic_bias.csv"

        create_dummy_image(images_dir, "ISIC_3", ".png")

        create_temp_isic_csv(csv_path)

        # CSV says .jpg
        csv_path = root / "temp_isic_bias.csv"

        normalize_image_extensions(root_path=root, images_path=images_dir)

        images = read_images(csv_path)
        self.assertEqual(images[2], "ISIC_3.png")

    def test_no_change_when_referenced_file_exists(self):
        # folder with temp generated CSV
        root = Path(self.temp_generated_folder)
        images_dir = root / "images"
        csv_path = root / "temp_isic_bias.csv"

        create_dummy_image(images_dir, "ISIC_1", ".png")

        create_temp_isic_csv(csv_path)

        csv_path = root / "temp_isic_bias.csv"

        normalize_image_extensions(root_path=root, images_path=images_dir)

        images = read_images(csv_path)
        self.assertEqual(images[0], "ISIC_1.png")

    def test_handles_multiple_rows(self):
        # folder with temp generated CSV
        root = Path(self.temp_generated_folder)
        images_dir = root / "images"
        csv_path = root / "temp_isic_bias.csv"

        create_dummy_image(images_dir, "ISIC_1", ".png")
        create_dummy_image(images_dir, "ISIC_2", ".jpg")

        create_temp_isic_csv(csv_path)

        csv_path = root / "temp_isic_bias.csv"

        normalize_image_extensions(root_path=root, images_path=images_dir)

        images = read_images(csv_path)
        self.assertEqual(images, ["ISIC_1.png", "ISIC_2.jpg", "ISIC_3.jpg"])
