import unittest
import tempfile
import shutil
from pathlib import Path
from textwrap import dedent

from pyhealth.datasets import NoisyMRCICUMortalityDataset


class TestNoisyMRCICUMortalityDataset(unittest.TestCase):
    """Test NoisyMRC ICU mortality dataset with mock csv."""

    def setUp(self):
        """Set up temp directory with mock mortality csvs and yaml config."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        # Create data_mortality directory
        data_dir = self.root / "data_mortality"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Minimal mock csv content: hospital_id is patient_id, y is label
        header = "hospital_id,y,age,bmi\n"
        rows = [
            "1,0,65,27.5\n",
            "2,1,72,30.1\n",
        ]

        # alsocat csv
        alsocat_csv = data_dir / "mortality_alsocat.csv"
        with alsocat_csv.open("w") as f:
            f.write(header)
            f.writelines(rows)

        # nocat csv
        nocat_csv = data_dir / "mortality_nocat.csv"
        with nocat_csv.open("w") as f:
            f.write(header)
            f.writelines(rows)

        # Minimal YAML config pointing to mock csvs
        self.config_path = self.root / "mrc_test.yaml"
        yaml_text = dedent(
            """
            version: "1.0"

            tables:
              mortality_alsocat:
                file_path: "data_mortality/mortality_alsocat.csv"
                patient_id: "hospital_id"
                timestamp: null
                attributes:
                  - "y"
                  - "age"
                  - "bmi"

              mortality_nocat:
                file_path: "data_mortality/mortality_nocat.csv"
                patient_id: "hospital_id"
                timestamp: null
                attributes:
                  - "y"
                  - "age"
                  - "bmi"
            """
        ).lstrip()

        with self.config_path.open("w") as f:
            f.write(yaml_text)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization_alsocat(self):
        """Test NoisyMRCICUMortalityDataset initialization for alsocat table."""
        dataset = NoisyMRCICUMortalityDataset(
            root=str(self.root),
            table="mortality_alsocat",
            config_path=str(self.config_path),
            dev=False,
        )

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.table_name, "mortality_alsocat")
        self.assertEqual(dataset.dataset_name, "mrc_mortality_alsocat")
        self.assertEqual(len(dataset.unique_patient_ids), 2)

    def test_dataset_initialization_nocat(self):
        """Test NoisyMRCICUMortalityDataset initialization for nocat table."""
        dataset = NoisyMRCICUMortalityDataset(
            root=str(self.root),
            table="mortality_nocat",
            config_path=str(self.config_path),
            dev=False,
        )

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.table_name, "mortality_nocat")
        self.assertEqual(dataset.dataset_name, "mrc_mortality_nocat")
        self.assertEqual(len(dataset.unique_patient_ids), 2)

    def test_invalid_table_raises(self):
        """Test that invalid table name raises ValueError."""
        with self.assertRaises(ValueError):
            NoisyMRCICUMortalityDataset(
                root=str(self.root),
                table="not_a_real_table",
                config_path=str(self.config_path),
            )

    def test_get_patient(self):
        """Test get_patient method works with mock data."""
        dataset = NoisyMRCICUMortalityDataset(
            root=str(self.root),
            table="mortality_alsocat",
            config_path=str(self.config_path),
            dev=False,
        )
        patient_id = dataset.unique_patient_ids[0]
        patient = dataset.get_patient(patient_id)

        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, patient_id)

    def test_stats_method(self):
        """Test stats method runs without error."""
        dataset = NoisyMRCICUMortalityDataset(
            root=str(self.root),
            table="mortality_alsocat",
            config_path=str(self.config_path),
            dev=True,
        )
        dataset.stats()


if __name__ == "__main__":
    unittest.main()