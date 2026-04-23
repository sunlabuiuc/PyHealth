import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pyhealth.datasets import ClinicalJargonDataset, split_by_patient
from pyhealth.tasks import ClinicalJargonVerification


class TestClinicalJargonDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "clinical_jargon"
        )
        cls.cache_dir = TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir.cleanup()

    def make_dataset(self):
        return ClinicalJargonDataset(
            root=str(self.root),
            cache_dir=self.cache_dir.name,
        )

    def test_dataset_initialization(self):
        dataset = self.make_dataset()
        self.assertEqual(dataset.dataset_name, "clinical_jargon")

    def test_missing_csv_requires_explicit_download(self):
        with TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                ClinicalJargonDataset(root=root, cache_dir=self.cache_dir.name)

    def test_default_task(self):
        dataset = self.make_dataset()
        self.assertIsInstance(dataset.default_task, ClinicalJargonVerification)

    def test_num_patients(self):
        dataset = self.make_dataset()
        self.assertEqual(len(dataset.unique_patient_ids), 9)

    def test_get_patient(self):
        dataset = self.make_dataset()
        patient = dataset.get_patient(dataset.unique_patient_ids[0])
        self.assertIsNotNone(patient)
        self.assertEqual(len(patient.get_events("examples")), 1)

    def test_release62_task_samples(self):
        dataset = self.make_dataset()
        task = ClinicalJargonVerification(benchmark="all", casi_variant="release62")
        samples = dataset.set_task(task)
        self.assertEqual(len(samples), 24)

    def test_paper59_task_samples(self):
        dataset = self.make_dataset()
        task = ClinicalJargonVerification(benchmark="all", casi_variant="paper59")
        samples = dataset.set_task(task)
        self.assertEqual(len(samples), 18)

    def test_medlingo_distractor_control(self):
        dataset = self.make_dataset()
        task = ClinicalJargonVerification(
            benchmark="medlingo",
            medlingo_distractors=1,
        )
        samples = dataset.set_task(task)
        self.assertEqual(len(samples), 6)

    def test_split_by_patient_keeps_candidate_rows_together(self):
        dataset = self.make_dataset()
        task = ClinicalJargonVerification(
            benchmark="medlingo",
            medlingo_distractors=1,
        )
        samples = dataset.set_task(task)
        splits = split_by_patient(samples, [0.6, 0.2, 0.2], seed=42)
        split_patient_ids = [
            {split_dataset[index]["patient_id"] for index in range(len(split_dataset))}
            for split_dataset in splits
        ]
        for left_index, left_ids in enumerate(split_patient_ids):
            for right_ids in split_patient_ids[left_index + 1 :]:
                self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_task_output_shape(self):
        dataset = self.make_dataset()
        task = ClinicalJargonVerification(benchmark="casi", casi_variant="paper59")
        samples = dataset.set_task(task)
        sample = samples[0]
        self.assertIn("paired_text", sample)
        self.assertIn("label", sample)
        self.assertIn("candidate_expansion", sample)

    def test_rejects_non_local_casi_cache_names(self):
        with patch.object(
            ClinicalJargonDataset,
            "_download_text",
            return_value='[{"name":"../escape.csv","download_url":"https://example.com"}]',
        ):
            with self.assertRaises(ValueError):
                ClinicalJargonDataset._fetch_casi_rows(Path(self.cache_dir.name))


if __name__ == "__main__":
    unittest.main()
