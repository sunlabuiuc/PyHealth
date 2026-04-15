import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch

from pyhealth.datasets import ECGQADataset
from pyhealth.tasks import ECGQA


class TestECGQADataset(unittest.TestCase):
    """Test ECG-QA dataset with synthetic test data."""

    def setUp(self):
        """Set up train/valid/test JSON files and a temp cache dir."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        for split in ("train", "valid", "test"):
            (self.root / split).mkdir()

        train_records = [
            {
                "ecg_id": [1],
                "question": "Does this ECG show normal sinus rhythm?",
                "answer": ["yes"],
                "question_type": "single-verify",
                "attribute_type": "scp_code",
                "template_id": 1,
                "question_id": 100,
                "sample_id": 1000,
                "attribute": ["NORM"],
            },
            {
                "ecg_id": [2],
                "question": "What rhythm does this ECG show?",
                "answer": ["sinus rhythm", "atrial fibrillation"],
                "question_type": "single-choose",
                "attribute_type": "rhythm",
                "template_id": 2,
                "question_id": 101,
                "sample_id": 1001,
                "attribute": ["SR", "AFIB"],
            },
            {
                "ecg_id": [3],
                "question": "Are both left axis deviation and right bundle branch block present?",
                "answer": ["yes"],
                "question_type": "comparison-verify",
                "attribute_type": "scp_code",
                "template_id": 3,
                "question_id": 102,
                "sample_id": 1002,
                "attribute": ["LAD", "RBBB"],
            },
        ]
        valid_records = [
            {
                "ecg_id": [4],
                "question": "What ECG abnormalities are present?",
                "answer": ["left axis deviation"],
                "question_type": "single-query",
                "attribute_type": "scp_code",
                "template_id": 4,
                "question_id": 103,
                "sample_id": 1003,
                "attribute": ["LAD"],
            },
        ]
        test_records = [
            {
                "ecg_id": [5],
                "question": "Is the heart rate above 100 beats per minute?",
                "answer": ["no"],
                "question_type": "single-verify",
                "attribute_type": "heart_rate",
                "template_id": 5,
                "question_id": 104,
                "sample_id": 1004,
                "attribute": ["bradycardia"],
            },
        ]

        (self.root / "train" / "00.json").write_text(json.dumps(train_records))
        (self.root / "valid" / "00.json").write_text(json.dumps(valid_records))
        (self.root / "test" / "00.json").write_text(json.dumps(test_records))

        # Redirect Path.home() into the temp dir so that prepare_metadata's
        # ~/.cache/pyhealth/ecg_qa fallback cannot find a pre-existing user
        # cache and shadow the test fixture with stale data.
        self._home_patch = patch.object(Path, "home", return_value=self.root)
        self._home_patch.start()

        self._cache_tmp = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        self._home_patch.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self._cache_tmp, ignore_errors=True)

    def test_dataset_initialization(self):
        """Test ECGQADataset initialization"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "ecg_qa")
        self.assertEqual(dataset.root, str(self.root))

    def test_metadata_file_created(self):
        """Test ecg-qa-pyhealth.csv is created in root"""
        ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        metadata_file = self.root / "ecg-qa-pyhealth.csv"
        self.assertTrue(metadata_file.exists())

    def test_patient_count(self):
        """Test only single-* records are loaded (4 of 5)"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        # ecg_ids 1, 2, 4, 5 survive; ecg_id 3 is comparison-verify and is filtered.
        self.assertEqual(len(dataset.unique_patient_ids), 4)
        self.assertEqual(
            sorted(dataset.unique_patient_ids), ["1", "2", "4", "5"]
        )

    def test_filters_non_single_question_types(self):
        """Test that comparison-verify records are dropped"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        self.assertNotIn("3", dataset.unique_patient_ids)

    def test_stats_method(self):
        """Test stats method runs without error"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        patient = dataset.get_patient("1")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "1")

    def test_get_patient_not_found(self):
        """Test that patient not found throws error."""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        with self.assertRaises(AssertionError):
            dataset.get_patient("999")

    def test_single_verify_event_fields(self):
        """Test a single-verify event surfaces the expected attributes"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        events = dataset.get_patient("1").get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["question_type"], "single-verify")
        self.assertEqual(events[0]["answer"], "yes")
        self.assertEqual(events[0]["attribute"], "NORM")

    def test_single_choose_event_joins_multi_valued_fields(self):
        """Test single-choose records join answer/attribute with ';'"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        events = dataset.get_patient("2").get_events()
        self.assertEqual(events[0]["answer"], "sinus rhythm;atrial fibrillation")
        self.assertEqual(events[0]["attribute"], "SR;AFIB")

    def test_default_task(self):
        """Test default_task returns an ECGQA instance"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        self.assertIsInstance(dataset.default_task, ECGQA)

    def test_set_task_ecgqa(self):
        """Test ECGQA task yields one sample per QA pair"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        samples = dataset.set_task(ECGQA())
        self.assertEqual(len(samples), 4)

    def test_invalid_ecg_source_raises(self):
        """Test ValueError on invalid ecg_source"""
        with self.assertRaises(ValueError):
            ECGQADataset(
                root=str(self.root),
                ecg_source="nope",
                cache_dir=self._cache_tmp,
            )


class TestECGQAVerifyData(unittest.TestCase):
    """Test the structural checks performed by ECGQADataset._verify_data."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_root_raises(self):
        """Test FileNotFoundError when root does not exist"""
        bogus = self.root / "does-not-exist"
        with self.assertRaises(FileNotFoundError):
            ECGQADataset(root=str(bogus))

    def test_missing_split_dir_raises(self):
        """Test FileNotFoundError when train/valid/test dirs are missing"""
        with self.assertRaises(FileNotFoundError):
            ECGQADataset(root=str(self.root))

    def test_empty_split_dir_raises(self):
        """Test ValueError when a split dir has no JSON files"""
        for split in ("train", "valid", "test"):
            (self.root / split).mkdir()
        with self.assertRaises(ValueError):
            ECGQADataset(root=str(self.root))


if __name__ == "__main__":
    unittest.main()
