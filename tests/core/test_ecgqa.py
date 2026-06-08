import unittest
import tempfile
import shutil
import json
from pathlib import Path

import torch

from pyhealth.datasets import ECGQADataset
from pyhealth.tasks import ECGQAPreprocessing


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

        self._cache_tmp = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self._cache_tmp, ignore_errors=True)

    def test_dataset_initialization(self):
        """Test ECGQADataset initialization"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "ecg_qa")
        self.assertEqual(dataset.root, str(self.root))

    def test_patient_count(self):
        """Test only single-* records are loaded (4 of 5)"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        # ecg_ids 1, 2, 4, 5 survive; ecg_id 3 is comparison-verify and is filtered.
        self.assertEqual(len(dataset.unique_patient_ids), 4)
        self.assertEqual(
            sorted(dataset.unique_patient_ids), ["00001", "00002", "00004", "00005"]
        )

    def test_filters_non_single_question_types(self):
        """Test that comparison-verify records are dropped"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        self.assertNotIn("00003", dataset.unique_patient_ids)

    def test_stats_method(self):
        """Test stats method runs without error"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        patient = dataset.get_patient("00001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "00001")

    def test_single_verify_event_fields(self):
        """Test a single-verify event surfaces the expected attributes"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        events = dataset.get_patient("00001").get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["question_type"], "single-verify")
        self.assertEqual(events[0]["answer"], "yes")
        self.assertEqual(events[0]["attribute"], "NORM")

    def test_single_choose_event_joins_multi_valued_fields(self):
        """Test single-choose records join answer/attribute with ';'"""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        events = dataset.get_patient("00002").get_events()
        self.assertEqual(events[0]["answer"], "sinus rhythm;atrial fibrillation")
        self.assertEqual(events[0]["attribute"], "SR;AFIB")


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


class TestECGQATask(unittest.TestCase):
    """Test the ECGQAPreprocessing task with synthetic data."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        for split in ("train", "valid", "test"):
            (self.root / split).mkdir()

        records = [
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
                "ecg_id": [1],
                "question": "What rhythm does this ECG show?",
                "answer": ["sinus rhythm"],
                "question_type": "single-query",
                "attribute_type": "rhythm",
                "template_id": 2,
                "question_id": 101,
                "sample_id": 1001,
                "attribute": ["SR"],
            },
        ]
        (self.root / "train" / "00.json").write_text(json.dumps(records))
        (self.root / "valid" / "00.json").write_text(json.dumps([records[0]]))
        (self.root / "test" / "00.json").write_text(json.dumps([records[1]]))

        self._cache_tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self._cache_tmp, ignore_errors=True)

    def test_text_only_mode(self):
        """Task with no signal_loader returns text-only samples."""
        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        samples = dataset.set_task(ECGQAPreprocessing())
        sample = samples[0]

        self.assertIn("question", sample)
        self.assertIn("answer", sample)
        self.assertIn("question_type", sample)
        self.assertIn("episode_class", sample)
        self.assertNotIn("signal", sample)

    def test_signal_loader_attaches_signal(self):
        """Task with signal_loader attaches signal tensor to samples."""
        def fake_loader(ecg_id):
            return torch.randn(12, 2500)

        dataset = ECGQADataset(root=str(self.root), cache_dir=self._cache_tmp)
        samples = dataset.set_task(ECGQAPreprocessing(signal_loader=fake_loader))
        sample = samples[0]

        self.assertIn("signal", sample)
        self.assertEqual(sample["signal"].shape, torch.Size([12, 2500]))


if __name__ == "__main__":
    unittest.main()
