import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import dask.dataframe as dd

from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.tasks.base_task import BaseTask
from pyhealth.processors.ignore_processor import IgnoreProcessor
from pyhealth.processors import RawProcessor

class MockTask(BaseTask):
    task_name = "test_task"
    input_schema = {
        "keep_field": "raw", 
        "ignore_field": "raw"
    }
    output_schema = {"label": "binary"}

    def __call__(self, patient):
        return [{
            "keep_field": "keep_val",
            "ignore_field": "ignore_val",
            "label": 0 if patient.patient_id == "1" else 1,
            "patient_id": patient.patient_id
        }]

class MockDataset(BaseDataset):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, tables=[], **kwargs)
    
    def load_data(self):
        return dd.from_pandas(
            pd.DataFrame({
                "patient_id": ["1", "2"],
                "event_type": ["visit", "visit"],
                "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")],
            }), 
            npartitions=1
        )

class TestIgnoreProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = self.tmp_dir
        self.dataset = MockDataset(root=self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_ignore_processor_with_set_task(self):
        task = MockTask()
        
        # 1. Normal set_task
        ds1 = self.dataset.set_task(task)
        self.assertIn("ignore_field", ds1.input_schema)
        
        # Check data
        # We need to access the first sample. 
        # Since SampleDataset is a StreamingDataset, we can index it or iterate.
        sample1 = ds1[0]
        self.assertIn("ignore_field", sample1)
        self.assertEqual(sample1["ignore_field"], "ignore_val")

        # 2. set_task with ignore processor
        # We MUST provide processors for ALL fields to avoid re-population logic in SampleBuilder
        ds2 = self.dataset.set_task(
            task, 
            input_processors={
                "keep_field": RawProcessor(),
                "ignore_field": IgnoreProcessor()
            }
        )
        
        # Expectation: "ignore_field" should be removed from input_schema of the dataset
        # This is what the user asked for: "result should be the input_schema & input_processors does not exists"
        
        # Note: Depending on current implementation, this might fail.
        self.assertNotIn("ignore_field", ds2.input_schema)
        self.assertNotIn("ignore_field", ds2.input_processors)
        
        sample2 = ds2[0]
        # Expectation: "ignore_field" should NOT be in the sample data
        self.assertNotIn("ignore_field", sample2)
        
        # 'keep_field' should still be there
        self.assertIn("keep_field", sample2)
        self.assertEqual(sample2["keep_field"], "keep_val")

if __name__ == "__main__":
    unittest.main()
