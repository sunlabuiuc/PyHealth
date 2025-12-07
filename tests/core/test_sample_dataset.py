import unittest
import pickle
from pyhealth.datasets.sample_dataset import create_sample_dataset

class TestSampleDataset(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"patient_id": "p1", "record_id": "r1", "feature": "a", "label": 1},
            {"patient_id": "p1", "record_id": "r2", "feature": "b", "label": 0},
            {"patient_id": "p2", "record_id": "r3", "feature": "c", "label": 1},
            {"patient_id": "p3", "record_id": "r4", "feature": "d", "label": 0},
            {"patient_id": "p3", "record_id": "r5", "feature": "e", "label": 1},
        ]
        self.input_schema = {"feature": "raw"}
        self.output_schema = {"label": "raw"}

    def test_sample_dataset_subset_slice(self):
        # Create SampleDataset (disk-based)
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            in_memory=False
        )
        
        # Define a slice object
        s = slice(1, 4) # Slice [1:4] -> 1, 2, 3
        
        subset = dataset.subset(s)
        
        self.assertEqual(len(subset), 3)
        
        # Check content
        subset_data = list(subset)
        self.assertEqual(subset_data[0]["feature"], "b")
        self.assertEqual(subset_data[1]["feature"], "c")
        self.assertEqual(subset_data[2]["feature"], "d")

    def test_in_memory_sample_dataset_behavior(self):
        # Create both datasets
        ds_disk = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            in_memory=False
        )
        
        ds_mem = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            in_memory=True
        )
        
        # 1. Test len
        self.assertEqual(len(ds_disk), len(ds_mem))
        self.assertEqual(len(ds_disk), 5)
        
        # 2. Test iter
        iter_disk = list(ds_disk)
        iter_mem = list(ds_mem)
        
        for d, m in zip(iter_disk, iter_mem):
            self.assertEqual(d["feature"], m["feature"])
            self.assertEqual(d["label"], m["label"])
            self.assertEqual(d["patient_id"], m["patient_id"])
            self.assertEqual(d["record_id"], m["record_id"])

        # 3. Test getitem
        for i in range(len(ds_disk)):
            d = ds_disk[i]
            m = ds_mem[i]
            self.assertEqual(d["feature"], m["feature"])
            self.assertEqual(d["label"], m["label"])

        # 4. Test subset with list
        indices = [0, 2, 4]
        sub_disk = ds_disk.subset(indices)
        sub_mem = ds_mem.subset(indices)
        
        self.assertEqual(len(sub_disk), len(sub_mem))
        
        for d, m in zip(sub_disk, sub_mem):
            self.assertEqual(d["feature"], m["feature"])
            self.assertEqual(d["label"], m["label"])

        # 5. Test subset with slice
        s = slice(0, 3)
        sub_disk_slice = ds_disk.subset(s)
        sub_mem_slice = ds_mem.subset(s)
        
        self.assertEqual(len(sub_disk_slice), len(sub_mem_slice))
        for d, m in zip(sub_disk_slice, sub_mem_slice):
            self.assertEqual(d["feature"], m["feature"])

if __name__ == "__main__":
    unittest.main()
