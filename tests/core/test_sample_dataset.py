import unittest
from pathlib import Path

from pyhealth.datasets.sample_dataset import SampleDataset, create_sample_dataset


class TestSampleDatasetParity(unittest.TestCase):
    def setUp(self):
        # Create a slightly larger dataset to make shuffling more obvious
        self.samples = [
            {"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i, "label": i % 2}
            for i in range(20)
        ]
        self.input_schema = {"feature": "raw"}
        self.output_schema = {"label": "raw"}

    def _get_datasets(self):
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
        return ds_disk, ds_mem

    def test_len(self):
        ds_disk, ds_mem = self._get_datasets()
        self.assertEqual(len(ds_disk), 20)
        self.assertEqual(len(ds_mem), 20)
        self.assertEqual(len(ds_disk), len(ds_mem))

    def test_getitem(self):
        ds_disk, ds_mem = self._get_datasets()
        for i in range(len(self.samples)):
            item_disk = ds_disk[i]
            item_mem = ds_mem[i]
            self.assertEqual(item_disk["feature"], item_mem["feature"])
            self.assertEqual(item_disk["label"], item_mem["label"])
            self.assertEqual(item_disk["patient_id"], item_mem["patient_id"])

    def test_iter(self):
        ds_disk, ds_mem = self._get_datasets()
        list_disk = list(ds_disk)
        list_mem = list(ds_mem)
        
        self.assertEqual(len(list_disk), len(list_mem))
        for d, m in zip(list_disk, list_mem):
            self.assertEqual(d["feature"], m["feature"])

    def test_subset_indices(self):
        ds_disk, ds_mem = self._get_datasets()
        indices = [0, 5, 10, 15, 19]
        
        sub_disk = ds_disk.subset(indices)
        sub_mem = ds_mem.subset(indices)
        
        self.assertEqual(len(sub_disk), len(sub_mem))
        self.assertEqual(len(sub_disk), 5)
        
        list_disk = list(sub_disk)
        list_mem = list(sub_mem)
        
        for d, m in zip(list_disk, list_mem):
            self.assertEqual(d["feature"], m["feature"])

    def test_subset_slice(self):
        ds_disk, ds_mem = self._get_datasets()
        s = slice(2, 18, 2)
        
        sub_disk = ds_disk.subset(s)
        sub_mem = ds_mem.subset(s)
        
        self.assertEqual(len(sub_disk), len(sub_mem))
        
        list_disk = list(sub_disk)
        list_mem = list(sub_mem)
        
        for d, m in zip(list_disk, list_mem):
            self.assertEqual(d["feature"], m["feature"])

    def test_subset_rebuilds_index_mappings(self):
        ds_disk, ds_mem = self._get_datasets()

        for dataset in (ds_disk, ds_mem):
            subset = dataset.subset([5, 2, 15])
            self.assertEqual(
                subset.patient_to_index,
                {"p5": [0], "p2": [1], "p15": [2]},
            )
            self.assertEqual(
                subset.record_to_index,
                {"r5": [0], "r2": [1], "r15": [2]},
            )

            nested_subset = subset.subset([2, 0])
            self.assertEqual(
                nested_subset.patient_to_index,
                {"p15": [0], "p5": [1]},
            )
            self.assertEqual(
                nested_subset.record_to_index,
                {"r15": [0], "r5": [1]},
            )

    def test_set_shuffle(self):
        ds_disk, ds_mem = self._get_datasets()
        
        # Test shuffle=True
        ds_disk.set_shuffle(True)
        ds_mem.set_shuffle(True)
        
        # Iterating should return all elements, but likely in different order than original
        # and potentially different order between disk and mem (implementation detail)
        # But the set of elements should be identical.
        
        items_disk = list(ds_disk)
        items_mem = list(ds_mem)
        
        self.assertEqual(len(items_disk), 20)
        self.assertEqual(len(items_mem), 20)
        
        # Check that we have the same set of features
        features_disk = sorted([x["feature"] for x in items_disk])
        features_mem = sorted([x["feature"] for x in items_mem])
        features_orig = sorted([x["feature"] for x in self.samples])
        
        self.assertEqual(features_disk, features_orig)
        self.assertEqual(features_mem, features_orig)
        
        # Test shuffle=False resets to original order
        ds_disk.set_shuffle(False)
        ds_mem.set_shuffle(False)
        
        items_disk_ordered = list(ds_disk)
        items_mem_ordered = list(ds_mem)
        
        for i in range(20):
            self.assertEqual(items_disk_ordered[i]["feature"], i)
            self.assertEqual(items_mem_ordered[i]["feature"], i)

    def test_close_preserves_caller_owned_directory(self):
        owner, _ = self._get_datasets()
        path = Path(owner.path)
        dataset = SampleDataset(path=str(path))

        try:
            dataset.close()
            self.assertTrue(path.exists())
            reopened = SampleDataset(path=str(path))
            self.assertEqual(reopened[0]["feature"], 0)
        finally:
            owner.close()

    def test_close_preserves_directory_shared_with_subset(self):
        owner, _ = self._get_datasets()
        path = Path(owner.path)
        subset = owner.subset([0])

        try:
            subset.close()
            self.assertTrue(path.exists())
        finally:
            owner.close()

    def test_close_removes_owned_temporary_directory(self):
        owner, _ = self._get_datasets()
        path = Path(owner.path)

        owner.close()

        self.assertFalse(path.exists())

if __name__ == "__main__":
    unittest.main()
