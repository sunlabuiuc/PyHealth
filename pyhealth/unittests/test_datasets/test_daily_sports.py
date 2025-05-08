import unittest
from pyhealth.datasets import DailySportsActivitiesDataset

class DailySportsActivitiesDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = DailySportsActivitiesDataset(
            root="data/Daily_and_Sports_Activity/data/",
            dev=True,
            refresh_cache=True
        )
        
    def test_basic_stats(self):
        self.assertEqual(len(self.dataset.patients), 8)  # 8 subjects
        self.assertGreaterEqual(len(self.dataset.samples), 100)  # At least 100 samples
        
    def test_sample_shape(self):
        sample = self.dataset.samples[0]
        self.assertEqual(sample["signal"].shape, (125, 45))  # Time steps x sensors
        self.assertIsInstance(sample["label"], int)
        
if __name__ == "__main__":
    unittest.main()
