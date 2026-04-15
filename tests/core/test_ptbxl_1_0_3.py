import unittest
from pyhealth.datasets import PTBXLDataset

class TestPTBXLDataset(unittest.TestCase):
	"""
	Test PTB-XL 1.0.3 dataset with demo data.
	"""
	def setUp(self):
		self.dataset = PTBXLDataset(root='../../../../', download=False, downsampled=True, dev=True)

	"""
	Verify if the dataset contains correct basic information 
	"""
	def testBasicInfo(self):
		self.assertEqual(self.dataset.dataset_name, 'PTB-XL1.0.3')
		self.assertIsInstance(self.dataset.filepath, str)

	"""
	Verify if the dataset contains expected data
	"""
	def testData(self):
		self.assertIsInstance(self.dataset.patients, dict)
		self.assertLessEqual(len(self.dataset.patients), 5)
		for pid, value in self.dataset.patients.items():
			self.assertIsNotNone(pid)
			self.assertIsInstance(pid, str)

			self.assertIsInstance(value, list)
			self.assertIsInstance(value[0], dict)
			self.assertIsNotNone(value[0]['load_from_path'])
			self.assertIsNotNone(value[0]['patient_id'])
			self.assertIsNotNone(value[0]['signal_file'])
			self.assertIsNotNone(value[0]['label_file'])
			self.assertIsNotNone(value[0]['save_to_path'])

if __name__ == "__main__":
    unittest.main()
