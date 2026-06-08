import unittest
import tempfile
import os
import shutil
from pyhealth.datasets import PTBXLDataset
from pathlib import Path

class TestPTBXLDataset(unittest.TestCase):
	"""
	Test PTB-XL 1.0.3 dataset with demo data.
	"""
	def setUp(self):
		sample_records = {
		'00001': {
		'load_from_path': 'sample/path',
		'signal_file': '00001_lr.dat',
		'label_file': '00001_lr.hea',
		'save_to_path': 'sample/path',
		},
		'00002': {
		'load_from_path': 'sample/path',
		'signal_file': '00002_lr.dat',
		'label_file': '00002_lr.hea',
		'save_to_path': 'sample/path',
		},
		'00003': {
		'load_from_path': 'sample/path',
		'signal_file': '00003_lr.dat',
		'label_file': '00003_lr.hea',
		'save_to_path': 'sample/path',
		},
		'00004': {
		'load_from_path': 'sample/path',
		'signal_file': '00004_lr.dat',
		'label_file': '00004_lr.hea',
		'save_to_path': 'sample/path',
		},
		'00005': {
		'load_from_path': 'sample/path',
		'signal_file': '00005_lr.dat',
		'label_file': '00005_lr.hea',
		'save_to_path': 'sample/path',
		},
		'00006': {
		'load_from_path': 'sample/path',
		'signal_file': '00006_lr.dat',
		'label_file': '00006_lr.hea',
		'save_to_path': 'sample/path',
		},
		}

		self.temp_dir = tempfile.mkdtemp()
		self.root = Path(self.temp_dir)

		os.makedirs(os.path.join(self.root, 'records100'))

		for i in sample_records.keys():
			with open(os.path.join(self.root, f'records100/{i}.dat'), 'w') as f:
				f.write('sample .dat data')
			with open(os.path.join(self.root, f'records100/{i}.hea'), 'w') as f:
				f.write('sample .hea data')

		self.dataset = PTBXLDataset(root=self.root, download=False, downsampled=True, dev=True)


	"""
	Verify if the dataset contains correct basic information 
	"""
	def testBasicInfo(self):
		self.assertEqual(self.dataset.dataset_name, 'PTB-XL')
		self.dataset.stats()

	"""
	Verify if the dataset contains expected data
	"""
	def testData(self):

		# Test if dev mode has been applied
		self.assertLessEqual(len(self.dataset.unique_patient_ids), 5)

		# Test info of the first patient
		patient = self.dataset.get_patient('00001')
		self.assertIsNotNone(patient)
		self.assertIsInstance(patient.patient_id, str)
		self.assertEqual(patient.patient_id, '00001')

		events = self.dataset.get_patient('00001').get_events()
		self.assertEqual(len(events), 1)
		self.assertIsNotNone(events[0]['load_from_path'])
		self.assertIsNotNone(events[0]['signal_file'])
		self.assertIsNotNone(events[0]['label_file'])
		self.assertIsNotNone(events[0]['save_to_path'])

	def tearDown(self):
		shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
	unittest.main()