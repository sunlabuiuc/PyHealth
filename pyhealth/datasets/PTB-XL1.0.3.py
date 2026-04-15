"""

Pyhealth dataset for the 1.0.3 PTB-XL dataset.

Dataset link: 
	https://physionet.org/content/ptb-xl/1.0.3/

Dataset paper: 
	J. Tang, T. Xia, Y. Lu, C. Mascolo, and A. Saeed, "Electrocardiogram-language model for few-shot question answering with meta learning," arXiv preprint arXiv:2410.14464, 2024.

Dataset paper link:
	https://arxiv.org/abs/2410.14464

Author:
	Yiyun Wang (yiyunw3@illinois.edu)

"""
import pandas as pd
import os
import urllib.request
import requests
from pyhealth.datasets import BaseSignalDataset
import zipfile
from pathlib import Path

"""
Dataset class for the PTB-XL 1.0.3 dataset.

Args:
    dataset_name: name of the dataset.
    root: root directory of the raw data (should contain many csv files).
    dev: whether to enable dev mode (only use a small subset of the data).
        Default is False.
    refresh_cache: whether to refresh the cache; if true, the dataset will
        be processed from scratch and the cache will be updated. Default is False.
"""
class PTBXLDataset(BaseSignalDataset):
	"""
	Initialize the PTB-XL dataset.
	
	Attributes: 
		root (str): Root directory of the raw data.
		download (bool): True iff requested to download dataset. Default to False.
	"""
	def __init__(self, 
		root: str = '.',
		download: bool = False,
		down_sampled: bool = False) -> None:

		# Determine the root path, where most of the data is stored
		# self.data_path: str = os.path.join(root, 'ptb_xl_processed_full.zip')
		self.data_path: str = os.path.join(root, 'test.zip')
		self.root = root

		# Determine signal path, where to fetch the signal samples
		signal_folder = 'records100' if down_sampled else 'records500'
		root_path = os.path.join(root, 'physionet.org/files/ptb-xl/1.0.3/') if download else root
		self.signal_path: str = os.path.join(root_path, signal_folder)

		# Download the dataset from online source to root if needed
		self._download(download)

		super().__init__(
			root=root,
			dataset_name='PTB-XL1.0.3',
		)


	"""
	Download PTB-XL dataset from public google drive sources. 
	It will contain both the original and downsampled versions, 
	in /records500 and /records100 folder respectively.
	"""
	def _download(self, download) -> None:
		
		if download:
			# zip_id = '1IE-4Co1fLRoEI9jez2pwuf9HPmFRzuLX' # full
			zip_id = '1Q9Ksxj4gSrsHVb6qqICI0nm0K8HWtDW2' # test
			response = requests.get(f'https://drive.google.com/uc?export=download&id={zip_id}')
			with open(self.data_path, 'wb') as file:
				file.write(response.content)

			with zipfile.ZipFile(self.data_path, "r") as z:
				z.extractall(self.root)

	"""
	Process and return a dictionary of the requested PTB-XL data for each patient.
	Each patient will have a corresponding object that contains 
	load_from_path, patient_id, signal_file, label_file, and save_to_path.
	"""
	def process_EEG_data(self):
		patients = {}

		for dirpath, dirnames, filenames in os.walk(self.signal_path):
			for filename in filenames:
				f = Path(filename).stem
				pid = f.split('_')[0]

				if pid not in patients:
					patients[pid] = [
						{
							"load_from_path": dirpath,
							"patient_id": pid,
							"signal_file": f + '.dat',
							"label_file": f + '.hea',
							"save_to_path": self.filepath,
						}
					]

		return patients

if __name__ == "__main__":
	dataset = PTBXLDataset(root='../../../../', download=False, down_sampled=True)
	dataset.stat()
	dataset.info()
	print(dataset.process_EEG_data())
