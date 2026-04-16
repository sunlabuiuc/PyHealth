"""
Pyhealth dataset for the 1.0.3 PTB-XL dataset.

Dataset link:
	https://physionet.org/content/ptb-xl/1.0.3/

Dataset paper:
	J. Tang, T. Xia, Y. Lu, C. Mascolo, and A. Saeed,
	"Electrocardiogram-language model for few-shot question answering with meta
	learning," arXiv preprint arXiv:2410.14464, 2024.

Dataset paper link:
	https://arxiv.org/abs/2410.14464

Author:
    Jovian Wang (jovianw2@illinois.edu)
    Matthew Pham (mdpham2@illinois.edu)
    Yiyun Wang (yiyunw3@illinois.edu)
"""
import pandas as pd
import os
import logging
import urllib.request
import requests
import zipfile
import random
import csv
from pathlib import Path
from typing import Optional
from pyhealth.datasets.utils import hash_str, MODULE_CACHE_PATH
from . import BaseDataset

logger = logging.getLogger(__name__)

"""
Dataset class for the PTB-XL 1.0.3 dataset.

Args:
	dataset_name: name of the dataset.
	root: root directory of the raw data.
		Expected to contain folders for original (records500) or
		downsampled (records100) data with determined names.
	dev: whether to enable dev mode (only use a small subset of the data).
		Default is False.
	refresh_cache: whether to refresh the cache; if true, the dataset will
		be processed from scratch and the cache will be updated.
		Default is False.
"""
class PTBXLDataset(BaseDataset):
	"""
	Initialize the PTB-XL dataset.
	
	Attributes: 
		root (str): Root directory of the raw data.
		download (bool): True iff requested to download dataset.
			Default to False.
		dev (bool): True iff enable dev mode.
		downsampled (bool): True iff use downsampled signal data.
	"""
	def __init__(
		self,
		root: str = ".",
		download: bool = False,
		dev: bool = False,
		downsampled: bool = False,
		config_path: Optional[str] = None,
		**kwargs) -> None:

		self.dev = dev

		# Determine the root path, where most of the data is stored
		self.data_path: str = os.path.join(root, "ptb_xl_processed_final.zip")
		self.root = root

		# Determine signal path, where to fetch the signal samples
		signal_folder = "records100" if downsampled else "records500"

		if download:
			root_path = os.path.join(root, "ptb_xl_processed_final/\
				ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
		else:
			root_path = root

		self.signal_path: str = os.path.join(root_path, signal_folder)

		# Download the dataset from online source to root if needed
		self._download(download)

		# Determine config_path if it isn't provided
		if config_path is None:
			logger.info("No config path provided. Using default config.")
			config_path = os.path.join(
				os.path.dirname(__file__), "configs", "ptbxl.yaml")

		# Validate data
		self._validate()

		self._prepare_metadata()

		super().__init__(
			root=root,
			dataset_name="PTB-XL",
			tables=["ptb-xl"],
			config_path=config_path,
			**kwargs,
		)


	"""
	Download PTB-XL dataset from public google drive sources.
	It will contain both the original and downsampled versions,
	in /records500 and /records100 folder respectively.
	"""
	def _download(self, download) -> None:
		if download:
			zip_id = "1btbPiHEOUBLNLfUYkLnKzs50ZTmgqdI2"
			response = requests.get(
				f"https://drive.google.com/uc?export=download&id={zip_id}")
			with open(self.data_path, "wb") as file:
				file.write(response.content)

			with zipfile.ZipFile(self.data_path, "r") as z:
				z.extractall(self.root)

	"""
	Verifies if the dataset directory exists and its structure.
	Check if specified records folders exists underneath the root,
	each patient directory inside contains at least one pair of .dat and .hea files,
	and there"s no other unexpected type of files in the directory.

	Raises:
		FileNotFoundError: if any directory is not found.
		ValueError: if a patient directory contains not .dat/.hea file
			or if there"s a mismatch of .dat/.hea.
	"""
	def _validate(self) -> None:
		if not os.path.exists(self.root):
			e = f"Dataset root path doesn't exist: {self.root}"
			logger.error(e)
			raise FileNotFoundError(e)

		if not os.path.exists(self.signal_path):
			e = f"Dataset signal path doesn't exist: {self.signal_path}"
			logger.error(e)
			raise FileNotFoundError(e)

		dat = set()
		hea = set()
		for dirpath, dirnames, filenames in os.walk(self.signal_path):
			for filename in filenames:
				f, suffix = filename.split(".")
				if suffix == "dat":
					dat.add(f)
				elif suffix == "hea":
					hea.add(f)
				else:
					e = f"Unexpected file format {suffix} in the directory"
					logger.error(e)
					raise ValueError(e)

		if len(dat ^ hea) != 0:
			e = f".dat and .hea files mismatch for patient id {dat ^ hea}."
			logger.error(e)
			raise ValueError(e)

	"""
	Process and return a dictionary of the requested PTB-XL data for each patient.
	Each patient will have a corresponding object that contains
	load_from_path, patient_id, signal_file, label_file, and save_to_path.
	"""
	def _prepare_metadata(self) -> None:
		patients = {}

		for dirpath, dirnames, filenames in os.walk(self.signal_path):
			for filename in filenames:
				f = Path(filename).stem
				pid = f.split("_")[0]

				if pid not in patients:
					patients[pid] = {
						"load_from_path": dirpath,
						"patient_id": pid,
						"signal_file": f + ".dat",
						"label_file": f + ".hea",
						"save_to_path": os.path.join(
							MODULE_CACHE_PATH, hash_str(filename)),
					}

		if self.dev:
			keys = random.sample(list(patients), min(len(patients), 5))
			values = [patients[k] for k in keys]
			patients = dict(zip(keys, values))

		with open(os.path.join(self.root, "ptbxl.csv"), "w") as file:
			w = csv.DictWriter(file, fieldnames=[
				"load_from_path",
				"patient_id",
				"signal_file",
				"label_file",
				"save_to_path"])
			w.writeheader()
			w.writerows(list(patients.values()))
		return None

if __name__ == "__main__":
	dataset = PTBXLDataset(root="../../", download=False, downsampled=True, dev=False)
	dataset.stats()
	print(dataset.load_data())
