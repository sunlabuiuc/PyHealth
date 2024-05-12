import gzip
import os
import subprocess
import zipfile

from torch.utils.data import Dataset

INFO_MSG = """
The Automated Cardiac Diagnosis Challenge (ACDC) dataset.
URL: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

The dataset contains the NIfTI-1 cardiac cine-MRI data.

Citation:
O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation 
and Diagnosis: Is the Problem Solved ?"
in IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018
doi: 10.1109/TMI.2018.2837502
"""
ACDC_DATASET_URL='https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download'
ACDC_DATASET_FILE='acdc.gz'

class ACDCDataset(Dataset):
    _data_prepared = False

    def __init__(self, url: str, target_dir: str, target_file: str, dataset_type: str):
        if dataset_type not in ('training', 'test'):
            raise ValueError("Unsupported dataset_type.")
        self.url = url
        self.target_dir = target_dir
        self.target_file = target_file
        self.dataset_name = 'ACDC'
        self.dataset_type = dataset_type
        self.train_samples = 100
        self.test_samples = 50
        if not ACDCDataset._data_prepared:
            self._download()
            self._prepare()
            ACDCDataset._data_prepared = True

    def _download(self):
        """Download the dataset from Internet.

        Download will be skipped if the target file already exists.
        """
        os.makedirs(self.target_dir, exist_ok=True)
        command = f"wget -nc -O {self.target_dir}/{self.target_file} {self.url}"
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Downloaded successfully.")
        except subprocess.CalledProcessError:
            print(f"Skipped or failure in download.")

    def _prepare(self):
        """Unzip the downloaded dataset file."""
        source_file = os.path.join(self.target_dir, self.target_file)
        with zipfile.ZipFile(source_file, 'r') as file:
            print(f'Extract files from {source_file} to {self.target_dir}')
            file.extractall(self.target_dir)

    def stat(self) -> str:
        """Returns statistics of the dataset."""
        lines = list()
        lines.append("")
        lines.append("Statistics of Dataset:")
        lines.append(f"\t- Dataset Name: {self.dataset_name}")
        lines.append(f"\t- Dataset Type: {self.dataset_type}")
        lines.append(f"\t- Number of samples: {self.__len__()}")
        lines.append("")
        lines = "\n".join(lines)
        print(lines)
        return lines

    @staticmethod
    def info():
        """Prints the dataset information."""
        print(INFO_MSG)

    def _load_data(self, index):
        """Load the sample data at the index."""
        id_str = f"{index+1:03}"
        if self.dataset_type == 'training':
            filepath = f"training/patient{id_str}/patient{id_str}_4d.nii.gz"
        else:
            filepath = f"testing/patient{id_str}/patient{id_str}_4d.nii.gz"
        filepath = os.path.join(self.target_dir, "database", filepath)
        with gzip.open(filepath, 'rb') as f:
            img = f.read()
        return img

    def __len__(self):
        """Returns the total number of samples."""
        if self.dataset_type == 'training':
            return self.train_samples
        elif self.dataset_type == 'test':
            return self.test_samples

    def __getitem__(self, index):
        """Fetch a sample at index."""
        if self.dataset_type == 'train':
            if index >= train_samples:
                raise IndexError("list index out of range")

        if self.dataset_type == 'test':
            if index >= test_samples:
                raise IndexError("list index out of range")

        return self._load_data(index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = ACDCDataset(ACDC_DATASET_URL, '/tmp', ACDC_DATASET_FILE, 'training')
    test_dataset = ACDCDataset(ACDC_DATASET_URL, '/tmp', ACDC_DATASET_FILE, 'test')

    train_dataset.info()
    train_dataset.stat()

    test_dataset.info()
    test_dataset.stat()

    train_data_generator = DataLoader(train_dataset)
    test_data_generator = DataLoader(test_dataset)

