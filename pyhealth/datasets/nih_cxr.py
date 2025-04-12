import os
import subprocess
import zipfile

from torch.utils.data import Dataset

INFO_MSG = """
The NIH Chest X-ray Dataset.
URL: https://www.kaggle.com/datasets/nih-chest-xrays/data

The dataset contains over 100,000 frontal-view X-ray images with 14 common thoracic disease labels.

Citation:
Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M.
"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases."
IEEE CVPR 2017, ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
doi: 10.1109/CVPR.2017.369
"""

NIH_DATASET_URL = 'https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data'
NIH_DATASET_FILE = 'nih_chestxray.zip'

class NIHChestXrayDataset(Dataset):
    _data_prepared = False

    def __init__(self, url: str, target_dir: str, target_file: str, dataset_type: str):
        """
        Initialize the NIH Chest X-ray Dataset.

        Args:
            url (str): The URL from which to download the dataset.
            target_dir (str): The directory where the dataset will be stored.
            target_file (str): The compressed dataset file name.
            dataset_type (str): Type of dataset, either 'training' or 'test'.
        """
        if dataset_type not in ('training', 'test'):
            raise ValueError("Unsupported dataset_type. Choose 'training' or 'test'.")
        self.url = url
        self.target_dir = target_dir
        self.target_file = target_file
        self.dataset_name = 'NIH Chest X-ray'
        self.dataset_type = dataset_type
        self.train_samples = 100
        self.test_samples = 20

        if not NIHChestXrayDataset._data_prepared:
            self._download()
            self._prepare()
            NIHChestXrayDataset._data_prepared = True

    def _download(self):
        """
        Download the dataset from the Internet.
        The download is skipped if the target file already exists.
        """
        os.makedirs(self.target_dir, exist_ok=True)
        file_path = os.path.join(self.target_dir, self.target_file)
        command = f"wget -nc -O {file_path} {self.url}"
        try:
            subprocess.run(command, check=True, shell=True)
            print("Downloaded successfully.")
        except subprocess.CalledProcessError:
            print("Download skipped or failed.")

    def _prepare(self):
        """
        Extract the downloaded dataset file.
        """
        source_file = os.path.join(self.target_dir, self.target_file)
        with zipfile.ZipFile(source_file, 'r') as file:
            print(f'Extracting files from {source_file} to {self.target_dir}')
            file.extractall(self.target_dir)

    def stat(self) -> str:
        """
        Returns and prints statistics of the dataset.
        """
        lines = []
        lines.append("")
        lines.append("Statistics of Dataset:")
        lines.append(f"\t- Dataset Name: {self.dataset_name}")
        lines.append(f"\t- Dataset Type: {self.dataset_type}")
        lines.append(f"\t- Number of samples: {len(self)}")
        lines.append("")
        stats_str = "\n".join(lines)
        print(stats_str)
        return stats_str

    @staticmethod
    def info():
        """
        Prints the dataset information.
        """
        print(INFO_MSG)

    def _load_data(self, index: int) -> bytes:
        """
        Load the image data at the specified index.

        Args:
            index (int): Sample index.

        Returns:
            bytes: The binary content of the image.
        """
        id_str = f"{index+1:05}"
        # Here we assume the extracted folder contains a subdirectory named "database"
        # with further "train" and "test" folders holding the JPEG images.
        if self.dataset_type == 'training':
            relative_path = os.path.join("train", f"image{id_str}.jpg")
        else:
            relative_path = os.path.join("test", f"image{id_str}.jpg")
        filepath = os.path.join(self.target_dir, "database", relative_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as f:
            img = f.read()
        return img

    def __len__(self):
        if self.dataset_type == 'training':
            return self.train_samples
        elif self.dataset_type == 'test':
            return self.test_samples

    def __getitem__(self, index: int) -> bytes:
        if self.dataset_type == 'training' and index >= self.train_samples:
            raise IndexError("Index out of range for training dataset.")
        if self.dataset_type == 'test' and index >= self.test_samples:
            raise IndexError("Index out of range for test dataset.")
        return self._load_data(index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Create dataset instances for training and test splits.
    train_dataset = NIHChestXrayDataset(
        NIH_DATASET_URL,
        '/tmp/nih_chestxray',
        NIH_DATASET_FILE,
        'training'
    )
    test_dataset = NIHChestXrayDataset(
        NIH_DATASET_URL,
        '/tmp/nih_chestxray',
        NIH_DATASET_FILE,
        'test'
    )

    # Display dataset information and statistics.
    train_dataset.info()
    train_dataset.stat()

    test_dataset.info()
    test_dataset.stat()

    # Create DataLoader generators.
    train_data_loader = DataLoader(train_dataset)
    test_data_loader = DataLoader(test_dataset)

    # Example: Load one sample from the training dataset.
    sample = train_dataset[0]
    print(f"Loaded sample of size: {len(sample)} bytes")
