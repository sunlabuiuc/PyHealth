from abc import ABC, abstractmethod
from typing import List, Optional

from torch.utils.data import Dataset

from pyhealth.data import Patient


class BaseDataset(ABC):
    """ Abstract base dataset class which will be inherited by specific datasets """

    def __init__(
            self,
            dataset_name: str,
            patients=List[Patient],
            dataset_info: Optional[dict] = None
    ):
        self.dataset_name = dataset_name
        self.patients = patients
        self.dataset_info = dataset_info

    def __len__(self):
        return len(self.patients)

    def __str__(self):
        return f"Dataset {self.dataset_name}"


class TaskDataset(ABC, Dataset):
    """ Abstract task dataset class which will be inherited by specific tasks """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.preprocess()
        self.set_all_tokens()

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def set_all_tokens(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
