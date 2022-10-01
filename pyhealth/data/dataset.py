from abc import ABC, abstractmethod
from typing import List, Optional

from torch.utils.data import Dataset

from pyhealth.data import Patient


class BaseDataset(ABC):
    """ Abstract dataset structure 
        will be inherited by specific datasets
    """

    def __init__(
            self,
            dataset_name: str,
            patients=List[Patient],
    ):
        self.dataset_name = dataset_name
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __str__(self):
        return f"Dataset {self.dataset_name}"

    def info(self):
        info = """
        ----- Output Data Structure -----
        Dataset.patients dict[str, Patient]
            - key: patient_id
            - value: <Patient> object
        
        <Patient>
            - patient_id: str
            - visits: dict[str, Visit]
                - key: visit_id
                - value: <Visit> object
        
        <Visit>
            - visit_id: str
            - patient_id: str
            - encounter_time: float
            - duration: float
            - mortality_status: bool = False,
            - conditions: List[Event] = [],
            - procedures: List[Event] = [],
            - drugs: List[Event] = [],
            - labs: List[Event] = [],
            - physicalExams: List[Event] = []

        <Event>
            - code: str
            - time: float
        """
        print (info)

class TaskDataset(ABC, Dataset):
    """ Abstract task dataset class which 
        will be inherited by specific tasks 
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        # a list of patients
        self.patients = None
        # from (0, N-1) to (patient_id, visit_id)
        self.index_map = None
        # [[index of patient 1], [index of patient 2], ...]
        self.index_group = None
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
