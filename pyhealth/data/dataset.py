import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from torch.utils.data import Dataset
from tqdm import tqdm

from pyhealth import CACHE_PATH
from pyhealth.data import Patient, Visit
from pyhealth.utils import load_pickle, save_pickle


class BaseDataset(ABC):
    """Abstract base dataset class.

    This abstract class defines a uniform interface for all datasets (e.g., MIMIC-III, MIMIC-IV, eICU, OMOP),
    such that all datasets can be used in the same way. Each specific dataset will be a subclass of this
    abstract class (e.g., MIMIC3BaseDataset, MIMIC4BaseDataset, eICUBaseDataset, OMOPBaseDataset).

    Only basic processing is done in this base class. All data are kept. No patient/visit/event is filtered out.
    More complex task-specific data cleaning should be done in the task-specific dataset class.

    Subclass should implement the following methods:
        process: process raw data and return patients and visits, which are two dicts with patient_id
            and visit_id as key.

    =====================================================================================
    Data structure:

        BaseDataset.patients: dict[str, Patient]
            - key: patient_id
            - value: <Patient> object

        BaseDataset.visits: dict[str, Visit]
            - key: visit_id
            - value: <Visit> object

        <Patient>
            - patient_id: str
            - visit_ids: list[str]
                - a list of visit_id
                - links to <Visit> objects in BaseDataset.visits
            - other attributes (e.g., birth_date, death_date, mortality_status, gender, ethnicity, etc.)

        <Visit>
            - visit_id: str
            - patient_id: str
                - links to <Patient> object in BaseDataset.patients
            - conditions, procedures, drugs, labs: list[Event]
                - a list of <Event> objects
            - other attributes (e.g., encounter_time, discharge_time, mortality_status, conditions, etc.)

        <Event>
            - code: str
            - domain: float
            - vocabulary: str
            - description: str
    =====================================================================================
    """

    def __init__(
            self,
            dataset_name: str,
            root: str,
            dev: bool = False,
            refresh_cache: bool = False
    ):
        """
        Will call self.process() to process raw data and return patients and visits, and save them to cache.

        Args:
            dataset_name: name of the dataset
            root: root directory of the raw data (should contain many csv files)
            dev: whether to enable dev mode (only use a small subset of the data)
            refresh_cache: whether to refresh the cache; if true, the dataset will be processed from scratch
                and the cache will be updated
        """
        self.dataset_name = dataset_name
        self.root = root
        self.dev = dev
        self.filename = f"{dataset_name.lower()}.data" if not dev else f"{dataset_name.lower()}_dev.data"
        self.filepath = os.path.join(CACHE_PATH, self.filename)
        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and not refresh_cache:
            self.patients, self.visits = self.load_from_cache()
        else:
            print(f"Processing {dataset_name} dataset...")
            self.patients, self.visits = self.process()
            self.save_to_cache()

    @abstractmethod
    def process(self):
        """Process the dataset.

        Will be called in __init__ if cache file does not exist or refresh_cache is True.

        Should be implemented by the specific dataset.
        """
        raise NotImplementedError

    def load_from_cache(self):
        """load data from cache.

        Returns:
            patients: dict[str, Patient], a dict of patients with patient_id as key
            visits: dict[str, Visit], a dict of visits with visit_id as key
        """
        print(f"Loaded data from {self.filepath}")
        return load_pickle(self.filepath)

    def save_to_cache(self):
        """Save data to cache.

        Save the following data structure to cache:
            self.patients: dict[str, Patient], a dict of patients with patient_id as key
            self.visits: dict[str, Visit], a dict of visits with visit_id as key
        """
        print(f"Saved data to {self.filepath}")
        save_pickle((self.patients, self.visits), self.filepath)

    def __len__(self):
        return len(self.patients)

    def __str__(self):
        return f"Dataset {self.dataset_name}"

    def stat(self):
        """Print some statistics of the dataset"""
        print()
        print(f"Statistics of {self.dataset_name} dataset (dev={self.dev}):")
        print(f"\t- Number of patients: {len(self.patients)}")
        print(f"\t- Number of visits: {len(self.visits)}")
        num_visits = [len(patient.visit_ids) for patient in self.patients.values()]
        print(f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}")
        for event_type in ["conditions", "procedures", "drugs", "labs"]:
            num_events = [len(getattr(visit, event_type)) for visit in self.visits.values()]
            print(f"\t- Number of {event_type} per visit: {sum(num_events) / len(num_events):.4f}")
        print()

    def info(self):
        """Print some information of the dataset"""
        print()
        print(self.__doc__)
        print()


class TaskDataset(ABC, Dataset):
    """Abstract task dataset class.

    This abstract class defines a uniform interface for all tasks (e.g., mortality prediction, length-of-stay
    estimation, drug recommendation). Each specific task will be a subclass of this abstract class.

    The task class takes a base dataset (e.g., MIMIC3BaseDataset, MIMIC4BaseDataset, eICUBaseDataset,
    OMOPBaseDataset) as input and processes the base dataset to generate the task-specific samples.

    Subclass should implement the following methods:
        process_single_patient: process a single patient and return a list of samples, which are dicts with
            patient_id, visit_id, and other task-specific attributes as key.

    =====================================================================================
    Data structure:

        TaskDataset.samples: List[dict[str, Any]]
            - a list of samples, each sample is a dict with patient_id, visit_id, and
                other task-specific attributes as key
    =====================================================================================
    """

    def __init__(self, task_name: str, base_dataset: BaseDataset):
        """Initialize the task dataset.

        Since the base dataset only performs basic processing and data cleaning, we will call self.process()
        to perform task-specific processing and return a list of samples (i.e., dict).

        Args:
            task_name: name of the task, e.g., mortality, length_of_stay, drug_recommendation
            base_dataset: a BaseDataset object
        """
        # TODO: maybe we should also support cache here
        self.task_name = task_name
        self.base_dataset = base_dataset
        self.samples = self.process()

    def process(self) -> List[Dict[str, Any]]:
        """Process the base dataset to generate the task-specific samples.

        This function will iterate through all patients in the base dataset and call self.process_single_patient().

        Returns:
            samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
                attributes as key
        """
        samples = []
        for patient_id, patient in tqdm(self.base_dataset.patients.items()):
            visits = {visit_id: self.base_dataset.visits[visit_id] for visit_id in patient.visit_ids}
            samples.extend(self.process_single_patient(patient, visits))
        return samples

    @abstractmethod
    def process_single_patient(self, patient: Patient, visits: Dict[str, Visit]) -> List[Dict[str, List]]:
        """Process a single patient.

        This function should be implemented by the specific task. It takes a patient and a dict of visits as input,
        and returns a list of samples, which are dicts with patient_id, visit_id, and other task-specific attributes
        as key. The samples will be concatenated to form the final samples of the task dataset.

        Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
        to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
        from the task dataset by returning an empty list.

        Args:
            patient: a Patient object
            visits: a dict of visits with visit_id as key

        Returns:
            samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
                attributes as key
        """
        raise NotImplementedError

    def get_all_tokens(self, domain: str, sort: bool = True) -> List[str]:
        """Get all tokens of a specific domain (e.g., conditions, procedures, drugs, labs) in the dataset.

        Args:
            domain: a string, one of "conditions", "procedures", "drugs", "labs"
            sort: whether to sort the tokens by alphabet order

        Returns:
            tokens: a list of tokens
        """
        tokens = []
        for sample in self.samples:
            tokens.extend(sample[domain][-1])
        tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return a sample by index.

        Note that the returned sample is a dict with patient_id, visit_id, and other task-specific attributes as key.
        Conversion to index/tensor will be done in the model.
        """
        return self.samples[index]

    def stat(self):
        """Print some statistics of the dataset"""
        print()
        print(f"Statistics of {self.task_name} task:")
        print(f"\t- Dataset: {self.base_dataset.dataset_name} (dev={self.base_dataset.dev})")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        print(f"\t- Number of patients: {num_patients}")
        print(f"\t- Number of visits: {len(self)}")
        print(f"\t- Number of visits per patient: {len(self) / num_patients:.4f}")
        # TODO: add more types once we support selecting domains with args
        for event_type in ["conditions", "procedures", "drugs"]:
            num_events = [len(sample[event_type][-1]) for sample in self.samples]
            print(f"\t- Number of {event_type} per visit: {sum(num_events) / len(num_events):.4f}")
            print(f"\t- Number of unique {event_type}: {len(self.get_all_tokens(event_type))}")
        print()

    def info(self):
        """Print some information of the dataset"""
        print()
        print(self.__doc__)
        print()
