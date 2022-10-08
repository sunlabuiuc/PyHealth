import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from pyhealth.medcode import CodeMap
from pyhealth import CACHE_PATH
from pyhealth.data import Patient
from pyhealth.utils import load_pickle, save_pickle, hash_str


class BaseDataset(ABC, Dataset):
    """Abstract base dataset class.

    This abstract class defines a uniform interface for all datasets (e.g., MIMIC-III, MIMIC-IV, eICU, OMOP)
    and all tasks (e.g., mortality prediction, length of stay prediction, etc.).

    Each specific dataset will be a subclass of this abstract class, which can adapt to different tasks by
    calling self.set_task().

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: Optional[Dict[str, str]], key is the table name, value is the code vocabulary to map to
            (e.g., {"DIAGNOSES_ICD": "CCS"}). Note that the source vocabulary will be automatically
            inferred from the table. Default is {}, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data). Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will be processed from scratch
            and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction"). Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with patient_id, visit_id, and
            other task-specific attributes as key. Default is None.
        patient_to_index: Optional[Dict[str, int]], a dict mapping patient_id to the index of the patient in
            self.samples. Default is None.
        visit_to_index: Optional[Dict[str, int]], a dict mapping visit_id to the index of the visit in
            self.samples. Default is None.
    """

    def __init__(
        self,
        dataset_name: str,
        root: str,
        tables: List[str],
        code_mapping: Optional[Dict[str, str]] = {},
        dev: bool = False,
        refresh_cache: bool = False,
    ):
        """Loads tables into a dict of patients and saves it to cache."""

        # base attributes
        self.dataset_name = dataset_name
        self.root = root
        self.tables = tables
        self.code_mapping = code_mapping
        self.dev = dev

        # task-specific attributes
        self.task: Optional[str] = None
        self.samples: Optional[List[Dict]] = None
        self.patient_to_index: Optional[Dict[str, int]] = None
        self.visit_to_index: Optional[Dict[str, int]] = None

        # cache
        if len(code_mapping) > 0:
            args_to_hash = (
                [dataset_name, root]
                + sorted(tables)
                + sorted(code_mapping.items())
                + [dev]
            )
        else:
            args_to_hash = [dataset_name, root] + sorted(tables) + [dev]
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
        self.filepath = os.path.join(CACHE_PATH, filename)

        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and not refresh_cache:
            print(f"Loaded {dataset_name} base dataset from {self.filepath}")
            self.patients = load_pickle(self.filepath)
        else:
            print(f"Processing {dataset_name} base dataset...")
            self.patients = self.parse_tables()
            print(f"Saved {dataset_name} base dataset to {self.filepath}")
            save_pickle(self.patients, self.filepath)

    @abstractmethod
    def parse_tables(self) -> Dict[str, Patient]:
        """Parses the tables in self.tables and return a dict of patients.

        Will be called in __init__ if cache file does not exist or refresh_cache is True.

        Should be implemented by the specific dataset.

        Returns:
            Dict[str, Patient], a dict mapping patient_id to Patient object.
        """
        raise NotImplementedError

    def map_code_in_table(
        self,
        df: pd.DataFrame,
        source_vocabulary: str,
        target_vocabulary: str,
        source_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Maps the codes in a table to a target vocabulary.

        Args:
            df: pandas DataFrame, the table to be mapped.
            source_vocabulary: str, the source vocabulary of the codes (e.g., "ICD9CM").
            target_vocabulary: str, the target vocabulary of the codes (e.g., "CCS").
            source_col: str name of the source column.
            target_col: str name of the target column.

        Returns:
            pandas DataFrame, the mapped table.
        """
        # TODO: save codemap to self
        codemap = CodeMap(source=source_vocabulary, target=target_vocabulary)
        df[target_col] = df[source_col].apply(codemap.map)
        # in case one code is mapped to multiple codes
        df = df.explode(target_col)
        return df

    def set_task(self, task, task_fn):
        """Processes the base dataset to generate the task-specific samples.

        This function will iterate through all patients in the base dataset and call task_fn which should be
        implemented by the specific task.

        Args:
            task: str, name of the task (e.g., "mortality prediction").
            task_fn: function, a function that takes a single patient and returns a list of
                samples (each sample is a dict with patient_id, visit_id, and other task-specific attributes
                as key). The samples will be concatenated to form the final samples of the task dataset.

        Returns:
            samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
                attributes as key.

        Note that in task_fn, a patient may be converted to multiple samples, e.g., a patient with three visits
        may be converted to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
        Patients can also be excluded from the task dataset by returning an empty list.
        """
        self.task = task
        self.task_fn = task_fn
        samples = []
        for patient_id, patient in tqdm(
            self.patients.items(), desc=f"Generating samples for {task}"
        ):
            samples.extend(self.task_fn(patient))
        self.samples = samples
        self.patient_to_index = self.index_patient()
        self.visit_to_index = self.index_visit()

    def index_patient(self) -> Dict[str, int]:
        """Index the samples by patient_id.

        This function will create a dict with patient_id as key and a list of sample indices as value.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            patient_to_index.setdefault(sample["patient_id"], []).append(idx)
        return patient_to_index

    def index_visit(self) -> Dict[str, int]:
        """Index the samples by visit_id.

        This function will create a dict with visit_id as key and a list of sample indices as value.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        visit_to_index = {}
        for idx, sample in enumerate(self.samples):
            visit_to_index.setdefault(sample["visit_id"], []).append(idx)
        return visit_to_index

    def get_all_tokens(self, key: str, sort: bool = True) -> List[str]:
        """Gets all tokens with a specific key in the samples.

        Args:
            key: str, the key of the tokens in the samples.
            sort: whether to sort the tokens by alphabet order.

        Returns:
            tokens: a list of tokens.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        tokens = []
        for sample in self.samples:
            tokens.extend(sample[key][-1])
        tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

    def __getitem__(self, index):
        """Returns a sample by index.

        Note that the returned sample is a dict with patient_id, visit_id, and other task-specific
        attributes as key. Conversion to index/tensor will be done in the model.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        return self.samples[index]

    def __str__(self):
        """Prints some information of the dataset."""
        if self.task is None:
            return f"Base {self.dataset_name} dataset"
        else:
            return f"{self.task} {self.dataset_name} dataset"

    def __len__(self):
        """Returns the number of samples in the dataset."""
        if self.task is None:
            raise ValueError("Please set task first.")
        return len(self.samples)

    def stat(self):
        """Prints some statistics of the dataset."""
        if self.task is None:
            self.base_stat()
        else:
            self.task_stat()

    def base_stat(self):
        """Prints some statistics of the base dataset."""
        print()
        print(f"Statistics of {self.dataset_name} dataset (dev={self.dev}):")
        print(f"\t- Number of patients: {len(self.patients)}")
        num_visits = [len(p) for p in self.patients.values()]
        print(f"\t- Number of visits: {sum(num_visits)}")
        print(
            f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        )

        for event_type in self.tables:
            num_events = [
                len(p[i].get_event_list(event_type))
                for p in self.patients.values()
                for i in range(len(p))
            ]
            print(
                f"\t- Number of {event_type} per visit: {sum(num_events) / len(num_events):.4f}"
            )
        print()

    def task_stat(self):
        """Prints some statistics of the task-specific dataset."""
        if self.task is None:
            raise ValueError("Please set task first.")
        print()
        print(f"Statistics of {self.task} task:")
        print(f"\t- Dataset: {self.dataset_name} (dev={self.dev})")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        print(f"\t- Number of patients: {num_patients}")
        print(f"\t- Number of visits: {len(self)}")
        print(f"\t- Number of visits per patient: {len(self) / num_patients:.4f}")
        # TODO: add more types once we support selecting domains with args
        for key in self.samples[0]:
            num_events = [len(sample[key][-1]) for sample in self.samples]
            print(
                f"\t- Number of {key} per visit: {sum(num_events) / len(num_events):.4f}"
            )
            print(f"\t- Number of unique {key}: {len(self.get_all_tokens(key))}")
        print()

    def info(self):
        """Prints the doc of the class."""
        print()
        print(self.__doc__)
        print()
