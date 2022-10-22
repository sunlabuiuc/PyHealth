import os
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Optional, List, Dict, Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from pyhealth import BASE_CACHE_PATH
from pyhealth.data import Patient, Event
from pyhealth.medcode import CrossMap
from pyhealth.utils import load_pickle, save_pickle, hash_str, create_directory

MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "datasets")
create_directory(MODULE_CACHE_PATH)


class BaseDataset(ABC, Dataset):
    """Abstract base dataset class.

    This abstract class defines a uniform interface for all datasets
        (e.g., MIMIC-III, MIMIC-IV, eICU, OMOP) and all tasks
        (e.g., mortality prediction, length of stay prediction).

    Each specific dataset will be a subclass of this abstract class, which can adapt
        to different tasks by calling set_task().

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded (e.g., ["DIAGNOSES_ICD",
            "PROCEDURES_ICD"]).
        code_mapping: Optional[Dict[str, str]], key is the source code vocabulary and
            value is the target code vocabulary (e.g., {"ICD9CM": "CCSCM"}).
            Default is empty dict, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: bool, whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.
    """

    def __init__(
        self,
        dataset_name: str,
        root: str,
        tables: List[str],
        code_mapping: Optional[Dict[str, str]] = None,
        dev: bool = False,
        refresh_cache: bool = False,
    ):
        """Loads tables into a dict of patients and saves it to cache."""

        if code_mapping is None:
            code_mapping = {}

        # base attributes
        self.dataset_name = dataset_name
        self.root = root
        self.tables = tables
        self.code_mapping = code_mapping
        self.dev = dev

        # task-specific attributes
        self.task: Optional[str] = None
        self.samples: Optional[List[Dict]] = None
        self.patient_to_index: Optional[Dict[str, List[int]]] = None
        self.visit_to_index: Optional[Dict[str, List[int]]] = None

        # load medcode for code mapping
        self.code_mapping_tools = self._load_code_mapping_tools()

        # hash filename for cache
        args_to_hash = (
            [dataset_name, root]
            + sorted(tables)
            + sorted(code_mapping.items())
            + ["dev" if dev else "prod"]
        )
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
        self.filepath = os.path.join(MODULE_CACHE_PATH, filename)

        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and (not refresh_cache):
            # load from cache
            print(f"Loaded {dataset_name} base dataset from {self.filepath}")
            self.patients = load_pickle(self.filepath)
        else:
            # load from raw data
            print(f"Processing {dataset_name} base dataset...")
            # parse tables
            patients = self._parse_tables()
            # convert codes
            patients = self._convert_code_in_patient_dict(patients)
            self.patients = patients
            # save to cache
            print(f"Saved {dataset_name} base dataset to {self.filepath}")
            save_pickle(self.patients, self.filepath)

    def _load_code_mapping_tools(self) -> Dict[str, CrossMap]:
        """Loads code mapping tools CrossMap for code mapping.

        Will be called in __init__().

        Returns:
            Dict[str, CrossMap], a dict whose key is the source and target code
                vocabulary and value is the CrossMap object.
        """
        code_mapping_tools = {}
        for source, target in self.code_mapping.items():
            # load code mapping from source to target
            code_mapping_tools[f"{source}_{target}"] = CrossMap(source, target)
        return code_mapping_tools

    @abstractmethod
    def _parse_tables(self) -> Dict[str, Patient]:
        """Parses the tables in self.tables and return a dict of patients.

        Will be called in __init__() if cache file does not exist or
            refresh_cache is True.

        Should be implemented by the specific dataset.

        Returns:
            Dict[str, Patient], a dict mapping patient_id to Patient object.
        """
        raise NotImplementedError

    @staticmethod
    def _strptime(s: str, format: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
        """Helper function which parses a string to datetime object.

        Will be called in _parse_tables().

        Args:
            s: str, string to be parsed.
            format: str, format of the string. Default is "%Y-%m-%d %H:%M:%S".

        Returns:
            Optional[datetime], parsed datetime object. If s is nan, return None.
        """
        # return None if s is nan
        if s != s:
            return None
        return datetime.strptime(s, format)

    @staticmethod
    def _add_event_to_patient_dict(
        patient_dict: Dict[str, Patient],
        event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict.

        Will be called in _parse_tables().

        Note that if the patient of the event is not in the patient dict, or the
            visit of the event is not in the patient, this function will do nothing.

        Args:
            patient_dict: Dict[str, Patient], a dict mapping patient_id to Patient
                object.
            event: Event, an event to be added to the patient dict.

        Returns:
            Dict[str, Patient], the updated patient dict.
        """
        patient_id = event.patient_id
        try:
            patient_dict[patient_id].add_event(event)
        except KeyError:
            pass
        return patient_dict

    def _convert_code_in_patient_dict(
        self,
        patients: Dict[str, Patient],
    ) -> Dict[str, Patient]:
        """Converts the codes for all patients in the patient dict.

        The codes to be converted are specified in self.code_mapping.

        Will be called in __init__() after _parse_tables().

        Args:
            patients: Dict[str, Patient], a dict mapping patient_id to Patient object.

        Returns:
            Dict[str, Patient], the updated patient dict.
        """
        for p_id, patient in tqdm(patients.items(), desc="Mapping codes"):
            patients[p_id] = self._convert_code_in_patient(patient)
        return patients

    def _convert_code_in_patient(self, patient: Patient) -> Patient:
        """Helper function which converts the codes for a single patient.

        Will be called in _convert_code_in_patient_dict().

        Args:
            patient: Patient, a Patient object.

        Returns:
            Patient, the updated patient object.
        """
        for visit in patient:
            for table in visit.available_tables:
                all_mapped_events = []
                for event in visit.get_event_list(table):
                    # an event may be mapped to multiple events after code conversion
                    mapped_events: List[Event]
                    mapped_events = self._convert_code_in_event(event)
                    all_mapped_events.extend(mapped_events)
                visit.set_event_list(table, all_mapped_events)
        return patient

    def _convert_code_in_event(self, event: Event) -> List[Event]:
        """Helper function which converts the code for a single event.

        Note that an event may be mapped to multiple events after code conversion.

        Will be called in _convert_code_in_patient().

        Args:
            event: Event, an Event object.

        Returns:
            List[Event], a list of Event objects after code conversion.
        """
        src_vocab = event.vocabulary
        if src_vocab in self.code_mapping:
            tgt_vocab = self.code_mapping[src_vocab]
            code_mapping_tool = self.code_mapping_tools[f"{src_vocab}_{tgt_vocab}"]
            mapped_code_list = code_mapping_tool.map(event.code)
            mapped_event_list = [deepcopy(event) for _ in range(len(mapped_code_list))]
            for i, mapped_event in enumerate(mapped_event_list):
                mapped_event.code = mapped_code_list[i]
                mapped_event.vocabulary = tgt_vocab
            return mapped_event_list
        # TODO: should normalize the code here
        return [event]

    def set_task(
        self,
        task_fn: Callable,
        task_name: Optional[str] = None,
    ) -> None:
        """Processes the base dataset to generate the task-specific samples.

        This function should be called by the user after the base dataset is
            initialized. It will iterate through all patients in the base dataset
            and call task_fn which should be implemented by the specific task.

        Args:
            task_fn: function, a function that takes a single patient and returns
                a list of samples (each sample is a dict with patient_id, visit_id,
                and other task-specific attributes as key). The samples will be
                concatenated to form the final samples of the task dataset.
            task_name: str, the name of the task. If None, the name of the task
                function will be used.

        Returns:
            samples: a list of samples, each sample is a dict with patient_id,
                visit_id, and other task-specific attributes as key.

        Note that in task_fn, a patient may be converted to multiple samples, e.g.,
            a patient with three visits may be converted to three samples ([visit 1],
            [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be
            excluded from the task dataset by returning an empty list.
        """
        if task_name is None:
            task_name = task_fn.__name__
        self.task = task_name
        self.task_fn = task_fn
        samples = []
        for patient_id, patient in tqdm(
            self.patients.items(), desc=f"Generating samples for {self.task}"
        ):
            samples.extend(self.task_fn(patient))
        self.samples = samples
        self.patient_to_index = self._index_patient()
        self.visit_to_index = self._index_visit()

    def _index_patient(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by patient_id.

        Will be called in set_task().

        Returns:
            patient_to_index: Dict[str, int], a dict mapping patient_id to a list
                of sample indices.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            patient_to_index.setdefault(sample["patient_id"], []).append(idx)
        return patient_to_index

    def _index_visit(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by visit_id.

        Will be called in set_task().

        Returns:
            visit_to_index: Dict[str, int], a dict mapping visit_id to a list
                of sample indices.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        visit_to_index = {}
        for idx, sample in enumerate(self.samples):
            visit_to_index.setdefault(sample["visit_id"], []).append(idx)
        return visit_to_index

    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the dataset.

        Returns:
            List[str], list of available tables.
        """
        tables = []
        for patient in self.patients.values():
            tables.extend(patient.available_tables)
        return list(set(tables))

    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List[str], list of available keys.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        keys = self.samples[0].keys()
        return list(keys)

    # TODO: check this
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
            # for multi-class classification
            if type(sample[key]) in [bool, int, str]:
                tokens.append(sample[key])
            # for multi-label classification
            elif type(sample[key][0]) in [bool, int, str]:
                tokens.extend(sample[key])
            elif type(sample[key][0]) == list:
                tokens.extend(sample[key][-1])
            else:
                raise ValueError(f"Unknown type of {key}: {type(sample[key])}")
        tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

    # TODO: check this
    def get_label_distribution(self) -> Dict[str, int]:
        """Gets the label distribution of the samples.

        Returns:
            label_distribution: a dict mapping label to count.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        label_distribution = {}
        for sample in self.samples:
            if type(sample["label"]) == list:
                for label in sample["label"]:
                    label_distribution.setdefault(label, 0)
                    label_distribution[label] += 1
            else:
                label_distribution.setdefault(sample["label"], 0)
                label_distribution[sample["label"]] += 1
        return label_distribution

    def __getitem__(self, index) -> Dict:
        """Returns a sample by index.

        Returns:
             Dict, a dict with patient_id, visit_id, and other task-specific
                attributes as key. Conversion to index/tensor will be done
                in the model.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        return self.samples[index]

    def __str__(self):
        """Prints some information of the dataset."""
        if self.task is None:
            return f"{self.dataset_name} base dataset"
        else:
            return f"{self.dataset_name} {self.task} dataset"

    def __len__(self):
        """Returns the number of samples in the dataset."""
        if self.task is None:
            raise ValueError("Please set task first.")
        return len(self.samples)

    def stat(self) -> None:
        """Prints some statistics of the dataset."""
        if self.task is None:
            self.base_stat()
        else:
            self.task_stat()

    def base_stat(self) -> None:
        """Prints some statistics of the base dataset."""
        print()
        print(f"Statistics of {self.dataset_name} dataset (dev={self.dev}):")
        print(f"\t- Number of patients: {len(self.patients)}")
        num_visits = [len(p) for p in self.patients.values()]
        print(f"\t- Number of visits: {sum(num_visits)}")
        print(
            f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        )
        for table in self.tables:
            num_events = [
                len(v.get_event_list(table)) for p in self.patients.values() for v in p
            ]
            print(
                f"\t- codes/visit in {table}: {sum(num_events) / len(num_events):.4f}"
            )
        print()

    # TODO: check this
    def task_stat(self) -> None:
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
            if key in ["patient_id", "visit_id"]:
                continue
            elif key == "label":
                print(f"\t- Label distribution: {self.get_label_distribution()}")
            else:
                # TODO: drugs[-1] is empty list
                num_events = [len(sample[key][-1]) for sample in self.samples]
                print(f"\t- #{key}/visit: {sum(num_events) / len(num_events):.4f}")
                print(f"\t- Number of unique {key}: {len(self.get_all_tokens(key))}")
        print()

    def info(self):
        """Prints the output format."""

        print(
            """
        dataset.patients: patient_id -> <Patient>
            
            <Patient>
                - visits: visit_id -> <Visit> 
                - other patient-level info.
        
                    <Visit>
                        - event_list_dict: table_name -> List[Event]
                        - other visit-level info.

                                <Event>
                                    - code: str
                                    - other event-level info.    
        """
        )
