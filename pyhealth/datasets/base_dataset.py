import logging
import os
from abc import ABC
from collections import Counter
from copy import deepcopy
from typing import Dict, Callable, Tuple, Union, List, Optional

from torch.utils.data import Dataset
from tqdm import tqdm

from pyhealth.data import Patient, Event
from pyhealth.datasets.utils import MODULE_CACHE_PATH
from pyhealth.datasets.utils import hash_str
from pyhealth.datasets.utils import list_nested_level, is_homo_list, flatten_list
from pyhealth.medcode import CrossMap
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)

INFO_MSG = """
dataset.patients: patient_id -> <Patient>

<Patient>
    - visits: visit_id -> <Visit> 
    - other patient-level info
    
    <Visit>
        - event_list_dict: table_name -> List[Event]
        - other visit-level info
    
        <Event>
            - code: str
            - other event-level info
"""


# TODO: parse_tables is too slow

class BaseDataset(ABC, Dataset):
    """Abstract base dataset class.

    This abstract class defines a uniform interface for all datasets
    (e.g., MIMIC-III, MIMIC-IV, eICU, OMOP) and all tasks (e.g., mortality
    prediction, length of stay prediction).

    Each specific dataset will be a subclass of this abstract class, which can adapt
    to different tasks by calling `self.set_task()`.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                - a str of the target code vocabulary;
                - a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
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
            root: str,
            tables: List[str],
            dataset_name: Optional[str] = None,
            code_mapping: Optional[Dict[str, Union[str, Tuple[str, Dict]]]] = None,
            dev: bool = False,
            refresh_cache: bool = False,
    ):
        """Loads tables into a dict of patients and saves it to cache."""

        if code_mapping is None:
            code_mapping = {}

        # base attributes
        self.dataset_name = self.__class__.__name__ if dataset_name is None else dataset_name
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
                [self.dataset_name, root]
                + sorted(tables)
                + sorted(code_mapping.items())
                + ["dev" if dev else "prod"]
        )
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
        self.filepath = os.path.join(MODULE_CACHE_PATH, filename)

        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and (not refresh_cache):
            # load from cache
            logger.debug(
                f"Loaded {self.dataset_name} base dataset from {self.filepath}")
            self.patients = load_pickle(self.filepath)
        else:
            # load from raw data
            logger.debug(f"Processing {self.dataset_name} base dataset...")
            # parse tables
            patients = self.parse_tables()
            # convert codes
            patients = self._convert_code_in_patient_dict(patients)
            self.patients = patients
            # save to cache
            logger.debug(f"Saved {self.dataset_name} base dataset to {self.filepath}")
            save_pickle(self.patients, self.filepath)

    def _load_code_mapping_tools(self) -> Dict[str, CrossMap]:
        """Loads code mapping tools CrossMap for code mapping.

        Will be called in `self.__init__()`.

        Returns:
            A dict whose key is the source and target code vocabulary and
                value is the `CrossMap` object.
        """
        code_mapping_tools = {}
        for s_vocab, target in self.code_mapping.items():
            if isinstance(target, tuple):
                assert len(target) == 2
                assert type(target[0]) == str
                assert type(target[1]) == dict
                assert target[1].keys() <= {"source_kwargs", "target_kwargs"}
                t_vocab = target[0]
            else:
                t_vocab = target
            # load code mapping from source to target
            code_mapping_tools[f"{s_vocab}_{t_vocab}"] = CrossMap(s_vocab, t_vocab)
        return code_mapping_tools

    def parse_tables(self) -> Dict[str, Patient]:
        """Parses the tables in `self.tables` and return a dict of patients.

        Will be called in `self.__init__()` if cache file does not exist or
            refresh_cache is True.

        Should be implemented by the specific dataset.

        Returns:
           A dict mapping patient_id to `Patient` object.
        """
        # process patients and admissions tables
        patients = self.parse_basic_info()
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                patients = getattr(self, f"parse_{table.lower()}")(patients)
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    @staticmethod
    def _add_event_to_patient_dict(
            patient_dict: Dict[str, Patient],
            event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict.

        Will be called in `self.parse_tables()`.

        Note that if the patient of the event is not in the patient dict, or the
        visit of the event is not in the patient, this function will do nothing.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            event: an event to be added to the patient dict.

        Returns:
            The updated patient dict.
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

        The codes to be converted are specified in `self.code_mapping`.

        Will be called in `self.__init__()` after `self.parse_tables()`.

        Args:
            patients: a dict mapping patient_id to `Patient` object.

        Returns:
            The updated patient dict.
        """
        for p_id, patient in tqdm(patients.items(), desc="Mapping codes"):
            patients[p_id] = self._convert_code_in_patient(patient)
        return patients

    def _convert_code_in_patient(self, patient: Patient) -> Patient:
        """Helper function which converts the codes for a single patient.

        Will be called in `self._convert_code_in_patient_dict()`.

        Args:
            patient:a `Patient` object.

        Returns:
            The updated `Patient` object.
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

        Will be called in `self._convert_code_in_patient()`.

        Args:
            event: an `Event` object.

        Returns:
            A list of `Event` objects after code conversion.
        """
        src_vocab = event.vocabulary
        if src_vocab in self.code_mapping:
            target = self.code_mapping[src_vocab]
            if isinstance(target, tuple):
                tgt_vocab, kwargs = target
                src_kwargs = kwargs.get("source_kwargs", {})
                tgt_kwargs = kwargs.get("target_kwargs", {})
            else:
                tgt_vocab = self.code_mapping[src_vocab]
                src_kwargs = {}
                tgt_kwargs = {}
            code_mapping_tool = self.code_mapping_tools[f"{src_vocab}_{tgt_vocab}"]
            mapped_code_list = code_mapping_tool.map(
                event.code, source_kwargs=src_kwargs, target_kwargs=tgt_kwargs
            )
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
        and call `task_fn` which should be implemented by the specific task.

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

        Note:
            In `task_fn`, a patient may be converted to multiple samples, e.g.,
                a patient with three visits may be converted to three samples
                ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
                Patients can also be excluded from the task dataset by returning
                an empty list.
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

        """
        Validate the samples.
        
        1. Check if all samples are of type dict.
        2. Check if all samples have the same keys.
        3. Check if "patient_id" and "visit_id" are in the keys.
        4. For each key, check if it is either:
            - a single value
            - a list of values of the sample type
            - a list of list of values of the same type
            
        Note that in check 4, we do not restrict the type of the values 
        to leave more flexibility for the user. But if the user wants to
        use some helper functions (e.g., `self.get_all_tokens()` and 
        `self.stat()`) in the dataset, we will further check the type of 
        the values.
        """
        assert all(isinstance(s, dict) for s in samples), "Each sample should be a dict"
        keys = samples[0].keys()
        assert all(set(s.keys()) == set(keys) for s in samples), \
            "All samples should have the same keys"
        assert "patient_id" in keys, "patient_id should be in the keys"
        assert "visit_id" in keys, "visit_id should be in the keys"
        # each feature has to be either a single value,
        # a list of values, or a list of list of values
        for key in keys:
            # check if all the samples have the same type of feature for the key
            check = is_homo_list([s[key] for s in samples])
            assert check, f"Key {key} has mixed types across samples"
            type_ = type(samples[0][key])

            # if key's feature is list
            if type_ == list:
                # All samples should either all be
                # (1) a list of values, i.e, 1 level nested list
                # (2) or a list of list of values, i.e., 2 level nested list
                levels = set([list_nested_level(s[key]) for s in samples])
                assert len(levels) == 1, \
                    f"Key {key} has mixed nested list levels across samples"
                level = levels.pop()
                assert level in [1, 2], \
                    f"Key {key} has unsupported nested list level across samples"
                # 1 level nested list
                if level == 1:
                    # a list of values of the same type
                    check = is_homo_list(flatten_list([s[key] for s in samples]))
                    assert check, \
                        f"Key {key} has mixed types in the nested list within samples"
                # 2 level nested list
                else:
                    # eliminate the case list [[1, 2], 3] where the
                    # nested level is 2 but some elements in the outer list
                    # are not list
                    check = [is_homo_list(s[key]) for s in samples]
                    assert all(check), \
                        f"Key {key} has mixed nested list levels within samples"
                    # a list of list of values of the same type
                    check = is_homo_list(
                        flatten_list([l for s in samples for l in s[key]])
                    )
                    assert check, \
                        f"Key {key} has mixed types in the nested list within samples"

        # set the samples
        self.samples = samples
        self.patient_to_index = self._index_patient()
        self.visit_to_index = self._index_visit()

    def _index_patient(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by patient_id.

        Will be called in `self.set_task()`.

        Returns:
            patient_to_index: a dict mapping patient_id to a list of sample indices.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            patient_to_index.setdefault(sample["patient_id"], []).append(idx)
        return patient_to_index

    def _index_visit(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by visit_id.

        Will be called in `self.set_task()`.

        Returns:
            visit_to_index: a dict mapping visit_id to a list of sample indices.
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
            List of available tables.
        """
        tables = []
        for patient in self.patients.values():
            tables.extend(patient.available_tables)
        return list(set(tables))

    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        keys = self.samples[0].keys()
        return list(keys)

    def get_all_tokens(
            self,
            key: str,
            remove_duplicates: bool = True,
            sort: bool = True
    ) -> List[str]:
        """Gets all tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.
            remove_duplicates: whether to remove duplicates. Default is True.
            sort: whether to sort the tokens by alphabet order. Default is True.

        Returns:
            tokens: a list of tokens.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        tokens = []
        for sample in self.samples:
            if type(sample[key]) == list:
                if len(sample[key]) == 0:
                    continue
                # a list of lists of values
                elif type(sample[key][0]) == list:
                    tokens.extend(flatten_list(sample[key]))
                # a list of values
                else:
                    tokens.extend(sample[key])
            # single value
            else:
                tokens.append(sample[key])
        types = set([type(t) for t in tokens])
        assert len(types) == 1, f"{key} tokens have mixed types"
        assert types.pop() in [int, float, str, bool], \
            f"{key} tokens have unsupported types"
        if remove_duplicates:
            tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

    def get_distribution_tokens(self, key: str) -> Dict[str, int]:
        """Gets the distribution of tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.

        Returns:
            distribution: a dict mapping token to count.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        tokens = self.get_all_tokens(key, remove_duplicates=False, sort=False)
        counter = Counter(tokens)
        return counter

    def __getitem__(self, index) -> Dict:
        """Returns a sample by index.

        Returns:
             A dict with patient_id, visit_id, and other task-specific
                attributes as key. Conversion to index/tensor will be done
                in the model.
        """
        if self.task is None:
            raise ValueError("Please set task first.")
        return self.samples[index]

    def __str__(self):
        if self.task is None:
            return f"{self.dataset_name} base dataset"
        else:
            return f"{self.dataset_name} {self.task} dataset"

    def __len__(self):
        """Returns the number of samples in the dataset."""
        if self.task is None:
            raise ValueError("Please set task first.")
        return len(self.samples)

    def stat(self) -> str:
        """Returns some statistics of the dataset."""
        if self.task is None:
            return self.base_stat()
        else:
            return self.task_stat()

    def base_stat(self) -> str:
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append(f"Statistics of {self.dataset_name} dataset (dev={self.dev}):")
        lines.append(f"\t- Number of patients: {len(self.patients)}")
        num_visits = [len(p) for p in self.patients.values()]
        lines.append(f"\t- Number of visits: {sum(num_visits)}")
        lines.append(
            f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        )
        for table in self.tables:
            num_events = [
                len(v.get_event_list(table)) for p in self.patients.values() for v in p
            ]
            lines.append(
                f"\t- Number of events per visit in {table}: "
                f"{sum(num_events) / len(num_events):.4f}"
            )
        print("\n".join(lines))
        return "\n".join(lines)

    def task_stat(self) -> str:
        """Returns some statistics of the task-specific dataset."""
        if self.task is None:
            raise ValueError("Please set task first.")
        lines = list()
        lines.append(f"Statistics of {self.task} task:")
        lines.append(f"\t- Dataset: {self.dataset_name} (dev={self.dev})")
        lines.append(f"\t- Number of samples: {len(self)}")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        lines.append(f"\t- Number of patients: {num_patients}")
        num_visits = len(set([sample["visit_id"] for sample in self.samples]))
        lines.append(f"\t- Number of visits: {num_visits}")
        lines.append(
            f"\t- Number of visits per patient: {len(self) / num_patients:.4f}"
        )
        for key in self.samples[0]:
            if key in ["patient_id", "visit_id"]:
                continue
            # key's feature is a list
            if type(self.samples[0][key]) == list:
                # check if the list also contains lists
                nested = [isinstance(e, list) for s in self.samples for e in s[key]]
                # key's feature is a list of lists
                if any(nested):
                    num_events = [
                        len(flatten_list(sample[key])) for sample in self.samples
                    ]
                # key's feature is a list of values
                else:
                    num_events = [len(sample[key]) for sample in self.samples]
            # key's feature is a single value
            else:
                num_events = [1 for sample in self.samples]
            lines.append(f"\t- {key}:")
            lines.append(f"\t\t- Number of {key} per sample: "
                         f"{sum(num_events) / len(num_events):.4f}")
            lines.append(
                f"\t\t- Number of unique {key}: {len(self.get_all_tokens(key))}"
            )
            distribution = self.get_distribution_tokens(key)
            top10 = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append(
                f"\t\t- Distribution of {key} (Top-10): {top10}")
        print("\n".join(lines))
        return "\n".join(lines)

    @staticmethod
    def info():
        """Prints the output format."""
        print(INFO_MSG)


class SampleDataset(ABC, Dataset):
    """Abstract sample dataset class.

    This dataset takes the processed data samples as an input list
    
    Args:
        samples: the processed data samples.

    Attributes:
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with:
            1. patient_id, 
            2. visit_id,
            3. other task-specific attributes as feature key,
            4. one label key
        
            E.g., 
                samples[0] = {
                    visit_id: 1,
                    patient_id: 1,
                    "codnition": ["A", "B"],
                    "procedure": ["C", "D"],
                    "label": 1
                }
                
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.
    """

    def __init__(self, samples, dataset_name="dataset"):
        self.dataset_name: str = dataset_name
        self.sanity_check(samples)
        self.samples: List[Dict] = samples
        self.patient_to_index: Optional[Dict[str, List[int]]] = self._index_patient()
        self.visit_to_index: Optional[Dict[str, List[int]]] = self._index_visit()

    def _index_patient(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by patient_id.

        Will be called in set_task().

        Returns:
            patient_to_index: Dict[str, int], a dict mapping patient_id to a list
                of sample indices.
        """
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
        visit_to_index = {}
        for idx, sample in enumerate(self.samples):
            visit_to_index.setdefault(sample["visit_id"], []).append(idx)
        return visit_to_index

    def get_all_tokens(
            self,
            key: str,
            remove_duplicates: bool = True,
            sort: bool = True
    ) -> List[str]:
        """Gets all tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.
            remove_duplicates: whether to remove duplicates. Default is True.
            sort: whether to sort the tokens by alphabet order. Default is True.

        Returns:
            tokens: a list of tokens.
        """
        tokens = []
        for sample in self.samples:
            if type(sample[key]) == list:
                if len(sample[key]) == 0:
                    continue
                # a list of lists of values
                elif type(sample[key][0]) == list:
                    tokens.extend(flatten_list(sample[key]))
                # a list of values
                else:
                    tokens.extend(sample[key])
            # single value
            else:
                tokens.append(sample[key])
        types = set([type(t) for t in tokens])
        assert len(types) == 1, f"{key} tokens have mixed types"
        assert types.pop() in [int, float, str, bool], \
            f"{key} tokens have unsupported types"
        if remove_duplicates:
            tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

    def sanity_check(self, samples):
        """
        Validate the samples.
        
        1. Check if all samples are of type dict.
        2. Check if all samples have the same keys.
        3. Check if "patient_id" and "visit_id" are in the keys.
        4. For each key, check if it is either:
            - a single value
            - a list of values of the sample type
            - a list of list of values of the same type
            
        Note that in check 4, we do not restrict the type of the values 
        to leave more flexibility for the user. But if the user wants to
        use some helper functions (e.g., `self.get_all_tokens()` and 
        `self.stat()`) in the dataset, we will further check the type of 
        the values.
        """
        assert all(isinstance(s, dict) for s in samples), "Each sample should be a dict"
        keys = samples[0].keys()
        assert all(set(s.keys()) == set(keys) for s in samples), \
            "All samples should have the same keys"
        assert "patient_id" in keys, "patient_id should be in the keys"
        assert "visit_id" in keys, "visit_id should be in the keys"
        # each feature has to be either a single value,
        # a list of values, or a list of list of values
        for key in keys:
            # check if all the samples have the same type of feature for the key
            check = is_homo_list([s[key] for s in samples])
            assert check, f"Key {key} has mixed types across samples"
            type_ = type(samples[0][key])

            # if key's feature is list
            if type_ == list:
                # All samples should either all be
                # (1) a list of values, i.e, 1 level nested list
                # (2) or a list of list of values, i.e., 2 level nested list
                levels = set([list_nested_level(s[key]) for s in samples])
                assert len(levels) == 1, \
                    f"Key {key} has mixed nested list levels across samples"
                level = levels.pop()
                assert level in [1, 2], \
                    f"Key {key} has unsupported nested list level across samples"
                # 1 level nested list
                if level == 1:
                    # a list of values of the same type
                    check = is_homo_list(flatten_list([s[key] for s in samples]))
                    assert check, \
                        f"Key {key} has mixed types in the nested list within samples"
                # 2 level nested list
                else:
                    # eliminate the case list [[1, 2], 3] where the
                    # nested level is 2 but some elements in the outer list
                    # are not list
                    check = [is_homo_list(s[key]) for s in samples]
                    assert all(check), \
                        f"Key {key} has mixed nested list levels within samples"
                    # a list of list of values of the same type
                    check = is_homo_list(
                        flatten_list([l for s in samples for l in s[key]])
                    )
                    assert check, \
                        f"Key {key} has mixed types in the nested list within samples"

    def get_distribution_tokens(self, key: str) -> Dict[str, int]:
        """Gets the distribution of tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.

        Returns:
            distribution: a dict mapping token to count.
        """

        tokens = self.get_all_tokens(key, remove_duplicates=False, sort=False)
        counter = Counter(tokens)
        return counter

    def __getitem__(self, index) -> Dict:
        """Returns a sample by index.

        Returns:
             Dict, a dict with patient_id, visit_id, and other task-specific
                attributes as key. Conversion to index/tensor will be done
                in the model.
        """
        return self.samples[index]

    def __str__(self):
        """Prints some information of the dataset."""
        return f"{self.dataset_name} base dataset"

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def stat(self) -> None:
        """Returns some statistics of the task-specific dataset."""
        lines = list()
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of samples: {len(self)}")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        lines.append(f"\t- Number of patients: {num_patients}")
        num_visits = len(set([sample["visit_id"] for sample in self.samples]))
        lines.append(f"\t- Number of visits: {num_visits}")
        lines.append(
            f"\t- Number of visits per patient: {len(self) / num_patients:.4f}"
        )
        for key in self.samples[0]:
            if key in ["patient_id", "visit_id"]:
                continue
            # key's feature is a list
            if type(self.samples[0][key]) == list:
                # check if the list also contains lists
                nested = [isinstance(e, list) for s in self.samples for e in s[key]]
                # key's feature is a list of lists
                if any(nested):
                    num_events = [
                        len(flatten_list(sample[key])) for sample in self.samples
                    ]
                # key's feature is a list of values
                else:
                    num_events = [len(sample[key]) for sample in self.samples]
            # key's feature is a single value
            else:
                num_events = [1 for sample in self.samples]
            lines.append(f"\t- {key}:")
            lines.append(f"\t\t- Number of {key} per sample: "
                         f"{sum(num_events) / len(num_events):.4f}")
            lines.append(
                f"\t\t- Number of unique {key}: {len(self.get_all_tokens(key))}"
            )
            distribution = self.get_distribution_tokens(key)
            top10 = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append(
                f"\t\t- Distribution of {key} (Top-10): {top10}")
        print("\n".join(lines))
        return "\n".join(lines)


if __name__ == "__main__":
    samples = [
        {"patient_id": "1", "visit_id": "1", "conditions": ["A", "B", "C"], "label": 0},
        {"patient_id": "1", "visit_id": "2", "conditions": ["A", "C", "D"], "label": 1}
    ]
    samples2 = [
        {'patient_id': 'patient-0',
         'visit_id': 'visit-0',
         'conditions': ['cond-33',
                        'cond-86',
                        'cond-80'],
         'procedures': ['prod-11',
                        'prod-8',
                        'prod-15',
                        'prod-66',
                        'prod-91',
                        'prod-94'],
         'label': 1},
        {'patient_id': 'patient-0',
         'visit_id': 'visit-0',
         'conditions': ['cond-33',
                        'cond-86',
                        'cond-80'],
         'procedures': ['prod-11',
                        'prod-8',
                        'prod-15',
                        'prod-66',
                        'prod-91',
                        'prod-94'],
         'label': 1}
    ]

    dataset = SampleDataset(
        samples=samples2,
        dataset_name="test")

    print(dataset.stat())
    data = iter(dataset)
    print(next(data))
    print(next(data))
