import logging
import time
import os
from abc import ABC
from collections import Counter
from copy import deepcopy
from typing import Dict, Callable, Tuple, Union, List, Optional

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from pyhealth.data import Patient, Event
from pyhealth.datasets.sample_dataset import SampleEHRDataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH, DATASET_BASIC_TABLES
from pyhealth.datasets.utils import hash_str
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


class BaseEHRDataset(ABC):
    """Abstract base dataset class.

    This abstract class defines a uniform interface for all EHR datasets
    (e.g., MIMIC-III, MIMIC-IV, eICU, OMOP).

    Each specific dataset will be a subclass of this abstract class, which can then
    be converted to samples dataset for different tasks by calling `self.set_task()`.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]). Basic tables will be loaded by default.
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                - a str of the target code vocabulary. E.g., {"NDC", "ATC"}.
                - a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method. E.g.,
                    {"NDC", ("ATC", {"target_kwargs": {"level": 3}})}.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
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
        self.dataset_name = (
            self.__class__.__name__ if dataset_name is None else dataset_name
        )
        self.root = root

        self.code_mapping = code_mapping
        self.dev = dev

        # if we are using a premade dataset, no basic tables need to be provided.
        if self.dataset_name in DATASET_BASIC_TABLES and [
            table
            for table in tables
            if table in DATASET_BASIC_TABLES[self.dataset_name]
        ]:
            raise AttributeError(
                f"Basic tables are parsed by default and do not need to be explicitly selected. Basic tables for {self.dataset_name}: {DATASET_BASIC_TABLES[self.dataset_name]}"
            )

        self.tables = tables

        # the medcode vocabularies of the dataset
        self.code_vocs = {}
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
                f"Loaded {self.dataset_name} base dataset from {self.filepath}"
            )
            try:
                self.patients, self.code_vocs = load_pickle(self.filepath)
            except:
                raise ValueError("Please refresh your cache by set refresh_cache=True")
        
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
            save_pickle((self.patients, self.code_vocs), self.filepath)

    def _load_code_mapping_tools(self) -> Dict[str, CrossMap]:
        """Helper function which loads code mapping tools CrossMap for code mapping.

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

        This function will first call `self.parse_basic_info()` to parse the
        basic patient information, and then call `self.parse_[table_name]()` to
        parse the table with name `table_name`. Both `self.parse_basic_info()` and
        `self.parse_[table_name]()` should be implemented in the subclass.

        Returns:
           A dict mapping patient_id to `Patient` object.
        """
        pandarallel.initialize(progress_bar=False)

        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process basic information (e.g., patients and visits)
        tic = time.time()
        patients = self.parse_basic_info(patients)
        print(
            "finish basic patient information parsing : {}s".format(time.time() - tic)
        )
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                tic = time.time()
                patients = getattr(self, f"parse_{table.lower()}")(patients)
                print(f"finish parsing {table} : {time.time() - tic}s")
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    def _add_events_to_patient_dict(
        self,
        patient_dict: Dict[str, Patient],
        group_df: pd.DataFrame,
    ) -> Dict[str, Patient]:
        """Helper function which adds the events column of a df.groupby object to the patient dict.

        Will be called at the end of each `self.parse_[table_name]()` function.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            group_df: a df.groupby object, having two columns: patient_id and events.
                - the patient_id column is the index of the patient
                - the events column is a list of <Event> objects

        Returns:
            The updated patient dict.
        """
        for _, events in group_df.items():
            for event in events:
                patient_dict = self._add_event_to_patient_dict(patient_dict, event)
        return patient_dict

    @staticmethod
    def _add_event_to_patient_dict(
        patient_dict: Dict[str, Patient],
        event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict.

        Will be called in `self._add_events_to_patient_dict`.

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
        """Helper function which converts the codes for all patients.

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
                source_kwargs = kwargs.get("source_kwargs", {})
                target_kwargs = kwargs.get("target_kwargs", {})
            else:
                tgt_vocab = self.code_mapping[src_vocab]
                source_kwargs = {}
                target_kwargs = {}
            code_mapping_tool = self.code_mapping_tools[f"{src_vocab}_{tgt_vocab}"]
            mapped_code_list = code_mapping_tool.map(
                event.code, source_kwargs=source_kwargs, target_kwargs=target_kwargs
            )
            mapped_event_list = [deepcopy(event) for _ in range(len(mapped_code_list))]
            for i, mapped_event in enumerate(mapped_event_list):
                mapped_event.code = mapped_code_list[i]
                mapped_event.vocabulary = tgt_vocab
            
            # update the code vocs
            for key, value in self.code_vocs.items():
                if value == src_vocab:
                    self.code_vocs[key] = tgt_vocab

            return mapped_event_list
        # TODO: should normalize the code here
        return [event]

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

    def __str__(self):
        """Prints some information of the dataset."""
        return f"Base dataset {self.dataset_name}"

    def stat(self) -> str:
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
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
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)

    @staticmethod
    def info():
        """Prints the output format."""
        print(INFO_MSG)

    def set_task(
        self,
        task_fn: Callable,
        task_name: Optional[str] = None,
    ) -> SampleEHRDataset:
        """Processes the base dataset to generate the task-specific sample dataset.

        This function should be called by the user after the base dataset is
        initialized. It will iterate through all patients in the base dataset
        and call `task_fn` which should be implemented by the specific task.

        Args:
            task_fn: a function that takes a single patient and returns a
                list of samples (each sample is a dict with patient_id, visit_id,
                and other task-specific attributes as key). The samples will be
                concatenated to form the sample dataset.
            task_name: the name of the task. If None, the name of the task
                function will be used.

        Returns:
            sample_dataset: the task-specific sample dataset.

        Note:
            In `task_fn`, a patient may be converted to multiple samples, e.g.,
                a patient with three visits may be converted to three samples
                ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
                Patients can also be excluded from the task dataset by returning
                an empty list.
        """
        if task_name is None:
            task_name = task_fn.__name__
        samples = []
        for patient_id, patient in tqdm(
            self.patients.items(), desc=f"Generating samples for {task_name}"
        ):
            samples.extend(task_fn(patient))

        sample_dataset = SampleEHRDataset(
            samples=samples,
            code_vocs=self.code_vocs,
            dataset_name=self.dataset_name,
            task_name=task_name,
        )
        return sample_dataset
