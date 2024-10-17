import logging
import time
import os
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import Dict, Callable, Tuple, Union, List, Optional
# from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH, DATASET_BASIC_TABLES
from pyhealth.datasets.utils import hash_str
from pyhealth.medcode import CrossMap
from pyhealth.utils import load_pickle, save_pickle
from pyhealth.data.cache import read_msgpack, read_msgpack_patients, write_msgpack, write_msgpack_patients # better to use msgpack than pickle
logger = logging.getLogger(__name__)

INFO_MSG = """
dataset.patients: patient_id -> <Patient>

<Patient>
    - events: List[Event]
    - other patient-level info
    <Event>
        - code: str
        - other event-level info
"""


# TODO: parse_tables is too slow

# Let's add our twist, because we have to define some type of tables even if there aren't any. 
class BaseDataset(ABC):
    """Abstract base dataset class."""

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        tables : List[str] = None, 
        additional_dirs : Optional[Dict[str, str]] = {} , 
        dev : bool = False,
        refresh_cache : bool = False,
        **kwargs,
    ):
        if dataset_name is None:
            dataset_name = self.__class__.__name__
        self.root = root
        self.dataset_name = dataset_name
        logger.debug(f"Processing {self.dataset_name} base dataset...")
        self.tables = tables 
        self.tables_dir = {table: None for table in tables} # root
        if additional_dirs:
            self.tables.extend(additional_dirs.keys()) 
            self.tables_dir.update(additional_dirs)
        self.tables_dir = {table: None for table in tables}
        self.tables_dir.update(additional_dirs)
        
        self.dev = dev
       
        # TODO: cache -> problem: It can be dataset specific in the sense that they might have unique args. 
        # hash filename for cache
        self.filepath = self.get_cache_path()
        # we should use messagepack
        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and (not refresh_cache):
            # load from cache
            logger.debug(
                f"Loaded {self.dataset_name} base dataset from {self.filepath}"
            )
            try:
                self.patients = read_msgpack_patients(self.filepath)
            except:
                raise ValueError("Please refresh your cache by set refresh_cache=True")
        
        else:
            # load from raw data
            logger.debug(f"Processing {self.dataset_name} base dataset...")
            # parse tables
            self.patients = self.process()
            # save to cache
            logger.debug(f"Saved {self.dataset_name} base dataset to {self.filepath}")
            write_msgpack_patients(self.patients, self.filepath)

        # return

    def __str__(self):
        return f"Base dataset {self.dataset_name}"

    def __len__(self):
        return len(self.patients)

    # Essentially, every dataset should have both a unique cache path and process method.
    @abstractmethod 
    def get_cache_path(self) -> str:
        args_to_hash = (
            [self.dataset_name, self.root]
            + sorted(self.tables)
            # + sorted(self.code_mapping.items())
            + ["dev" if self.dev else "prod"]
            + sorted([(k, v) for k, v in self.tables_dir.items()])
        )
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".msgpack"
        return filename
        # raise NotImplementedError

    @abstractmethod
    def process(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def stat(self):
        print(f"Statistics of {self.dataset_name}:")
        return

    # @property
    # def default_task(self) -> Optional[TaskTemplate]:
    #     return None

    # def set_task(self, task: Optional[TaskTemplate] = None) -> SampleDataset:
    #     """Processes the base dataset to generate the task-specific sample dataset.
    #     """
    #     # TODO: cache?
    #     if task is None:
    #         # assert default tasks exist in attr
    #         assert self.default_task is not None, "No default tasks found"
    #         task = self.default_task

    #     # load from raw data
    #     logger.debug(f"Setting task for {self.dataset_name} base dataset...")

    #     samples = []
    #     for patient_id, patient in tqdm(
    #         self.patients.items(), desc=f"Generating samples for {task.task_name}"
    #     ):
    #         samples.extend(task(patient))
       
    #     sample_dataset = SampleDataset(
    #         samples,
    #         input_schema=task.input_schema,
    #         output_schema=task.output_schema,
    #         dataset_name=self.dataset_name,
    #         task_name=task,
    #     )
    #     return sample_dataset

