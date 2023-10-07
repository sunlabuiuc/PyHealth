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
"""


class BaseNoteDataset(ABC):
    
    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        dev: bool = False,
        refresh_cache: bool = False,
    ):
        
        """Loads tables into a dict of patients and saves it to cache."""

        # base attributes
        self.dataset_name = (
            self.__class__.__name__ if dataset_name is None else dataset_name
        )
        self.root = root

        self.dev = dev

        # hash filename for cache
        args_to_hash = (
            [self.dataset_name, root]
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
            self.patients = load_pickle(self.filepath)
        else:
            # load from raw data
            logger.debug(f"Processing {self.dataset_name} base dataset...")
            # parse tables
            patients = self.parse_tables()

            self.patients = patients
            # save to cache
            logger.debug(f"Saved {self.dataset_name} base dataset to {self.filepath}")
            save_pickle(self.patients, self.filepath)
            
            
            
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

        return patients
    

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
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)