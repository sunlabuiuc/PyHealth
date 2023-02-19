from typing import Dict, Optional, Tuple, Union, Callable
import os
import logging

from abc import ABC
import pandas as pd
from pandarallel import pandarallel

from pyhealth.datasets.utils import hash_str, MODULE_CACHE_PATH
from pyhealth.datasets.sample_dataset import SampleSignalDataset
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)

INFO_MSG = """
dataset.patients:
    - key: patient id
    - value: recoding file paths
"""


class BaseSignalDataset(ABC):
    """Abstract base Signal dataset class.

    This abstract class defines a uniform interface for all EEG datasets
    (e.g., SleepEDF, SHHS).

    Each specific dataset will be a subclass of this abstract class, which can then
    be converted to samples dataset for different tasks by calling `self.set_task()`.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        dev: bool = False,
        refresh_cache: bool = False,
        **kwargs: Optional[Dict],
    ):

        # base attributes
        self.dataset_name = (
            self.__class__.__name__ if dataset_name is None else dataset_name
        )
        self.root = root
        self.dev = dev
        self.refresh_cache = refresh_cache

        # hash filename for cache
        args_to_hash = [self.dataset_name, root] + ["dev" if dev else "prod"]
        filename = hash_str("+".join([str(arg) for arg in args_to_hash]))
        self.filepath = os.path.join(MODULE_CACHE_PATH, filename)

        # for task-specific attributes
        self.kwargs = kwargs

        self.patients = self.process_EEG_data()

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
        num_records = [len(p) for p in self.patients.values()]
        lines.append(f"\t- Number of recodings: {sum(num_records)}")
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
    ) -> SampleSignalDataset:
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
            sample_dataset: the task-specific sample (Base) dataset.

        Note:
            In `task_fn`, a patient may be converted to multiple samples, e.g.,
                a patient with three visits may be converted to three samples
                ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
                Patients can also be excluded from the task dataset by returning
                an empty list.
        """
        if task_name is None:
            task_name = task_fn.__name__

        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and (not self.refresh_cache):
            """
            It obtains the signal samples (with path only) to ```self.filepath.pkl``` file
            """
            # load from cache
            logger.debug(
                f"Loaded {self.dataset_name} base dataset from {self.filepath}"
            )
            samples = load_pickle(self.filepath + ".pkl")
        else:
            """
            It stores the actual data and label to ```self.filepath/``` folder
            It also stores the signal samples (with path only) to ```self.filepath.pkl``` file
            """
            # load from raw data
            logger.debug(f"Processing {self.dataset_name} base dataset...")

            pandarallel.initialize(progress_bar=False)

            # transform dict to pandas dataframe
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)
            patients = pd.DataFrame(self.patients.items(), columns=["pid", "records"])
            patients.records = patients.records.parallel_apply(lambda x: task_fn(x))

            samples = []
            for _, records in patients.values:
                samples.extend(records)

            # save to cache
            logger.debug(f"Saved {self.dataset_name} base dataset to {self.filepath}")
            save_pickle(samples, self.filepath + ".pkl")

        sample_dataset = SampleSignalDataset(
            samples,
            dataset_name=self.dataset_name,
            task_name=task_name,
        )
        return sample_dataset
