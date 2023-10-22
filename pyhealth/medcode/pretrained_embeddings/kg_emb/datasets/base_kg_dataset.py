import logging
import os
from abc import ABC

from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from typing import Callable, Optional
from pyhealth.datasets.utils import MODULE_CACHE_PATH, hash_str
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import SampleKGDataset
from pyhealth.utils import load_pickle, save_pickle


logger = logging.getLogger(__name__)

INFO_MSG = """
dataset.triples:
    Array((<head_entity>, <relation>, <tail_entity>))
"""


class BaseKGDataset(ABC):
    """Abstract base Knowledge Graph class
    
    This abstract class defines a uniform

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
        refresh_cache: bool = False
    ):
        self.dataset_name = (
            self.__class__.__name__ if dataset_name is None else dataset_name
        )
        self.root = root
        self.dev = dev
        self.triples = []
        self.samples = []
        self.task_name = "Null"
        self.entity_num = 0
        self.relation_num = 0
        self.entity2id = None
        self.relation2id = None
        self.refresh_cache = refresh_cache

        # hash filename for cache
        args_to_hash = [self.dataset_name, root] + ["dev" if dev else "prod"]
        filename = hash_str("+".join([str(arg) for arg in args_to_hash]))
        self.filepath = os.path.join(MODULE_CACHE_PATH, filename)
        self.raw_graph_process()

        self.len = len(self.triples)


    def __str__(self):
        """Prints some information of the dataset."""
        return f"Base dataset {self.dataset_name}"

    def __len__(self):
        return self.len

    def raw_graph_process(self):
        """Process the raw graph to triples (a list of triple)
        """
        raise NotImplementedError

    @staticmethod
    def info():
        """Prints the output format."""
        print(INFO_MSG)

    def stat(self):
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of triples: {len(self.triples)}")
        lines.append(f"\t- Number of entities: {self.entity_num}")
        lines.append(f"\t- Number of relations: {self.relation_num}")
        lines.append(f"\t- Task name: {self.task_name}")
        lines.append(f"\t- Number of samples: {len(self.samples)}")
        lines.append("")
        print("\n".join(lines))
        return 

    
    def set_task(
        self,
        task_fn: Callable,
        task_name: Optional[str] = None,
        save: bool = True,
        **kwargs
    ) -> SampleKGDataset:
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

        """

        if task_name is None:
            self.task_name = task_fn.__name__

        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath + '.pkl') and (not self.refresh_cache):
            # load from cache
            print(
                f"Loading {self.dataset_name} base dataset from {self.filepath}.pkl"
            )
            self.samples = load_pickle(self.filepath + ".pkl")
        
        else:
            print(f"Processing {self.dataset_name} base dataset...")
            self.samples = task_fn(self.triples)

            # save to cache
            print(f"Saving {self.dataset_name} base dataset to {self.filepath}")
            if save == True:
                save_pickle(self.samples, self.filepath + ".pkl")

        sample_dataset = SampleKGDataset(
            samples=self.samples,
            dataset_name=self.dataset_name,
            task_name=self.task_name,
            dev=self.dev,
            entity_num=self.entity_num,
            relation_num=self.relation_num,
            entity2id=self.entity2id,
            relation2id=self.relation2id,
            **kwargs
        )

        return sample_dataset
