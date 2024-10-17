from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import os
import hashlib
from pyhealth.data.cache import read_msgpack, write_msgpack
from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.datasets.utils import hash_str
@dataclass
class TaskTemplate(ABC):
    task_name: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    dataset: BaseDataset = None
    cache_dir: str = "./cache"
    refresh_cache: bool = False
    samples: List[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.cache_path = self.get_cache_path()
        if os.path.exists(self.cache_path) and not self.refresh_cache:
            try:
                self.samples = read_msgpack(self.cache_path)
                print(f"Loaded {self.task_name} task data from {self.cache_path}")
                return
            except:
                print(f"Failed to load cache for {self.task_name}. Processing from scratch.")
        else:
            if self.dataset is None:
                raise ValueError("Dataset is required when cache doesn't exist or refresh_cache is True")
            self.samples = self.process()
            # Save to cache
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            write_msgpack(self.samples, self.cache_path)
            print(f"Saved {self.task_name} task data to {self.cache_path}")

    def get_cache_path(self) -> str:
        schema_str = f"input_{'-'.join(self.input_schema.values())}_output_{'-'.join(self.output_schema.values())}"
        hash_object = hash_str(schema_str)
        hash_num = int(hash_object, 16)
        short_hash = str(hash_num)[-10:]
        cache_filename = f"{self.task_name}_{short_hash}.msgpack"
        return os.path.join(self.cache_dir, cache_filename)

    @abstractmethod
    def process(self) -> List[Any]:
        raise NotImplementedError

    def to_torch_dataset(self) -> SampleDataset:
        dataset_name = self.dataset.dataset_name if self.dataset else "Unknown"
        return SampleDataset(
            self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name=dataset_name,
            task_name=self.task_name,
        )

    @classmethod
    def from_cache(cls, task_name: str, input_schema: Dict[str, str], output_schema: Dict[str, str], cache_dir: str = "./cache"):
        task = cls(task_name, input_schema, output_schema, dataset=None, cache_dir=cache_dir)
        if os.path.exists(task.cache_path):
            task.samples = read_msgpack(task.cache_path)
            print(f"Loaded {task.task_name} task data from {task.cache_path}")
        else:
            raise FileNotFoundError(f"Cache file not found for {task_name}")
        return task