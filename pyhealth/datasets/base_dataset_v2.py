import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict

from tqdm import tqdm

from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.tasks.task_template import TaskTemplate

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base dataset class."""

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        if dataset_name is None:
            dataset_name = self.__class__.__name__
        self.root = root
        self.dataset_name = dataset_name
        logger.debug(f"Processing {self.dataset_name} base dataset...")
        self.patients = self.process()
        # TODO: cache
        return

    def __str__(self):
        return f"Base dataset {self.dataset_name}"

    def __len__(self):
        return len(self.patients)

    @abstractmethod
    def process(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def stat(self):
        print(f"Statistics of {self.dataset_name}:")
        return

    @property
    def default_task(self) -> Optional[TaskTemplate]:
        return None

    def set_task(self, task: Optional[TaskTemplate] = None) -> SampleDataset:
        """Processes the base dataset to generate the task-specific sample dataset.
        """
        # TODO: cache?
        if task is None:
            # assert default tasks exist in attr
            assert self.default_task is not None, "No default tasks found"
            task = self.default_task

        # load from raw data
        logger.debug(f"Setting task for {self.dataset_name} base dataset...")

        samples = []
        for patient_id, patient in tqdm(
            self.patients.items(), desc=f"Generating samples for {task.task_name}"
        ):
            samples.extend(task(patient))
        sample_dataset = SampleDataset(
            samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task,
        )
        return sample_dataset
