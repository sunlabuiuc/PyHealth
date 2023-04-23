import logging
import os
from abc import ABC
from typing import Optional, Callable

import pandas as pd
from tqdm import tqdm

from pyhealth.datasets.sample_dataset import SampleImageCaptionDataset

logger = logging.getLogger(__name__)

INFO_MSG = """
dataset.patients:
    - key: patient id
    - value: a dict of image paths, caption, and other information
"""


class BaseImageCaptionDataset(ABC):
    """Abstract base Image Caption Generation dataset class.

    This abstract class defines a uniform interface for all
    image caption generation datasets.

    Each specific dataset will be a subclass of this abstract class, which can
    then be converted to samples dataset for different tasks by calling
    `self.set_task()`.

    Args:
        root: root directory of the raw data (should contain many csv files).
        dataset_name: name of the dataset. Default is the name of the class.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated.
            Default is False.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        dev: bool = False,
        refresh_cache: bool = False,
    ):
        # base attributes
        self.dataset_name = (
            self.__class__.__name__ if dataset_name is None else dataset_name
        )
        self.root = root
        # TODO: dev seems unnecessary for image and signal?
        self.dev = dev
        if dev:
            logger.warning("WARNING: dev has no effect \
                            for image caption generation datasets.")
        # TODO: refresh_cache seems unnecessary for image and signal?
        self.refresh_cache = refresh_cache
        if refresh_cache:
            logger.warning("WARNING: refresh_cache has no effect \
                           for image caption generation datasets.")

        self.metadata = pd.read_json(os.path.join(root,
                                                  "metadata.jsonl"), lines=True)
        if "patient_id" not in self.metadata.columns:
            # no patient_id in metadata, sequentially assign patient_id
            self.metadata["patient_id"] = self.metadata.index

        # group by patient_id
        self.patients = dict()
        for patient_id, group in self.metadata.groupby("patient_id"):
            self.patients[patient_id] = group.to_dict(orient="records")

        return

    def __len__(self):
        return len(self.patients)

    def __str__(self):
        """Prints some information of the dataset."""
        return f"Base dataset {self.dataset_name}"

    def stat(self) -> str:
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of images: {len(self)}")
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
    ) -> SampleImageCaptionDataset:
        """Processes the base dataset to generate the task-specific
            sample dataset.

        This function should be called by the user after the base dataset is
        initialized. It will iterate through all patients in the base dataset
        and call `task_fn` which should be implemented by the specific task.

        Args:
            task_fn: a function that takes a single patient and returns a
                list of samples (each sample is a dict with patient_id,
                image_path_list, caption and other task-specific attributes
                as key). The samples will be concatenated to form the
                sample dataset.
            task_name: the name of the task. If None, the name of the task
                function will be used.

        Returns:
            sample_dataset: the task-specific sample (Base) dataset.

        Note:
            In `task_fn`, a patient may have one or multiple images associated
                to a caption, for e.g. a patient can have single report
                for xrays taken from diffrent views that may be combined to
                have a single sample such as
                (
                    {'patient_id': 1,
                    'image_path_list': [frontal_img_path,lateral_img_path],
                    'caption': 'report_text'}
                )
                Patients can also be excluded from the task dataset by
                returning an empty list.
        """
        if task_name is None:
            task_name = task_fn.__name__

        # load from raw data
        logger.debug(f"Processing {self.dataset_name} base dataset...")

        samples = []
        for patient_id, patient in tqdm(
            self.patients.items(), desc=f"Generating samples for {task_name}"):
            samples.extend(task_fn(patient))

        sample_dataset = SampleImageCaptionDataset(
            samples,
            dataset_name=self.dataset_name,
            task_name=task_name,
        )
        return sample_dataset