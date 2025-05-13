"""
Author: Alex Hu
NetID: alexxh2
Paper Title: Diagnosing Aortic Stenosis from Echocardiogram Videos via Multi-Instance Learning
Paper Link: https://proceedings.mlr.press/v219/huang23a/huang23a.pdf
Description:
This dataset class implements support for the TMED2 view_and_diagnosis_labeled_set
consisting of labeled echocardiogram still frames in .png format. It loads studies
based on folder names and patient_study mappings. It is designed for supervised
classification tasks, particularly aortic stenosis (AS) severity classification.
"""

import os
import logging
from typing import List, Optional, Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from pyhealth.datasets import BaseDataset
from pyhealth.datasets.utils import strptime

logger = logging.getLogger(__name__)


class TMED2Echo(BaseDataset):
    """TMED2 Echo Dataset for frame-level classification tasks."""

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = "tmed2_echo",
        config_path: Optional[str] = None,
        dev: bool = False,
        **kwargs,
    ):
        """
        Initializes the TMED2Echo dataset.

        Args:
            root (str): Path to the root folder containing TMED2 dataset.
            tables (List[str]): List containing "tmed2_echo".
            dataset_name (Optional[str]): Dataset identifier name.
            config_path (Optional[str]): YAML config file for metadata.
            dev (bool): If True, loads only a small subset.
        """
        super().__init__(root, tables, dataset_name, config_path, dev, **kwargs)

    def load_data(self):
        """
        Loads PNG image files and corresponding labels for each study.
        Expects a directory of folders named with study ID (e.g. 1234s1),
        each containing multiple PNG files. Labels are inferred from a
        CSV file containing query_key -> diagnosis_label.
        """
        image_root = os.path.join(self.root, "view_and_diagnosis_labeled_set", "structured", "labeled")
        label_csv_path = os.path.join(self.root, "labels_per_image.csv")

        assert os.path.exists(image_root), f"Directory not found: {image_root}"
        assert os.path.exists(label_csv_path), f"Missing label CSV: {label_csv_path}"

        labels_df = pd.read_csv(label_csv_path)
        examples = []

        for _, row in labels_df.iterrows():
            image_name = row["query_key"]
            label = row["diagnosis_label"]
            study_id = image_name.split("_")[0]

            image_path = os.path.join(image_root, study_id, image_name)
            if not os.path.exists(image_path):
                logger.warning(f"Skipping missing file: {image_path}")
                continue

            example = {
                "patient_id": study_id,
                "visit_id": image_name,
                "image_path": image_path,
                "label": label,
            }
            examples.append(example)

        self.examples = examples
        logger.info(f"Loaded {len(self.examples)} labeled examples from TMED2Echo dataset")

    def __getitem__(self, index: int):
        """
        Retrieves an image and label at the given index.

        Args:
            index (int): Index of the sample.

        Returns:
            Dict: A dictionary containing the image (PIL), label (str), and IDs.
        """
        sample = self.examples[index]
        image = Image.open(sample["image_path"]).convert("L")

        return {
            "patient_id": sample["patient_id"],
            "visit_id": sample["visit_id"],
            "image": image,
            "label": sample["label"],
        }

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Number of examples in the dataset.
        """
        return len(self.examples)


# Main usage example for testing the dataset class
if __name__ == "__main__":
    dataset = TMED2Echo(
        root="/path/to/tmed2",
        tables=["tmed2_echo"],
        config_path="/path/to/config.yaml",
        dev=True,
    )

    print("Loaded dataset with", len(dataset), "examples")
    print("First example:", dataset[0])