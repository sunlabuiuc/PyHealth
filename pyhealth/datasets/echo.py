import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from pyhealth.datasets import BaseDataset
import os
DiagnosisStr_to_Int_Mapping = {
    'no_AS': 0,
    'mild_AS': 1,
    'mildtomod_AS': 1,
    'moderate_AS': 2,
    'severe_AS': 2
}

class EchoBagDataset(BaseDataset):
    """ Echocardiogram Aortic Stenosis Bagged Dataset for Multiple Instance Learning.

    This dataset consists of echocardiogram images grouped into 
    "bags" per patient study, with corresponding aortic stenosis severity labels.
    Each bag contains multiple grayscale images , and is labeled
    according to the severity of aortic stenosis diagnosed by clinical experts.

    Data is organized into images and a summary CSV containing patient study IDs
    and diagnosis labels. The dataset can be used for multiple instance learning
    (MIL) tasks such as study-level classification.

    Dataset is available at:
    https://tmed.cs.tufts.edu/data_access.html

    Args:
        root_dir: Root directory containing the image files and summary CSV.
        dataset_name: Name of the dataset. Defaults to "echocardiogram_as".
        config_path: Path to the dataset configuration YAML file. If None, uses default config.
        transform_fn: Optional image transformation function to apply to each image.
        sampling_strategy: Strategy for sampling images in each study ("first_frame", etc).
        training_seed: Random seed for any randomized operations.

    Attributes:
        root_dir: Directory containing the raw images and summary table.
        dataset_name: Name of the dataset.
        config_path: Path to the YAML configuration file.
        transform_fn: Image transformation function.
        summary_table: DataFrame containing patient study IDs and diagnosis labels.
        bag_of_PatientStudy_images: List of image bags
        bag_of_PatientStudy_DiagnosisLabels: Corresponding labels for each image bag.

    Examples:
        >>> from pyhealth.datasets import EchoBagDataset
        >>> dataset = EchoBagDataset(
        ...     root_dir="path/to/echobag",
        ...     config_path="path/to/echobag.yaml",
        ...     transform_fn=some_transform
        ... )
        >>> dataset.stats()
        >>> bag, label = dataset[0]
        >>> print(bag.shape, label)
    """

    def __init__(self, root_dir, summary_table, transform_fn=None, sampling_strategy="first_frame"):
        

        config_path = os.path.join(os.path.dirname(__file__), "configs", "echo.yaml")
        super().__init__(
            root=root_dir,
            tables=["echo_images"],
            dataset_name="EchoBagDataset",
            config_path=config_path,
        )
        self.root_dir = root_dir
        self.summary_table = summary_table  
        self.transform_fn = transform_fn
        self.sampling_strategy = sampling_strategy
        self.patient_studies = self.summary_table['patient_study'].unique()
        self.data, self.labels = self._create_bags()

    def _create_bags(self):
        data, labels = [], []
        for study in self.patient_studies:
            study_dir = os.path.join(self.root_dir)
            images = sorted([f for f in os.listdir(study_dir) if study in f and f.endswith(".png")])
            if len(images) == 0:
                continue

            bag_images = []
            for img_file in images:
                img_path = os.path.join(study_dir, img_file)
                img = np.array(Image.open(img_path).convert("RGB"))
                assert img.shape == (112, 112, 3), f"Image shape error: {img.shape}"
                bag_images.append(img)

            bag_images = np.array(bag_images)
            data.append(bag_images)

            label_str = self.summary_table[self.summary_table['patient_study'] == study]['diagnosis_label'].iloc[0]
            label = DiagnosisStr_to_Int_Mapping[label_str]
            labels.append(label)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag = self.data[idx]
        if self.transform_fn:
            bag = torch.stack([self.transform_fn(Image.fromarray(img)) for img in bag])
        label = self.labels[idx]
        return bag, label
