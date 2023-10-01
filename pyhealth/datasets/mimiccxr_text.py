import os
from collections import Counter

import pandas as pd

from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.tasks.chest_xray_generation import ChestXrayGeneration


class MIMICCXRDataset(BaseDataset):
    """MIMIC-CXR data
    
    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        root: root directory of the raw data (should contain many csv files).
        dataset_name: name of the dataset. Default is the name of the class.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Examples:
        >>> dataset = MIMICCXRDataset(
                root="/home/xucao2/xucao/PIEMedApp/checkpoints/mimic_cxr",
            )
        >>> print(dataset[0])
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process(self):
        df = pd.read_csv(f"{self.root}/mimiccxr_text.csv", index_col=0)

        # create patient dict
        patients = {}
        for index, row in df.iterrows():
            patients[index] = row.to_dict()
        return patients

    def stat(self):
        super().stat()
        print(f"Number of samples: {len(self.patients)}")

    @property
    def default_task(self):
        return ChestXrayGeneration()


if __name__ == "__main__":
    dataset = MIMICCXRDataset(
        root="/home/xucao2/xucao/PIEMedApp/checkpoints/mimic_cxr",
    )
    print(list(dataset.patients.items())[0])
    dataset.stat()
    samples = dataset.set_task()
    print(samples[0])
    