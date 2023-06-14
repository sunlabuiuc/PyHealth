import os
from collections import Counter

import pandas as pd

from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.tasks.medical_transcriptions_classification import MedicalTranscriptionsClassification


class MedicalTranscriptionsDataset(BaseDataset):
    """Medical transcription data scraped from mtsamples.com

    Dataset is available at https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

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
        >>> dataset = MedicalTranscriptionsDataset(
                root="/srv/local/data/zw12/raw_data/MedicalTranscriptions",
            )
        >>> print(dataset[0])
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process(self):
        df = pd.read_csv(f"{self.root}/mtsamples.csv", index_col=0)

        # create patient dict
        patients = {}
        for index, row in df.iterrows():
            patients[index] = row.to_dict()
        return patients

    def stat(self):
        super().stat()
        print(f"Number of samples: {len(self.patients)}")
        count = Counter([v['medical_specialty'] for v in self.patients.values()])
        print(f"Number of classes: {len(count)}")
        print(f"Class distribution: {count}")

    @property
    def default_task(self):
        return MedicalTranscriptionsClassification()


if __name__ == "__main__":
    dataset = MedicalTranscriptionsDataset(
        root="/srv/local/data/zw12/raw_data/MedicalTranscriptions",
    )
    print(list(dataset.patients.items())[0])
    dataset.stat()
    samples = dataset.set_task()
    print(samples[0])
