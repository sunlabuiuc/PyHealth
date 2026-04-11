import os

import numpy as np

from typing import Optional, List
from itertools import islice

from pyhealth.datasets import BaseSignalDataset


class CardiologyDataset(BaseSignalDataset):
    """Base ECG dataset for Cardiology

    Dataset is available at https://physionet.org/content/challenge-2020/1.0.2/

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
        chosen_dataset: a list of (0,1) of length 6 indicting which datasets will be used. Default: [1, 1, 1, 1, 1, 1]
            The datasets contain "cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ptb-xl", "st_petersburg_incart".
            eg. [0,1,1,1,1,1] indicates that "cpsc_2018_extra", "georgia", "ptb", "ptb-xl" and "st_petersburg_incart" will be used.

    Attributes:
        task: Optional[str], name of the task (e.g., "sleep staging").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> dataset = CardiologyDataset(
        ...         root="/srv/local/data/physionet.org/files/challenge-2020/1.0.2/training",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(self, root: str, chosen_dataset: List[int] = [1,1,1,1,1,1], dataset_name: Optional[str] = None, dev: bool = False, refresh_cache: bool = False):
        self.chosen_dataset = chosen_dataset

        super().__init__(dataset_name=dataset_name, root=root, dev=dev, refresh_cache=refresh_cache) 
        self.root = root
        self.dev = dev
        self.refresh_cache = refresh_cache

    def process_EEG_data(self):
        
        # get all file names depending on user-defined dataset
        dataset_lists = ["cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ptb-xl", "st_petersburg_incart"]

        all_files = []
        for idx in range(6):
            if self.chosen_dataset[idx] == 0:
                all_files.append([])
            else:
                dataset_root = os.path.join(self.root, dataset_lists[idx])
                dataset_samples = []
                for patient in range(len(os.listdir(dataset_root)) - 1): #exclude RECORDS 
                    patient_id = "g" + str(patient+1)
                    patient_root = os.path.join(dataset_root, patient_id)
                    dataset_samples.append([i.split(".")[0] for i in os.listdir(patient_root) if i != "RECORDS" and i != "index.html"])
                all_files.append(dataset_samples)  #[dataset:[patient:[sample1, sample2...]...]...]
        
        #print(all_files)
        # get all patient ids
        patient_ids = []
        for dataset_idx in range(len(all_files)):
            if all_files[dataset_idx] != []:
                for patient_idx in range(len(all_files[dataset_idx])):
                    cur_id = "{}_{}".format(dataset_idx, patient_idx)
                    patient_ids.append(cur_id)

        #print(patient_ids)
        
        if self.dev:
            patient_ids = patient_ids[:5]

        # get patient to record maps
        #    - key: pid:
        #    - value: [{"load_from_path": None, "signal_file": None, "label_file": None, "save_to_path": None}, ...]
        patients = {
            pid: []
            for pid in patient_ids
        }
        
        for dataset_idx in range(len(all_files)):
            if all_files[dataset_idx] != []:
                for patient_idx in range(len(all_files[dataset_idx])):
                    pid = "{}_{}".format(dataset_idx, patient_idx)
                    if pid in patient_ids:
                        for sample in all_files[dataset_idx][patient_idx]: 
                            patients[pid].append({
                                "load_from_path": os.path.join(self.root, dataset_lists[dataset_idx], "g{}".format(patient_idx+1)),
                                "patient_id": pid,
                                "signal_file": sample + ".mat",
                                "label_file": sample + ".hea",
                                "save_to_path": self.filepath,
                            })

                    
        return patients


if __name__ == "__main__":
    dataset = CardiologyDataset(
        root="/srv/local/data/physionet.org/files/challenge-2020/1.0.2/training",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    # the number of records for the first patient
    keys = list(dataset.patients.keys())
    print(len(dataset.patients[keys[0]]))
