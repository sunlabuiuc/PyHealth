import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset


class TUABDataset(BaseSignalDataset):
    """Base EEG dataset for the TUH Abnormal EEG Corpus

    Dataset is available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

    The TUAB dataset (or Temple University Hospital EEG Abnormal Corpus) is a collection of EEG data acquired at the Temple University Hospital. 
    
    The dataset contains both normal and abnormal EEG readings.

    Files are named in the form aaaaamye_s001_t000.edf. This includes the subject identifier ("aaaaamye"), the session number ("s001") and a token number ("t000"). EEGs are split into a series of files starting with *t000.edf, *t001.edf, ...

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "EEG_abnormal").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import TUABDataset
        >>> dataset = TUABDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process_EEG_data(self):
        # create a map for data sets for latter mapping patients
        data_map = {
            "train/abnormal": "0",
            "train/normal": "1",
            "eval/abnormal": "2",
            "eval/normal": "3",
        }

        data_map_reverse = {
            "0": "train/abnormal",
            "1": "train/normal",
            "2": "eval/abnormal",
            "3": "eval/normal",
        }

        # get all file names
        all_files = {}

        train_abnormal_files = os.listdir(os.path.join(self.root, "train/abnormal/01_tcp_ar"))
        all_files["train/abnormal"] = train_abnormal_files

        train_normal_files = os.listdir(os.path.join(self.root, "train/normal/01_tcp_ar"))
        all_files["train/normal"] = train_normal_files

        eval_abnormal_files = os.listdir(os.path.join(self.root, "eval/abnormal/01_tcp_ar"))
        all_files["eval/abnormal"] = eval_abnormal_files

        eval_normal_files = os.listdir(os.path.join(self.root, "eval/normal/01_tcp_ar"))
        all_files["eval/normal"] = eval_normal_files


        # get all patient ids
        patient_ids = []
        for field, sub_data in all_files.items():
            patient_ids.extend(["{}_{}".format(data_map[field], data.split("_")[0]) for data in sub_data])

        patient_ids = list(set(patient_ids))

        if self.dev:
            patient_ids = patient_ids[:20]

        # get patient to record maps
        #    - key: pid:
        #    - value: [{"load_from_path": None, "patient_id": None, "signal_file": None, "label_file": None, "save_to_path": None}, ...]
        patients = {
            pid: []
            for pid in patient_ids
        }
           
        for pid in patient_ids:
            data_field = data_map_reverse[pid.split("_")[0]]
            patient_visits = [file for file in all_files[data_field] if file.split("_")[0] == pid.split("_")[1]]
            
            for visit in patient_visits:
                patients[pid].append({
                    "load_from_path": os.path.join(self.root, data_field, "01_tcp_ar"),
                    "patient_id": pid,
                    "visit_id": visit.strip(".edf").strip(pid.split("_")[1])[1:],
                    "signal_file": visit,
                    "label_file": visit,
                    "save_to_path": self.filepath,
                })
        
        return patients


if __name__ == "__main__":
    dataset = TUABDataset(
        root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
