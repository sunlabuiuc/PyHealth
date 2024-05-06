import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset


class TUEVDataset(BaseSignalDataset):
    """Base EEG dataset for the TUH EEG Events Corpus

    Dataset is available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

    This corpus is a subset of TUEG that contains annotations of EEG segments as one of six classes: (1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), (4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).

    Files are named in the form of bckg_032_a_.edf in the eval partition:
        bckg: this file contains background annotations.
		032: a reference to the eval index	
		a_.edf: EEG files are split into a series of files starting with a_.edf, a_1.ef, ... These represent pruned EEGs, so the  original EEG is split into these segments, and uninteresting parts of the original recording were deleted.
    or in the form of 00002275_00000001.edf in the train partition:
        00002275: a reference to the train index. 
		0000001: indicating that this is the first file inssociated with this patient. 

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "EEG_events").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> dataset = TUEVDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process_EEG_data(self):
        # get all file names
        all_files = {}

        train_files = os.listdir(os.path.join(self.root, "train/"))
        for id in train_files:
            if id != ".DS_Store":
                all_files["0_{}".format(id)] = [name for name in os.listdir(os.path.join(self.root, "train/", id)) if name.endswith(".edf")]

        eval_files = os.listdir(os.path.join(self.root, "eval/"))
        for id in eval_files:
            if id != ".DS_Store":
                all_files["1_{}".format(id)] = [name for name in os.listdir(os.path.join(self.root, "eval/", id)) if name.endswith(".edf")]

        # get all patient ids
        patient_ids = list(set(list(all_files.keys())))

        if self.dev:
            patient_ids = patient_ids[:20]
            # print(patient_ids)

        # get patient to record maps
        #    - key: pid:
        #    - value: [{"load_from_path": None, "patient_id": None, "signal_file": None, "label_file": None, "save_to_path": None}, ...]
        patients = {
            pid: []
            for pid in patient_ids
        }
           
        for pid in patient_ids:
            split = "train" if pid.split("_")[0] == "0" else "eval"
            id = pid.split("_")[1]

            patient_visits = all_files[pid]
            
            for visit in patient_visits:
                if split == "train":
                    visit_id = visit.strip(".edf").split("_")[1]
                else:
                    visit_id = visit.strip(".edf")
                    
                patients[pid].append({
                    "load_from_path": os.path.join(self.root, split, id),
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "signal_file": visit,
                    "label_file": visit,
                    "save_to_path": self.filepath,
                })
        
        return patients


if __name__ == "__main__":
    dataset = TUEVDataset(
        root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
