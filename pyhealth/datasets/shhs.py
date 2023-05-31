import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset


class SHHSDataset(BaseSignalDataset):
    """Base EEG dataset for Sleep Heart Health Study (SHHS)

    Dataset is available at https://sleepdata.org/datasets/shhs

    The Sleep Heart Health Study (SHHS) is a multi-center cohort study implemented by the National Heart Lung & Blood Institute to determine the cardiovascular and other consequences of sleep-disordered breathing. It tests whether sleep-related breathing is associated with an increased risk of coronary heart disease, stroke, all cause mortality, and hypertension.  In all, 6,441 men and women aged 40 years and older were enrolled between November 1, 1995 and January 31, 1998 to take part in SHHS Visit 1. During exam cycle 3 (January 2001- June 2003), a second polysomnogram (SHHS Visit 2) was obtained in 3,295 of the participants. CVD Outcomes data were monitored and adjudicated by parent cohorts between baseline and 2011. More than 130 manuscripts have been published investigating predictors and outcomes of sleep disorders.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

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
        >>> from pyhealth.datasets import SHHSDataset
        >>> dataset = SHHSDataset(
        ...         root="/srv/local/data/SHHS/",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def parse_patient_id(self, file_name):
        """
        Args:
            file_name: the file name of the shhs datasets. e.g., shhs1-200001.edf
        Returns:
            patient_id: the patient id of the shhs datasets. e.g., 200001
        """
        return file_name.split("-")[1].split(".")[0]

    def process_EEG_data(self):

        # get shhs1
        shhs1 = []
        if os.path.exists(os.path.join(self.root, "edfs/shhs1")):
            print("shhs1 exists and load shhs1")
            shhs1 = os.listdir(os.path.join(self.root, "edfs/shhs1"))
        else:
            print("shhs1 does not exist")

        # get shhs2
        shhs2 = []
        if os.path.exists(os.path.join(self.root, "edfs/shhs2")):
            print("shhs2 exists and load shhs2")
            shhs2 = os.listdir(os.path.join(self.root, "edfs/shhs2"))
        else:
            print("shhs2 does not exist")

        # get all patient ids
        patient_ids = np.unique([self.parse_patient_id(file) for file in shhs1 + shhs2])
        if self.dev:
            patient_ids = patient_ids[:5]
        # get patient to record maps
        #    - key: pid:
        #    - value: [{"load_from_path": None, "file": None, "save_to_path": None}, ...]
        patients = {pid: [] for pid in patient_ids}

        # parse shhs1
        for file in shhs1:
            pid = self.parse_patient_id(file)
            if pid in patient_ids:
                patients[pid].append(
                    {
                        "load_from_path": self.root,
                        "signal_file": os.path.join("edfs/shhs1", file),
                        "label_file": os.path.join("annotations-events-profusion/shhs1", f"shhs1-{pid}-profusion.xml"),
                        "save_to_path": os.path.join(self.filepath),
                    }
                )

        # parse shhs2
        for file in shhs2:
            pid = self.parse_patient_id(file)
            if pid in patient_ids:
                patients[pid].append(
                    {
                        "load_from_path": self.root,
                        "signal_file": os.path.join("edfs/shhs2", file),
                        "label_file": os.path.join("annotations-events-profusion/label", f"shhs2-{pid}-profusion.xml"),
                        "save_to_path": os.path.join(self.filepath),
                    }
                )
        return patients


if __name__ == "__main__":
    dataset = SHHSDataset(
        root="/srv/local/data/SHHS/polysomnography",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
