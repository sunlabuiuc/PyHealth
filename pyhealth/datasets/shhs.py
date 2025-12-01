import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset

from pyhealth.datasets.utils import read_edf_data, save_to_npz
from tqdm import tqdm

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

    def __init__(self, root, dev=False, refresh_cache=False, **kwargs):
        """Initialize SHHS Dataset"""
        super().__init__()
        self.root = root
        self.dev = dev
        self.refresh_cache = refresh_cache
        self.filepath = os.path.join(os.path.expanduser("~"), ".cache", "pyhealth_shhs")
        self.patients = self.process_EEG_data()

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

def process_ECG_data(self, out_dir, target_fs=None, select_chs=["ECG"]):
        """
        Extract SHHS ECG signals + labels and save them as .npz files.

        Args:
            out_dir: Destination directory for generated .npz files.
            target_fs: Optional int, target sampling rate (e.g., 100 Hz).
            select_chs: list of channels to extract, default ECG.

        Expected SHHS directory structure:
            root/
                edfs/shhs1/*.edf
                edfs/shhs2/*.edf
                annotations-events-profusion/shhs1/*.xml
                annotations-events-profusion/shhs2/*.xml
        """

        shhs_dirs = [
            os.path.join(self.root, "edfs", "shhs1"),
            os.path.join(self.root, "edfs", "shhs2"),
        ]

        os.makedirs(out_dir, exist_ok=True)

        for shhs_dir in shhs_dirs:
            if not os.path.exists(shhs_dir):
                print(f"Directory missing: {shhs_dir}")
                continue

            dir_label = os.path.basename(os.path.normpath(shhs_dir))
            files = [f for f in os.listdir(shhs_dir) if f.endswith(".edf")]

            print(f"Processing ECG for {dir_label}: {len(files)} EDF files found")

            for file in tqdm(files):
                sid = self.parse_patient_id(file)
                data_path = os.path.join(shhs_dir, file)

                # Label XML file
                label_path = os.path.join(
                    self.root,
                    "annotations-events-profusion",
                    dir_label,
                    f"{file.split('.')[0]}-profusion.xml",
                )

                if not os.path.exists(label_path):
                    print(f"Missing annotation for {sid}: {label_path}")
                    continue

                try:
                    data, fs, stages = read_edf_data(
                        data_path=data_path,
                        label_path=label_path,
                        dataset="SHHS",
                        select_chs=select_chs,
                        target_fs=target_fs,
                    )

                    outfile = os.path.join(out_dir, f"{dir_label}-{sid}.npz")
                    save_to_npz(outfile, data, stages, fs)

                except Exception as e:
                    print(f"Error processing patient {sid}: {e}")

        print("ECG extraction completed.")
        return True

if __name__ == "__main__":
    dataset = SHHSDataset(
        root="/srv/local/data/SHHS/polysomnography",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
