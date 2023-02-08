import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset


class SleepEDFCassetteDataset(BaseSignalDataset):
    """Base EEG dataset for SleepEDF Cassette portion

    Dataset is available at https://www.physionet.org/content/sleep-edfx/1.0.0/

    For the Sleep Cassette Study portion:
        - The 153 SC* files (SC = Sleep Cassette) were obtained in a 1987-1991 study of age effects on sleep in healthy Caucasians aged 25-101, without any sleep-related medication [2]. Two PSGs of about 20 hours each were recorded during two subsequent day-night periods at the subjects homes. Subjects continued their normal activities but wore a modified Walkman-like cassette-tape recorder described in chapter VI.4 (page 92) of Bob's 1987 thesis [7].

        - Files are named in the form SC4ssNEO-PSG.edf where ss is the subject number, and N is the night. The first nights of subjects 36 and 52, and the second night of subject 13, were lost due to a failing cassette or laserdisk.

        - The EOG and EEG signals were each sampled at 100 Hz. The submental-EMG signal was electronically highpass filtered, rectified and low-pass filtered after which the resulting EMG envelope expressed in uV rms (root-mean-square) was sampled at 1Hz. Oro-nasal airflow, rectal body temperature and the event marker were also sampled at 1Hz.

        - Subjects and recordings are further described in the file headers, the descriptive spreadsheet SC-subjects.xls, and in [2].

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
        >>> from pyhealth.datasets import SleepEDFCassetteDataset
        >>> dataset = SleepEDFCassetteDataset(
        ...         root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process_EEG_data(self):

        # get all file names
        all_files = os.listdir(self.root)
        # get all patient ids
        patient_ids = np.unique([file[:6] for file in all_files])
        if self.dev:
            patient_ids = patient_ids[:5]
        # get patient to record maps
        #    - key: pid:
        #    - value: [(PSG.edf, Hypnogram.edf), ...]
        patients = {
            pid: [
                {
                    "load_from_path": self.root,
                    "signal_file": None,
                    "label_file": None,
                    "save_to_path": self.filepath,
                }
            ]
            for pid in patient_ids
        }
        for record in all_files:
            pid = record[:6]
            if pid in patient_ids:
                if "PSG" in record:
                    patients[pid][0]["signal_file"] = record
                elif "Hypnogram" in record:
                    patients[pid][0]["label_file"] = record
                else:
                    raise ValueError(f"Unknown record: {record}")
        return patients


if __name__ == "__main__":
    dataset = SleepEDFCassetteDataset(
        root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
