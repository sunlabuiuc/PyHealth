import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset


class SleepEDFDataset(BaseSignalDataset):
    """Base EEG dataset for SleepEDF

    Dataset is available at https://www.physionet.org/content/sleep-edfx/1.0.0/

    For the Sleep Cassette Study portion:
        - The 153 SC* files (SC = Sleep Cassette) were obtained in a 1987-1991 study of age effects on sleep in healthy Caucasians aged 25-101, without any sleep-related medication [2]. Two PSGs of about 20 hours each were recorded during two subsequent day-night periods at the subjects homes. Subjects continued their normal activities but wore a modified Walkman-like cassette-tape recorder described in chapter VI.4 (page 92) of Bob's 1987 thesis [7].

        - Files are named in the form SC4ssNEO-PSG.edf where ss is the subject number, and N is the night. The first nights of subjects 36 and 52, and the second night of subject 13, were lost due to a failing cassette or laserdisk.

        - The EOG and EEG signals were each sampled at 100 Hz. The submental-EMG signal was electronically highpass filtered, rectified and low-pass filtered after which the resulting EMG envelope expressed in uV rms (root-mean-square) was sampled at 1Hz. Oro-nasal airflow, rectal body temperature and the event marker were also sampled at 1Hz.

        - Subjects and recordings are further described in the file headers, the descriptive spreadsheet SC-subjects.xls, and in [2].

    For the Sleep Telemetry portoin:
        - The 44 ST* files (ST = Sleep Telemetry) were obtained in a 1994 study of temazepam effects on sleep in 22 Caucasian males and females without other medication. Subjects had mild difficulty falling asleep but were otherwise healthy. The PSGs of about 9 hours were recorded in the hospital during two nights, one of which was after temazepam intake, and the other of which was after placebo intake. Subjects wore a miniature telemetry system with very good signal quality described in [8].

        - Files are named in the form ST7ssNJ0-PSG.edf where ss is the subject number, and N is the night.

        - EOG, EMG and EEG signals were sampled at 100 Hz, and the event marker at 1 Hz. The physical marker dimension ID+M-E relates to the fact that pressing the marker (M) button generated two-second deflections from a baseline value that either identifies the telemetry unit (ID = 1 or 2 if positive) or marks an error (E) in the telemetry link if negative. Subjects and recordings are further described in the file headers, the descriptive spreadsheet ST-subjects.xls, and in [1].

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
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
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> dataset = SleepEDFDataset(
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
        #    - value: [{"load_from_path": None, "signal_file": None, "label_file": None, "save_to_path": None}, ...]
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
    dataset = SleepEDFDataset(
        root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-telemetry",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
