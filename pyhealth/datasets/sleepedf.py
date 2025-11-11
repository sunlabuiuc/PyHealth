import logging
import os
from typing import Optional

import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.tasks.sleep_staging_v2 import SleepStagingSleepEDF

logger = logging.getLogger(__name__)


class SleepEDFDataset(BaseDataset):
    """Base EEG dataset for SleepEDF

    Dataset is available at https://www.physionet.org/content/sleep-edfx/1.0.0/

    For the Sleep Cassette Study portion:
        - The 153 SC* files (SC = Sleep Cassette) were obtained in a 1987-1991 study of age effects on sleep in healthy Caucasians aged 25-101, without any sleep-related medication [2]. Two PSGs of about 20 hours each were recorded during two subsequent day-night periods at the subjects homes. Subjects continued their normal activities but wore a modified Walkman-like cassette-tape recorder described in chapter VI.4 (page 92) of Bob's 1987 thesis.
        - Files are named in the form SC4ssNEO-PSG.edf where ss is the subject number, and N is the night. The first nights of subjects 36 and 52, and the second night of subject 13, were lost due to a failing cassette or laserdisk.
        - The EOG and EEG signals were each sampled at 100 Hz. The submental-EMG signal was electronically highpass filtered, rectified and low-pass filtered after which the resulting EMG envelope expressed in uV rms (root-mean-square) was sampled at 1Hz. Oro-nasal airflow, rectal body temperature and the event marker were also sampled at 1Hz.
        - Subjects and recordings are further described in the file headers, the descriptive spreadsheet SC-subjects.xls.

    For the Sleep Telemetry portion:
        - The 44 ST* files (ST = Sleep Telemetry) were obtained in a 1994 study of temazepam effects on sleep in 22 Caucasian males and females without other medication. Subjects had mild difficulty falling asleep but were otherwise healthy. The PSGs of about 9 hours were recorded in the hospital during two nights, one of which was after temazepam intake, and the other of which was after placebo intake. Subjects wore a miniature telemetry system with very good signal quality.
        - Files are named in the form ST7ssNJ0-PSG.edf where ss is the subject number, and N is the night.
        - EOG, EMG and EEG signals were sampled at 100 Hz, and the event marker at 1 Hz. The physical marker dimension ID+M-E relates to the fact that pressing the marker (M) button generated two-second deflections from a baseline value that either identifies the telemetry unit (ID = 1 or 2 if positive) or marks an error (E) in the telemetry link if negative. Subjects and recordings are further described in the file headers, the descriptive spreadsheet ST-subjects.xls.
    Args:
        root: str, root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dataset_name: Optional[str], name of the dataset. Default is None.
        config_path: Optional[str], path to the config file. Default is None.

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
        ...         root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "sleepedf_cassette.yaml"
            )
        if not os.path.exists(os.path.join(root, "sleepedf-metadata-pyhealth.csv")):
            self.prepare_metadata(root)
        default_tables = ["subjects"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "sleepedf",
            config_path=config_path,
        )

    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the SleepEDF dataset.
        Args:
            root: Root directory containing the dataset files.

        This method processes the raw metadata files and saves a processed CSV file.
        """

        sleep_edf_cassette = pd.read_excel(os.path.join(root, "SC-subjects.xls"))

        sleep_edf_cassette = sleep_edf_cassette.rename(columns={"sex (F=1)": "sex"})

        for files in os.listdir(os.path.join(root, "sleep-cassette")):
            if files.endswith("-PSG.edf"):
                subject_id = int(files[3:5])
                night = files[5]
                if subject_id in sleep_edf_cassette["subject"].values:
                    sleep_edf_cassette.loc[
                        (sleep_edf_cassette["subject"] == subject_id)
                        & (sleep_edf_cassette["night"] == int(night)),
                        "signal_file",
                    ] = os.path.join(root, "sleep-cassette", files)
            elif files.endswith("-Hypnogram.edf"):
                subject_id = int(files[3:5])
                night = files[5]
                if subject_id in sleep_edf_cassette["subject"].values:
                    sleep_edf_cassette.loc[
                        (sleep_edf_cassette["subject"] == subject_id)
                        & (sleep_edf_cassette["night"] == int(night)),
                        "label_file",
                    ] = os.path.join(root, "sleep-cassette", files)

        sleep_edf_cassette.to_csv(
            os.path.join(root, "sleepedf-metadata-pyhealth.csv"), index=False
        )

    @property
    def default_task(self) -> SleepStagingSleepEDF:
        """Returns the default task for this dataset.

        Returns:
            SleepStagingSleepEDF: The default task instance.
        """
        return SleepStagingSleepEDF()
