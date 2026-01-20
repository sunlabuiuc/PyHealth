import logging
import warnings
import os
from typing import Optional, List, Dict, Tuple, Union
from typing import Dict, Iterator, List, Optional

import pandas as pd
from tqdm import tqdm
from datetime import datetime

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset
from pyhealth.datasets.utils import strptime, padyear

from .base_dataset import BaseDataset

import polars as pl

logger = logging.getLogger(__name__)


class eICUDataset(BaseDataset):
    """Base dataset for eICU dataset.

    The eICU dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://eicu-crd.mit.edu/.

    The basic information is stored in the following tables:
        - patient: defines a patient (uniquepid), a hospital admission
            (patienthealthsystemstayid), and a ICU stay (patientunitstayid)
            in the database.
        - hospital: contains information about a hospital (e.g., region).

    Note that in eICU, a patient can have multiple hospital admissions and each
    hospital admission can have multiple ICU stays. The data in eICU is centered
    around the ICU stay and all timestamps are relative to the ICU admission time.
    Thus, we only know the order of ICU stays within a hospital admission, but not
    the order of hospital admissions within a patient. As a result, we use `Patient`
    object to represent a hospital admission of a patient, and use `Visit` object to
    store the ICU stays within that hospital admission.

    We further support the following tables:
        - diagnosis: contains ICD diagnoses (ICD9CM and ICD10CM code)
            and diagnosis information (under attr_dict) for patients
        - treatment: contains treatment information (eICU_TREATMENTSTRING code)
            for patients.
        - medication: contains medication related order entries (eICU_DRUGNAME
            code) for patients.
        - lab: contains laboratory measurements (eICU_LABNAME code)
            for patients
        - physicalExam: contains all physical exam (eICU_PHYSICALEXAMPATH)
            conducted for patients.
        - admissionDx:  table contains the primary diagnosis for admission to
            the ICU per the APACHE scoring criteria. (eICU_ADMITDXPATH)

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> dataset = eICUDataset(
        ...         root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...         tables=["diagnosis", "medication", "lab", "treatment", "physicalExam", "admissionDx"],
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: str = "eicu",
        config_path: Optional[str] = None,
        **kwargs
    ):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "eicu.yaml")
            logger.info(f"Using default eICU config: {config_path}")

        #log_memory_usage(f"Before initializing {dataset_name}")
        default_tables = ["patient","hospital"]
        tables = tables + default_tables

        # store a mapping from visit_id to patient_id
        # will be used to parse clinical tables as they only contain visit_id
        self.visit_id_to_patient_id: Dict[str, str] = {}
        self.visit_id_to_encounter_time: Dict[str, datetime] = {}

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )
#        log_memory_usage(f"After initializing {dataset_name}")


    def iter_patients(self, df: Optional[pl.LazyFrame] = None) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset.

        Yields:
            Iterator[Patient]: An iterator over Patient objects.
        """
        if df is None:
            df = self.collected_global_event_df
        grouped = df.group_by("patient_id")

        for patient_id, patient_df in grouped:
            patient_id = patient_id[0]
            yield Patient(patient_id=patient_id, data_source=patient_df)

    def stats(self) -> None:
        """Prints statistics about the dataset."""
        df = self.collected_global_event_df
        print(f"Dataset: {self.dataset_name}")
        print(f"Dev mode: {self.dev}")
        print(f"Number of tables: {len(self.tables)}")
        for table in self.tables:
            print(f"Table: {table}")
 #       print(f"Number of patients: {df['patient'].n_unique()}")
        print(f"Number of events: {df.height}")

