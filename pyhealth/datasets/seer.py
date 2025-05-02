"""
Developer: Elizabeth Amundsen (ecreigh2)
Paper: Reproducible Survival Prediction with SEER Cancer Data
Paper link: https://proceedings.mlr.press/v85/hegselmann18a/hegselmann18a.pdf
Description: Dataset for importing SEER (Surveillance, Epidemiology, and End Results) cancer incidences data from SEER*Stat program.
"""

import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import polars as pl

from .base_dataset import BaseDataset
from pyhealth.data import Event, Visit, Patient
from ..tasks import BaseTask
from .configs import load_yaml_config

logger = logging.getLogger(__name__)


class SEERIncidencesDataset(BaseDataset):
    """
    SEER Incidences dataset.    
    
    SEER is an authoritative source for cancer statistics in the United States.
    The Surveillance, Epidemiology, and End Results (SEER) Program provides information 
    on cancer statistics in an effort to reduce the cancer burden among the U.S. population. 
    See: https://seer.cancer.gov/

    This class is responsible for loading and managing data from the SEER Incidences dataset,
    which contains data on cancer occurances with tumor details, treatment information, and mortality.
    Dataset has only one table with dynamic attributes to account for the SEER database 
    being a live database with columns changing with each publication.    

    Attributes:
        root (str): The root directory where the dataset is stored.
        dataset_name (Optional[str]): The name of the dataset.
        PATIENT_ID_STR (str) : "Patient ID", the patient unique identifier
        TUMOR_ID_STR (str) : "Sequence Number", the tumor unique identifier
        global_event_df (pl.LazyFrame) : The SEER incidences raw data.
        dev (bool): Whether to enable dev mode (limit to 1000 patients).
        
    Examples:
        >>> from pyhealth.datasets import SEERIncidencesDataset
        >>> dataset = SEERIncidencesDataset(root="")
        >>> dataset.stat()
        
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = "seer",
        dev: bool = False,  # Added dev parameter
    ):
        """Initializes the SEERIncidencesDataset.

        Args:
            root (str): The root directory where dataset files are stored.
            dataset_name (Optional[str]): Name of the dataset. Defaults to class name.
            config_path (Optional[str]): Path to the configuration YAML file.
            dev (bool): Whether to run in dev mode (limits to 1000 patients).
        """
        self.root = root
        self.dataset_name = dataset_name or self.__class__.__name__
        self.config = None # .yaml configuration is not supported due to dynamic database
        self.dev = dev  # Store dev mode flag
        
        self.PATIENT_ID_STR = "Patient ID"
        self.TUMOR_ID_STR = "Sequence Number"

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        self.global_event_df = self.load_data()

        # Cached attributes
        self._collected_global_event_df = None
        self._unique_patient_ids = None

    @property
    def collected_global_event_df(self) -> pl.DataFrame:
        """Collects and returns the global event data frame.

        Returns:
            pl.DataFrame: The collected global event data frame.
        """
        if self._collected_global_event_df is None:
            logger.info("Collecting global event dataframe...")

            # Collect the dataframe - with dev mode limiting if applicable
            df = self.global_event_df
            # TODO: dev doesn't seem to improve the speed / memory usage
            if self.dev:
                # Limit the number of patients in dev mode
                logger.info("Dev mode enabled: limiting to 1000 patients")
                limited_patients = (
                    df.select(pl.col(self.PATIENT_ID_STR))
                    .unique()
                    .limit(1000)
                )
                df = df.join(limited_patients, on=self.PATIENT_ID_STR, how="inner")

            self._collected_global_event_df = df.collect()
            logger.info(f"Collected dataframe with shape: {self._collected_global_event_df.shape}")

        return self._collected_global_event_df

    def load_data(self) -> pl.LazyFrame:
        """Loads SEER incidences data from export.txt (SEER*Stat export file).
        
        The text file is expected to contain comma-separated data from SEER*Stat with headers.

        Returns:
            pl.LazyFrame: A lazy frame of raw data.
        """
        file_path = Path(self.root) / "export.txt"
        
        if not os.path.exists(file_path) :
            logger.error("ERROR:", file_path, "does not exist")
            raise FileNotFoundError(file_path, "does not exist")
        
        try :
            tmp_data = pl.scan_csv(file_path)            
            return tmp_data
        except Exception as e :
            logger.error("ERROR: could not load data from", file_path)
            raise RuntimeError("could not load data from", file_path)
            

    @property
    def unique_patient_ids(self) -> List[str]:
        """Returns a list of unique patient IDs.

        Returns:
            List[str]: List of unique patient IDs.
        """
        if self._unique_patient_ids is None:
            self._unique_patient_ids = (
                self.collected_global_event_df.select(self.PATIENT_ID_STR)
                .unique()
                .to_series()
                .to_list()
            )
            logger.info(f"Found {len(self._unique_patient_ids)} unique patient IDs")
        return self._unique_patient_ids

    def get_patient(self, patient_id: str) -> Patient:
        """Retrieves a Patient object for the given patient ID.

        Args:
            patient_id (str): The ID of the patient to retrieve.

        Returns:
            Patient: The Patient object for the given ID.

        Raises:
            AssertionError: If the patient ID is not found in the dataset.
        """
        assert (
            patient_id in self.unique_patient_ids
        ), f"Patient {patient_id} not found in dataset"
        df = self.collected_global_event_df.filter(
            pl.col(self.PATIENT_ID_STR) == patient_id
        )
        return Patient(patient_id=patient_id, data_source=df)

    def iter_patients(self, df: Optional[pl.LazyFrame] = None) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset.

        Yields:
            Iterator[Patient]: An iterator over Patient objects.
        """
        if df is None:
            df = self.collected_global_event_df
        grouped = df.group_by(self.PATIENT_ID_STR)

        for patient_id, patient_df in grouped:
            patient_id = patient_id[0]
            yield Patient(patient_id=patient_id, data_source=patient_df)

    def stats(self) -> None: 
        """Prints statistics about the dataset."""
        df = self.collected_global_event_df
        print(f"Dataset: {self.dataset_name}")
        print(f"Dev mode: {self.dev}")
        print(f"Number of patients: {df[self.PATIENT_ID_STR].n_unique()}")
        print(f"Number of events: {df.height}")

