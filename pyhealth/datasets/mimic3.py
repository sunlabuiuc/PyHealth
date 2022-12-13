import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset
from pyhealth.datasets.utils import strptime


# TODO: add other tables


class MIMIC3Dataset(BaseDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, SUBJECT_ID.
        - ADMISSIONS: defines a patient's hospital admission, HADM_ID.

    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (NDC code)
            for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_ITEMID code)
            for patients

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
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")
        # load patients
        for p_id, p_info in tqdm(df_group, desc="Parsing PATIENTS and ADMISSIONS"):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses DIAGNOSES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to None.
        """
        table = "DIAGNOSES_ICD"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby(["SUBJECT_ID", "HADM_ID"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for code in v_info["ICD9_CODE"]:
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="ICD9CM",
                    visit_id=v_id,
                    patient_id=p_id,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
                table, so we set it to None.
        """
        table = "PROCEDURES_ICD"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby(["SUBJECT_ID", "HADM_ID"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for code in v_info["ICD9_CODE"]:
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="ICD9PROC",
                    visit_id=v_id,
                    patient_id=p_id,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby(["SUBJECT_ID", "HADM_ID"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["STARTDATE"], v_info["NDC"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="NDC",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby(["SUBJECT_ID", "HADM_ID"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["CHARTTIME"], v_info["ITEMID"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients


if __name__ == "__main__":
    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()

    # dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    #     refresh_cache=False,
    # )
    # print(dataset.stat())
    # print(dataset.available_tables)
    # print(list(dataset.patients.values())[4])
