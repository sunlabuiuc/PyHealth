import os
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset


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
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded. Must be a subset of the
            following tables: DIAGNOSES_ICD, PROCEDURES_ICD, PRESCRIPTIONS, LABEVENTS.
        code_mapping: Optional[Dict[str, str]], key is the source code vocabulary and
            value is the target code vocabulary (e.g., {"ICD9CM": "CCSCM"}).
            Default is empty dict, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: bool, whether to refresh the cache; if true, the dataset will
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

    def __init__(
            self,
            root: str,
            tables: List[str],
            code_mapping: Optional[Dict[str, str]] = None,
            dev=False,
            refresh_cache=False,
    ):
        super(MIMIC3Dataset, self).__init__(
            dataset_name="MIMIC-III",
            root=root,
            tables=tables,
            code_mapping=code_mapping,
            dev=dev,
            refresh_cache=refresh_cache,
        )

    def _parse_tables(self) -> Dict[str, Patient]:
        """This function overrides the _parse_tables() function in BaseDataset.

        It parses the corresponding tables and creates a dict of patients which
            will be cached later.

        Returns:
            patients: a dictionary of Patient objects indexed by patient_id.
        """
        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process patients and admissions tables
        patients = self._parse_basic_info(patients)
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                patients = getattr(self, f"_parse_{table.lower()}")(patients)
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    def _parse_basic_info(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses PATIENTS and ADMISSIONS tables.

        Will be called in _parse_tables().

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/
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
                birth_datetime=self._strptime(p_info["DOB"].values[0]),
                death_datetime=self._strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=self._strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=self._strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def _parse_diagnoses_icd(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses DIAGNOSES_ICD table.

        Will be called in _parse_tables().

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Note that MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
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

    def _parse_procedures_icd(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses PROCEDURES_ICD table.

        Will be called in _parse_tables().

        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/

        Note that MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
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

    def _parse_prescriptions(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses PRESCRIPTIONS table.

        Will be called in _parse_tables().

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/
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
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_labevents(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses LABEVENTS table.

        Will be called in _parse_tables().

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/
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
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients


if __name__ == "__main__":
    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
        dev=False,
        code_mapping={"NDC": "ATC"},
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()

    # dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"ICD9CM": "CCSCM"},
    #     refresh_cache=False,
    # )
    # dataset.stat()
    # print(dataset.available_tables)
    # print(list(dataset.patients.values())[4])