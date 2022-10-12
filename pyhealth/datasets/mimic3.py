import os
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset


# TODO: add cptevents
# TODO: add drgcodes
# TODO: add noteevents
# TODO: add microbiologyevents
# TODO: add procedureevents_mv


class MIMIC3Dataset(BaseDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    We support the following tables:
        - PATIENTS.csv: defines each SUBJECT_ID in the database,
            i.e. defines a single patient.
        - ADMISSIONS.csv: defines a patient's hospital admission, HADM_ID.
        - DIAGNOSES_ICD.csv: contains ICD diagnoses for patients, most notably
            ICD-9 diagnoses.
        - PROCEDURES_ICD.csv: contains ICD procedures for patients, most notably
            ICD-9 procedures.
        - PRESCRIPTIONS.csv: contains medication related order entries,
            i.e. prescriptions.
        - LABEVENTS.csv: contains all laboratory measurements for a given patient,
            including out patient data.

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded (e.g., ["DIAGNOSES_ICD",
            "PROCEDURES_ICD"]).
        code_mapping: Optional[Dict[str, str]], key is the table name, value is the
            code vocabulary to map to (e.g., {"DIAGNOSES_ICD": "CCS"}). Note that
            the source vocabulary will be automatically inferred from the table.
            Default is empty dict, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will be
            processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, int]], a dict mapping patient_id to the
            index of the patient in self.samples. Default is None.
        visit_to_index: Optional[Dict[str, int]], a dict mapping visit_id to the index
            of the visit in self.samples. Default is None.
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

    def parse_tables(self) -> Dict[str, Patient]:
        """This function overrides the parse_tables function in BaseDataset.

        It parses the corresponding tables and creates a dict of patients which will
        be cached later.

        Returns:
            patients: a dictionary of Patient objects indexed by patient_id.
        """
        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process patients and admissions tables
        patients = self.parse_patients_and_admissions(patients)
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                patients = getattr(self, f"parse_{table.lower()}")(patients)
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    @staticmethod
    def strptime(s: str) -> Optional[datetime]:
        """Parses a string to datetime object."""
        if pd.isna(s):
            return None
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    def parse_patients_and_admissions(self, patients) -> Dict[str, Patient]:
        """function to parse PATIENTS and ADMISSIONS tables"""
        # read patient table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admission table
        admission_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admission_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # load patients
        for p_id, p_info in tqdm(
                df.groupby("SUBJECT_ID"), desc="Parsing PATIENTS and ADMISSIONS"
        ):
            patient = Patient(
                patient_id=p_id,
                # TODO: convert to datetime object
                birth_datetime=p_info["DOB"].values[0],
                death_datetime=p_info["DOD_HOSP"].values[0],
                # TODO: should categorize the gender
                gender=p_info["GENDER"].values[0],
                # TODO: should categorize the ethnicity
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    # TODO: convert to datetime object
                    encounter_time=v_info["ADMITTIME"].values[0],
                    discharge_time=v_info["DISCHTIME"].values[0],
                    # TODO: should categorize the discharge_status
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def parse_diagnoses_icd(self, patients) -> Dict[str, Patient]:
        """function to parse DIAGNOSES_ICD table.

        Note that MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
        table, so we set it to None.
        """
        table = "DIAGNOSES_ICD"
        col = "ICD9_CODE"
        vocabulary = "ICD9CM"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # code mapping
        if table in self.code_mapping:
            df = self.map_code_in_table(
                df,
                source_vocabulary=vocabulary,
                target_vocabulary=self.code_mapping[table],
                source_col=col,
                target_col=self.code_mapping[table],
            )
            vocabulary = self.code_mapping[table]
            col = self.code_mapping[table]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", col])
        # sort by sequence number (i.e., disease priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # update patients
        for (p_id, v_id), v_info in tqdm(
                df.groupby(["SUBJECT_ID", "HADM_ID"]), desc=f"Parsing {table}"
        ):
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_procedures_icd(self, patients) -> Dict[str, Patient]:
        """function to parse PROCEDURES_ICD table.

        Note that MIMIC-III does not provide specific timestamps in PROCEDURES_ICD table, so we set it to None.
        """
        table = "PROCEDURES_ICD"
        col = "ICD9_CODE"
        vocabulary = "ICD9PROC"
        # read procedures table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # code mapping
        if table in self.code_mapping:
            df = self.map_code_in_table(
                df,
                source_vocabulary=vocabulary,
                target_vocabulary=self.code_mapping[table],
                source_col=col,
                target_col=self.code_mapping[table],
            )
            vocabulary = self.code_mapping[table]
            col = self.code_mapping[table]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", col])
        # sort by sequence number (i.e., procedure priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # update patients and visits
        for (p_id, v_id), v_info in tqdm(
                df.groupby(["SUBJECT_ID", "HADM_ID"]), desc=f"Parsing {table}"
        ):
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_prescriptions(self, patients) -> Dict[str, Patient]:
        """function to parse PRESCRIPTIONS table."""
        table = "PRESCRIPTIONS"
        col = "NDC"
        vocabulary = "NDC"
        # read prescriptions table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # code mapping
        if table in self.code_mapping:
            df = self.map_code_in_table(
                df,
                source_vocabulary=vocabulary,
                target_vocabulary=self.code_mapping[table],
                source_col=col,
                target_col=self.code_mapping[table],
            )
            vocabulary = self.code_mapping[table]
            col = self.code_mapping[table]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", col])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # update patients and visits
        for (p_id, v_id), v_info in tqdm(
                df.groupby(["SUBJECT_ID", "HADM_ID"]), desc=f"Parsing {table}"
        ):
            for timestamp, code in zip(v_info["STARTDATE"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_labevents(self, patients) -> Dict[str, Patient]:
        """function to parse LABEVENTS table."""
        table = "LABEVENTS"
        col = "ITEMID"
        vocabulary = "MIMIC3_ITEMID"
        # read labevents table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # code mapping
        if table in self.code_mapping:
            df = self.map_code_in_table(
                df,
                source_vocabulary=vocabulary,
                target_vocabulary=self.code_mapping[table],
                source_col=col,
                target_col=self.code_mapping[table],
            )
            vocabulary = self.code_mapping[table]
            col = self.code_mapping[table]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", col])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # update patients and visits-
        for (p_id, v_id), v_info in tqdm(
                df.groupby(["SUBJECT_ID", "HADM_ID"]), desc=f"Parsing {table}"
        ):
            for timestamp, code in zip(v_info["CHARTTIME"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients


if __name__ == "__main__":
    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"PRESCRIPTIONS": "ATC"},
        refresh_cache=False,
    )
    dataset.stat()
    dataset.info()
