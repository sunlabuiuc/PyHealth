import os
import sys
from typing import Optional, List, Dict

import pandas as pd

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")

from pyhealth.data import Event, Visit, Patient, BaseDataset
from tqdm import tqdm


# TODO: add cptevents
# TODO: add drgcodes
# TODO: add noteevents
# TODO: add microbiologyevents
# TODO: add procedureevents_mv


class MIMIC4Dataset(BaseDataset):
    """Base dataset for MIMIC-IV dataset.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU patients.
    The dataset is available at https://mimic.physionet.org/.

    We support the following tables:
        - patients.csv: defines each subject_id in the database, i.e. defines a single patient.
        - admission.csv: define a patient's hospital admission, hadm_id.
        - diagnoses_icd.csv: contains ICD diagnoses for patients, most notably ICD-9 diagnoses.
        - procedures_icd.csv: contains ICD procedures for patients, most notably ICD-9 procedures.
        - prescriptions.csv: contains medication related order entries, i.e. prescriptions.
        - labevents.csv: contains all laboratory measurements for a given patient, including out patient data.

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "procedures_icd"]).
        code_mapping: Optional[Dict[str, str]], key is the table name, value is the code vocabulary to map to
            (e.g., {"DIAGNOSES_ICD": "CCS"}). Note that the source vocabulary will be automatically
            inferred from the table. Default is None, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data). Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will be processed from scratch
            and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction"). Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with patient_id, visit_id, and
            other task-specific attributes as key. Default is None.
        patient_to_index: Optional[Dict[str, int]], a dict mapping patient_id to the index of the patient in
            self.samples. Default is None.
        visit_to_index: Optional[Dict[str, int]], a dict mapping visit_id to the index of the visit in
            self.samples. Default is None.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        code_mapping: Optional[Dict[str, str]] = {},
        dev=False,
        refresh_cache=False,
    ):
        super(MIMIC4Dataset, self).__init__(
            dataset_name="MIMIC-IV",
            root=root,
            tables=tables,
            code_mapping=code_mapping,
            dev=dev,
            refresh_cache=refresh_cache,
        )

    def parse_tables(self) -> Dict[str, Patient]:
        """This function overrides the parse_tables function in BaseDataset.

        It parses the corresponding tables and creates a dict of patients which will be cached later.

        Returns:
            patients: a dictionary of Patient objects indexed by patient_id
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

    def parse_patients_and_admissions(self, patients) -> Dict[str, Patient]:
        """function to parse PATIENTS and ADMISSIONS tables"""
        # read patient table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admission table
        admission_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admission_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # load patients
        for p_id, p_info in tqdm(
            df.groupby("subject_id"), desc="Parsing patients and admissions"
        ):
            patient = Patient(
                patient_id=p_id,
                # no birth datetime in MIMIC-IV, use anchor_year to replace
                birth_datetime=p_info["anchor_year"].values[0],
                death_datetime=p_info["dod"].values[0],
                # TODO: should categorize the gender
                gender=p_info["gender"].values[0],
                # no ethnicity in MIMIC-IV, use "unknown" to replace
                ethnicity="unknown",
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    # TODO: convert to datetime object
                    encounter_time=v_info["admittime"].values[0],
                    discharge_time=v_info["dischtime"].values[0],
                    # TODO: should categorize the discharge_status
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def parse_diagnoses_icd(self, patients) -> Dict[str, Patient]:
        """function to parse diagnosis_icd table.

        Note that MIMIC-III does not provide specific timestamps in diagnoses_icd table, so we set it to None.
        """
        table = "diagnoses_icd"
        col = "icd_code"
        vocabulary = "ICD9CM"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
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
        df = df.dropna(subset=["subject_id", "hadm_id", col])
        # sort by sequence number (i.e., disease priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # update patients
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["subject_id", "hadm_id"]), desc=f"Parsing {table}"
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
        """function to parse procedures_icd table.

        Note that MIMIC-III does not provide specific timestamps in procedures_icd table, so we set it to None.
        """
        table = "procedures_icd"
        col = "icd_code"
        vocabulary = "ICD9PROC"
        # read procedures table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
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
        df = df.dropna(subset=["subject_id", "hadm_id", "seq_num", col])
        # sort by sequence number (i.e., procedure priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # update patients and visits
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["subject_id", "hadm_id"]), desc=f"Parsing {table}"
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
        table = "prescriptions"
        col = "ndc"
        vocabulary = "NDC"
        # read prescriptions table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
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
        df = df.dropna(subset=["subject_id", "hadm_id", col])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # update patients and visits
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["subject_id", "hadm_id"]), desc=f"Parsing {table}"
        ):
            for timestamp, code in zip(v_info["starttime"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=timestamp,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_labevents(self, patients) -> Dict[str, Patient]:
        """function to parse labevents table."""
        table = "labevents"
        col = "itemid"
        vocabulary = "MIMIC4_ITEMID"
        # read labevents table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
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
        df = df.dropna(subset=["subject_id", "hadm_id", col])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # update patients and visits
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["subject_id", "hadm_id"]), desc=f"Parsing {table}"
        ):
            for timestamp, code in zip(v_info["charttime"], v_info[col]):
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
    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"prescriptions": "ATC"},
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
