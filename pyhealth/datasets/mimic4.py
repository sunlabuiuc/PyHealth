import os
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset


# TODO: add other tables


class MIMIC4Dataset(BaseDataset):
    """Base dataset for MIMIC-IV dataset.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
        patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: define a patient's hospital admission, hadm_id.

    We further support the following tables:
        - diagnoses_icd: contains ICD diagnoses (ICD9CM and ICD10CM code)
            for patients.
        - procedures_icd: contains ICD procedures (ICD9PROC and ICD10PROC
            code) for patients.
        - prescriptions: contains medication related order entries (NDC code)
            for patients.
        - labevents: contains laboratory measurements (MIMIC4_ITEMID code)
            for patients

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded. Must be a subset of the
            following tables: diagnoses_icd, procedures_icd, prescriptions, labevents.
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
        super(MIMIC4Dataset, self).__init__(
            dataset_name="MIMIC-IV",
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
        """Helper functions which parses patients and admissions tables.

        Will be called in _parse_tables().

        Docs:
            - patients:https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")
        # load patients
        for p_id, p_info in tqdm(df_group, desc="Parsing patients and admissions"):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=self._strptime(str(birth_year), "%Y"),
                # no exact time, use 00:00:00
                death_datetime=self._strptime(p_info["dod"].values[0], "%Y-%m-%d"),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=self._strptime(v_info["admittime"].values[0]),
                    discharge_time=self._strptime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def _parse_diagnoses_icd(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses diagnosis_icd table.

        Will be called in _parse_tables().

        Docs:
            - diagnosis_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/

        Note that MIMIC-IV does not provide specific timestamps in diagnoses_icd
            table, so we set it to None.
        """
        table = "diagnoses_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby(["subject_id", "hadm_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary=f"ICD{version}CM",
                    visit_id=v_id,
                    patient_id=p_id,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_procedures_icd(self, patients) -> Dict[str, Patient]:
        """function to parse procedures_icd table.

        Will be called in _parse_tables().

        Docs:
            - procedures_icd: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/

        Note that MIMIC-IV does not provide specific timestamps in procedures_icd
            table, so we set it to None.
        """
        table = "procedures_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby(["subject_id", "hadm_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary=f"ICD{version}PROC",
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
            - prescriptions: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions/
        """
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby(["subject_id", "hadm_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["starttime"], v_info["ndc"]):
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
        """Helper functions which parses labevents table.

        Will be called in _parse_tables().

        Docs:
            - labevents: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/
        """
        table = "labevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby(["subject_id", "hadm_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["charttime"], v_info["itemid"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients


if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        dev=False,
        code_mapping={"NDC": "ATC"},
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
