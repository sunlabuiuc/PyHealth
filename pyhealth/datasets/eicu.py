from tqdm import tqdm
from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset
import os
import pandas as pd
from typing import Optional, List, Dict

# TODO: add cptevents
# TODO: add drgcodes
# TODO: add noteevents
# TODO: add microbiologyevents
# TODO: add procedureevents_mv


class eICUDataset(BaseDataset):
    """Base dataset for eICU dataset.

    The eICU dataset is a large dataset of de-identified health records of ICU patients.
    The dataset is available at https://eicu-crd.mit.edu/.

    We support the following tables:
        - patient.csv: defines each uniquepid in the database, i.e. defines a single patient.
        - diagnosis.csv: contains ICD diagnoses for patients, most notably ICD-9 diagnoses.
        - treatment.csv: contains treatment code information for patients, via treatmentstring.
        - medication.csv: contains medication related order entries, i.e. prescriptions.
        - lab.csv: contains all laboratory measurements for a given patient, including out patient data.
        - physicalExam.csv: contains all physical exam types for a given patient.

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
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
        # store a mapping from visit_id to patient_id
        self.visit_to_patient = {}

        super(eICUDataset, self).__init__(
            dataset_name="eICU",
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
        patients = self.parse_patients(patients)
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

    def parse_patients(self, patients) -> Dict[str, Patient]:
        """function to parse patients tables"""
        # read patient table
        df = pd.read_csv(
            os.path.join(self.root, "patient.csv"),
            dtype={"uniquepid": str, "patientunitstayid": str},
            nrows=5000 if self.dev else None,
        )
        # sort by admission and discharge time
        df = df.sort_values(
            [
                "uniquepid",
                "patientunitstayid",
                "hospitaladmitoffset",
                "unitdischargeoffset",
            ],
            ascending=True,
        )
        # load patients
        for p_id, p_info in tqdm(df.groupby("uniquepid"), desc="Parsing patients"):
            patient = Patient(
                patient_id=p_id,
                # no dob, let us use age
                birth_datetime=p_info["age"].values[0],
                # no death time, let us use "unknown"
                death_datetime="unknown",
                # TODO: should categorize the gender
                gender=p_info["gender"].values[0],
                # TODO: should categorize the ethnicity
                ethnicity=p_info["ethnicity"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("patientunitstayid"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    # TODO: convert to datetime object
                    encounter_time=v_info["hospitaladmitoffset"].values[0],
                    discharge_time=v_info["unitdischargeoffset"].values[0],
                    # TODO: should categorize the discharge_status
                    discharge_status=v_info["unitdischargestatus"].values[0],
                )
                # add visit
                patient.add_visit(visit)

                # add visit to patient mapping
                self.visit_to_patient[v_id] = p_id

            # add patient
            patients[p_id] = patient
        return patients

    def parse_diagnosis(self, patients) -> Dict[str, Patient]:
        """function to parse diagnosis table."""

        table = "diagnosis"
        col = "icd9code"
        vocabulary = "ICD9CM"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "icd9code": str},
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
        df = df.dropna(subset=["patientunitstayid", col])
        df = df.sort_values(["patientunitstayid"], ascending=True)
        # update patients
        for v_id, v_info in tqdm(
            df.groupby("patientunitstayid"), desc=f"Parsing {table}"
        ):
            if v_id not in self.visit_to_patient:
                continue
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=self.visit_to_patient[v_id],
                )
                try:
                    patients[self.visit_to_patient[v_id]].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_medication(self, patients) -> Dict[str, Patient]:
        """function to parse medication table."""
        table = "medication"
        col = "drugname"
        vocabulary = "EICU_DRUGNAME"
        # read prescriptions table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"patientunitstayid": str, "drugname": str},
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
        df = df.dropna(subset=["patientunitstayid", col])
        # sort by drugstartoffset
        df = df.sort_values(["patientunitstayid", "drugstartoffset"], ascending=True)
        # update patients and visits
        for v_id, v_info in tqdm(
            df.groupby(["patientunitstayid"]), desc=f"Parsing {table}"
        ):
            if v_id not in self.visit_to_patient:
                continue
            for timestamp, code in zip(v_info["drugstartoffset"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=self.visit_to_patient[v_id],
                    timestamp=timestamp,
                )
                try:
                    patients[self.visit_to_patient[v_id]].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_lab(self, patients) -> Dict[str, Patient]:
        """function to parse lab table."""
        table = "lab"
        col = "labname"
        vocabulary = "EICU_LABNAME"
        # read labevents table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "labname": str},
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
        df = df.dropna(subset=["patientunitstayid", col])
        # sort by labresultoffset
        df = df.sort_values(["patientunitstayid", "labresultoffset"], ascending=True)
        # update patients and visits
        for v_id, v_info in tqdm(
            df.groupby("patientunitstayid"), desc=f"Parsing {table}"
        ):
            if v_id not in self.visit_to_patient:
                continue
            for timestamp, code in zip(v_info["labresultoffset"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=self.visit_to_patient[v_id],
                    timestamp=timestamp,
                )
                try:
                    patients[self.visit_to_patient[v_id]].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_treatment(self, patients) -> Dict[str, Patient]:
        """function to parse treatment table.
        Note: treatment value is not available for the first fewer patients (e.g., ~20,000).
        """
        table = "treatment"
        col = "treatmentstring"
        vocabulary = "EICU_TREATMENTSTRING"
        # read labevents table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "treatmentstring": str},
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
        df = df.dropna(subset=["patientunitstayid", col])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "treatmentoffset"], ascending=True)
        # update visits
        for v_id, v_info in tqdm(
            df.groupby("patientunitstayid"), desc=f"Parsing {table}"
        ):
            if v_id not in self.visit_to_patient:
                continue
            for timestamp, code in zip(v_info["treatmentoffset"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=self.visit_to_patient[v_id],
                    timestamp=timestamp,
                )
                try:
                    patients[self.visit_to_patient[v_id]].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_physicalexam(self, patients) -> Dict[str, Patient]:
        """function to parse physicalExam table."""
        table = "physicalExam"
        col = "physicalexampath"
        vocabulary = "EICU_PHYSICALEXAMPATH"
        # read labevents table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "physicalexampath": str},
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
        df = df.dropna(subset=["patientunitstayid", col])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "physicalexamoffset"], ascending=True)
        # update visits
        for v_id, v_info in tqdm(
            df.groupby("patientunitstayid"), desc=f"Parsing {table}"
        ):
            if v_id not in self.visit_to_patient:
                continue
            for timestamp, code in zip(v_info["physicalexamoffset"], v_info[col]):
                # TODO: convert to datetime object
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=self.visit_to_patient[v_id],
                    timestamp=timestamp,
                )
                try:
                    patients[self.visit_to_patient[v_id]].add_event(event)
                except KeyError:
                    continue
        return patients


if __name__ == "__main__":
    dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "lab", "treatment", "physicalExam"],
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
