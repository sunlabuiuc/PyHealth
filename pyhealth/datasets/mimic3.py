import os
import sys
from typing import Dict

import pandas as pd

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")

from pyhealth.data import Event, Visit, Patient, BaseDataset
from tqdm import tqdm


# TODO: we should include all the following tables from hospital database
# TODO: add cptevents
# TODO: add drgcodes
# TODO: add noteevents
# TODO: add microbiologyevents
# TODO: add procedureevents_mv

# TODO: should convert all timestamps into datetime objects, right now they are strings

# TODO: where should we filter out codes that do not make sense? (e.g., 0 in NDC)

class MIMIC3BaseDataset(BaseDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU patients.
    The dataset is available at https://mimic.physionet.org/.

    Here, we only use the following tables:
        PATIENTS.csv: defines each SUBJECT_ID in the database, i.e. defines a single patient.
        ADMISSIONS.csv: define a patient’s hospital admission, HADM_ID.
        DIAGNOSES_ICD.csv: contains ICD diagnoses for patients, most notably ICD-9 diagnoses.
        PROCEDURES_ICD.csv: contains ICD procedures for patients, most notably ICD-9 procedures.
        PRESCRIPTIONS.csv: contains medication related order entries, i.e. prescriptions.
        LABEVENTS.csv: contains all laboratory measurements for a given patient, including out patient data.

    =====================================================================================
    Data structure:

        BaseDataset.patients: dict[str, Patient]
            - key: patient_id, value: <Patient> object

        BaseDataset.visits: dict[str, Visit]
            - key: visit_id, value: <Visit> object

        <Patient>
            - patient_id: str
            - visit_ids: list[str]
                - links to <Visit> objects in BaseDataset.visits
            - other attributes (e.g., birth_date, death_date, mortality_status, gender, ethnicity, etc.)

        <Visit>
            - visit_id: str
            - patient_id: str
                - links to <Patient> object in BaseDataset.patients
            - conditions, procedures, drugs, labs: list[Event]
                - a list of <Event> objects
            - other attributes (e.g., encounter_time, discharge_time, mortality_status, conditions, etc.)

        <Event>
            - code: str
            - domain: float
            - vocabulary: str
            - description: str
    =====================================================================================
    """

    def __init__(self, root, dev=False, refresh_cache=False):
        """
        Args:
            root: root directory of the raw data (should contain many csv files)
            dev: whether to enable dev mode (only use a small subset of the data)
            refresh_cache: whether to refresh the cache; if true, the dataset will be processed from scratch
             and the cache will be updated
        """
        super(MIMIC3BaseDataset, self).__init__(dataset_name="MIMIC-III",
                                                root=root,
                                                dev=dev,
                                                refresh_cache=refresh_cache)

    def process(self):
        """This function overrides the process function in BaseDataset.

        It parses the corresponding tables and create patients and visits which will be cached later.

        Returns:
            patients: a dictionary of Patient objects indexed by patient_id
            visits: a dictionary of Visit objects indexed by visit_id
        """
        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient]
        # visits is a dict of Visit objects indexed by visit_id
        visits: Dict[str, Visit]
        # process patients and admissions features
        patients, visits = self.parse_patients_and_admissions()
        # process clinical features
        patients, visits = self.parse_diagnoses_icd(patients=patients, visits=visits)
        patients, visits = self.parse_procedures_icd(patients=patients, visits=visits)
        patients, visits = self.parse_prescriptions(patients=patients, visits=visits)
        patients, visits = self.parse_labevents(patients=patients, visits=visits)
        return patients, visits

    def parse_patients_and_admissions(self):
        """function to parse patients and admissions tables"""
        # read patient table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admission table
        admission_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str}
        )
        # merge patient and admission tables
        merged_df = pd.merge(patients_df, admission_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        merged_df = merged_df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # create patients and visits
        patients = {}
        visits = {}
        for patient_id, patient_info in tqdm(merged_df.groupby("SUBJECT_ID"), desc="Parsing patients and admissions"):
            # load patient info
            patient = Patient(patient_id=patient_id,
                              birth_date=patient_info["DOB"].values[0],
                              death_date=patient_info["DOD_HOSP"].values[0],
                              mortality_status=patient_info["EXPIRE_FLAG"].values[0],
                              # TODO: should categorize the gender
                              gender=patient_info["GENDER"].values[0],
                              # TODO: should categorize the ethnicity
                              ethnicity=patient_info["ETHNICITY"].values[0],
                              visit_ids=patient_info["HADM_ID"].tolist())
            patients[patient_id] = patient
            # load visit info
            for visit_id, visit_info in patient_info.groupby("HADM_ID"):
                visit = Visit(visit_id=visit_id,
                              patient_id=patient_id,
                              encounter_time=visit_info["ADMITTIME"].values[0],
                              discharge_time=visit_info["DISCHTIME"].values[0],
                              mortality_status=visit_info["HOSPITAL_EXPIRE_FLAG"].values[0])
                visits[visit_id] = visit
        return patients, visits

    def parse_diagnoses_icd(self, patients, visits):
        """function to parse diagnoses table.

        Note that MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD table,
        so we use the admission time as the timestamp for all diagnoses.
        """
        # read diagnoses table
        diagnoses_icd_df = pd.read_csv(
            os.path.join(self.root, "DIAGNOSES_ICD.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # sort by sequence number (i.e., disease priority)
        diagnoses_icd_df = diagnoses_icd_df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # update patients and visits
        for (patient_id, visit_id), visit_info in tqdm(diagnoses_icd_df.groupby(["SUBJECT_ID", "HADM_ID"]),
                                                       desc="Parsing diagnoses"):
            if (patient_id not in patients) or (visit_id not in visits):
                continue
            domain = "condition"
            vocabulary = "ICD9"
            timestamp = visits[visit_id].encounter_time
            for code in visit_info["ICD9_CODE"]:
                event = Event(code=code,
                              domain=domain,
                              vocabulary=vocabulary,
                              timestamp=timestamp)
                visits[visit_id].conditions.append(event)
        return patients, visits

    def parse_procedures_icd(self, patients, visits):
        """function to parse procedures table.

        Note that MIMIC-III does not provide specific timestamps in PROCEDURES_ICD table,
        so we use the admission time as the timestamp for all diagnoses.
        """
        # read procedures table
        procedures_icd_df = pd.read_csv(
            os.path.join(self.root, "PROCEDURES_ICD.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # sort by sequence number (i.e., procedure priority)
        procedures_icd_df = procedures_icd_df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # update patients and visits
        for (patient_id, visit_id), visit_info in tqdm(procedures_icd_df.groupby(["SUBJECT_ID", "HADM_ID"]),
                                                       desc="Parsing procedures"):
            if (patient_id not in patients) or (visit_id not in visits):
                continue
            domain = "procedure"
            vocabulary = "ICD9"
            timestamp = visits[visit_id].encounter_time
            for code in visit_info["ICD9_CODE"]:
                event = Event(code=code,
                              domain=domain,
                              vocabulary=vocabulary,
                              timestamp=timestamp)
                visits[visit_id].procedures.append(event)
        return patients, visits

    def parse_prescriptions(self, patients, visits):
        """function to parse prescriptions table"""
        # read prescriptions table
        prescriptions_df = pd.read_csv(
            os.path.join(self.root, "PRESCRIPTIONS.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # sort by start date and end date
        prescriptions_df = prescriptions_df.sort_values(["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"],
                                                        ascending=True)
        # update patients and visits
        for (patient_id, visit_id), visit_info in tqdm(prescriptions_df.groupby(["SUBJECT_ID", "HADM_ID"]),
                                                       desc="Parsing prescriptions"):
            if (patient_id not in patients) or (visit_id not in visits):
                continue
            domain = "drug"
            vocabulary = "NDC"
            for timestamp, code in zip(visit_info["STARTDATE"], visit_info["NDC"]):
                event = Event(code=code,
                              domain=domain,
                              vocabulary=vocabulary,
                              timestamp=timestamp)
                visits[visit_id].drugs.append(event)
        return patients, visits

    def parse_labevents(self, patients, visits):
        """function to parse labevents table"""
        # read labevents table
        labevents_df = pd.read_csv(
            os.path.join(self.root, "LABEVENTS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # sort by charttime
        labevents_df = labevents_df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # update patients and visits
        for (patient_id, visit_id), visit_info in tqdm(labevents_df.groupby(["SUBJECT_ID", "HADM_ID"]),
                                                       desc="Parsing labevents"):
            if (patient_id not in patients) or (visit_id not in visits):
                continue
            domain = "lab"
            vocabulary = "MIMIC-III ITEMID"
            for timestamp, code in zip(visit_info["CHARTTIME"], visit_info["ITEMID"]):
                event = Event(code=code,
                              domain=domain,
                              vocabulary=vocabulary,
                              timestamp=timestamp)
                visits[visit_id].labs.append(event)
        return patients, visits


if __name__ == "__main__":
    dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4", dev=False, refresh_cache=False)
    dataset.stat()
    dataset.info()
