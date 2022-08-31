import os
from pathlib import Path
import pandas as pd
# import sys
# sys.path.append('/home/chaoqiy2/github/PyHealth-OMOP')

from pyhealth.data import Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, pickle_dump, pickle_load

class MIMIC3BaseDataset(BaseDataset):
    """ Base dataset for MIMIC-III """

    def __init__(self, root):
        self.root = root
        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/mimic3.data")):
            patients_df = self.parse_patients()
            admissions_df = self.parse_admissions()
            diagnoses_icd_df = self.parse_diagnoses_icd()
            procedures_icd_df = self.parse_procedures_icd()
            prescriptions_df = self.parse_prescriptions()
            patients = self.merge_data(patients_df,
                                       admissions_df,
                                       diagnoses_icd_df,
                                       procedures_icd_df,
                                       prescriptions_df)
            create_directory(os.path.join(str(Path.home()), ".cache/pyhealth"))
            pickle_dump(patients, os.path.join(str(Path.home()), ".cache/pyhealth/mimic3.data"))
        else:
            patients = pickle_load(os.path.join(str(Path.home()), ".cache/pyhealth/mimic3.data"))
        super(MIMIC3BaseDataset, self).__init__(dataset_name="MIMIC-III", patients=patients)

    def parse_patients(self):
        patients_df = pd.read_csv(os.path.join(self.root, "PATIENTS.csv"),
                                  dtype={'SUBJECT_ID': str})
        return patients_df

    def parse_admissions(self):
        admissions_df = pd.read_csv(os.path.join(self.root, "ADMISSIONS.csv"),
                                    dtype={'SUBJECT_ID': str, 'HADM_ID': str})
        return admissions_df

    def parse_diagnoses_icd(self):
        diagnoses_icd_df = pd.read_csv(os.path.join(self.root, "DIAGNOSES_ICD.csv"),
                                       dtype={'SUBJECT_ID': str, "HADM_ID": str, "ICD9_CODE": str})
        diagnoses_icd_df = diagnoses_icd_df.sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], ascending=True)
        diagnoses_icd_df = diagnoses_icd_df.groupby(['SUBJECT_ID', 'HADM_ID']).ICD9_CODE.agg(DIAG=list)
        diagnoses_icd_df = diagnoses_icd_df.reset_index()
        return diagnoses_icd_df

    def parse_procedures_icd(self):
        procedures_icd_df = pd.read_csv(os.path.join(self.root, "PROCEDURES_ICD.csv"),
                                        dtype={'SUBJECT_ID': str, "HADM_ID": str, "ICD9_CODE": str})
        procedures_icd_df = procedures_icd_df.sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], ascending=True)
        procedures_icd_df = procedures_icd_df.groupby(['SUBJECT_ID', 'HADM_ID']).ICD9_CODE.agg(PROC=list)
        procedures_icd_df = procedures_icd_df.reset_index()
        return procedures_icd_df

    def parse_prescriptions(self):
        prescriptions_df = pd.read_csv(os.path.join(self.root, "PRESCRIPTIONS.csv"),
                                       dtype={'SUBJECT_ID': str, "HADM_ID": str, "NDC": str})
        prescriptions_df = prescriptions_df.groupby(['SUBJECT_ID', 'HADM_ID']).NDC.agg(PRES=list)
        prescriptions_df = prescriptions_df.reset_index()
        return prescriptions_df

    @staticmethod
    def merge_data(patients_df, admissions_df, diagnoses_icd_df, procedures_icd_df, prescriptions_df):
        data = patients_df.merge(admissions_df, on='SUBJECT_ID', how="outer")
        data = data.merge(diagnoses_icd_df, on=['SUBJECT_ID', 'HADM_ID'], how="outer")
        data = data.merge(procedures_icd_df, on=['SUBJECT_ID', 'HADM_ID'], how="outer")
        data = data.merge(prescriptions_df, on=['SUBJECT_ID', 'HADM_ID'], how="outer")
        data = data.sort_values(['SUBJECT_ID', 'ADMITTIME'], ascending=True)
        visit_id_to_visit_dict = {}
        for idx, row in data.iterrows():
            visit = Visit(visit_id=row.HADM_ID,
                          patient_id=row.SUBJECT_ID,
                          conditions=row.DIAG,
                          procedures=row.PROC,
                          drugs=row.PRES)
            visit_id_to_visit_dict[row.HADM_ID] = visit
        patients = []
        for patient_id, row in data.groupby("SUBJECT_ID"):
            visit_ids = row.HADM_ID.tolist()
            visits = [visit_id_to_visit_dict[visit_id] for visit_id in sorted(visit_ids)]
            patient = Patient(patient_id=patient_id, visits=visits)
            patients.append(patient)
        return patients


if __name__ == "__main__":
    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")
    print(base_dataset)
    print(type(base_dataset))
    print(len(base_dataset))
