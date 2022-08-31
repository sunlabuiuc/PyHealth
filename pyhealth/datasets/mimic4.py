import os
from pathlib import Path

import pandas as pd
# import sys
# sys.path.append('/home/chaoqiy2/github/PyHealth-OMOP')

from pyhealth.data import Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, pickle_dump, pickle_load


class MIMIC4BaseDataset(BaseDataset):
    """ Base dataset for MIMIC-IV """

    def __init__(self, root):
        self.root = root
        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/mimic4.data")):
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
            pickle_dump(patients, os.path.join(str(Path.home()), ".cache/pyhealth/mimic4.data"))
        else:
            patients = pickle_load(os.path.join(str(Path.home()), ".cache/pyhealth/mimic4.data"))
        super(MIMIC4BaseDataset, self).__init__(dataset_name="MIMIC-IV", patients=patients)

    def parse_patients(self):
        patients_df = pd.read_csv(os.path.join(self.root, "patients.csv"),
                                  dtype={'subject_id': str})
        return patients_df

    def parse_admissions(self):
        admissions_df = pd.read_csv(os.path.join(self.root, "admissions.csv"),
                                    dtype={'subject_id': str, 'hadm_id': str})
        return admissions_df

    def parse_diagnoses_icd(self):
        diagnoses_icd_df = pd.read_csv(os.path.join(self.root, "diagnoses_icd.csv"),
                                       dtype={'subject_id': str, "hadm_id": str, "icd_code": str})
        diagnoses_icd_df = diagnoses_icd_df.sort_values(['subject_id', 'hadm_id', 'seq_num'], ascending=True)
        diagnoses_icd_df = diagnoses_icd_df.groupby(['subject_id', 'hadm_id']).icd_code.agg(diag=list)
        diagnoses_icd_df = diagnoses_icd_df.reset_index()
        return diagnoses_icd_df

    def parse_procedures_icd(self):
        procedures_icd_df = pd.read_csv(os.path.join(self.root, "procedures_icd.csv"),
                                        dtype={'subject_id': str, "hadm_id": str, "icd_code": str})
        procedures_icd_df = procedures_icd_df.sort_values(['subject_id', 'hadm_id', 'seq_num'], ascending=True)
        procedures_icd_df = procedures_icd_df.groupby(['subject_id', 'hadm_id']).icd_code.agg(proc=list)
        procedures_icd_df = procedures_icd_df.reset_index()
        return procedures_icd_df

    def parse_prescriptions(self):
        prescriptions_df = pd.read_csv(os.path.join(self.root, "prescriptions.csv"),
                                       dtype={'subject_id': str, "hadm_id": str, "ndc": str})
        prescriptions_df = prescriptions_df.groupby(['subject_id', 'hadm_id']).ndc.agg(pres=list)
        prescriptions_df = prescriptions_df.reset_index()
        return prescriptions_df

    @staticmethod
    def merge_data(patients_df, admissions_df, diagnoses_icd_df, procedures_icd_df, prescriptions_df):
        data = patients_df.merge(admissions_df, on='subject_id', how="outer")
        data = data.merge(diagnoses_icd_df, on=['subject_id', 'hadm_id'], how="outer")
        data = data.merge(procedures_icd_df, on=['subject_id', 'hadm_id'], how="outer")
        data = data.merge(prescriptions_df, on=['subject_id', 'hadm_id'], how="outer")
        data = data.sort_values(['subject_id', 'admittime'], ascending=True)
        visit_id_to_visit_dict = {}
        for idx, row in data.iterrows():
            visit = Visit(visit_id=row.hadm_id,
                          patient_id=row.subject_id,
                          conditions=row.diag,
                          procedures=row.proc,
                          drugs=row.pres)
            visit_id_to_visit_dict[row.hadm_id] = visit
        patients = []
        for patient_id, row in data.groupby("subject_id"):
            visit_ids = row.hadm_id.tolist()
            visits = [visit_id_to_visit_dict[visit_id] for visit_id in visit_ids]
            patient = Patient(patient_id=patient_id, visits=visits)
            patients.append(patient)
        return patients


if __name__ == "__main__":
    base_dataset = MIMIC4BaseDataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp")
    print(base_dataset)
    print(type(base_dataset))
    print(len(base_dataset))
