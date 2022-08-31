import os
from pathlib import Path

import pandas as pd
import sys
sys.path.append('/home/chaoqiy2/github/PyHealth-OMOP')
from pyhealth.data import Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, pickle_dump, pickle_load


class OMOPBaseDataset(BaseDataset):
    """ Base dataset for OMOP based database """

    def __init__(self, root):
        self.root = root
        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/omop.data")):
            patients_df = self.parse_patients()
            visits_df = self.parse_visits()
            diagnoses_df = self.parse_diagnoses()
            procedures_df = self.parse_procedures()
            prescriptions_df = self.parse_prescriptions()
            patients = self.merge_data(patients_df,
                                       visits_df,
                                       diagnoses_df,
                                       procedures_df,
                                       prescriptions_df)
            create_directory(os.path.join(str(Path.home()), ".cache/pyhealth"))
            pickle_dump(patients, os.path.join(str(Path.home()), ".cache/pyhealth/omop.data"))
        else:
            patients = pickle_load(os.path.join(str(Path.home()), ".cache/pyhealth/omop.data"))
        super(OMOPBaseDataset, self).__init__(dataset_name="OMOP", patients=patients)

    def parse_patients(self):
        patients_df = pd.read_csv(os.path.join(self.root, "person.csv"), sep="\t",
                        dtype={'person_id': str})
        return patients_df

    def parse_visits(self):
        visits_df = pd.read_csv(os.path.join(self.root, "visit_occurrence.csv"), sep="\t",
                        dtype={'person_id': str, "visit_occurrence_id": str})
        return visits_df

    def parse_diagnoses(self):
        diagnoses_df = pd.read_csv(os.path.join(self.root, "condition_occurrence.csv"), sep="\t",
                                       dtype={'person_id': str, "visit_occurrence_id": str, "condition_concept_id": str})
        diagnoses_df = diagnoses_df[['person_id', 'visit_occurrence_id', 'condition_concept_id']]
        diagnoses_df = diagnoses_df.sort_values(['person_id', 'visit_occurrence_id'], ascending=True)
        diagnoses_df = diagnoses_df.groupby(['person_id', 'visit_occurrence_id']).condition_concept_id.agg(DIAG=list)
        diagnoses_df = diagnoses_df.reset_index()
        return diagnoses_df

    def parse_procedures(self):
        procedures_df = pd.read_csv(os.path.join(self.root, "procedure_occurrence.csv"), sep="\t",
                                       dtype={'person_id': str, "visit_occurrence_id": str, "procedure_concept_id": str})
        procedures_df = procedures_df[['person_id', 'visit_occurrence_id', 'procedure_concept_id']]
        procedures_df = procedures_df.sort_values(['person_id', 'visit_occurrence_id'], ascending=True)
        procedures_df = procedures_df.groupby(['person_id', 'visit_occurrence_id']).procedure_concept_id.agg(PROC=list)
        procedures_df = procedures_df.reset_index()
        return procedures_df

    def parse_prescriptions(self):
        prescriptions_df = pd.read_csv(os.path.join(self.root, "drug_exposure.csv"), sep="\t", 
                                       dtype={'person_id': str, "visit_occurrence_id": str, "drug_concept_id": str})
        prescriptions_df = prescriptions_df[['person_id', 'visit_occurrence_id', 'drug_concept_id']]
        prescriptions_df = prescriptions_df.sort_values(['person_id', 'visit_occurrence_id'], ascending=True)
        prescriptions_df = prescriptions_df.groupby(['person_id', 'visit_occurrence_id']).drug_concept_id.agg(PRES=list)
        prescriptions_df = prescriptions_df.reset_index()
        return prescriptions_df

    @staticmethod
    def merge_data(patients_df, visits_df, diagnoses_icd_df, procedures_icd_df, prescriptions_df):
        data = patients_df.merge(visits_df, on='person_id', how="outer")
        data = data.merge(diagnoses_icd_df, on=['person_id', 'visit_occurrence_id'], how="outer")
        data = data.merge(procedures_icd_df, on=['person_id', 'visit_occurrence_id'], how="outer")
        data = data.merge(prescriptions_df, on=['person_id', 'visit_occurrence_id'], how="outer")
        data = data.sort_values(['person_id', 'visit_start_date'], ascending=True)
        visit_id_to_visit_dict = {}
        for idx, row in data.iterrows():
            visit = Visit(visit_id=row.visit_occurrence_id,
                          patient_id=row.person_id,
                          conditions=row.DIAG,
                          procedures=row.PROC,
                          drugs=row.PRES)
            visit_id_to_visit_dict[row.visit_occurrence_id] = visit
        patients = []
        for patient_id, row in data.groupby("person_id"):
            visit_ids = row.visit_occurrence_id.tolist()
            visits = [visit_id_to_visit_dict[visit_id] for visit_id in visit_ids]
            patient = Patient(patient_id=patient_id, visits=visits)
            patients.append(patient)
        return patients


if __name__ == "__main__":
    base_dataset = OMOPBaseDataset(root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2")
    print(base_dataset)
    print(type(base_dataset))
    print(len(base_dataset))
