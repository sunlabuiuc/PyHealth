import os
from pathlib import Path
import numpy as np
import pandas as pd
# import sys
# sys.path.append('/home/chaoqiy2/github/PyHealth-OMOP')

from pyhealth.data import Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, pickle_dump, pickle_load


class eICUBaseDataset(BaseDataset):
    """ Base dataset for eICU """

    def __init__(self, root):
        self.root = root
        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/eicu.data")):
            patients_df, unique_visits = self.parse_patients()
            diagnosis_df, diagnosis_dict = self.parse_diagnosis(unique_visits)
            # lab_df, lab_dict = self.parse_lab(unique_visits)
            medication_df, medication_dict = self.parse_medication(unique_visits)
            treatment_df, treatment_dict = self.parse_treatment(unique_visits)
            # physicalExam_df, physicalExam_dict = self.parse_physicalExam(unique_visits)
            patients = self.merge_data(
                patients_df,
                diagnosis_dict,
            #    lab_dict,
                medication_dict,
                treatment_dict,
                # physicalExam_dict,
            )
            create_directory(os.path.join(str(Path.home()), ".cache/pyhealth"))
            pickle_dump(patients, os.path.join(str(Path.home()), ".cache/pyhealth/eicu.data"))
        else:
            patients = pickle_load(os.path.join(str(Path.home()), ".cache/pyhealth/eicu.data"))
        super(eICUBaseDataset, self).__init__(dataset_name="eICU", patients=patients)

    def _solve_nested_list(self, ls):
        out_ls = []
        for item in ls:
            if type(item) == type("string"):
                out_ls += item.split(', ')
        return out_ls
    def parse_patients(self):
        patients_df = pd.read_csv(os.path.join(self.root, "patient.csv"))
        unique_visits = patients_df.patientunitstayid.unique()
        return patients_df, unique_visits

    def parse_diagnosis(self, unique_visits):
        diagnosis_df = pd.read_csv(os.path.join(self.root, "diagnosis.csv"))
        diagnosis_dict = dict.fromkeys(unique_visits)
        for stay_id, content in diagnosis_df.groupby("patientunitstayid"):
            diagnosis_dict[stay_id] = np.unique(self._solve_nested_list(content["icd9code"].tolist()))
        return diagnosis_df, diagnosis_dict
    
    def parse_lab(self, unique_visits):
        lab_df = pd.read_csv(os.path.join(self.root, "lab.csv"))
        lab_dict = dict.fromkeys(unique_visits)
        for stay_id, content in lab_df.groupby("patientunitstayid"):
            lab_dict[stay_id] = np.unique(self._solve_nested_list(content["labname"].tolist()))
        return lab_df, lab_dict

    def parse_medication(self, unique_visits):
        medication_df = pd.read_csv(os.path.join(self.root, "medication.csv"))
        medication_dict = dict.fromkeys(unique_visits)
        for stay_id, content in medication_df.groupby("patientunitstayid"):
            medication_dict[stay_id] = np.unique(self._solve_nested_list(content["drugname"].tolist()))
        return medication_df, medication_dict

    def parse_treatment(self, unique_visits):
        treatment_df = pd.read_csv(os.path.join(self.root, "treatment.csv"))
        treatment_dict = dict.fromkeys(unique_visits)
        for stay_id, content in treatment_df.groupby("patientunitstayid"):
            treatment_dict[stay_id] = np.unique(self._solve_nested_list(content["treatmentstring"].tolist()))
        return treatment_df, treatment_dict

    def parse_physicalExam(self, unique_visits):
        physicalExam_df = pd.read_csv(os.path.join(self.root, "physicalExam.csv"))
        physicalExam_dict = dict.fromkeys(unique_visits)
        for stay_id, content in physicalExam_df.groupby("patientunitstayid"):
            physicalExam_dict[stay_id] = np.unique(self._solve_nested_list(content["physicalExamPath"].tolist()))
        return physicalExam_df, physicalExam_dict

    @staticmethod
    def merge_data(
        patients_df,
        diagnosis_dict,
        # lab_dict,
        medication_dict,
        treatment_dict,
        # physicalExam_dict
    ):
        """
        conditions: diagnosis 
        procedures: treatment
        drugs: medications
        """
        patients = []
        # enumerate patients
        for patient_id, p_content in patients_df.groupby("uniquepid"):
            visits = []
            # enumerate visits
            for visit_id in p_content.patientunitstayid.tolist():
                if (visit_id in diagnosis_dict) and (visit_id in medication_dict) and (visit_id in treatment_dict):
                    visit = Visit(visit_id=visit_id,
                            patient_id=patient_id,
                            conditions=diagnosis_dict[visit_id],
                            procedures=treatment_dict[visit_id],
                            drugs=medication_dict[visit_id])
                    visits.append(visit)
            patient = Patient(patient_id=patient_id, visits=visits)
            patients.append(patient)
        return patients


if __name__ == "__main__":
    base_dataset = eICUBaseDataset(root="/srv/local/data/physionet.org/files/eicu-crd/2.0")
    print(base_dataset)
    print(type(base_dataset))
    print(len(base_dataset))
