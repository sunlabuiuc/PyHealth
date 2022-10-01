import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append('/home/chaoqiy2/github/PyHealth-OMOP')

from pyhealth.data import Event, Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, dump_pickle, load_pickle
from tqdm import tqdm
from datetime import datetime


class MIMIC4BaseDataset(BaseDataset):
    """ Base dataset for MIMIC-IV
        1. it contains a superset of information used for all relevant tasks
        2. it will be inputted into a task module for further data cleaning
    """

    def __init__(self, root, files=['conditions', 'procedures', 'drugs']):
        """
        INPUT
            - root: root directory of the dataset
            - files: list of files to be parsed (no order), i.e., conditions, procedures, drugs, labs, physicalExams
        OUTPUT
            - patients: a <pyhealth.data.Patient> object

        NOTICE:
            - the encounter time stored in <Visit> is the absolute time
            - the time stored in <Event> is the relative time offset w.r.t. the encounter time
        """
        self.root = root
        self.files = files
        self.all_support_files = ['conditions', 'procedures', 'drugs', 'labs']

        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/mimic4.data")):
            # get visit-level static features
            visits = self.parse_patients()
            patients = {}

            print("structured all patients and visits")
            # process based on self.files
            if 'conditions' in self.files:
                self.parse_diagnoses_icd(visits, patients)
                print("processed conditions")
            if 'procedures' in self.files:
                self.parse_procedures_icd(visits, patients)
                print("processed procedures")
            if 'drugs' in self.files:
                self.parse_prescriptions(visits, patients)
                print("processed drugs")
            if 'labs' in self.files:
                self.parse_lab_results(visits, patients)
                print("processed labs")

            # save to cache
            create_directory(os.path.join(str(Path.home()), ".cache/pyhealth"))
            dump_pickle(patients, os.path.join(str(Path.home()), ".cache/pyhealth/mimic4.data"))
        else:
            patients = load_pickle(os.path.join(str(Path.home()), ".cache/pyhealth/mimic4.data"))
        super(MIMIC4BaseDataset, self).__init__(dataset_name="MIMIC-IV", patients=patients)

    def parse_patients(self):
        """ func to parse patient table """
        patients_df = pd.read_csv(os.path.join(self.root, "patients.csv"), \
                                  dtype={'subject_id': str})
        patients_df = patients_df[['subject_id', 'gender', 'dod']]
        admission_df = pd.read_csv(os.path.join(self.root, "admissions.csv"), \
                                   dtype={'subject_id': str, "hadm_id": str, "icd_code": str})

        admission_df = admission_df[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type', \
                                     'admission_location', 'marital_status', 'hospital_expire_flag']]
        patients_admission_df = admission_df.merge(patients_df, on='subject_id')

        visits = {}
        for patient_id, patient_info in tqdm(patients_admission_df.groupby("subject_id")):
            for visit_id, visit_info in patient_info.groupby("hadm_id"):
                # visit statistics
                encounter_time = visit_info["admittime"].values[0]
                duration = self.diffhours(encounter_time, visit_info["dischtime"].values[0])
                mortality = visit_info["hospital_expire_flag"].values[0]
                cur_visit = Visit(
                    visit_id,
                    patient_id,
                    encounter_time,
                    duration,
                    mortality,
                )
                visits[visit_id] = cur_visit
        return visits

    def parse_diagnoses_icd(self, visits, patients):
        """ func to parse diagnoses table
            for diagnosis time, MIMIC-III seems to perform diagnosis when admitted into hospital
            thus, we use the admission time as diagnosis time
        """
        print(len(visits))
        diagnoses_icd_df = pd.read_csv(os.path.join(self.root, "diagnoses_icd.csv"),
                                       dtype={'subject_id': str, "hadm_id": str, "icd_code": str})
        diagnoses_icd_df = diagnoses_icd_df.sort_values(['subject_id', 'hadm_id', 'seq_num'], ascending=True)
        for visit_id, visit_info in tqdm(diagnoses_icd_df.groupby("hadm_id")):
            if visit_id not in visits: continue
            # load diagnosis with time info
            cur_diagnosis = []
            encounter_time = visits[visit_id].encounter_time
            for code in visit_info['icd_code'].values:
                cur_diagnosis += self.process_nested_code(code, self.diffhours(encounter_time, encounter_time))

            if len(cur_diagnosis) == 0: continue
            # add diagnosis to patient dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if visit_id not in patients[patient_id].visits:  # register visit if not exist
                visits[visit_id].conditions = cur_diagnosis
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].conditions = cur_diagnosis

    def parse_procedures_icd(self, visits, patients):
        """ func to parse procedures table """
        procedures_icd_df = pd.read_csv(os.path.join(self.root, "procedures_icd.csv"),
                                        dtype={'subject_id': str, "hadm_id": str, "icd_code": str})
        procedures_icd_df = procedures_icd_df.sort_values(['subject_id', 'hadm_id'], ascending=True)
        for visit_id, visit_info in tqdm(procedures_icd_df.groupby("hadm_id")):
            if visit_id not in visits: continue

            # load procedures with time info
            cur_procedures = []
            encounter_time = visits[visit_id].encounter_time
            for code, time in visit_info[['icd_code', 'chartdate']].values:
                cur_procedures += self.process_nested_code(code, self.diffhours(encounter_time, time))

            if len(cur_procedures) == 0: continue
            # add procedures to patient dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if visit_id not in patients[patient_id].visits:  # register visit if not exist
                visits[visit_id].procedures = cur_procedures
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].procetures = cur_procedures

    def parse_prescriptions(self, visits, patients):
        prescriptions_df = pd.read_csv(os.path.join(self.root, "prescriptions.csv"), low_memory=False,
                                       dtype={'subject_id': str, "hadm_id": str, "ndc": str})
        for visit_id, visit_info in tqdm(prescriptions_df.groupby("hadm_id")):
            if visit_id not in visits: continue

            # load prescription with time info
            cur_prescriptions = []
            encounter_time = visits[visit_id].encounter_time
            for code, time in visit_info[['ndc', 'starttime']].values:
                if time == time:
                    cur_prescriptions += self.process_nested_code(code, self.diffhours(encounter_time, time))

            if len(cur_prescriptions) == 0: continue
            # add prescription to patient dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if visit_id not in patients[patient_id].visits:  # register visit if not exist
                visits[visit_id].drugs = cur_prescriptions
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].drugs = cur_prescriptions

    def parse_lab_results(self, visits, patients):
        lab_results_df = pd.read_csv(os.path.join(self.root, "labevents.csv"),
                                     dtype={'subject_id': str, "hadm_id": str, "itemid": str})
        for visit_id, visit_info in tqdm(lab_results_df.groupby("hadm_id")):
            if visit_id not in visits: continue
            # load lab results with time info
            cur_lab_results = []
            encounter_time = visits[visit_id].encounter_time
            for code, time in visit_info[['itemid', 'charttime']].values:
                cur_lab_results += self.process_nested_code(code, self.diffhours(encounter_time, time))

            if len(cur_lab_results) == 0: continue
            # add lab results to patient dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if visit_id not in patients[patient_id].visits:  # register visit if not exist
                visits[visit_id].labs = cur_lab_results
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].labs = cur_lab_results

    @staticmethod
    def diffhours(start_time, end_time):
        """
        start_time: str
        end_time: str
        """
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        try:
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        except:
            end_time = datetime.strptime(end_time, '%Y-%m-%d')
        return ((end_time - start_time).total_seconds() / 3600)

    @staticmethod
    def process_nested_code(code_ls, time):
        event_ls = []
        if code_ls != code_ls:
            return event_ls
        elif type(code_ls) == type("string"):
            out_ls = code_ls.split(", ")
        else:
            out_ls = [code_ls]

        for code in out_ls:
            event_ls.append(Event(code, time))
        return event_ls


if __name__ == "__main__":
    dataset = MIMIC4BaseDataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp", \
                                files=['conditions', 'procedures', 'drugs', 'labs'])
    print(dataset)
    print(type(dataset))
    print(len(dataset))
