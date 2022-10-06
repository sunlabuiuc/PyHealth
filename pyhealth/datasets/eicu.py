import os
import sys
from pathlib import Path

import pandas as pd
import sys

sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")

from pyhealth.data import Event, Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, save_pickle, load_pickle
from tqdm import tqdm


# TODO: will update this later

class eICUBaseDataset(BaseDataset):
    """Base dataset for eICU
    1. it contains a superset of information used for all relevant tasks
    2. it will be inputted into a task module for further data cleaning
    """

    def __init__(self, root, files=["conditions", "procedures", "drugs"]):
        """
        INPUT
            - root: root directory of the dataset
            - files: list of files to be parsed (no order), i.e., conditions, procedures, drugs, labs, physicalExams
        OUTPUT
            - patients: a <pyhealth.data.Patient> object
        """
        self.root = root
        self.files = files
        self.all_support_files = [
            "conditions",
            "procedures",
            "drugs",
            "labs",
            "physicalExams",
        ]

        if not os.path.exists(
            os.path.join(str(Path.home()), ".cache/pyhealth/eicu.data")
        ):
            # get visit-level static features
            visits = self.parse_patients()
            patients = {}

            print("structured all patients and visits")
            # process based on self.files
            if "conditions" in self.files:
                self.parse_diagnosis(visits, patients)
                print("processed conditions")

            if "procedures" in self.files:
                self.parse_treatment(visits, patients)
                print("processed procedures")
            if "drugs" in self.files:
                self.parse_medication(visits, patients)
                print("processed drugs")
            if "labs" in self.files:
                self.parse_lab(visits, patients)
                print("processed labs")
            if "physicalExams" in self.files:
                self.parse_physicalExam(visits, patients)
                print("processed physicalExams")

            # save to cache
            create_directory(os.path.join(str(Path.home()), ".cache/pyhealth"))
            dump_pickle(
                patients, os.path.join(str(Path.home()), ".cache/pyhealth/eicu.data")
            )
        else:
            patients = load_pickle(
                os.path.join(str(Path.home()), ".cache/pyhealth/eicu.data")
            )
        super(eICUBaseDataset, self).__init__(dataset_name="eICU", patients=patients)

    def parse_patients(self):
        """func to parse patient table"""
        patients_df = pd.read_csv(os.path.join(self.root, "patient.csv"))
        visits = {}
        for patient_id, patient_info in tqdm(patients_df.groupby("uniquepid")):
            for visit_id, visit_info in patient_info.groupby("patientunitstayid"):
                # visit statistics
                encounter_time = float(visit_info["hospitaladmitoffset"].values[0])
                duration = float(visit_info["unitdischargeoffset"].values[0])
                mortality = visit_info["unitdischargestatus"].values[0] == "Expire"
                cur_visit = Visit(
                    visit_id,
                    patient_id,
                    encounter_time,
                    duration,
                    mortality,
                )
                visits[visit_id] = cur_visit
        return visits

    def parse_diagnosis(self, visits, patients):
        """func to parse diagnosis table"""
        diagnosis_df = pd.read_csv(os.path.join(self.root, "diagnosis.csv"))
        for visit_id, visit_info in tqdm(diagnosis_df.groupby("patientunitstayid")):
            if visit_id not in visits:
                continue

            # load diagnosis with time info
            cur_diagnosis = []
            for code, time in visit_info[["icd9code", "diagnosisoffset"]].values:
                cur_diagnosis += self.process_nested_code(code, time)

            if len(cur_diagnosis) == 0:
                continue
            # add diagnosis to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].conditions = cur_diagnosis
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].conditions = cur_diagnosis

    def parse_lab(self, visits, patients):
        """func to parse lab table"""
        lab_df = pd.read_csv(os.path.join(self.root, "lab.csv")).iloc[:5000]
        for visit_id, visit_info in tqdm(lab_df.groupby("patientunitstayid")):
            if visit_id not in visits:
                continue

            # load lab with time info
            cur_lab = []
            for code, time in visit_info[["labname", "labresultoffset"]].values:
                cur_lab += self.process_nested_code(code, time)

            if len(cur_lab) == 0:
                continue
            # add labs to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].labs = cur_lab
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].labs = cur_lab

    def parse_medication(self, visits, patients):
        """func to parse medication table"""
        medication_df = pd.read_csv(
            os.path.join(self.root, "medication.csv"), low_memory=False
        )
        for visit_id, visit_info in tqdm(medication_df.groupby("patientunitstayid")):
            if visit_id not in visits:
                continue

            # load drugs with time info
            cur_medication = []
            for code, time in visit_info[["drugname", "drugstartoffset"]].values:
                cur_medication += self.process_nested_code(code, time)

            if len(cur_medication) == 0:
                continue
            # add medication to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].drugs = cur_medication
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].drugs = cur_medication

    def parse_treatment(self, visits, patients):
        """func to parse treatment table"""
        treatment_df = pd.read_csv(os.path.join(self.root, "treatment.csv"))
        for visit_id, visit_info in tqdm(treatment_df.groupby("patientunitstayid")):
            if visit_id not in visits:
                continue

            # load procedures with time info
            cur_treatment = []
            for code, time in visit_info[["treatmentstring", "treatmentoffset"]].values:
                cur_treatment += self.process_nested_code(code, time)

            if len(cur_treatment) == 0:
                continue
            # add medication to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].procedures = cur_treatment
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].procedures = cur_treatment

    def parse_physicalExam(self, visits, patients):
        """func to parse physicalExam table"""
        physicalExam_df = pd.read_csv(os.path.join(self.root, "physicalExam.csv"))
        for visit_id, visit_info in tqdm(physicalExam_df.groupby("patientunitstayid")):
            if visit_id not in visits:
                continue

            # load physicalExam with time info
            cur_physicalExam = []
            for code, time in visit_info[
                ["physicalexampath", "physicalexamoffset"]
            ].values:
                cur_physicalExam += self.process_nested_code(code, time)

            if len(cur_physicalExam) == 0:
                continue
            # add medication to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].physicalExams = cur_physicalExam
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].physicalExams = cur_physicalExam

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
    dataset = eICUBaseDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        files=["conditions", "procedures", "drugs"],
    )
    print(dataset)
    print(type(dataset))
    print(len(dataset))
