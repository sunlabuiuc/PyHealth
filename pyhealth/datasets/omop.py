import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
from pyhealth.data import Event, Visit, Patient, BaseDataset
from pyhealth.utils import create_directory, dump_pickle, load_pickle
from tqdm import tqdm


class OMOPBaseDataset(BaseDataset):
    """Base dataset for OMOP based database
    1. it contains a superset of information used for all relevant tasks
    2. it will be inputted into a task module for further data cleaning
    """

    def __init__(
        self, root, files=["conditions", "procedures", "prescriptions", "labs"]
    ):
        """
        INPUT
            - root: root directory of the dataset
            - files: list of files to be parsed (no order), i.e., conditions, procedures, drugs, labs, physicalExams
        OUTPUT
            - patients: a <pyhealth.data.Patient> object
        """
        self.root = root
        self.files = files
        self.all_support_files = ["conditions", "procedures", "drugs", "labs"]

        if not os.path.exists(
            os.path.join(str(Path.home()), ".cache/pyhealth/omop.data")
        ):
            # get visit-level static features
            visits = self.parse_visit_occurrence()
            patients = {}

            print("structured all patients and visits")
            # process based on self.files
            if "conditions" in self.files:
                self.parse_conditions(visits, patients)
                print("processed conditions")
            if "procedures" in self.files:
                self.parse_procedures(visits, patients)
                print("processed procedures")
            if "drugs" in self.files:
                self.parse_prescriptions(visits, patients)
                print("processed drugs")
            if "labs" in self.files:
                self.parse_labs(visits, patients)
                print("processed labs")

            create_directory(os.path.join(str(Path.home()), ".cache/pyhealth"))
            dump_pickle(
                patients, os.path.join(str(Path.home()), ".cache/pyhealth/omop.data")
            )
        else:
            patients = load_pickle(
                os.path.join(str(Path.home()), ".cache/pyhealth/omop.data")
            )
        super(OMOPBaseDataset, self).__init__(dataset_name="OMOP", patients=patients)

    def parse_visit_occurrence(self):
        """func to parse patient table"""
        visit_occurrence_df = pd.read_csv(
            os.path.join(self.root, "visit_occurrence.csv"),
            sep="\t",
            dtype={"person_id": str, "visit_occurrence_id": str},
        )
        death_df = pd.read_csv(
            os.path.join(self.root, "death_filled.csv"),
            sep="\t",
            dtype={"person_id": str},
        )
        visit_occurrence_df = pd.merge(
            visit_occurrence_df, death_df, on="person_id", how="left"
        )
        visits = {}
        for visit_id, visit_info in visit_occurrence_df.groupby("visit_occurrence_id"):
            # visit statistics
            # from datetime to seconds
            patient_id = str(visit_info["person_id"].values[0])
            encounter_time = float(visit_info["visit_start_datetime"].values[0])
            duration = float(
                visit_info["visit_end_datetime"].values[0]
                - visit_info["visit_start_datetime"].values[0]
            )
            if visit_info["death_date"].values[0] == visit_info["death_date"].values[0]:
                mortality = (
                    visit_info["visit_start_date"].values[0]
                    <= visit_info["death_date"].values[0]
                    <= visit_info["visit_end_date"].values[0]
                )
            else:
                mortality = False
            # mortality = False
            cur_visit = Visit(
                visit_id,
                patient_id,
                encounter_time,
                duration,
                mortality,
            )
            visits[visit_id] = cur_visit
        return visits

    def parse_conditions(self, visits, patients):
        """func to parse condition table"""
        condition_df = pd.read_csv(
            os.path.join(self.root, "condition_occurrence.csv"),
            sep="\t",
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "condition_concept_id": str,
            },
        )
        for visit_id, visit_info in tqdm(condition_df.groupby("visit_occurrence_id")):
            if visit_id not in visits:
                continue

            # load condition with time info
            cur_condition = []
            for code, time_datetime, time_date in visit_info[
                [
                    "condition_concept_id",
                    "condition_start_datetime",
                    "condition_start_date",
                ]
            ].values:
                if time_date == time_date:
                    cur_condition += self.process_nested_code(code, time_date)
                else:
                    cur_condition += self.process_nested_code(code, time_datetime)

            if len(cur_condition) == 0:
                continue
            # add condition to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].conditions = cur_condition
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].conditions = cur_condition

    def parse_procedures(self, visits, patients):
        """func to parse procedure table"""
        procedure_df = pd.read_csv(
            os.path.join(self.root, "procedure_occurrence.csv"),
            sep="\t",
            dtype={
                "person_id": str,
                "visit_procedure_id": str,
                "procedure_concept_id": str,
            },
        )
        for visit_id, visit_info in tqdm(procedure_df.groupby("visit_occurrence_id")):
            if visit_id not in visits:
                continue

            # load procedure with time info
            cur_procedure = []
            for code, time_datetime, time_date in visit_info[
                [
                    "procedure_concept_id",
                    "procedure_start_datetime",
                    "procedure_start_date",
                ]
            ].values:
                if time_date == time_date:
                    cur_procedure += self.process_nested_code(code, time_date)
                else:
                    cur_procedure += self.process_nested_code(code, time_datetime)

            if len(cur_procedure) == 0:
                continue
            # add procedure to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].procedures = cur_procedure
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].procedures = cur_procedure

    def parse_prescriptions(self, visits, patients):
        """func to parse condition table"""
        drugs_df = pd.read_csv(
            os.path.join(self.root, "drug_exposure.csv"),
            sep="\t",
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "drug_concept_id": str,
            },
        )
        for visit_id, visit_info in tqdm(drugs_df.groupby("visit_occurrence_id")):
            if visit_id not in visits:
                continue

            # load drugs with time info
            cur_drugs = []
            for code, time_datetime, time_date in visit_info[
                [
                    "drug_concept_id",
                    "drug_exposure_start_datetime",
                    "drug_exposure_start_date",
                ]
            ].values:
                if time_date == time_date:
                    cur_drugs += self.process_nested_code(code, time_date)
                else:
                    cur_drugs += self.process_nested_code(code, time_datetime)

            if len(cur_drugs) == 0:
                continue
            # add drugs to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].drugs = cur_drugs
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].drugs = cur_drugs

    def parse_labs(self, visits, patients):
        """func to parse condition table"""
        labs_df = pd.read_csv(
            os.path.join(self.root, "measurement_filled.csv"),
            sep="\t",
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "measurement_concept_id": str,
            },
        )
        for visit_id, visit_info in tqdm(labs_df.groupby("visit_occurrence_id")):
            if visit_id not in visits:
                continue

            # load labs with time info
            cur_labs = []
            for code, time_datetime, time_date in visit_info[
                ["measurement_concept_id", "measurement_datetime", "measurement_date"]
            ].values:
                if time_date == time_date:
                    cur_labs += self.process_nested_code(code, time_date)
                else:
                    cur_labs += self.process_nested_code(code, time_datetime)

            if len(cur_labs) == 0:
                continue
            # add labs to patients dict
            patient_id = visits[visit_id].patient_id
            if patient_id not in patients:  # register patient if not exist
                patients[patient_id] = Patient(patient_id)
            if (
                visit_id not in patients[patient_id].visits
            ):  # register visit if not exist
                visits[visit_id].labs = cur_labs
                patients[patient_id].visits[visit_id] = visits[visit_id]
            else:
                patients[patient_id].visits[visit_id].labs = cur_labs

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
    dataset = OMOPBaseDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        files=["conditions", "procedures", "drugs"],
    )
    print(dataset)
    print(type(dataset))
    print(len(dataset))
