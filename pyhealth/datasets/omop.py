import os
import sys
from typing import Optional, List, Dict

import pandas as pd

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")

from pyhealth.data import Event, Visit, Patient, BaseDataset
from tqdm import tqdm


# TODO: add cptevents
# TODO: add drgcodes
# TODO: add noteevents
# TODO: add microbiologyevents
# TODO: add procedureevents_mv


class OMOPDataset(BaseDataset):
    """Base dataset for OMOP dataset.

    The OMOP dataset is a large dataset of de-identified health records of ICU patients.
    The dataset is available at https://www.ohdsi.org/data-standardization/the-common-data-model/.

    We support the following tables:
        - visit_occurrence.csv: defines each visit_occurrence_id in the database, i.e. defines a single patient visit.
        - death.csv: define the death information of the patient.
        - condition_occurrence.csv: contains the condition information of patients' visits.
        - procedure_occurrence.csv: contains the procedure information of patients' visits.
        - drug_exposure.csv: contains the drug information of patients' visits.
        - measurement.csv: contains all laboratory measurements of patients' visits.

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
        super(OMOPDataset, self).__init__(
            dataset_name="OMOP",
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
        patients = self.parse_occurrence(patients)
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

    def parse_occurrence(self, patients) -> Dict[str, Patient]:
        """function to parse visit_occurrence"""
        # read the person table
        person_df = pd.read_csv(
            os.path.join(self.root, "person.csv"),
            dtype={"person_id": str},
            nrows=1000 if self.dev else None,
            sep="\t",
        )
        # read visit occurrence table
        visit_occurrence_df = pd.read_csv(
            os.path.join(self.root, "visit_occurrence.csv"),
            dtype={"person_id": str, "visit_occurrence_id": str},
            sep="\t",
        )
        # read the death table
        death_df = pd.read_csv(
            os.path.join(self.root, "death.csv"),
            sep="\t",
            dtype={"person_id": str},
        )
        person_df = pd.merge(person_df, visit_occurrence_df, on="person_id", how="left")
        df = pd.merge(person_df, death_df, on="person_id", how="left")
        # sort by admission and discharge time
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "visit_start_datetime"], ascending=True
        )
        # load patients
        for p_id, p_info in tqdm(
            df.groupby("person_id"), desc="Parsing person, visit_occurrence and death"
        ):
            patient = Patient(
                patient_id=p_id,
                # TODO: convert to datetime object
                birth_datetime=p_info["day_of_birth"].values[0],
                death_datetime=p_info["death_date"].values[0],
                # TODO: should categorize the gender
                gender=p_info["gender_concept_id"].values[0],
                # TODO: should categorize the ethnicity
                ethnicity=p_info["ethnicity_concept_id"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    # TODO: convert to datetime object
                    encounter_time=v_info["visit_start_datetime"].values[0],
                    discharge_time=v_info["visit_end_datetime"].values[0],
                    # TODO: should categorize the discharge_status
                    discharge_status=v_info["discharge_to_concept_id"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def parse_condition_occurrence(self, patients) -> Dict[str, Patient]:
        """function to parse condition_occurrence table."""
        table = "condition_occurrence"
        col = "condition_concept_id"
        vocabulary = "CONDITION_CONCEPT_ID"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str, "visit_occurrence_id": str, col: str},
            sep="\t",
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
        df = df.dropna(subset=["person_id", "visit_occurrence_id", col])
        # sort by sequence number (i.e., disease priority)
        df = df.sort_values(["person_id", "visit_occurrence_id"], ascending=True)
        # update patients
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["person_id", "visit_occurrence_id"]), desc=f"Parsing {table}"
        ):
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_procedure_occurrence(self, patients) -> Dict[str, Patient]:
        """function to parse procedure_occurrence table."""
        table = "procedure_occurrence"
        col = "procedure_concept_id"
        vocabulary = "PROCEDURE_CONCEPT_ID"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str, "visit_occurrence_id": str, col: str},
            sep="\t",
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
        df = df.dropna(subset=["person_id", "visit_occurrence_id", col])
        # sort by sequence number (i.e., disease priority)
        df = df.sort_values(["person_id", "visit_occurrence_id"], ascending=True)
        # update patients
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["person_id", "visit_occurrence_id"]), desc=f"Parsing {table}"
        ):
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_drug_exposure(self, patients) -> Dict[str, Patient]:
        """function to parse drug_exposure table."""
        table = "drug_exposure"
        col = "drug_concept_id"
        vocabulary = "DRUG_CONCEPT_ID"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str, "visit_occurrence_id": str, col: str},
            sep="\t",
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
        df = df.dropna(subset=["person_id", "visit_occurrence_id", col])
        # sort by sequence number (i.e., disease priority)
        df = df.sort_values(["person_id", "visit_occurrence_id"], ascending=True)
        # update patients
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["person_id", "visit_occurrence_id"]), desc=f"Parsing {table}"
        ):
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients

    def parse_measurement(self, patients) -> Dict[str, Patient]:
        """function to parse measurement table."""
        table = "measurement"
        col = "measurement_concept_id"
        vocabulary = "MEASUREMENT_CONCEPT_ID"
        # read diagnoses table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str, "visit_occurrence_id": str, col: str},
            sep="\t",
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
        df = df.dropna(subset=["person_id", "visit_occurrence_id", col])
        # sort by sequence number (i.e., disease priority)
        df = df.sort_values(["person_id", "visit_occurrence_id"], ascending=True)
        # update patients
        for (p_id, v_id), v_info in tqdm(
            df.groupby(["person_id", "visit_occurrence_id"]), desc=f"Parsing {table}"
        ):
            for code in v_info[col]:
                event = Event(
                    code=code,
                    event_type=table,
                    vocabulary=vocabulary,
                    visit_id=v_id,
                    patient_id=p_id,
                )
                try:
                    patients[p_id].add_event(event)
                except KeyError:
                    continue
        return patients


if __name__ == "__main__":
    dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=[
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "measurement",
        ],
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
