import os
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset


# TODO: add other tables


class OMOPDataset(BaseDataset):
    """Base dataset for OMOP dataset.

    The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM)
        is an open community data standard, designed to standardize the structure
        and content of observational data and to enable efficient analyses that
        can produce reliable evidence.

    See: https://www.ohdsi.org/data-standardization/the-common-data-model/.

    The basic information is stored in the following tables:
        - person: contains records that uniquely identify each person or patient,
            and some demographic information.
        - visit_occurrence: contains info for how a patient engages with the
            healthcare system for a duration of time.
        - death: contains info for how and when a patient dies.

    We further support the following tables:
        - condition_occurrence.csv: contains the condition information
            (CONDITION_CONCEPT_ID code) of patients' visits.
        - procedure_occurrence.csv: contains the procedure information
            (PROCEDURE_CONCEPT_ID code) of patients' visits.
        - drug_exposure.csv: contains the drug information (DRUG_CONCEPT_ID code)
            of patients' visits.
        - measurement.csv: contains all laboratory measurements
            (MEASUREMENT_CONCEPT_ID code) of patients' visits.

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded. Must be a subset of the
            following tables: condition_occurrence, procedure_occurrence,
            drug_exposure, measurement.
        code_mapping: Optional[Dict[str, str]], key is the source code vocabulary and
            value is the target code vocabulary (e.g., {"ICD9CM": "CCSCM"}).
            Default is empty dict, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: bool, whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.
    """

    def __init__(
            self,
            root: str,
            tables: List[str],
            code_mapping: Optional[Dict[str, str]] = None,
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

    def _parse_tables(self) -> Dict[str, Patient]:
        """This function overrides the _parse_tables() function in BaseDataset.

        It parses the corresponding tables and creates a dict of patients which
            will be cached later.

        Returns:
            patients: a dictionary of Patient objects indexed by patient_id.
        """
        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process person, visit_occurrence, and death tables
        patients = self._parse_basic_info(patients)
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                patients = getattr(self, f"_parse_{table.lower()}")(patients)
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    def _parse_basic_info(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses person, visit_occurrence, and death tables.

        Will be called by _parse_tables().

        Docs:
            - person: http://ohdsi.github.io/CommonDataModel/cdm53.html#PERSON
            - visit_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#VISIT_OCCURRENCE
            - death: http://ohdsi.github.io/CommonDataModel/cdm53.html#DEATH
        """
        # read person table
        person_df = pd.read_csv(
            os.path.join(self.root, "person.csv"),
            dtype={"person_id": str},
            nrows=1000 if self.dev else None,
            sep="\t",
        )
        # read visit_occurrence table
        visit_occurrence_df = pd.read_csv(
            os.path.join(self.root, "visit_occurrence.csv"),
            dtype={"person_id": str, "visit_occurrence_id": str},
            sep="\t",
        )
        # read death table
        death_df = pd.read_csv(
            os.path.join(self.root, "death.csv"),
            sep="\t",
            dtype={"person_id": str},
        )
        # merge
        df = pd.merge(person_df, visit_occurrence_df, on="person_id", how="left")
        df = pd.merge(df, death_df, on="person_id", how="left")
        # sort by admission time
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "visit_start_datetime"],
            ascending=True
        )
        # group by patient
        df_group = df.groupby("person_id")
        # load patients
        for p_id, p_info in tqdm(
                df_group, desc="Parsing person, visit_occurrence and death"
        ):
            birth_y = p_info["year_of_birth"].values[0]
            birth_m = p_info["month_of_birth"].values[0]
            birth_d = p_info["day_of_birth"].values[0]
            birth_date = f"{birth_y}-{birth_m}-{birth_d}"
            patient = Patient(
                patient_id=p_id,
                # no exact time, use 00:00:00
                birth_datetime=self._strptime(birth_date, "%Y-%m-%d"),
                death_datetime=self._strptime(p_info["death_date"].values[0],
                                              "%Y-%m-%d"),
                gender=p_info["gender_concept_id"].values[0],
                ethnicity=p_info["race_concept_id"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                death_date = v_info["death_date"].values[0]
                visit_start_date = v_info["visit_start_date"].values[0]
                visit_end_date = v_info["visit_end_date"].values[0]
                if pd.isna(death_date):
                    discharge_status = 0
                elif death_date > visit_end_date:
                    discharge_status = 0
                else:
                    discharge_status = 1
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=self._strptime(visit_start_date, "%Y-%m-%d"),
                    discharge_time=self._strptime(visit_end_date, "%Y-%m-%d"),
                    discharge_status=discharge_status,
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

    def _parse_condition_occurrence(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses condition_occurrence table.

        Will be called by _parse_tables().

        Docs:
            - condition_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#CONDITION_OCCURRENCE
        """
        table = "condition_occurrence"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str,
                   "visit_occurrence_id": str,
                   "condition_concept_id": str},
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "condition_concept_id"]
        )
        # sort by condition_start_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "condition_start_datetime"],
            ascending=True
        )
        # group by patient and visit
        group_df = df.groupby(["person_id", "visit_occurrence_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["condition_start_datetime"],
                                       v_info["condition_concept_id"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="CONDITION_CONCEPT_ID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_procedure_occurrence(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses procedure_occurrence table.

        Will be called by _parse_tables().

        Docs:
            - procedure_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#PROCEDURE_OCCURRENCE
        """
        table = "procedure_occurrence"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str,
                   "visit_occurrence_id": str,
                   "procedure_concept_id": str},
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "procedure_concept_id"]
        )
        # sort by procedure_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "procedure_datetime"],
            ascending=True
        )
        # group by patient and visit
        group_df = df.groupby(["person_id", "visit_occurrence_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["procedure_datetime"],
                                       v_info["procedure_concept_id"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="PROCEDURE_CONCEPT_ID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_drug_exposure(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses drug_exposure table."""
        table = "drug_exposure"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str,
                   "visit_occurrence_id": str,
                   "drug_concept_id": str},
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(subset=["person_id", "visit_occurrence_id", "drug_concept_id"])
        # sort by drug_exposure_start_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "drug_exposure_start_datetime"],
            ascending=True
        )
        # group by patient and visit
        group_df = df.groupby(["person_id", "visit_occurrence_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["drug_exposure_start_datetime"],
                                       v_info["drug_concept_id"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="DRUG_CONCEPT_ID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_measurement(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses measurement table.

        Will be called by _parse_tables().

        Docs:
            - measurement: http://ohdsi.github.io/CommonDataModel/cdm53.html#MEASUREMENT
        """
        table = "measurement"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"person_id": str,
                   "visit_occurrence_id": str,
                   "measurement_concept_id": str},
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "measurement_concept_id"]
        )
        # sort by measurement_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "measurement_datetime"],
            ascending=True
        )
        # group by patient and visit
        group_df = df.groupby(["person_id", "visit_occurrence_id"])
        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(group_df, desc=f"Parsing {table}"):
            for timestamp, code in zip(v_info["measurement_datetime"],
                                       v_info["measurement_concept_id"]):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="MEASUREMENT_CONCEPT_ID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=self._strptime(timestamp),
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
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
        dev=False,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
