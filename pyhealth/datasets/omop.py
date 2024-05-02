import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime


# TODO: add other tables


class OMOPDataset(BaseEHRDataset):
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
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
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

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> dataset = OMOPDataset(
        ...         root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        ...         tables=["condition_occurrence", "procedure_occurrence", "drug_exposure", "measurement",],
        ...     )
        >>> dataset.stat()
        >>> dataset.info()

    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses person, visit_occurrence, and death tables.

        Will be called in `self.parse_tables()`

        Docs:
            - person: http://ohdsi.github.io/CommonDataModel/cdm53.html#PERSON
            - visit_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#VISIT_OCCURRENCE
            - death: http://ohdsi.github.io/CommonDataModel/cdm53.html#DEATH

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
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
            ["person_id", "visit_occurrence_id", "visit_start_datetime"], ascending=True
        )
        # group by patient
        df_group = df.groupby("person_id")

        # parallel unit of basic informatin (per patient)
        def basic_unit(p_info):
            p_id = p_info["person_id"].values[0]
            birth_y = p_info["year_of_birth"].values[0]
            birth_m = p_info["month_of_birth"].values[0]
            birth_d = p_info["day_of_birth"].values[0]
            birth_date = f"{birth_y}-{birth_m}-{birth_d}"
            patient = Patient(
                patient_id=p_id,
                # no exact time, use 00:00:00
                birth_datetime=strptime(birth_date),
                death_datetime=strptime(p_info["death_date"].values[0]),
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
                    encounter_time=strptime(visit_start_date),
                    discharge_time=strptime(visit_end_date),
                    discharge_status=discharge_status,
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(lambda x: basic_unit(x))
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_condition_occurrence(
        self, patients: Dict[str, Patient]
    ) -> Dict[str, Patient]:
        """Helper function which parses condition_occurrence table.

        Will be called in `self.parse_tables()`

        Docs:
            - condition_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#CONDITION_OCCURRENCE

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "condition_occurrence"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "condition_concept_id": str,
            },
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "condition_concept_id"]
        )
        # sort by condition_start_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "condition_start_datetime"],
            ascending=True,
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of condition occurrence (per patient)
        def condition_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["condition_start_datetime"], v_info["condition_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="CONDITION_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: condition_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_procedure_occurrence(
        self, patients: Dict[str, Patient]
    ) -> Dict[str, Patient]:
        """Helper function which parses procedure_occurrence table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedure_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#PROCEDURE_OCCURRENCE

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "procedure_occurrence"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "procedure_concept_id": str,
            },
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "procedure_concept_id"]
        )
        # sort by procedure_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "procedure_datetime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of procedure occurrence (per patient)
        def procedure_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["procedure_datetime"], v_info["procedure_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="PROCEDURE_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: procedure_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_drug_exposure(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses drug_exposure table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedure_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#DRUG_EXPOSURE

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "drug_exposure"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "drug_concept_id": str,
            },
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(subset=["person_id", "visit_occurrence_id", "drug_concept_id"])
        # sort by drug_exposure_start_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "drug_exposure_start_datetime"],
            ascending=True,
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of drug exposure (per patient)
        def drug_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["drug_exposure_start_datetime"], v_info["drug_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="DRUG_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: drug_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_measurement(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses measurement table.

        Will be called in `self.parse_tables()`

        Docs:
            - measurement: http://ohdsi.github.io/CommonDataModel/cdm53.html#MEASUREMENT

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "measurement"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "measurement_concept_id": str,
            },
            sep="\t",
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "measurement_concept_id"]
        )
        # sort by measurement_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "measurement_datetime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of measurement (per patient)
        def measurement_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["measurement_datetime"], v_info["measurement_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MEASUREMENT_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: measurement_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

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
