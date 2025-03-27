import os
import re
import time
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from pyhealth.data.data import Event, Patient
from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.datasets.utils import strptime


# Function to extract specific sections from clinical notes
def get_section(text, section_header="Past Medical History"):
    """Extract a specific section from clinical notes text using regex.
    
    Args:
        text: The full text of the clinical note
        section_header: The section header to extract (e.g., "Past Medical History")
        
    Returns:
        The extracted section text or None if the section is not found
    """
    pattern = re.escape(section_header) + "(.*?)(?=\n[A-Za-z ]+:|$)"

    # Search for the pattern in the text
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class MIMIC3Dataset(BaseDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, SUBJECT_ID.
        - ADMISSIONS: defines a patient's hospital admission, HADM_ID.

    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (NDC code)
            for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_ITEMID code)
            for patients
        - NOTEEVENTS: contains discharge summaries and other clinical notes

    Args:
        root: root directory of the raw data (should contain many csv files).
        dataset_name: name of the dataset.
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        note_sections: List of sections to extract from clinical notes (e.g., 
            ["Past Medical History", "Medications on Admission"]).
            If ["all"], the entire note will be used.
            Default is None, meaning the entire note will be used.
        concatenate_notes: If True, all specified sections will be concatenated into
            a single event. If False, each section will be a separate event.
            Default is True.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        tables: List[str] = None,
        note_sections: Optional[List[str]] = None,
        concatenate_notes: bool = True,
        dev: bool = False,
        **kwargs
    ):
        self.note_sections = note_sections or ["all"]
        self.concatenate_notes = concatenate_notes
        self.dev = dev
        self.tables = tables or []
        super().__init__(root=root, dataset_name=dataset_name, **kwargs)

    def process(self) -> Dict[str, Patient]:
        """Parses the tables in `self.tables` and return a dict of patients.

        This function will first call `self.parse_basic_info()` to parse the
        basic patient information, and then call `self.parse_[table_name]()` to
        parse the table with name `table_name`.

        Returns:
           A dict mapping patient_id to `Patient` object.
        """
        pandarallel.initialize(progress_bar=False)

        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process basic information (e.g., patients and visits)
        tic = time.time()
        patients = self.parse_basic_info(patients)
        print(
            "finish basic patient information parsing : {}s".format(time.time() - tic)
        )
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                tic = time.time()
                patients = getattr(self, f"parse_{table.lower()}")(patients)
                print(f"finish parsing {table} : {time.time() - tic}s")
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # Create patient object with basic information
            attr_dict = {
                "birth_datetime": strptime(p_info["DOB"].values[0]),
                "death_datetime": strptime(p_info["DOD_HOSP"].values[0]),
                "gender": p_info["GENDER"].values[0],
                "ethnicity": p_info["ETHNICITY"].values[0],
            }
            patient = Patient(patient_id=p_id, attr_dict=attr_dict)
            
            # Add admission events
            for idx, row in p_info.iterrows():
                admission_attr = {
                    "visit_id": row["HADM_ID"],
                    "discharge_time": strptime(row["DISCHTIME"]),
                    "discharge_status": row["HOSPITAL_EXPIRE_FLAG"],
                    "insurance": row["INSURANCE"],
                    "language": row["LANGUAGE"],
                    "religion": row["RELIGION"],
                    "marital_status": row["MARITAL_STATUS"],
                    "patient_id": p_id
                }
                event = Event(
                    type="admissions",
                    timestamp=strptime(row["ADMITTIME"]),
                    attr_dict=admission_attr
                )
                patient.add_event(event)
            
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses DIAGNOSES_ICD table.

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "DIAGNOSES_ICD"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    attr_dict = {
                        "code": code,
                        "vocabulary": "ICD9CM",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    event = Event(
                        type=table,
                        timestamp=None,  # MIMIC-III does not provide specific timestamps
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PROCEDURES_ICD"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    attr_dict = {
                        "code": code,
                        "vocabulary": "ICD9PROC",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    event = Event(
                        type=table,
                        timestamp=None,  # MIMIC-III does not provide specific timestamps
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["STARTDATE"], v_info["NDC"]):
                    attr_dict = {
                        "code": code,
                        "vocabulary": "NDC",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    event = Event(
                        type=table,
                        timestamp=strptime(timestamp),
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for lab (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["CHARTTIME"], v_info["ITEMID"]):
                    attr_dict = {
                        "code": code,
                        "vocabulary": "MIMIC3_ITEMID",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    event = Event(
                        type=table,
                        timestamp=strptime(timestamp),
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_noteevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses NOTEEVENTS table with support for section extraction.

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "NOTEEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "TEXT"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for note (per patient)
        def note_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for _, row in v_info.iterrows():
                    full_text = row["TEXT"]
                    
                    # Case 1: Use the entire note
                    if self.note_sections[0] == "all":
                        attr_dict = {
                            "code": full_text,
                            "vocabulary": "note",
                            "visit_id": v_id,
                            "patient_id": p_id,
                        }
                        event = Event(
                            type=table,
                            timestamp=strptime(row["CHARTTIME"]),
                            attr_dict=attr_dict,
                        )
                        events.append(event)
                    
                    # Case 2: Extract and concatenate specified sections
                    elif self.concatenate_notes:
                        combined_text = ""
                        for section in self.note_sections:
                            section_text = get_section(full_text, section)
                            if section_text:
                                combined_text += f"<{section}> {section_text} "
                        
                        if combined_text:  # Only add if we found at least one section
                            attr_dict = {
                                "code": combined_text.strip(),
                                "vocabulary": "note",
                                "visit_id": v_id,
                                "patient_id": p_id,
                            }
                            event = Event(
                                type=table,
                                timestamp=strptime(row["CHARTTIME"]),
                                attr_dict=attr_dict,
                            )
                            events.append(event)
                    
                    # Case 3: Create separate events for each section
                    else:
                        for section in self.note_sections:
                            section_text = get_section(full_text, section)
                            if section_text:
                                attr_dict = {
                                    "code": section_text,
                                    "vocabulary": f"note_{section.replace(' ', '_').lower()}",
                                    "visit_id": v_id,
                                    "patient_id": p_id,
                                }
                                event = Event(
                                    type=f"{table}_{section.replace(' ', '_').lower()}",
                                    timestamp=strptime(row["CHARTTIME"]),
                                    attr_dict=attr_dict,
                                )
                                events.append(event)
            
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: note_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def _add_events_to_patient_dict(
        self,
        patient_dict: Dict[str, Patient],
        group_df: pd.DataFrame,
    ) -> Dict[str, Patient]:
        """Helper function which adds the events column of a df.groupby object to the patient dict.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            group_df: a df.groupby object with patient_id as index and events as values.

        Returns:
            The updated patient dict.
        """
        for pat_id, events in group_df.items():
            for event in events:
                self._add_event_to_patient_dict(patient_dict, event)
        return patient_dict

    @staticmethod
    def _add_event_to_patient_dict(
        patient_dict: Dict[str, Patient],
        event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            event: an event to be added to the patient dict.

        Returns:
            The updated patient dict.
        """
        patient_id = event.attr_dict["patient_id"]
        try:
            patient_dict[patient_id].add_event(event)
        except KeyError:
            pass
        return patient_dict

    def stat(self) -> str:
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of patients: {len(self.patients)}")
        
        # Count admission events to estimate number of visits
        num_visits = []
        for p in self.patients.values():
            visits = p.get_events_by_type("admissions")
            num_visits.append(len(visits))
        
        lines.append(f"\t- Number of visits: {sum(num_visits)}")
        lines.append(
            f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        )
        
        # Count events by type
        for table in self.tables:
            table_lower = table.lower()
            num_events = []
            for p in self.patients.values():
                events = p.get_events_by_type(table_lower)
                num_events.append(len(events))
            if sum(num_events) > 0:
                lines.append(
                    f"\t- Number of events in {table}: {sum(num_events)}"
                )
                lines.append(
                    f"\t- Average events per patient in {table}: {sum(num_events) / len(num_events):.4f}"
                )
        
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)
    

def main():
    root = "/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4"
    dataset = MIMIC3Dataset(
        root=root,
        dataset_name="mimic3",  
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
            "LABEVENTS",
            "NOTEEVENTS"
        ],
        code_mapping={"NDC": "ATC"},
        dev=True,
    )
    dataset.stat()

if __name__ == "__main__":
    main()