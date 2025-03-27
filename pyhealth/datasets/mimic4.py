import os
import time
import re
from datetime import timedelta
from typing import List, Dict, Optional

import pandas as pd
from pandarallel import pandarallel

from pyhealth.data import Event, Patient
from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.datasets.utils import strptime


class MIMIC4Dataset(BaseDataset):
    """Base dataset for MIMIC-IV dataset.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: define a patient's hospital admission, hadm_id.

    We further support the following tables:
        - diagnoses_icd: contains ICD diagnoses (ICD9CM and ICD10CM code)
            for patients.
        - procedures_icd: contains ICD procedures (ICD9PROC and ICD10PROC
            code) for patients.
        - prescriptions: contains medication related order entries (NDC code)
            for patients.
        - labevents: contains laboratory measurements (MIMIC4_ITEMID code)
            for patients
        - clinical notes: contains clinical notes for patients (requires note_root)
        - xrays: contains chest x-ray images for patients (requires cxr_root)

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
        note_root: optional root directory for clinical notes. Default is None (no notes loaded).
        cxr_root: optional root directory for x-ray images. Default is None (no x-rays loaded).
        note_sections: list of note sections to extract. Default is ["Past Medical History"].
        lab_events: list of lab event names to extract. Default is None (all lab events).
        chart_events: list of chart event names to extract. Default is None (all chart events).
        concatenate_notes: whether to concatenate note sections. Default is True.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        dev_patients: number of patients to use in dev mode. Default is 1000.
        refresh_cache: whether to refresh the cache. Default is False.
    """

    def __init__(
        self,
        root: str,
        dev=False,
        tables: List[str] = None,
        note_root: Optional[str] = None,
        cxr_root: Optional[str] = None,
        note_sections: List[str] = ["Past Medical History"],
        lab_events: Optional[List[str]] = None,
        chart_events: Optional[List[str]] = None,
        concatenate_notes: bool = True,
        dev_patients: int = 1000,
    ):
        self.dev = dev
        self.tables = tables if tables is not None else []
        self.dev_patients = dev_patients
        
        # Multimodal features (optional)
        self.note_root = note_root
        self.cxr_root = cxr_root
        self.note_sections = note_sections
        self.lab_events = lab_events
        self.chart_events = chart_events
        self.concatenate_notes = concatenate_notes
        
        # Initialize item IDs if specified
        self.lab_event_ids = set()
        self.chart_event_ids = set()
        self.to_lab_event_names = {}
        self.to_chart_event_names = {}
        
        # If lab_events or chart_events are specified, set up mapping
        if (self.lab_events or self.chart_events) and os.path.exists(os.path.join(root, "hosp", "d_labitems.csv")):
            self._initialize_item_mappings(root)
        
        super().__init__(root)

    def _initialize_item_mappings(self, root):
        """Initialize mappings for lab and chart events if needed."""
        hosp_path = os.path.join(root, "hosp")
        icu_path = os.path.join(root, "icu")
        
        # Read lab and chart event table mappings if needed
        if self.lab_events and os.path.exists(os.path.join(hosp_path, "d_labitems.csv")):
            lab_event_ids_df = pd.read_csv(
                os.path.join(hosp_path, "d_labitems.csv"), 
                dtype={"itemid": str}
            )
            self.lab_event_ids = self._get_item_ids(self.lab_events, lab_event_ids_df)
            self.to_lab_event_names = lab_event_ids_df.set_index("itemid").to_dict()["label"]
            
        if self.chart_events and os.path.exists(os.path.join(icu_path, "d_items.csv")):
            chart_event_ids_df = pd.read_csv(
                os.path.join(icu_path, "d_items.csv"), 
                dtype={"itemid": str}
            )
            self.chart_event_ids = self._get_item_ids(self.chart_events, chart_event_ids_df)
            self.to_chart_event_names = chart_event_ids_df.set_index("itemid").to_dict()["label"]

    def _get_item_ids(self, item_names, item_df):
        """Get item IDs for the specified item names."""
        item_set = set()
        for specific_label in item_names:
            # Handle NA/NaN values by replacing them with an empty string
            item_df['label'] = item_df['label'].str.lower().fillna('')
            if specific_label.lower() in ["ph"]:
                matching_ids = item_df[item_df["label"] == specific_label.lower()]['itemid'].to_list()
            else:
                # Use str.contains correctly and handle NA/NaN values
                matching_ids = item_df[item_df["label"].str.contains(specific_label.lower())]['itemid'].to_list()
            item_set = item_set.union(set(matching_ids))
        return item_set

    def _dev_mode(self, df):
        """Limit dataframe to development mode if enabled."""
        if self.dev:
            unique_patients = df['subject_id'].unique()
            limited_patients = unique_patients[:self.dev_patients]
            return df[df['subject_id'].isin(limited_patients)]
        return df

    def process(self) -> Dict[str, Patient]:
        """Parses the tables and return a dict of patients."""
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
        
        # Process multimodal data if root directories are provided
        if self.note_root and "notes" not in self.tables:
            tic = time.time()
            patients = self.parse_notes(patients)
            print(f"finish parsing notes : {time.time() - tic}s")
            
        if self.cxr_root and "xrays" not in self.tables:
            tic = time.time()
            patients = self.parse_xrays(patients)
            print(f"finish parsing xrays : {time.time() - tic}s")
            
        return patients

    def parse_notes(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Parse clinical notes from the note_root directory.
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            
        Returns:
            The updated patients dict.
        """
        if not self.note_root or not os.path.exists(self.note_root):
            print("Note root directory not found, skipping notes parsing.")
            return patients
            
        print("Reading discharge notes...")
        note_path = os.path.join(self.note_root, "discharge.csv")
        if not os.path.exists(note_path):
            print(f"Note file {note_path} not found, skipping notes parsing.")
            return patients
            
        note_df = pd.read_csv(note_path)
        note_df = note_df.dropna(subset=["subject_id", "text", "charttime"])
        print(f"Read {len(note_df)} note events.")
        note_df = note_df.sort_values(["subject_id", "charttime"], ascending=True)
        
        note_df['subject_id'] = note_df['subject_id'].astype(str)
        note_df = self._dev_mode(note_df)
        
        # Define function to extract sections from notes
        def get_section(text, section_header):
            pattern = re.escape(section_header) + "(.*?)(?=\n[A-Za-z ]+:|$)"
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        
        # Group by patient and process notes
        for patient_id, patient_notes in note_df.groupby("subject_id"):
            if patient_id not in patients:
                continue
                
            for _, row in patient_notes.iterrows():
                text = row['text']
                
                # Process note sections based on configuration
                if self.note_sections[0] == "all":
                    # Add entire note as a single event
                    event = Event(
                        type="clinical_note",
                        timestamp=strptime(row['charttime']),
                        attr_dict={
                            "value": text,
                            "section": "all",
                            "patient_id": patient_id
                        }
                    )
                    patients[patient_id].add_event(event)
                else:
                    if self.concatenate_notes:
                        # Concatenate all specified sections
                        combined_text = ""
                        for section in self.note_sections:
                            section_text = get_section(text.lower(), section.lower())
                            if section_text:
                                combined_text += f"<{section}> {section_text} </{section}> "
                                
                        if combined_text:
                            event = Event(
                                type="clinical_note",
                                timestamp=strptime(row['charttime']),
                                attr_dict={
                                    "value": combined_text.strip(),
                                    "section": "combined",
                                    "patient_id": patient_id
                                }
                            )
                            patients[patient_id].add_event(event)
                    else:
                        # Add each section as a separate event
                        for section in self.note_sections:
                            section_text = get_section(text.lower(), section.lower())
                            if section_text:
                                event = Event(
                                    type="clinical_note",
                                    timestamp=strptime(row['charttime']),
                                    attr_dict={
                                        "value": section_text,
                                        "section": section,
                                        "patient_id": patient_id
                                    }
                                )
                                patients[patient_id].add_event(event)
        
        return patients

    def parse_xrays(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Parse X-ray metadata from the cxr_root directory.
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            
        Returns:
            The updated patients dict.
        """
        if not self.cxr_root or not os.path.exists(self.cxr_root):
            print("CXR root directory not found, skipping X-ray parsing.")
            return patients
            
        print("Reading CXR metadata...")
        metadata_path = os.path.join(self.cxr_root, "mimic-cxr-2.0.0-metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"X-ray metadata file {metadata_path} not found, skipping X-ray parsing.")
            return patients
            
        cxr_jpg_meta_df = pd.read_csv(metadata_path)
        
        # Process datetime columns
        cxr_jpg_meta_df.StudyDate = cxr_jpg_meta_df.StudyDate.astype(str)
        cxr_jpg_meta_df.StudyTime = cxr_jpg_meta_df.StudyTime.astype(str).str.split(".").str[0]
        cxr_jpg_meta_df["StudyDateTime"] = pd.to_datetime(
            cxr_jpg_meta_df.StudyDate + cxr_jpg_meta_df.StudyTime,
            format="%Y%m%d%H%M%S",
            errors="coerce"
        )
        
        # Filter and prepare dataframe
        cxr_df = cxr_jpg_meta_df[["subject_id", "study_id", "dicom_id", "StudyDateTime"]]
        cxr_df = cxr_df.dropna(subset=["subject_id", "dicom_id", "StudyDateTime"])
        cxr_df = cxr_df.sort_values(["subject_id", "StudyDateTime"], ascending=True)
        print(f"Read {len(cxr_df)} x-ray events.")
        
        cxr_df['subject_id'] = cxr_df['subject_id'].astype(str)
        cxr_df = self._dev_mode(cxr_df)
        
        # Process each patient's X-rays
        for patient_id, patient_xrays in cxr_df.groupby("subject_id"):
            if patient_id not in patients:
                continue
                
            for _, row in patient_xrays.iterrows():
                dicom_id = row['dicom_id']
                image_path = os.path.join(self.cxr_root, "images", f"{dicom_id}.jpg")
                
                event = Event(
                    type="xray",
                    timestamp=row['StudyDateTime'],
                    attr_dict={
                        "dicom_id": dicom_id,
                        "study_id": row['study_id'],
                        "image_path": image_path,
                        "patient_id": patient_id
                    }
                )
                patients[patient_id].add_event(event)
                
        return patients

    # Keep all the original methods unchanged
    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables."""
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            attr_dict = {
                # no exact month, day, and time, use Jan 1st, 00:00:00
                "birth_datetime": strptime(str(birth_year)),
                # no exact time, use 00:00:00
                "death_datetime": strptime(p_info["dod"].values[0]),
                "gender": p_info["gender"].values[0],
                "ethnicity": p_info["race"].values[0],
                "anchor_year_group": p_info["anchor_year_group"].values[0],
            }
            patient = Patient(
                patient_id=p_id,
                attr_dict=attr_dict,
            )
            # load admissions
            for v_id, v_info in p_info.groupby("hadm_id"):
                attr_dict = {
                    "visit_id": v_id,
                    "discharge_time": strptime(v_info["dischtime"].values[0]),
                    "discharge_status": v_info["hospital_expire_flag"].values[0],
                }
                event = Event(
                    type="admissions",
                    timestamp=strptime(v_info["admittime"].values[0]),
                    attr_dict=attr_dict,
                )
                # add visit
                patient.add_event(event)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses diagnosis_icd table."""
        table = "diagnoses_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # load admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = df.merge(
            admissions_df[["subject_id", "hadm_id", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="inner"
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            # iterate over each patient and visit
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    attr_dict = {
                        "code": code,
                        "vocabulary": f"ICD{version}CM",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    event = Event(
                        type=table,
                        timestamp=strptime(v_info["dischtime"].values[0]) - timedelta(
                            seconds=1),
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses procedures_icd table."""
        table = "procedures_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # load admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = df.merge(
            admissions_df[["subject_id", "hadm_id", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="inner"
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    attr_dict = {
                        "code": code,
                        "vocabulary": f"ICD{version}PROC",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    event = Event(
                        type=table,
                        timestamp=strptime(v_info["dischtime"].values[0]) - timedelta(
                            seconds=1),
                        attr_dict=attr_dict,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses prescriptions table."""
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["starttime"], v_info["ndc"]):
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
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses labevents table."""
        table = "labevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        
        # Filter by lab_event_ids if specified
        if self.lab_events and self.lab_event_ids:
            df = df[df["itemid"].isin(self.lab_event_ids)]
            
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of labevent (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code, value in zip(v_info["charttime"], v_info["itemid"], v_info.get("valuenum", [None] * len(v_info))):
                    attr_dict = {
                        "code": code,
                        "vocabulary": "MIMIC4_ITEMID",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    
                    # Add value if available
                    if value is not None:
                        attr_dict["value"] = value
                    
                    event = Event(
                        type=table,
                        timestamp=strptime(timestamp),
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_chartevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses chartevents table."""
        table = "chartevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, "icu", f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        
        # Filter by chart_event_ids if specified
        if self.chart_events and self.chart_event_ids:
            df = df[df["itemid"].isin(self.chart_event_ids)]
            
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of chartevent (per patient)
        def chart_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code, value in zip(v_info["charttime"], v_info["itemid"], v_info.get("valuenum", [None] * len(v_info))):
                    attr_dict = {
                        "code": code,
                        "vocabulary": "MIMIC4_CHART_ITEMID",
                        "visit_id": v_id,
                        "patient_id": p_id,
                    }
                    
                    # Add value if available
                    if value is not None:
                        attr_dict["value"] = value
                        
                    event = Event(
                        type=table,
                        timestamp=strptime(timestamp),
                        attr_dict=attr_dict,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: chart_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def _add_events_to_patient_dict(
        self,
        patient_dict: Dict[str, Patient],
        group_df: pd.DataFrame,
    ) -> Dict[str, Patient]:
        """Helper function which adds the events column of a df.groupby object to the patient dict."""
        for _, events in group_df.items():
            for event in events:
                patient_dict = self._add_event_to_patient_dict(patient_dict, event)
        return patient_dict

    @staticmethod
    def _add_event_to_patient_dict(
        patient_dict: Dict[str, Patient],
        event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict."""
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
        num_visits = [len(p.get_events_by_type("admissions")) for p in
                      self.patients.values()]
        lines.append(f"\t- Number of visits: {sum(num_visits)}")
        lines.append(
            f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        )
        for table in self.tables:
            num_events = [
                len(p.get_events_by_type(table)) for p in self.patients.values()
            ]
            lines.append(
                f"\t- Number of events per patient in {table}: "
                f"{sum(num_events) / len(num_events):.4f}"
            )
            
        # Add stats for multimodal data if available
        if hasattr(self, 'note_root') and self.note_root:
            num_notes = [
                len(p.get_events_by_type("clinical_note")) for p in self.patients.values()
            ]
            if sum(num_notes) > 0:
                lines.append(
                    f"\t- Number of clinical notes per patient: "
                    f"{sum(num_notes) / len(num_notes):.4f}"
                )
                
        if hasattr(self, 'cxr_root') and self.cxr_root:
            num_xrays = [
                len(p.get_events_by_type("xray")) for p in self.patients.values()
            ]
            if sum(num_xrays) > 0:
                lines.append(
                    f"\t- Number of X-rays per patient: "
                    f"{sum(num_xrays) / len(num_xrays):.4f}"
                )
                
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)
    


def main():
    # make sure to change this to your own datasets
    mimic_cxr_path = "/srv/local/data/MIMIC-CXR"
    mimic_iv_path = "/srv/local/data/MIMIC-IV/2.0/hosp"
    mimic_note_directory = "/srv/local/data/MIMIC-IV/2.0/note"
    # Original usage (unchanged)
    dataset = MIMIC4Dataset(
        root=mimic_iv_path,
        tables=["diagnoses_icd", "procedures_icd"],
        dev=True
    )
    print("Finished Processing Base Codes Dataset")
    # With multimodality features
    dataset_multimodal = MIMIC4Dataset(
        root=mimic_iv_path,
        tables=["diagnoses_icd", "procedures_icd"],
        note_root=mimic_note_directory,
        cxr_root=mimic_cxr_path,
        note_sections=["all"], # "Past Medical History", "Medications on Admission"
        lab_events=["Hematocrit", "Platelet", "WBC"],
        chart_events=["Heart Rate", "Blood Pressure"],
        dev=True
    )
    print("Finshed Processing True Multimodal Dataset")


if __name__ == "__main__":
    main()