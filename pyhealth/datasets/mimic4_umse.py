from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=8)
import logging
import re
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime
from typing import Optional, Dict
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union



def get_img_path(images_path, dicom_id):
    img_path = f"{dicom_id}.jpg"
    img_path = os.path.join(images_path, img_path)
    return img_path

def get_img(path, transform):
    img = Image.open(path)
    img_tensor = transform(img)
    return img_tensor

def get_section(text, section_header="Past Medical History"):
    pattern =  re.escape(section_header) + "(.*?)(?=\n[A-Za-z ]+:|$)"

    # Search for the pattern in the text
    match = re.search(pattern, text, flags=re.DOTALL)
    past_medical_history_section = None
    if match:
        past_medical_history_section = match.group(1)
        # print(past_medical_history_section)
    else:
        print(f"Section '{section_header}:' not found.")
    return past_medical_history_section[1:] # for the colon


# UMSE includes explicitly, the following:
# 1. Continuous Time Series Vital Signal and Lab Measurement Data
# 2. Clinical Notes
# 3. X-ray Images
# each list contains an EventUMSE (time, value, feature_type)


# Set of all observable tuples for a specific data type (e.g. lab measurements, vital signals, xrays, etc.)
# patient_id is the patient's id for the set of observables
# data_type is the type of data (e.g. lab measurements, vital signals, xrays, etc.)
# observables is a list of EventUMSE
class SetUMSE:
    def __init__(self, patient_id, data_type, observables):
        self.patient_id = patient_id
        self.data_type = data_type
        self.observables = observables

# EventUMSE = namedtuple('EventUMSE', ['time', 'value', 'feature_type'])
class EventUMSE:
    def __init__(self, time, feature_type, value):
        self.time = time
        self.feature_type = feature_type
        self.value = value

class PatientUMSE:
    def __init__(self, patient_id: str, 
                 notes : SetUMSE = None, 
                 lab : SetUMSE = None, 
                 chart : SetUMSE = None,
                 x_rays : SetUMSE = None, 
                 birth_datetime: Optional[datetime] = None,
                 death_datetime: Optional[datetime] = None,
                 initial_admittime : Optional[datetime] = None,
                 final_discharge_time : Optional[datetime] = None,
                 gender=None,
                 ethnicity=None,
                 age=None,
                 outcome_events=None):
        
        self.patient_id = patient_id
        self.birth_datetime = birth_datetime
        self.death_datetime = death_datetime
        self.admittime = initial_admittime
        self.discharge_datetime = final_discharge_time
        self.gender = gender
        self.ethnicity = ethnicity
        self.notes = notes
        self.lab = lab
        self.chart = chart
        self.x_rays = x_rays
        self.age = age
        self.outcome_events = outcome_events
        self.logger = logging.getLogger(__name__)

    def info(self):
        print(f"Patient ID: {self.patient_id}")
        print(f"Birth Date: {self.birth_datetime}")
        print(f"Death Date: {self.death_datetime}")
        print(f"Age: {self.age}")
        print(f"Gender:{self.gender}")
        print(f"Ethnicity: {self.ethnicity}")
        print("First Admittime:", self.admittime)
        print("Final Discharge Time:", self.discharge_datetime)
        print("Total Number of Notes:", len(self.notes.observables))
        print("Total Number of X-rays:", len(self.x_rays.observables))
        print("Total Number of Lab Measurements:", len(self.lab.observables))
        print("Total Number of Chart Measurements:", len(self.chart.observables))

# This is where the dirty stuff is going to happen.
# Time-Series 
# Hematocrit, Platelet, WBC, Bilirubin, pH, bicarbonate, Creatinine, Lactate, Potassium, and Sodium - Lab Events Used
# heart rate, respiration rate, diastolic and systolic blood pressure, temperature, and pulse oximetry. - ChartEvents Used
# Note Sections to Extract
# Past Medical History, Medications on Admission, Chief Medical Complaint (may or may not exist)
class MIMIC4UMSE(BaseEHRDataset):

    def dev_mode(self, df):
        if self.dev:
            unique_patients = df['subject_id'].unique()
            limited_patients = unique_patients[:self.dev_patients]
            limited_df = df[df['subject_id'].isin(limited_patients)]
            return limited_df
        else: 
            return df

    def get_item_ids(self, item_names, item_df):
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

    def __init__(
        self,
        root: str,
        cxr_root : str, 
        note_root : str,
        dataset_name: Optional[str] = None,
        note_sections = ["Past Medical History", "Medications on Admission", "Chief Medical Complaint"], 
        lab_events = ["Hematocrit", "Platelet", "WBC", "Bilirubin", "pH", "bicarbonate", "Creatinine", "Lactate", "Potassium", "Sodium"], 
        chart_events = ["Heart Rate", "respiratory rate", "blood pressure", "temperature", "pulseoxymetry"],
        outcome_events = ["mortality", "intubation", "vasopressor", "icd"],
        dev : bool = False,
        use_parquet : bool = False, 
        use_relative_time = False,
        time_unit = "day",
        exclude_negative_time = False, # set if you want to exclude events that exist before their ED or ICU admission.
        concatenate_notes = True, # Set if you want note sections to be separate events or all of them to be just one event 
        dev_patients : int = 1000, # number of patients to use in dev mode
        **kwargs,
    ):
        if dataset_name is None:
            dataset_name = self.__class__.__name__
        self.root = root
        self.cxr_root = cxr_root
        self.note_root = note_root
        self.hosp_path = os.path.join(self.root, "hosp")
        self.icu_path = os.path.join(self.root, "icu")
        self.dataset_name = dataset_name
        
        # Items to Extract
        self.note_sections = note_sections
        self.lab_events = lab_events
        self.chart_events = chart_events
        self.outcome_events = outcome_events
        
        # Dataset Processing Details
        self.image_transform = transforms.Compose([transforms.ToTensor()])
        self.dev = dev
        self.dev_patients = dev_patients
        self.use_parquet = use_parquet
        self.time_unit = time_unit 
        self.exclude_negative_time = exclude_negative_time
        self.concatenate_notes = concatenate_notes

        # read lab and chart event table mappings
        lab_event_ids_df = pd.read_csv(os.path.join(self.hosp_path, "d_labitems.csv"), dtype={"itemid": str})
        chart_event_ids_df = pd.read_csv(os.path.join(self.icu_path, "d_items.csv"), dtype={"itemid": str})

        # sets of lab event ids that we want to keep measurements of.
        self.lab_event_ids = self.get_item_ids(lab_events, lab_event_ids_df)
        self.chart_event_ids = self.get_item_ids(chart_events, chart_event_ids_df)

        # Convert from id to label
        self.to_lab_event_names = lab_event_ids_df.set_index("itemid").to_dict()["label"]
        self.to_chart_event_names = chart_event_ids_df.set_index("itemid").to_dict()["label"]
        
        self.logger.debug(f"Processing {self.dataset_name} base dataset...")

        self.patients = self.process(**kwargs)
        if use_relative_time:
            self.patients = self.set_patient_occurence_time(self.time_unit)

        # + 1 for the xrays
        self.num_feature_types = len(self.lab_event_ids) + len(self.chart_event_ids) + len(self.note_sections) + 1

    def process(self, **kwargs) -> Dict[str, PatientUMSE]:
        patients = dict()
        
        # load patients info
        patients = self.parse_basic_info(patients)
        patients = self.parse_lab_events(patients)
        patients = self.parse_chart_events(patients)
        patients = self.parse_notes(patients)
        patients = self.parse_xrays(patients)
    
        return patients
    

    def add_observations_to_patients(self, patients: Dict[str, PatientUMSE], modality, df) -> Dict[str, PatientUMSE]:
        for pid, set_umse in df.items():
            assert pid == set_umse.patient_id
            patient_id = set_umse.patient_id
            if patient_id in patients:
                if modality == "lab":
                    patients[patient_id].lab = set_umse
                elif modality == "chart":
                    patients[patient_id].chart = set_umse
                elif modality == "note":
                    patients[patient_id].notes = set_umse
                elif modality == "xray":
                    patients[patient_id].x_rays = set_umse
                else:
                    AssertionError("Modality not recognized!")

        return patients

    def parse_basic_info(self, patients: Dict[str, PatientUMSE]) -> Dict[str, PatientUMSE]:
        def process_patient(self, pid, p_info):
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            admit_date = strptime(p_info["admittime"].values[0])
            discharge_date = strptime(p_info["dischtime"].values[-1]) # final
            p_outcome_events = {}
        
            if "mortality" in self.outcome_events:
                expired = int(1 in p_info["hospital_expire_flag"].values) # if patient died in hospital, label is 1, else 0 
                p_outcome_events["mortality"] = expired 
            
            # load observables
            patient = PatientUMSE(
                patient_id=pid,
                notes=SetUMSE(pid, "note", []),
                lab=SetUMSE(pid, "lab", []),
                chart=SetUMSE(pid, "chart", []),
                x_rays=SetUMSE(pid, "xray", []),
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                initial_admittime=admit_date,
                final_discharge_time=discharge_date,
                age=anchor_age,
                outcome_events=p_outcome_events
            )
            
            return patient
        
        print("Reading Patients and Admissions!")
        patients_df = pd.read_csv(
            os.path.join(self.hosp_path, "patients.csv"),
            dtype={"subject_id": str},
            nrows=self.dev_patients if self.dev else None
        )
        print("Total Number of Patient Records:", len(patients_df))
        admissions_df = pd.read_csv(
            os.path.join(self.hosp_path, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str}
        )

        # Now merge DataFrames
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        df = df.dropna(subset=["subject_id", "admittime", "dischtime"])
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)

        # group by patient
        df_group = df.groupby("subject_id")
        print("Parsing Basic Info!")
        df_group = df_group.parallel_apply(
            lambda x: process_patient(self, x.subject_id.unique()[0], x)
            )
        
        for pid, pat in df_group.items():
            patients[pid] = pat

        return patients
    
    # Get all specified lab events for each patient
    def parse_lab_events(self, patients: Dict[str, PatientUMSE]) -> Dict[str, PatientUMSE]:
        
        def parse_lab_event(self, pid, p_info):
            lab_measurements = []
            for idx, row in p_info.iterrows():
                # Check if the 'itemid' is in the list of lab event IDs and
                # the 'category' is 'Routine Vital Signs'
                if str(row['itemid']) in self.lab_event_ids:
                        # Convert charttime to datetime and handle parsing inside strptime function
                        charttime_datetime = strptime(row['charttime'])
                        event = EventUMSE(time=charttime_datetime, 
                                          feature_type = str(row['itemid']),
                                          value = row['valuenum'])
                        lab_measurements.append(event)

            lab_measurements = SetUMSE(pid, "lab", lab_measurements)
            return lab_measurements
        
        # read lab event data
        print("Reading Lab Events!")
        lab_events_df = None
        if self.use_parquet:
            path = os.path.join(self.hosp_path, "labevents.parquet")
            if os.path.exists(path):
                print("Loading Existing Parquet Path!")
                lab_events_df = pd.read_parquet(path) 
            else: 
                print("Creating New Parquet File!")
                lab_events_df = pd.read_csv(os.path.join(self.hosp_path, "labevents.csv"))
                lab_events_df.to_parquet(path, index=False)
        else:
            lab_events_df = pd.read_csv(os.path.join(self.hosp_path, "labevents.csv"))
        
        print(f"Read {len(lab_events_df)} lab events.")
        print("Parsing Lab Events!")
        lab_events_df = lab_events_df.dropna(subset=["subject_id", "itemid", "valuenum", "charttime"])
        lab_events_df = lab_events_df.sort_values(["subject_id", "itemid", "charttime"], ascending=True)
        lab_events_df['subject_id'] = lab_events_df['subject_id'].astype(str)
        lab_events_df = self.dev_mode(lab_events_df)
        lab_events_df = lab_events_df.groupby("subject_id")
        lab_events_df = lab_events_df.parallel_apply(lambda x: parse_lab_event(self, x.subject_id.unique()[0], x))
        patients = self.add_observations_to_patients(patients, "lab", lab_events_df)
        return patients

    
    def parse_chart_events(self, patients : Dict[str, PatientUMSE]) -> Dict[str, PatientUMSE]:
        def parse_chart_event(self, pid, p_info):
            chart_measurements = []
            for idx, row in p_info.iterrows():
                # want (feature type, charttime, valuenum)
                if str(row['itemid']) in self.chart_event_ids:
                    event = EventUMSE(time=strptime(row['charttime']),
                                    feature_type=str(row['itemid']),
                                    value=row['valuenum'])
                    chart_measurements.append(event)
            chart_measurements = SetUMSE(pid, "chart", chart_measurements)
            return chart_measurements
        
        print("Reading Chart Events!")
        icu_chart_events_df = None
        if self.use_parquet:
            path = os.path.join(self.icu_path, "chartevents.parquet")
            if os.path.exists(path):
                print("Loading Existing Parquet Path!")
                icu_chart_events_df = pd.read_parquet(path)
            else: 
                print("Creating New Parquet File!")
                icu_chart_events_df = pd.read_csv(os.path.join(self.icu_path, "chartevents.csv"))
                icu_chart_events_df.to_parquet(path, index=False)
        else:
            icu_chart_events_df = pd.read_csv(os.path.join(self.icu_path, "chartevents.csv"))
        print(f"Read {len(icu_chart_events_df)} chart events.")

        print("Parsing Chart Events!")
        icu_chart_events_df = icu_chart_events_df.dropna(subset=["subject_id", "itemid", "valuenum", "charttime"])
        icu_chart_events_df = icu_chart_events_df.sort_values(["subject_id", "itemid", "charttime"], ascending=True)
        icu_chart_events_df['subject_id'] = icu_chart_events_df['subject_id'].astype(str)
        icu_chart_events_df = self.dev_mode(icu_chart_events_df)
        icu_chart_events_df = icu_chart_events_df.groupby("subject_id")
        icu_chart_events_df = icu_chart_events_df.parallel_apply(lambda x: parse_chart_event(self, x.subject_id.unique()[0], x))
        patients = self.add_observations_to_patients(patients, "chart", icu_chart_events_df)
        return patients
    
    def parse_notes(self, patients : Dict[str, PatientUMSE]) -> Dict[str, PatientUMSE]:
        def parse_note(self, pid, p_info):
            notes = []
            for idx, row in p_info.iterrows():
                # want (feature type, time, text)
                text = row['text']
                if self.note_sections[0] == "all":
                    event = EventUMSE(time=row['charttime'], feature_type="note", value=text)
                    notes.append(event)
                else:
                    if self.concatenate_notes:
                        combined_text = " "
                        for section in self.note_sections:
                            if section in text:
                                combined_text += "<SEP>" + get_section(text.lower(), section.lower())
                        event = EventUMSE(time=row['charttime'], feature_type="note", value=combined_text)
                        notes.append(event)
                    else:
                        for section in self.note_sections:
                            if section in text:
                                event = EventUMSE(time=row['charttime'], feature_type=section, value=get_section(text.lower(), section.lower()))
                                notes.append(event)

            notes = SetUMSE(pid, "note", notes)
            return notes
        
        # Read Note Data
        print("Reading Note Data!")
        note_df = None
        if self.use_parquet:
            path = os.path.join(self.note_root, "discharge.parquet")
            if os.path.exists(path):
                print("Loading Existing Parquet Path!")
                note_df = pd.read_parquet(path)
            else: 
                print("Creating New Parquet File!")
                note_df = pd.read_csv(os.path.join(self.note_root, "discharge.csv"))
                note_df.to_parquet(path, index=False)
        else:
            note_df = pd.read_csv(os.path.join(self.note_root, "discharge.csv"))
        note_df = note_df.dropna(subset=["subject_id", "text", "charttime"])
        print(f"Read {len(note_df)} note events.")
        note_df = note_df.sort_values(["subject_id", "charttime"], ascending=True)
        
        note_df['subject_id'] = note_df['subject_id'].astype(str)
        note_df = self.dev_mode(note_df)
        note_df = note_df.groupby("subject_id")
        print("Parsing Notes!")
        note_df = note_df.parallel_apply(lambda x: parse_note(self, x.subject_id.unique()[0], x))
        
        patients = self.add_observations_to_patients(patients, "note", note_df)
        return patients
    
    def parse_xrays(self, patients : Dict[str, PatientUMSE]) -> Dict[str, PatientUMSE]:
        def process_xray(self, pid, p_info):
            xrays = []
            for idx, row in p_info.iterrows():
                # want ("xray", time, image)     
                dicom_id = row['dicom_id']
                image_path = get_img_path(os.path.join(self.cxr_root, "images"), dicom_id)
                event = EventUMSE(time=row['StudyDateTime'], feature_type="xray", value=image_path)
                xrays.append(event)
            xrays = SetUMSE(pid, "xray", xrays)
            return xrays
        # read mimic-cxr metadata
        print("Reading CXR metadata!")
        cxr_jpg_meta_df = pd.read_csv(os.path.join(self.cxr_root, "mimic-cxr-2.0.0-metadata.csv"))
        cxr_jpg_meta_df.StudyDate = cxr_jpg_meta_df.StudyDate.astype(str)
        cxr_jpg_meta_df.StudyTime = cxr_jpg_meta_df.StudyTime.astype(str).str.split(".").str[0]
        cxr_jpg_meta_df["StudyDateTime"] = pd.to_datetime(cxr_jpg_meta_df.StudyDate + cxr_jpg_meta_df.StudyTime,
                                                            format="%Y%m%d%H%M%S",
                                                            errors="coerce")
        
        cxr_df = cxr_jpg_meta_df[cxr_jpg_meta_df.StudyDateTime.isna()]
        cxr_df = cxr_jpg_meta_df[["subject_id", "study_id", "dicom_id", "StudyDateTime"]]
        cxr_df = cxr_df.dropna(subset=["subject_id", "dicom_id", "StudyDateTime"])
        cxr_df = cxr_df.sort_values(["subject_id", "StudyDateTime"], ascending=True)
        print(f"Read {len(cxr_df)} x-ray events.")
        cxr_df['subject_id'] = cxr_df['subject_id'].astype(str)
        cxr_df = self.dev_mode(cxr_df)
        cxr_df = cxr_df.groupby("subject_id")
        print("Parsing X-rays!")
        cxr_df = cxr_df.parallel_apply(lambda x: process_xray(self, x.subject_id.unique()[0], x))
        patients = self.add_observations_to_patients(patients, "xray", cxr_df)

        return patients
    
    def unit_factor(self, difference, unit):
        if unit == 'hour':
            return difference / 3600
        elif unit == 'day':
            return difference / (3600 * 24)
        elif unit == 'minute':
            return difference / 60
        elif unit == 'second':
            return difference
        else:
            raise ValueError("Unit not recognized")

    # t_occurence = t - t_admit 
    # t_current = t_discharge - t
    def set_patient_occurence_time(self, unit='day'):
        for patient_id, patient in tqdm(self.patients.items(), desc="Setting all Charttimes to Relative Time from Admittime"):
            
            if patient.notes:
                for note in patient.notes.observables:
                    note_time = datetime.strptime(note.time, "%Y-%m-%d %H:%M:%S") if isinstance(note.time, str) else note.time
                    note.time = self.unit_factor((note_time - patient.admittime).total_seconds(), unit)
                    
            if patient.x_rays:
                for xray in patient.x_rays.observables:
                    xray_time = datetime.strptime(xray.time, "%Y-%m-%d %H:%M:%S") if isinstance(xray.time, str) else xray.time
                    xray.time = self.unit_factor((xray_time - patient.admittime).total_seconds(), unit)

            if patient.lab:
                for lab_event in patient.lab.observables:
                    lab_event_time = datetime.strptime(lab_event.time, "%Y-%m-%d %H:%M:%S") if isinstance(lab_event.time, str) else lab_event.time
                    lab_event.time = self.unit_factor((lab_event_time - patient.admittime).total_seconds(), unit)

            if patient.chart:
                for chart_event in patient.chart.observables:
                    chart_event_time = datetime.strptime(chart_event.time, "%Y-%m-%d %H:%M:%S") if isinstance(chart_event.time, str) else chart_event.time
                    chart_event.time = self.unit_factor((chart_event_time - patient.admittime).total_seconds(), unit)
            
        return self.patients



def save_to_pkl(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    


if __name__ == "__main__":
    mimic_cxr_path = "/home/johnwu3/projects/serv4/Medical_Tri_Modal_Pilot/data/physionet.org/files/MIMIC-CXR"
    mimic_cxr_jpg_path = "/home/johnwu3/projects/serv4/Medical_Tri_Modal_Pilot/data/physionet.org/files/MIMIC-CXR"
    mimic_iv_path = "/home/johnwu3/projects/serv4/Medical_Tri_Modal_Pilot/data/physionet.org/files/MIMIC-IV/2.0/"
    mimic_note_directory = "/home/johnwu3/projects/serv4/Medical_Tri_Modal_Pilot/data/physionet.org/files/mimic-iv-note/2.2/note"
    dataset = MIMIC4UMSE(root=mimic_iv_path, 
                    cxr_root=mimic_cxr_jpg_path, 
                    note_root=mimic_note_directory, 
                    dev=True, 
                    dev_patients=2000,
                    use_parquet=True,
                    use_relative_time=True)