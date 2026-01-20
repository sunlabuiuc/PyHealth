"""OhioT1DM Dataset for Blood Glucose Level Prediction.

This module provides the OhioT1DMDataset class for loading the OhioT1DM dataset,
which contains continuous glucose monitoring (CGM), insulin, physiological sensor,
and self-reported life-event data for blood glucose prediction research.

Dataset Information:
    - Name: OhioT1DM (Ohio Type 1 Diabetes Mellitus)
    - Subjects: 12 (IDs: 540, 544, 552, 567, 584, 596, 559, 563, 570, 575, 588, 591)
    - Duration: 8 weeks per subject
    - CGM: Blood glucose every 5 minutes
    - Insulin: Basal and bolus doses
    - Life events: Meals, exercise, sleep, stress, illness
    - Physiological: Heart rate, GSR, skin temperature, steps (from fitness bands)
    
Dataset Reference:
    Marling, C., & Bunescu, R. (2020). The OhioT1DM Dataset for Blood Glucose 
    Level Prediction: Update 2020. CEUR Workshop Proceedings, 2675, 71-74.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/

Dataset Download:
    - Official: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
    - Kaggle: https://www.kaggle.com/datasets/ryanmouton/ohiot1dm

Example:
    >>> from pyhealth.datasets import OhioT1DMDataset
    >>> dataset = OhioT1DMDataset(root="/path/to/OhioT1DM/")
    >>> dataset.stat()
"""

import os
import pickle
import logging
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime

import numpy as np

from pyhealth.datasets import BaseSignalDataset

logger = logging.getLogger(__name__)

# Subject IDs (2018 cohort: Basis band, 2020 cohort: Empatica band)
SUBJECT_IDS_2018 = [559, 563, 570, 575, 588, 591]  # Basis Peak band
SUBJECT_IDS_2020 = [540, 544, 552, 567, 584, 596]  # Empatica Embrace band
ALL_SUBJECT_IDS = SUBJECT_IDS_2020 + SUBJECT_IDS_2018


class OhioT1DMDataset(BaseSignalDataset):
    """Dataset class for OhioT1DM (Ohio Type 1 Diabetes Mellitus).

    The OhioT1DM dataset contains 8 weeks of data for 12 people with type 1 
    diabetes on insulin pump therapy with continuous glucose monitoring.

    Data includes:
        - CGM blood glucose levels (every 5 minutes)
        - Finger stick blood glucose measurements
        - Insulin doses (basal and bolus)
        - Self-reported meals with carbohydrate estimates
        - Self-reported exercise, sleep, work, stress, illness
        - Physiological data from fitness bands (heart rate, GSR, temperature, steps)

    Args:
        root: Root directory containing XML files (train/ and test/ subdirs or flat).
        dataset_name: Name of the dataset. Default is "ohiot1dm".
        dev: If True, only load 3 subjects. Default is False.
        refresh_cache: If True, reprocess data. Default is False.

    Example:
        >>> from pyhealth.datasets import OhioT1DMDataset
        >>> dataset = OhioT1DMDataset(root="/path/to/OhioT1DM/", dev=True)
        >>> dataset.stat()
        >>> patient = dataset.get_patient("540")
        >>> glucose = patient["glucose_level"]
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "ohiot1dm",
        dev: bool = False,
        refresh_cache: bool = False,
    ) -> None:
        self.root = root
        self.dataset_name = dataset_name
        self.dev = dev
        self.refresh_cache = refresh_cache

        self.filepath = os.path.join(
            os.path.dirname(os.path.abspath(root.rstrip("/"))),
            f".cache_{dataset_name}",
        )
        os.makedirs(self.filepath, exist_ok=True)

        self.task: Optional[str] = None
        self.samples: Optional[List[Dict]] = None
        self.patient_to_index: Optional[Dict[str, List[int]]] = None
        self.visit_to_index: Optional[Dict[str, List[int]]] = None
        self.patients: Dict[str, Dict[str, Any]] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load data from cache or process raw files."""
        cache_file = os.path.join(self.filepath, "processed_data.pkl")

        if os.path.exists(cache_file) and not self.refresh_cache:
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                self.patients = pickle.load(f)
        else:
            logger.info("Processing raw OhioT1DM XML data...")
            self._process_raw_data()
            with open(cache_file, "wb") as f:
                pickle.dump(self.patients, f)

    def _find_xml_files(self, subject_id: int) -> List[str]:
        """Find XML files for a given subject."""
        xml_files = []
        patterns = [
            f"{subject_id}-ws-training.xml",
            f"{subject_id}-ws-testing.xml",
            f"{subject_id}_training.xml",
            f"{subject_id}_testing.xml",
            f"{subject_id}.xml",
        ]
        
        # Search in root and common subdirectories
        search_dirs = [
            self.root,
            os.path.join(self.root, "train"),
            os.path.join(self.root, "test"),
            os.path.join(self.root, "training"),
            os.path.join(self.root, "testing"),
            os.path.join(self.root, "2018"),
            os.path.join(self.root, "2020"),
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for pattern in patterns:
                filepath = os.path.join(search_dir, pattern)
                if os.path.exists(filepath) and filepath not in xml_files:
                    xml_files.append(filepath)
            # Also search for any XML with subject ID
            if os.path.isdir(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.xml') and str(subject_id) in f:
                        filepath = os.path.join(search_dir, f)
                        if filepath not in xml_files:
                            xml_files.append(filepath)
        
        return xml_files

    def _parse_xml_file(self, xml_path: str) -> Dict[str, Any]:
        """Parse a single XML file and extract all data."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        data = {
            "glucose_level": [],      # CGM readings
            "finger_stick": [],       # Manual BG readings
            "basal": [],              # Basal insulin rates
            "temp_basal": [],         # Temporary basal rates
            "bolus": [],              # Bolus insulin doses
            "meal": [],               # Meal events with carbs
            "exercise": [],           # Exercise events
            "sleep": [],              # Sleep events
            "work": [],               # Work events
            "stressors": [],          # Stress events
            "illness": [],            # Illness events
            "hypo_event": [],         # Hypoglycemic events
            "basis_heart_rate": [],   # Heart rate (Basis band)
            "basis_gsr": [],          # Galvanic skin response
            "basis_skin_temperature": [],  # Skin temperature
            "basis_air_temperature": [],   # Air temperature (Basis only)
            "basis_steps": [],        # Step count (Basis only)
            "basis_sleep": [],        # Band-detected sleep
            "acceleration": [],       # Acceleration (Empatica only)
        }
        
        # Parse patient info
        patient_elem = root.find(".//patient")
        if patient_elem is not None:
            data["patient_id"] = patient_elem.get("id", "")
            data["insulin_type"] = patient_elem.get("insulin_type", "")
        
        # Parse glucose levels (CGM)
        for elem in root.findall(".//glucose_level/event"):
            ts = elem.get("ts")
            value = elem.get("value")
            if ts and value:
                data["glucose_level"].append({
                    "ts": ts,
                    "value": float(value)
                })
        
        # Parse finger sticks
        for elem in root.findall(".//finger_stick/event"):
            ts = elem.get("ts")
            value = elem.get("value")
            if ts and value:
                data["finger_stick"].append({
                    "ts": ts,
                    "value": float(value)
                })
        
        # Parse basal insulin
        for elem in root.findall(".//basal/event"):
            ts = elem.get("ts")
            value = elem.get("value")
            if ts and value:
                data["basal"].append({
                    "ts": ts,
                    "value": float(value)
                })
        
        # Parse temp basal
        for elem in root.findall(".//temp_basal/event"):
            ts_begin = elem.get("ts_begin")
            ts_end = elem.get("ts_end")
            value = elem.get("value")
            if ts_begin and value:
                data["temp_basal"].append({
                    "ts_begin": ts_begin,
                    "ts_end": ts_end,
                    "value": float(value)
                })
        
        # Parse bolus insulin
        for elem in root.findall(".//bolus/event"):
            ts_begin = elem.get("ts_begin")
            ts_end = elem.get("ts_end")
            dose = elem.get("dose")
            btype = elem.get("type", "normal")
            if ts_begin and dose:
                data["bolus"].append({
                    "ts_begin": ts_begin,
                    "ts_end": ts_end,
                    "dose": float(dose),
                    "type": btype
                })
        
        # Parse meals
        for elem in root.findall(".//meal/event"):
            ts = elem.get("ts")
            carbs = elem.get("carbs")
            meal_type = elem.get("type", "")
            if ts:
                data["meal"].append({
                    "ts": ts,
                    "carbs": float(carbs) if carbs else 0.0,
                    "type": meal_type
                })
        
        # Parse exercise
        for elem in root.findall(".//exercise/event"):
            ts = elem.get("ts")
            duration = elem.get("duration")
            intensity = elem.get("intensity")
            if ts:
                data["exercise"].append({
                    "ts": ts,
                    "duration": float(duration) if duration else 0.0,
                    "intensity": int(intensity) if intensity else 0
                })
        
        # Parse sleep
        for elem in root.findall(".//sleep/event"):
            ts_begin = elem.get("ts_begin")
            ts_end = elem.get("ts_end")
            quality = elem.get("quality")
            if ts_begin:
                data["sleep"].append({
                    "ts_begin": ts_begin,
                    "ts_end": ts_end,
                    "quality": int(quality) if quality else 0
                })
        
        # Parse physiological data (basis_heart_rate, basis_gsr, etc.)
        for field in ["basis_heart_rate", "basis_gsr", "basis_skin_temperature", 
                      "basis_air_temperature", "basis_steps", "acceleration"]:
            for elem in root.findall(f".//{field}/event"):
                ts = elem.get("ts")
                value = elem.get("value")
                if ts and value:
                    data[field].append({
                        "ts": ts,
                        "value": float(value)
                    })
        
        # Parse stressors
        for elem in root.findall(".//stressors/event"):
            ts = elem.get("ts")
            if ts:
                data["stressors"].append({"ts": ts})
        
        # Parse illness
        for elem in root.findall(".//illness/event"):
            ts = elem.get("ts")
            if ts:
                data["illness"].append({"ts": ts})
        
        return data

    def _merge_patient_data(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Merge data from multiple XML files (train + test)."""
        merged = {}
        
        for data in data_list:
            for key, value in data.items():
                if key in ["patient_id", "insulin_type"]:
                    merged[key] = value
                elif isinstance(value, list):
                    if key not in merged:
                        merged[key] = []
                    merged[key].extend(value)
        
        # Sort time-series data by timestamp
        for key in merged:
            if isinstance(merged[key], list) and len(merged[key]) > 0:
                if isinstance(merged[key][0], dict):
                    ts_key = "ts" if "ts" in merged[key][0] else "ts_begin"
                    if ts_key in merged[key][0]:
                        merged[key].sort(key=lambda x: x.get(ts_key, ""))
        
        return merged

    def _process_raw_data(self) -> None:
        """Process raw XML files from the dataset."""
        subject_ids = ALL_SUBJECT_IDS[:3] if self.dev else ALL_SUBJECT_IDS

        for sid in subject_ids:
            xml_files = self._find_xml_files(sid)

            if not xml_files:
                logger.warning(f"No XML files found for subject {sid}")
                continue

            try:
                data_list = []
                for xml_file in xml_files:
                    data = self._parse_xml_file(xml_file)
                    data_list.append(data)
                    logger.info(f"Parsed {xml_file}")

                merged_data = self._merge_patient_data(data_list)
                merged_data["patient_id"] = str(sid)
                merged_data["subject_id"] = sid
                merged_data["cohort"] = "2020" if sid in SUBJECT_IDS_2020 else "2018"
                
                # Convert glucose to numpy array for easier processing
                if merged_data.get("glucose_level"):
                    glucose_values = [g["value"] for g in merged_data["glucose_level"]]
                    merged_data["glucose_array"] = np.array(glucose_values, dtype=np.float32)
                
                self.patients[str(sid)] = merged_data
                logger.info(f"Loaded subject {sid}: {len(merged_data.get('glucose_level', []))} CGM readings")
                
            except Exception as e:
                logger.error(f"Error loading subject {sid}: {e}")

    def stat(self) -> None:
        """Print dataset statistics."""
        print("=" * 70)
        print(f"Dataset: {self.dataset_name.upper()}")
        print("=" * 70)
        print(f"Subjects: {len(self.patients)}")
        print(f"IDs: {list(self.patients.keys())}")
        print("\nPer-subject statistics:")
        print("-" * 70)
        print(f"{'ID':<8} {'Cohort':<8} {'CGM':<10} {'Meals':<8} {'Bolus':<8} {'Exercise':<10}")
        print("-" * 70)
        
        for pid, patient in self.patients.items():
            cohort = patient.get("cohort", "?")
            n_glucose = len(patient.get("glucose_level", []))
            n_meals = len(patient.get("meal", []))
            n_bolus = len(patient.get("bolus", []))
            n_exercise = len(patient.get("exercise", []))
            print(f"{pid:<8} {cohort:<8} {n_glucose:<10} {n_meals:<8} {n_bolus:<8} {n_exercise:<10}")

    def info(self) -> None:
        """Print dataset info."""
        print(f"Dataset: {self.dataset_name}")
        print(f"Root: {self.root}")
        print(f"Subjects: {len(self.patients)}")

    def set_task(self, task_fn: Callable) -> "OhioT1DMDataset":
        """Apply task function to create samples."""
        self.samples = []
        self.patient_to_index = {}

        for patient_id, patient_data in self.patients.items():
            start_idx = len(self.samples)
            self.samples.extend(task_fn(patient_data))
            self.patient_to_index[patient_id] = list(range(start_idx, len(self.samples)))

        return self

    def get_patient(self, patient_id: str) -> Dict:
        """Get patient data by ID."""
        if patient_id not in self.patients:
            raise KeyError(f"Patient {patient_id} not found")
        return self.patients[patient_id]

    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self):
        for pid in self.patients:
            yield self.patients[pid]

    def __getitem__(self, patient_id: str) -> Dict:
        return self.get_patient(patient_id)
