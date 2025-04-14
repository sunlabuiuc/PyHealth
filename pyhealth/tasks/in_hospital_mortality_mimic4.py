from datetime import datetime, timedelta
from typing import Any, Dict, List, ClassVar

import polars as pl

from .base_task import BaseTask

class InHospitalMortalityMIMIC4(BaseTask):
    """Task for predicting in-hospital mortality using MIMIC-IV dataset.

    This task uses lab values from the first 48 hours of a patient's hospital stay
    to predict whether they will die during that hospital stay.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema for the task.
        output_schema (Dict[str, str]): The output schema for the task.
        LAB_CATEGORIES (Dict[str, Dict[str, List[str]]]): Organized lab test IDs by system/category.
        LABITEMS (List[str]): Flattened list of all lab item IDs used in the task.
    """
    task_name: str = "InHospitalMortalityMIMIC4"
    input_schema: Dict[str, str] = {"labs": "timeseries"}
    output_schema: Dict[str, str] = {"mortality": "binary"}

    # Organize lab items by category for better maintainability
    LAB_CATEGORIES: ClassVar[Dict[str, Dict[str, List[str]]]] = {
        "Electrolytes & Metabolic": {
            "Sodium": ["50824", "52455", "50983", "52623"],
            "Potassium": ["50822", "52452", "50971", "52610"],
            "Chloride": ["50806", "52434", "50902", "52535"],
            "Bicarbonate": ["50803", "50804"],
            "Glucose": ["50809", "52027", "50931", "52569"],
            "Calcium": ["50808", "51624"],
            "Magnesium": ["50960"],
            "Anion Gap": ["50868", "52500"],
            "Osmolality": ["52031", "50964", "51701"],
            "Phosphate": ["50970"],
        },
        # Add more categories as needed
        # "Renal": {...},
        # "Liver": {...},
        # etc.
    }
    
    # Create flat list of all lab items for easy lookup
    LABITEMS: ClassVar[List[str]] = [
        item for category in LAB_CATEGORIES.values() 
        for subcategory in category.values() 
        for item in subcategory
    ]

    def __init__(self, input_window_hours: int = 48, *args, **kwargs):
        """Initialize the task with configurable parameters.
        
        Args:
            input_window_hours: Number of hours from admission to use for prediction
                                Default is 48 hours (2 days)
        """
        super().__init__(*args, **kwargs)
        self.input_window_hours = input_window_hours

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the in-hospital mortality prediction task.

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a dict with patient_id,
            admission_id, labs, and mortality as keys.
        """
        samples = []

        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
            
        demographics = demographics[0]
        try:
            anchor_age = int(demographics.anchor_age)
            if anchor_age < 18:
                return []
        except (ValueError, AttributeError):
            # Skip if age can't be determined
            return []
    
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            try:
                admission_dischtime = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
                duration_hour = (admission_dischtime - admission.timestamp).total_seconds() / 3600
                if duration_hour <= self.input_window_hours:
                    continue
                predict_time = admission.timestamp + timedelta(hours=self.input_window_hours)
            except (ValueError, AttributeError):
                # Skip if timestamps can't be processed
                continue

            # Get lab values within the prediction window
            labevents_df = self._process_lab_events(patient, admission, predict_time)
            if labevents_df is None or labevents_df.height == 0:
                continue
            
            timestamps = labevents_df["timestamp"].to_list()
            lab_values = labevents_df.drop("timestamp").to_numpy()

            # Determine mortality label
            try:
                mortality = int(admission.hospital_expire_flag)
            except (ValueError, AttributeError):
                mortality = 0

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "admission_id": admission.hadm_id,
                    "labs": (timestamps, lab_values),
                    "mortality": mortality,
                }
            )

        return samples
    
    def _process_lab_events(self, patient, admission, predict_time):
        """Process lab events for the given patient and admission.
        
        Args:
            patient: Patient object
            admission: Admission event
            predict_time: End of prediction window
            
        Returns:
            Processed lab events dataframe or None if no relevant labs
        """
        # Get lab events within the prediction window
        labevents_df = patient.get_events(
            event_type="labevents",
            start=admission.timestamp,
            end=predict_time,
            return_df=True
        )
        
        # Return None if no lab events were found
        if labevents_df is None or labevents_df.height == 0:
            return None
            
        # Filter by relevant lab items
        labevents_df = labevents_df.filter(
            pl.col("labevents/itemid").is_in(self.LABITEMS)
        )
        
        # Process timestamps
        try:
            labevents_df = labevents_df.with_columns(
                pl.col("labevents/storetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )
            labevents_df = labevents_df.filter(
                (pl.col("labevents/storetime") <= predict_time)
            )
        except Exception:
            # If timestamp processing fails, return None
            return None
            
        if labevents_df.height == 0:
            return None

        # Select and process relevant columns
        labevents_df = labevents_df.select(
            pl.col("timestamp"),
            pl.col("labevents/itemid"),
            pl.col("labevents/valuenum").cast(pl.Float64)
        )
        
        # Pivot to wide format with timestamps as rows and lab items as columns
        labevents_df = labevents_df.pivot(
            index="timestamp",
            columns="labevents/itemid",
            values="labevents/valuenum"
        )
        labevents_df = labevents_df.sort("timestamp")

        # Add missing columns with null values
        existing_cols = set(labevents_df.columns) - {"timestamp"}
        missing_cols = [item for item in self.LABITEMS if item not in existing_cols]
        for col in missing_cols:
            labevents_df = labevents_df.with_columns(pl.lit(None).alias(col))
    
        # Reorder columns by LABITEMS order for consistency
        labevents_df = labevents_df.select(
            "timestamp",
            *self.LABITEMS
        )
        
        return labevents_df