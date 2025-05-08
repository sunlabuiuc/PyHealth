from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List

import polars as pl

from .base_task import BaseTask


class InHospitalMortalityMIMIC4(BaseTask):
    """Task for predicting in-hospital mortality using MIMIC-IV dataset.

    This task leverages lab results to predict the likelihood of in-hospital
    mortality.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - labs: A timeseries of lab results.    
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - mortality: A binary indicator of mortality.
    """
    task_name: str = "InHospitalMortalityMIMIC4"
    input_schema: Dict[str, str] = {"labs": "timeseries"}
    output_schema: Dict[str, str] = {"mortality": "binary"}

    # Organize lab items by category
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
    }

    # Create flat list of all lab items for use in the function
    LABITEMS: ClassVar[List[str]] = [
        item for category in LAB_CATEGORIES.values() 
        for subcategory in category.values() 
        for item in subcategory
    ]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        input_window_hours = 48
        samples = []

        demographics = patient.get_events(event_type="patients")
        assert len(demographics) == 1
        demographics = demographics[0]
        anchor_age = int(demographics.anchor_age)        
        if anchor_age < 18:
            return []

        admissions = patient.get_events(event_type="admissions")
        for admission in admissions:
            admission_dischtime = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
            duration_hour = (admission_dischtime - admission.timestamp).total_seconds() / 3600
            if duration_hour <= input_window_hours:
                continue
            predict_time = admission.timestamp + timedelta(hours=input_window_hours)

            labevents_df = patient.get_events(
                event_type="labevents",
                start=admission.timestamp,
                end=predict_time,
                return_df=True
            )
            labevents_df = labevents_df.filter(
                pl.col("labevents/itemid").is_in(self.LABITEMS)
            )
            labevents_df = labevents_df.with_columns(
                pl.col("labevents/storetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )
            labevents_df = labevents_df.filter(
                (pl.col("labevents/storetime") <= predict_time)
            )
            if labevents_df.height == 0:
                continue

            labevents_df = labevents_df.select(
                pl.col("timestamp"),
                pl.col("labevents/itemid"),
                pl.col("labevents/valuenum").cast(pl.Float64)
            )
            labevents_df = labevents_df.pivot(
                index="timestamp",
                columns="labevents/itemid",
                values="labevents/valuenum",
                # in case of multiple values for the same timestamp
                aggregate_function="first",
            )
            labevents_df = labevents_df.sort("timestamp")

            # Add missing columns with NaN values
            existing_cols = set(labevents_df.columns) - {"timestamp"}
            missing_cols = [item for item in self.LABITEMS if item not in existing_cols]
            for col in missing_cols:
                labevents_df = labevents_df.with_columns(pl.lit(None).alias(col))

            # Reorder columns by LABITEMS
            labevents_df = labevents_df.select(
                "timestamp",
                *self.LABITEMS
            )

            timestamps = labevents_df["timestamp"].to_list()
            lab_values = labevents_df.drop("timestamp").to_numpy()

            mortality = int(admission.hospital_expire_flag)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "admission_id": admission.hadm_id,
                    "labs": (timestamps, lab_values),
                    "mortality": mortality,
                }
            )

        return samples
