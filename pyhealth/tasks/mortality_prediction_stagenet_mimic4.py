from datetime import datetime
from typing import Any, ClassVar, Dict, List

import polars as pl

from .base_task import BaseTask


class MortalityPredictionStageNetMIMIC4(BaseTask):
    """Task for predicting mortality using MIMIC-IV with StageNet format.

    This task leverages diagnosis codes, procedure codes, and lab results to
    predict the likelihood of in-hospital mortality. Data is formatted for
    StageNet with time intervals based on differences between visits.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - conditions: Diagnosis codes (stagenet format)
            - procedures: Procedure codes (stagenet format)
            - labs: Lab results (stagenet_tensor format)
        output_schema (Dict[str, str]): The schema for output data:
            - mortality: Binary indicator of in-hospital mortality
    """

    task_name: str = "MortalityPredictionStageNetMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "stagenet",
        "procedures": "stagenet",
        "labs": "stagenet_tensor",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    # Organize lab items by category (same as InHospitalMortalityMIMIC4)
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

    # Flat list of all lab items
    LABITEMS: ClassVar[List[str]] = [
        item
        for category in LAB_CATEGORIES.values()
        for subcategory in category.values()
        for item in subcategory
    ]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create mortality prediction samples.

        Args:
            patient: Patient object with get_events method

        Returns:
            List of samples, each with patient_id, visit_id, conditions,
            procedures, labs, and mortality label
        """
        samples = []

        # Filter patients by age (>= 18)
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]
        try:
            anchor_age = int(demographics.anchor_age)
            if anchor_age < 18:
                return []
        except (ValueError, TypeError, AttributeError):
            # If age can't be determined, skip patient
            return []

        # Get all admissions
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 1:
            return []

        # Process each admission (predict mortality for current admission)
        for i in range(len(admissions)):
            admission = admissions[i]

            # Get mortality label from current admission
            try:
                mortality = int(admission.hospital_expire_flag)
            except (ValueError, TypeError, AttributeError):
                # Skip if mortality label not available
                continue

            # Parse admission and discharge times
            try:
                admission_time = admission.timestamp
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # Skip if timestamps invalid
                continue

            # Calculate time interval from previous visit (in hours)
            if i == 0:
                # First visit: time interval is 0
                time_interval = 0.0
            else:
                prev_admission = admissions[i - 1]
                try:
                    prev_dischtime = datetime.strptime(
                        prev_admission.dischtime, "%Y-%m-%d %H:%M:%S"
                    )
                    time_diff = admission_time - prev_dischtime
                    time_interval = time_diff.total_seconds() / 3600.0
                except (ValueError, AttributeError):
                    time_interval = 0.0

            # Get diagnosis codes (conditions)
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission_time,
                end=admission_dischtime,
            )
            conditions = [
                event.icd_code
                for event in diagnoses_icd
                if hasattr(event, "icd_code") and event.icd_code
            ]

            # Get procedure codes
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission_time,
                end=admission_dischtime,
            )
            procedures_list = [
                event.icd_code
                for event in procedures_icd
                if hasattr(event, "icd_code") and event.icd_code
            ]

            # Get lab events
            labevents_df = patient.get_events(
                event_type="labevents",
                start=admission_time,
                end=admission_dischtime,
                return_df=True,
            )

            # Filter to relevant lab items
            labevents_df = labevents_df.filter(
                pl.col("labevents/itemid").is_in(self.LABITEMS)
            )

            # Parse storetime and filter
            if labevents_df.height > 0:
                labevents_df = labevents_df.with_columns(
                    pl.col("labevents/storetime").str.strptime(
                        pl.Datetime, "%Y-%m-%d %H:%M:%S"
                    )
                )
                labevents_df = labevents_df.filter(
                    pl.col("labevents/storetime") <= admission_dischtime
                )

            # Skip if no clinical data
            if (
                len(conditions) == 0
                and len(procedures_list) == 0
                and labevents_df.height == 0
            ):
                continue

            # Process lab events into time series format
            lab_timestamps = []
            lab_values = []

            if labevents_df.height > 0:
                labevents_df = labevents_df.select(
                    pl.col("timestamp"),
                    pl.col("labevents/itemid"),
                    pl.col("labevents/valuenum").cast(pl.Float64),
                )
                labevents_df = labevents_df.pivot(
                    index="timestamp",
                    columns="labevents/itemid",
                    values="labevents/valuenum",
                    aggregate_function="first",
                )
                labevents_df = labevents_df.sort("timestamp")

                # Add missing columns with NaN values
                existing_cols = set(labevents_df.columns) - {"timestamp"}
                missing_cols = [
                    item for item in self.LABITEMS if item not in existing_cols
                ]
                for col in missing_cols:
                    labevents_df = labevents_df.with_columns(pl.lit(None).alias(col))

                # Reorder columns by LABITEMS
                labevents_df = labevents_df.select("timestamp", *self.LABITEMS)

                lab_timestamps = labevents_df["timestamp"].to_list()
                lab_values = labevents_df.drop("timestamp").to_numpy().tolist()

            # Format conditions for StageNet (with time intervals)
            conditions_data = {
                "value": conditions if conditions else [],
                "time": [time_interval] if conditions else [],
            }

            # Format procedures for StageNet (with time intervals)
            procedures_data = {
                "value": procedures_list if procedures_list else [],
                "time": [time_interval] if procedures_list else [],
            }

            # Format labs for StageNet (time from admission start)
            if lab_values:
                # Calculate time differences from admission start
                lab_time_intervals = [
                    (ts - admission_time).total_seconds() / 3600.0
                    for ts in lab_timestamps
                ]
                labs_data = {"value": lab_values, "time": lab_time_intervals}
            else:
                labs_data = {"value": [], "time": None}

            # Create sample
            sample = {
                "patient_id": patient.patient_id,
                "visit_id": admission.hadm_id,
                "conditions": conditions_data,
                "procedures": procedures_data,
                "labs": labs_data,
                "mortality": mortality,
            }

            samples.append(sample)

        return samples
