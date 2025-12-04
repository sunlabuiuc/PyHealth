from datetime import datetime
from typing import Any, ClassVar, Dict, List, Tuple

import polars as pl

from .base_task import BaseTask


class MortalityPredictionStageNetMIMIC4(BaseTask):
    """Task for predicting mortality using MIMIC-IV with StageNet format.

    This task creates PATIENT-LEVEL samples (not visit-level) by aggregating
    all admissions for each patient. ICD codes (diagnoses + procedures) and
    lab results across all visits are combined with time intervals calculated
    from the patient's first admission timestamp.

    Time Calculation:
        - ICD codes: Hours from previous admission (0 for first visit,
          then time intervals between consecutive visits)
        - Labs: Hours from admission start (within-visit measurements)

    Lab Processing:
        - 10-dimensional vectors (one per lab category)
        - Multiple itemids per category → take first observed value
        - Missing categories → None/NaN in vector

    Args:
        padding: Additional padding for StageNet processor to handle
            sequences longer than observed during training. Default: 0.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - icd_codes: Combined diagnosis + procedure ICD codes
              (stagenet format, nested by visit)
            - labs: Lab results (stagenet_tensor, 10D vectors per timestamp)
        output_schema (Dict[str, str]): The schema for output data:
            - mortality: Binary indicator (1 if any admission had mortality)
    """

    task_name: str = "MortalityPredictionStageNetMIMIC4"

    def __init__(self, padding: int = 0):
        """Initialize task with optional padding parameter.

        Args:
            padding: Additional padding for nested sequences. Default: 0.
        """
        self.padding = padding
        # Use tuple format to pass kwargs to processor
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

    # Organize lab items by category
    # Each category will map to ONE dimension in the output vector
    LAB_CATEGORIES: ClassVar[Dict[str, List[str]]] = {
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
    }

    # Ordered list of category names (defines vector dimension order)
    LAB_CATEGORY_NAMES: ClassVar[List[str]] = [
        "Sodium",
        "Potassium",
        "Chloride",
        "Bicarbonate",
        "Glucose",
        "Calcium",
        "Magnesium",
        "Anion Gap",
        "Osmolality",
        "Phosphate",
    ]

    # Flat list of all lab item IDs for filtering
    LABITEMS: ClassVar[List[str]] = [
        item for itemids in LAB_CATEGORIES.values() for item in itemids
    ]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create mortality prediction samples.

        Creates ONE sample per patient with all admissions aggregated.
        Time intervals are calculated between consecutive admissions.

        Args:
            patient: Patient object with get_events method

        Returns:
            List with single sample containing patient_id, all conditions,
            procedures, labs across visits, and final mortality label
        """
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

        # Initialize aggregated data structures
        # List of ICD codes (diagnoses + procedures) per visit
        all_icd_codes = []
        all_icd_times = []  # Time from previous admission per visit
        all_lab_values = []  # List of 10D lab vectors
        all_lab_times = []  # Time from admission start per measurement

        # Track previous admission timestamp for interval calculation
        previous_admission_time = None

        # Track if patient had any mortality event
        final_mortality = 0

        # Process each admission
        for i, admission in enumerate(admissions):
            # Parse admission and discharge times
            try:
                admission_time = admission.timestamp
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # Skip if timestamps invalid
                continue

            # Skip if discharge is before admission (data quality issue)
            if admission_dischtime < admission_time:
                continue

            # Calculate time from previous admission (in hours)
            # First admission will have time = 0
            if previous_admission_time is None:
                time_from_previous = 0.0
            else:
                time_from_previous = (
                    admission_time - previous_admission_time
                ).total_seconds() / 3600.0

            # Update previous admission time for next iteration
            previous_admission_time = admission_time

            # Update mortality label if this admission had mortality
            try:
                if int(admission.hospital_expire_flag) == 1:
                    final_mortality = 1
            except (ValueError, TypeError, AttributeError):
                pass

            # Get diagnosis codes for this admission using hadm_id
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            visit_diagnoses = [
                event.icd_code
                for event in diagnoses_icd
                if hasattr(event, "icd_code") and event.icd_code
            ]

            # Get procedure codes for this admission using hadm_id
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            visit_procedures = [
                event.icd_code
                for event in procedures_icd
                if hasattr(event, "icd_code") and event.icd_code
            ]

            # Combine diagnoses and procedures into single ICD code list
            visit_icd_codes = visit_diagnoses + visit_procedures

            if visit_icd_codes:
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)

            # Get lab events for this admission
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

                if labevents_df.height > 0:
                    # Select relevant columns
                    labevents_df = labevents_df.select(
                        pl.col("timestamp"),
                        pl.col("labevents/itemid"),
                        pl.col("labevents/valuenum").cast(pl.Float64),
                    )

                    # Group by timestamp and aggregate into 10D vectors
                    # For each timestamp, create vector of lab categories
                    unique_timestamps = sorted(
                        labevents_df["timestamp"].unique().to_list()
                    )

                    for lab_ts in unique_timestamps:
                        # Get all lab events at this timestamp
                        ts_labs = labevents_df.filter(pl.col("timestamp") == lab_ts)

                        # Create 10-dimensional vector (one per category)
                        lab_vector = []
                        for category_name in self.LAB_CATEGORY_NAMES:
                            category_itemids = self.LAB_CATEGORIES[category_name]

                            # Find first matching value for this category
                            category_value = None
                            for itemid in category_itemids:
                                matching = ts_labs.filter(
                                    pl.col("labevents/itemid") == itemid
                                )
                                if matching.height > 0:
                                    category_value = matching["labevents/valuenum"][0]
                                    break

                            lab_vector.append(category_value)

                        # Calculate time from admission start (hours)
                        time_from_admission = (
                            lab_ts - admission_time
                        ).total_seconds() / 3600.0

                        all_lab_values.append(lab_vector)
                        all_lab_times.append(time_from_admission)

        # Skip if no lab events (required for this task)
        if len(all_lab_values) == 0:
            return []

        # Also skip if no ICD codes across all admissions
        if len(all_icd_codes) == 0:
            return []

        # Format as tuples: (time, values)
        # ICD codes: nested list with times
        icd_codes_data = (all_icd_times, all_icd_codes)

        # Labs: list of 10D vectors with times
        labs_data = (all_lab_times, all_lab_values)

        # Create single patient-level sample
        sample = {
            "patient_id": patient.patient_id,
            "icd_codes": icd_codes_data,
            "labs": labs_data,
            "mortality": final_mortality,
        }
        return [sample]
