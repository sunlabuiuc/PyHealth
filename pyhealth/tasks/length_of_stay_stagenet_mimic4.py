from datetime import datetime
from typing import Any, ClassVar, Dict, List, Tuple

import polars as pl

from .base_task import BaseTask
from .length_of_stay_prediction import categorize_los


class LengthOfStayStageNetMIMIC4(BaseTask):
    """StageNet-format length of stay prediction for MIMIC-IV.

    Creates a single PATIENT-LEVEL sample by aggregating all admissions. Input
    structures mirror ``MortalityPredictionStageNetMIMIC4`` (ICD codes and labs
    with StageNet encodings). The target is the length-of-stay category for the
    most recent valid admission, using the same 10-category scheme as
    ``LengthOfStayPredictionMIMIC4``.

    Time handling
    ------------
    - ICD codes: hours since previous admission (first admission uses 0)
    - Labs: hours from admission start (within-visit)

    Lab aggregation
    ---------------
    - 10D vectors, one value per lab category (first observed per category per
      timestamp, missing -> None)

    Args:
        padding: Optional padding forwarded to the StageNet processor for nested
            sequences. Default is 0.
    """

    task_name: str = "LengthOfStayStageNetMIMIC4"

    def __init__(self, padding: int = 0):
        self.padding = padding
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"los": "multiclass"}

    # Lab item categories reused from the StageNet mortality task
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

    LABITEMS: ClassVar[List[str]] = [
        item for itemids in LAB_CATEGORIES.values() for item in itemids
    ]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        # Age filter
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]
        try:
            anchor_age = int(demographics.anchor_age)
            if anchor_age < 18:
                return []
        except (ValueError, TypeError, AttributeError):
            return []

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 1:
            return []

        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[Any]] = []
        all_lab_times: List[float] = []

        previous_admission_time = None
        target_los_category = None

        for admission in admissions:
            try:
                admission_time = admission.timestamp
                discharge_time = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                continue

            if discharge_time < admission_time:
                continue

            # Label from the most recent valid admission encountered
            los_days = (discharge_time - admission_time).days
            target_los_category = categorize_los(los_days)

            if previous_admission_time is None:
                time_from_previous = 0.0
            else:
                time_from_previous = (
                    admission_time - previous_admission_time
                ).total_seconds() / 3600.0

            previous_admission_time = admission_time

            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            visit_diagnoses = [
                event.icd_code
                for event in diagnoses_icd
                if hasattr(event, "icd_code") and event.icd_code
            ]

            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            visit_procedures = [
                event.icd_code
                for event in procedures_icd
                if hasattr(event, "icd_code") and event.icd_code
            ]

            visit_icd_codes = visit_diagnoses + visit_procedures

            if visit_icd_codes:
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)

            labevents_df = patient.get_events(
                event_type="labevents",
                start=admission_time,
                end=discharge_time,
                return_df=True,
            )

            labevents_df = labevents_df.filter(
                pl.col("labevents/itemid").is_in(self.LABITEMS)
            )

            if labevents_df.height > 0:
                labevents_df = labevents_df.with_columns(
                    pl.col("labevents/storetime").str.strptime(
                        pl.Datetime, "%Y-%m-%d %H:%M:%S"
                    )
                )
                labevents_df = labevents_df.filter(
                    pl.col("labevents/storetime") <= discharge_time
                )

                if labevents_df.height > 0:
                    labevents_df = labevents_df.select(
                        pl.col("timestamp"),
                        pl.col("labevents/itemid"),
                        pl.col("labevents/valuenum").cast(pl.Float64),
                    )

                    unique_timestamps = sorted(
                        labevents_df["timestamp"].unique().to_list()
                    )

                    for lab_ts in unique_timestamps:
                        ts_labs = labevents_df.filter(pl.col("timestamp") == lab_ts)

                        lab_vector: List[Any] = []
                        for category_name in self.LAB_CATEGORY_NAMES:
                            category_itemids = self.LAB_CATEGORIES[category_name]
                            category_value = None
                            for itemid in category_itemids:
                                matching = ts_labs.filter(
                                    pl.col("labevents/itemid") == itemid
                                )
                                if matching.height > 0:
                                    category_value = matching["labevents/valuenum"][0]
                                    break
                            lab_vector.append(category_value)

                        time_from_admission = (
                            lab_ts - admission_time
                        ).total_seconds() / 3600.0

                        all_lab_values.append(lab_vector)
                        all_lab_times.append(time_from_admission)

        if target_los_category is None:
            return []

        if len(all_lab_values) == 0 or len(all_icd_codes) == 0:
            return []

        icd_codes_data = (all_icd_times, all_icd_codes)
        labs_data = (all_lab_times, all_lab_values)

        sample = {
            "patient_id": patient.patient_id,
            "icd_codes": icd_codes_data,
            "labs": labs_data,
            "los": target_los_category,
        }
        return [sample]
