from typing import Any, Dict, List
from .base_task import BaseTask
import polars as pl


class CXRBiasMitigationSamplingTask(BaseTask):
    """
    A task that performs sampling of chest X-ray metadata and associated patient information,
    targeting bias mitigation by filtering for the availability of demographic data.
    """
    task_name: str = "CXRBiasMitigationSamplingTask"
    input_schema: Dict[str, str] = {"path": "image"}
    output_schema: Dict[str, str] = {
        "dicom_id": "raw",
        "subject_id": "raw",
        "study_id": "raw",
        "ViewPosition": "raw",
        "path": "raw",
        "gender": "raw",
        "insurance": "raw",
        "ethnicity": "raw",
        "marital_status": "raw",
        "Atelectasis": "binary",
        "Cardiomegaly": "binary",
        "Consolidation": "binary",
        "Edema": "binary",
        "Enlarged Cardiomediastinum": "binary",
        "Fracture": "binary",
        "Lung Lesion": "binary",
        "Lung Opacity": "binary",
        "No Finding": "binary",
        "Pleural Effusion": "binary",
        "Pleural Other": "binary",
        "Pneumonia": "binary",
        "Pneumothorax": "binary",
        "Support Devices": "binary"
    }

    def __init__(self):
        self.allowed_view_positions = ["PA", "AP"]

    def pre_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the input DataFrame to retain only patients who have both gender, race, and
        insurance information available.
        """
        patients_with_gender = (
            df.filter((pl.col("event_type") == "patients") & pl.col("patients/gender").is_not_null())
            .select("patient_id").unique().get_column("patient_id").to_list()
        )

        patients_with_race = (
            df.filter((pl.col("event_type") == "admissions") & pl.col("admissions/insurance").is_not_null())
            .select("patient_id").unique().get_column("patient_id").to_list()
        )

        patients_with_insurance = (
            df.filter((pl.col("event_type") == "admissions") & pl.col("admissions/race").is_not_null())
            .select("patient_id").unique().get_column("patient_id").to_list()
        )

        valid_ids = set(patients_with_gender) & set(patients_with_race) & set(patients_with_insurance)

        return df.filter(pl.col("patient_id").is_in(valid_ids))

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """
        Extracts and merges bias-related metadata for a given patient,
        returning the processed data as a list of dictionaries.
        """
        md_df = patient.get_events(event_type="metadata", return_df=True)
        pt_df = patient.get_events(event_type="patients", return_df=True)
        ad_df = patient.get_events(event_type="admissions", return_df=True)
        lb_df = patient.get_events(event_type="chexpert", return_df=True)

        # Exit early if no metadata exits
        if md_df.is_empty():
            return []

        md_df = md_df.filter(pl.col("metadata/viewposition").is_in(self.allowed_view_positions))

        # Again exit if no metadata exists after filtering
        if md_df.is_empty():
            return []

        base = md_df.select([
            pl.col("metadata/dicom_id").alias("dicom_id"),
            pl.col("patient_id").alias("subject_id"),
            pl.col("metadata/study_id").alias("study_id"),
            pl.col("metadata/viewposition").alias("ViewPosition"),
            pl.col("metadata/image_path").alias("path")
        ])
        base = base.with_columns(pl.lit(patient.patient_id).alias("subject_id"))

        # Join gender data from patient info if available
        if not pt_df.is_empty():
            pt_df = pt_df.select([
                pl.lit(patient.patient_id).alias("subject_id"),
                pl.col("patients/gender").alias("gender")
            ])
            base = base.join(pt_df, on="subject_id", how="left")

        # Join insurance, ethnicity, and marital status from admission data if available
        if not ad_df.is_empty():
            ad_df = (
                ad_df.select([
                    pl.lit(patient.patient_id).alias("subject_id"),
                    pl.col("admissions/insurance").alias("insurance"),
                    pl.col("admissions/race").alias("ethnicity"),
                    pl.col("admissions/marital_status").alias("marital_status")
                ])
                .unique(subset=["subject_id"], keep="last")
            )
            base = base.join(ad_df, on="subject_id", how="left")

        # Join CheXpert labels if available
        if not lb_df.is_empty():
            label_cols = [
                "atelectasis", "cardiomegaly", "consolidation", "edema",
                "enlarged cardiomediastinum", "fracture", "lung lesion",
                "lung opacity", "no finding", "pleural effusion",
                "pleural other", "pneumonia", "pneumothorax", "support devices"
            ]
            rename_map = {f"chexpert/{c}": c.title() for c in label_cols}
            lb_df = lb_df.rename(rename_map)

            base = base.join(lb_df, left_on="dicom_id", right_on="chexpert/dicom_id", how="left")
        selected_keys = list(self.output_schema.keys())
        base = base.select([k for k in selected_keys if k in base.columns])

        return base.to_dicts()
