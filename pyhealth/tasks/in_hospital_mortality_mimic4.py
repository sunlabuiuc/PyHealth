from datetime import datetime, timedelta
from typing import Any, Dict, List, ClassVar
import logging

import polars as pl

from .base_task import BaseTask

# Set up logging
logger = logging.getLogger(__name__)

class InHospitalMortalityMIMIC4(BaseTask):
    """Task for predicting in-hospital mortality using MIMIC-IV dataset."""
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
    
    # Create flat list of all lab items
    LABITEMS: ClassVar[List[str]] = [
        item for category in LAB_CATEGORIES.values() 
        for subcategory in category.values() 
        for item in subcategory
    ]

    def __init__(self, input_window_hours: int = 48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_window_hours = input_window_hours
        # Log our lab item IDs to check formatting
        logger.info(f"Initialized with {len(self.LABITEMS)} lab items: {self.LABITEMS[:5]}...")

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient for mortality prediction."""
        samples = []
        logger.info(f"Processing patient {patient.patient_id}")

        demographics = patient.get_events(event_type="patients")
        if not demographics:
            logger.info(f"No demographics for patient {patient.patient_id}")
            return []
            
        demographics = demographics[0]
        
        # More lenient age checking
        try:
            anchor_age = int(float(demographics.anchor_age))
            if anchor_age < 18:
                logger.info(f"Patient {patient.patient_id} under 18 (age: {anchor_age})")
                return []
        except (ValueError, AttributeError) as e:
            logger.info(f"Could not determine age for patient {patient.patient_id}: {e}")
            # Continue anyway
    
        admissions = patient.get_events(event_type="admissions")
        logger.info(f"Patient {patient.patient_id} has {len(admissions)} admissions")

        for idx, admission in enumerate(admissions):
            try:
                admission_dischtime = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
                duration_hour = (admission_dischtime - admission.timestamp).total_seconds() / 3600
                logger.info(f"Admission {idx} duration: {duration_hour} hours")
                
                if duration_hour <= self.input_window_hours:
                    logger.info(f"Admission {idx} too short (needs > {self.input_window_hours} hours)")
                    continue
                    
                predict_time = admission.timestamp + timedelta(hours=self.input_window_hours)
            except (ValueError, AttributeError) as e:
                logger.info(f"Error processing timestamps for admission {idx}: {e}")
                continue

            # Get lab events
            labevents_raw = patient.get_events(
                event_type="labevents",
                start=admission.timestamp,
                end=predict_time,
                return_df=True
            )
            
            if labevents_raw is None or labevents_raw.height == 0:
                logger.info(f"No lab events for admission {idx}")
                continue
                
            # Check what itemids are available 
            if "labevents/itemid" in labevents_raw.columns:
                available_ids = set(labevents_raw["labevents/itemid"].to_list())
                matching_ids = [id for id in self.LABITEMS if id in available_ids]
                logger.info(f"Found {len(matching_ids)}/{len(self.LABITEMS)} matching lab IDs")
                if not matching_ids:
                    # Try with integer IDs instead of strings
                    try:
                        int_labitems = [int(id) for id in self.LABITEMS]
                        available_int_ids = set(int(id) for id in available_ids if id.isdigit())
                        matching_int_ids = [id for id in int_labitems if id in available_int_ids]
                        logger.info(f"Found {len(matching_int_ids)}/{len(int_labitems)} matching integer lab IDs")
                        
                        # Use integer IDs for filtering if that's what we found
                        if matching_int_ids:
                            labevents_df = labevents_raw.filter(
                                pl.col("labevents/itemid").cast(pl.Int64).is_in(matching_int_ids)
                            )
                        else:
                            logger.info("No matching lab IDs found in either string or integer format")
                            continue
                    except Exception as e:
                        logger.info(f"Error trying integer IDs: {e}")
                        continue
                else:
                    # Use string IDs as planned
                    labevents_df = labevents_raw.filter(
                        pl.col("labevents/itemid").is_in(matching_ids)
                    )
            else:
                logger.info(f"Missing labevents/itemid column. Available columns: {labevents_raw.columns}")
                continue

            # Continue processing as before
            try:
                # Process timestamps
                if "labevents/storetime" in labevents_df.columns:
                    labevents_df = labevents_df.with_columns(
                        pl.col("labevents/storetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                    )
                    labevents_df = labevents_df.filter(
                        (pl.col("labevents/storetime") <= predict_time)
                    )
                
                if labevents_df.height == 0:
                    logger.info(f"No lab events after timestamp filtering for admission {idx}")
                    continue

                # Process values
                if "labevents/valuenum" in labevents_df.columns:
                    labevents_df = labevents_df.select(
                        pl.col("timestamp"),
                        pl.col("labevents/itemid"),
                        pl.col("labevents/valuenum").cast(pl.Float64)
                    )
                else:
                    logger.info(f"Missing valuenum column. Available columns: {labevents_df.columns}")
                    continue
                    
                # Pivot and process
                try:
                    labevents_df = labevents_df.pivot(
                        index="timestamp",
                        columns="labevents/itemid",
                        values="labevents/valuenum"
                    )
                    labevents_df = labevents_df.sort("timestamp")
                except Exception as e:
                    logger.info(f"Error during pivot: {e}")
                    continue

                # Add any missing columns and reorder
                existing_cols = set(labevents_df.columns) - {"timestamp"}
                logger.info(f"Available columns after pivot: {len(existing_cols)} columns")
                
                # Use the matching IDs we found earlier instead of all LABITEMS
                for col in matching_ids:
                    if col not in existing_cols:
                        labevents_df = labevents_df.with_columns(pl.lit(None).alias(col))
                
                # Select only timestamp and available columns
                labevents_df = labevents_df.select(
                    "timestamp",
                    *[col for col in matching_ids if col in labevents_df.columns]
                )
                
                # Create sample
                timestamps = labevents_df["timestamp"].to_list()
                lab_values = labevents_df.drop("timestamp").to_numpy()
                
                # Check if we have enough data
                if len(timestamps) == 0 or lab_values.size == 0:
                    logger.info(f"Empty lab values for admission {idx}")
                    continue

                # Determine mortality label
                try:
                    mortality = int(admission.hospital_expire_flag)
                except (ValueError, AttributeError):
                    logger.info(f"Using default mortality=0 for admission {idx}")
                    mortality = 0

                samples.append({
                    "patient_id": patient.patient_id,
                    "admission_id": admission.hadm_id,
                    "labs": (timestamps, lab_values),
                    "mortality": mortality,
                })
                logger.info(f"Successfully created sample for admission {idx}")
                
            except Exception as e:
                logger.info(f"Error processing lab events for admission {idx}: {e}")
                continue

        logger.info(f"Generated {len(samples)} samples for patient {patient.patient_id}")
        return samples