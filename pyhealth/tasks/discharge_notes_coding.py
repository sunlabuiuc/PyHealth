# Author: Fabricio Brigagao
# NetID: fb8
# Description: Generates discharge notes associated with 19 top-level ICD-9 categories.
# Reason: Used by the paper Classifying Unstructured Clinical Notes via Automatic Weak Supervision (https://arxiv.org/abs/2206.12088).

import logging
from dataclasses import field
from datetime import datetime
from typing import Dict, List

import polars as pl

from pyhealth.data.data import Patient

from .base_task import BaseTask

logger = logging.getLogger(__name__)

class MIMIC3DischargeNotesICD9Coding(BaseTask):
    """Medical Discharge Notes Task for MIMIC-III mapped to 19 Top-Level ICD-9 Categories.
    
    This task generates discharge notes associated with 19 Top-Level ICD-9 Categories used by the KeyClass paper:
        - Classifying Unstructured Clinical Notes via Automatic Weak Supervision (https://arxiv.org/abs/2206.12088).
    It builds upon the dataset generation from FasTag (used by KeyClass), mapping discharge notes to 19 top-level categories. 
        https://github.com/rivas-lab/FasTag/blob/master/src/textPreprocessing/createAdmissionNoteTable.R
    
    ICD-9 top-level ICD9 Categories Mappings for KeyClass
    Range (3-digit)   →   Cat code   →   Category name
    ------------------------------------------------------------------------
        001 to 139    →   cat:01     →   Infectious & parasitic diseases
        140 to 239    →   cat:02     →   Neoplasms
        240 to 279    →   cat:03     →   Endocrine, nutritional & metabolic
        280 to 289    →   cat:04     →   Diseases of the blood & blood-forming organs
        290 to 319    →   cat:05     →   Mental disorders
        320 to 359    →   cat:06     →   Diseases of the nervous system
        360 to 389    →   cat:07     →   Diseases of the sense organs
        390 to 459    →   cat:08     →   Diseases of the circulatory system
        460 to 519    →   cat:09     →   Diseases of the respiratory system
        520 to 579    →   cat:10     →   Diseases of the digestive system
        580 to 629    →   cat:11     →   Diseases of the genitourinary system
        630 to 679    →   cat:12     →   Pregnancy, childbirth & puerperium
        680 to 709    →   cat:13     →   Diseases of the skin & subcutaneous tissue
        710 to 739    →   cat:14     →   Diseases of the musculoskeletal system & connective tissue
        740 to 759    →   cat:15     →   Congenital anomalies
        760 to 779    →   cat:16     →   Certain conditions originating in the perinatal period
        800 to 999    →   cat:17     →   Injury & poisoning
        E-codes       →   cat:18     →   External causes of injury
        V-codes       →   cat:19     →   Supplementary factors influencing health status

    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """
    task_name: str = "mimic3_discharge_notes_icd9_coding"
    input_schema: Dict[str, str] = {"text": "text"}
    output_schema: Dict[str, str] = {"top_level_icd9_cats": "multilabel"}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        filtered_df = df.filter(
            pl.col("patient_id").is_in(
                df.filter(pl.col("event_type") == "noteevents")
                .select("patient_id")
                .unique()
                .to_series()
            )
        )
        return filtered_df
    
    def map_to_toplevel_ICD9(self, code: str):
        """Return the top-level ICD-9 category for a raw ICD-9 code string."""
        if code is None:
            return None

        icd9 = code.strip()
        if not icd9:
            return None

        if icd9.startswith("E"):
            return "cat:18" # "External causes of injury",
        elif icd9.startswith("V"):
            return "cat:19" # "Supplementary"

        try:
            num = int(icd9[:3])
        except ValueError:
            return None          # not a valid numeric code

        if   1   <= num <= 139:  return "cat:01" # Infectious & parasitic"
        elif 140 <= num <= 239:  return "cat:02" # "Neoplasms",
        elif 240 <= num <= 279:  return "cat:03" # "Endocrine, nutritional and metabolic",
        elif 280 <= num <= 289:  return "cat:04" # "Blood & blood-forming organs",
        elif 290 <= num <= 319:  return "cat:05" # "Mental disorders",
        elif 320 <= num <= 359:  return "cat:06" # "Nervous system",
        elif 360 <= num <= 389:  return "cat:07" # "Sense organs",
        elif 390 <= num <= 459:  return "cat:08" # "Circulatory system",
        elif 460 <= num <= 519:  return "cat:09" # "Respiratory system",
        elif 520 <= num <= 579:  return "cat:10" # "Digestive system",
        elif 580 <= num <= 629:  return "cat:11" # "Genitourinary system",
        elif 630 <= num <= 679:  return "cat:12" # "Pregnancy & childbirth complications",
        elif 680 <= num <= 709:  return "cat:13" # "Skin & subcutaneous tissue",
        elif 710 <= num <= 739:  return "cat:14" # "Musculoskeletal system & connective tissue",
        elif 740 <= num <= 759:  return "cat:15" # "Congenital anomalies",
        elif 760 <= num <= 779:  return "cat:16" # "Perinatal period conditions",
        elif 800 <= num <= 999:  return "cat:17" # "Injury and poisoning",
        else:
            return None          # range not covered

    
    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract discharge notes and ICD-9 codes.
        
        Args:
            patient: Patient object containing events
            
        Returns:
            List of samples, each containing the text of the discharge notes and associated top-level ICD-9 Categories.
        """
        samples = []
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            
            selected_note_event = None
            notes_on_earliest_date = None
            earliest_chartdate = None
            noteevents = None

            # For each admission look for DISCHARGE NOTES if they exist.
            noteevents = patient.get_events(
                event_type="noteevents",
                filters = [ ("hadm_id", "==", admission.hadm_id), 
                           ("category", "==", "Discharge summary"),
                        ]
            )

            if noteevents and len(noteevents) > 0:

                # If only 1 discharge note was returned.
                if len(noteevents) == 1:         
                    selected_note_event = noteevents[0]
                else:
                    
                    # More than 1 discharge note was returned.
                    # FasTag's preprocessing (used by KeyClass) maintains the one with min. discharge date.
                    earliest_chartdate = min(ne.chartdate for ne in noteevents if ne.chartdate)
                    notes_on_earliest_date = [ ne for ne in noteevents if ne.chartdate == earliest_chartdate ]
                    
                    # Following FasTag's and KeyClass's approach, keep the first record even if there are multiple discharge notes 
                    # for the same HADM and on the minimum date. FasTag's does not elaborate on this.
                    selected_note_event = notes_on_earliest_date[0]

                if selected_note_event:

                    icd_codes = set()
                    icd_top_level_cats = set()

                    # Get diagnoses ICDs for patient and admission.
                    diagnoses_icd = patient.get_events(
                        event_type="diagnoses_icd",
                        filters = [ ("hadm_id", "==", str(admission.hadm_id)) ]
                    )

                    if len(diagnoses_icd) > 0:

                        # Map ICD-9 codes to an array
                        diagnoses_icd_codes = [event.icd9_code for event in diagnoses_icd]
                        
                        # Drop null/None values and deduplicate
                        icd_codes = list({code for code in diagnoses_icd_codes if code not in (None, "", "nan")})
                        # Get respective top-level ICD-9 category for each ICD-9 code assigned to the discharge notes.
                        icd_top_level = [self.map_to_toplevel_ICD9(code) for code in icd_codes]
                        # Keep only the unique values (removes None and duplicates).
                        icd_top_level_cats = list({cat for cat in icd_top_level if cat not in (None, "", "nan")})

                        if not selected_note_event.text.strip() or not icd_codes:
                            continue    # skip samples with empty text OR no valid ICD-9 codes
                        
                        # Add sample to list
                        samples.append({
                            "patient_id": patient.patient_id,
                            "text": selected_note_event.text,
                            "top_level_icd9_cats": icd_top_level_cats
                        })

        return samples

def main():
    # Test case for MIMIC3DischargeNotesICD9Coding
    from pyhealth.datasets import MIMIC3Dataset
    
    MIMIC_DB_LOCATION = "../mimicdatabase"
    print("Testing MIMIC3DischargeNotesICD9Coding task...")
    dataset = MIMIC3Dataset(
        root=MIMIC_DB_LOCATION,
        dataset_name="mimic3",  
        tables=[
        "diagnoses_icd",
        "noteevents"
        ],
        dev=True,
    )
    mimic3_coding = MIMIC3DischargeNotesICD9Coding()
    # print(len(mimic3_coding.samples))
    samples = dataset.set_task(mimic3_coding)
    # Print sample information
    print(f"Total samples generated: {len(samples)}")
    if len(samples) > 0:
        print("First sample:")
        print(f"  - Patient ID: {samples[0]['patient_id']}")
        print(f"  - Text length: {len(samples[0]['text'])} characters")
        print(f"  - Number of Top Level ICD9 Categories: {len(samples[0]['top_level_icd9_cats'])}")
        if len(samples[0]['top_level_icd9_cats']) > 0:
            print(f"  - Top-Level ICD9 Categories: {samples[0]['top_level_icd9_cats']}")

if __name__ == "__main__":
    main()