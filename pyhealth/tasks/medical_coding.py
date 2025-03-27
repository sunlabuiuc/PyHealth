# Author: John Wu
# NetID: johnwu3
# Description: Medical coding tasks for MIMIC-III and MIMIC-IV datasets

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from tqdm import tqdm

from pyhealth.data.data import Patient, Event
from pyhealth.tasks.task_template import TaskTemplate
from pyhealth.datasets.base_dataset_v2 import BaseDataset

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MIMIC3ICD9Coding(TaskTemplate):
    """Medical coding task for MIMIC-III using ICD-9 codes.
    
    This task uses clinical notes to predict ICD-9 codes for a patient.
    
    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """
    task_name: str = "mimic3_icd9_coding"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"text": "str"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"icd_codes": "List[str]"})

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract the clinical notes and ICD-9 codes.
        
        Args:
            patient: Patient object containing events
            
        Returns:
            List of samples, each containing text and ICD codes
        """
        text = ""
        icd_codes = set()
        
        for event in patient.events:
            event_type = event.type.upper() if isinstance(event.type, str) else ""
            
            if event_type == "NOTEEVENTS":
                if "code" in event.attr_dict:
                    text += event.attr_dict["code"]
            
            if event_type == "DIAGNOSES_ICD":
                if "code" in event.attr_dict:
                    icd_codes.add(event.attr_dict["code"])
            
            if event_type == "PROCEDURES_ICD":
                if "code" in event.attr_dict:
                    icd_codes.add(event.attr_dict["code"])
        
        if text == "" or len(icd_codes) < 1:
            return []
        
        return [{"text": text, "icd_codes": list(icd_codes)}]


@dataclass(frozen=True)
class MIMIC4ICD9Coding(TaskTemplate):
    """Medical coding task for MIMIC-IV using ICD-9 codes.
    
    This task uses discharge notes to predict ICD-9 codes for a patient.
    
    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """
    task_name: str = "mimic4_icd9_coding"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"text": "str"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"icd_codes": "List[str]"})

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract the discharge notes and ICD-9 codes."""
        text = ""
        icd_codes = set()
        
        for event in patient.events:
            event_type = event.type.lower() if isinstance(event.type, str) else ""
            
            # Look for "value" instead of "code" for clinical notes
            if event_type == "clinical_note":
                if "value" in event.attr_dict:
                    text += event.attr_dict["value"]
            
            vocabulary = event.attr_dict.get("vocabulary", "").upper()
            if vocabulary == "ICD9CM":
                if event_type == "diagnoses_icd" or event_type == "procedures_icd":
                    if "code" in event.attr_dict:
                        icd_codes.add(event.attr_dict["code"])
        
        if text == "" or len(icd_codes) < 1:
            return []
        
        return [{"text": text, "icd_codes": list(icd_codes)}]


@dataclass(frozen=True)
class MIMIC4ICD10Coding(TaskTemplate):
    """Medical coding task for MIMIC-IV using ICD-10 codes.
    
    This task uses discharge notes to predict ICD-10 codes for a patient.
    
    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """
    task_name: str = "mimic4_icd10_coding"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"text": "str"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"icd_codes": "List[str]"})

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract the discharge notes and ICD-9 codes."""
        text = ""
        icd_codes = set()
        
        for event in patient.events:
            event_type = event.type.lower() if isinstance(event.type, str) else ""
            
            # Look for "value" instead of "code" for clinical notes
            if event_type == "clinical_note":
                if "value" in event.attr_dict:
                    text += event.attr_dict["value"]
            
            vocabulary = event.attr_dict.get("vocabulary", "").upper()
            if vocabulary == "ICD10CM":
                if event_type == "diagnoses_icd" or event_type == "procedures_icd":
                    if "code" in event.attr_dict:
                        icd_codes.add(event.attr_dict["code"])
        
        if text == "" or len(icd_codes) < 1:
            return []
        
        return [{"text": text, "icd_codes": list(icd_codes)}]


def main():
    # Test case for MIMIC4ICD9Coding and MIMIC3
    from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
    
    root = "/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4"
    print("Testing MIMIC3ICD9Coding task...")
    dataset = MIMIC3Dataset(
        root=root,
        dataset_name="mimic3",  
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "NOTEEVENTS"
        ],
        code_mapping={"NDC": "ATC"},
        dev=True,
    )
    mimic3_coding = MIMIC3ICD9Coding()
    # print(len(mimic3_coding.samples))
    samples = dataset.set_task(mimic3_coding)
    # Print sample information
    print(f"Total samples generated: {len(samples)}")
    if len(samples) > 0:
        print("First sample:")
        print(f"  - Text length: {len(samples[0]['text'])} characters")
        print(f"  - Number of ICD codes: {len(samples[0]['icd_codes'])}")
        if len(samples[0]['icd_codes']) > 0:
            print(f"  - Sample ICD codes: {samples[0]['icd_codes'][:5] if len(samples[0]['icd_codes']) > 5 else samples[0]['icd_codes']}")

    # Initialize the dataset with dev mode enabled
    print("Testing MIMIC4ICD9Coding task...")
    dataset = MIMIC4Dataset(
        root="/srv/local/data/MIMIC-IV/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd"],
        note_root="/srv/local/data/MIMIC-IV/2.0/note",
        dev=True,
    )
    # Create the task instance
    mimic4_coding = MIMIC4ICD9Coding()
    
    # Generate samples
    samples = dataset.set_task(mimic4_coding)
    
    # Print sample information
    print(f"Total samples generated: {len(samples)}")
    if len(samples) > 0:
        print("First sample:")
        print(f"  - Text length: {len(samples[0]['text'])} characters")
        print(f"  - Number of ICD codes: {len(samples[0]['icd_codes'])}")
        if len(samples[0]['icd_codes']) > 0:
            print(f"  - Sample ICD codes: {samples[0]['icd_codes'][:5] if len(samples[0]['icd_codes']) > 5 else samples[0]['icd_codes']}")

    print("Testing MIMIC4ICD10Coding task... ")

    mimic4_coding = MIMIC4ICD10Coding()
    
    # Generate samples
    samples = dataset.set_task(mimic4_coding)
    
    # Print sample information
    print(f"Total samples generated: {len(samples)}")
    if len(samples) > 0:
        print("First sample:")
        print(f"  - Text length: {len(samples[0]['text'])} characters")
        print(f"  - Number of ICD codes: {len(samples[0]['icd_codes'])}")
        if len(samples[0]['icd_codes']) > 0:
            print(f"  - Sample ICD codes: {samples[0]['icd_codes'][:5] if len(samples[0]['icd_codes']) > 5 else samples[0]['icd_codes']}")
if __name__ == "__main__":
    main()