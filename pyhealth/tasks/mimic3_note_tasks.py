# pyhealth/tasks/mimic3_note_tasks.py

import re
import os
import nltk
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataclasses import dataclass, field
from pyhealth.data import Patient
#from pyhealth.datasets import MIMIC3NoteDataset
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# Ensure nltk packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class MIMIC3NoteReplaceDeIdTask(BaseTask):
    """Task for de-identifying personal health information in MIMIC-III clinical notes.
    
    This task identifies and removes/masks PHI in clinical notes based on pattern detection
    and rule-based methods. It directly updates the patient entity with deidentified text.
    
    """
    task_name: str = "Deidentifying MIMIC-III Clinical Notes"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"text": "text"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"masked_text": "text"})

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes a patient for the de-identification marking task and updates the patient entity directly.
        
        Args:
            patient: Patient object containing clinical notes
            
        Returns:
            List of dictionaries with patient_id, text, masked_text and timestamps fields
        """
        samples = []
        
        # Get all note events
        note_events = patient.get_events(event_type="noteevents")
        
        # Process each note event
        for event in note_events:
            text = event.attr_dict.get("text", "")
            if not text:
                continue
                
            # Process the note and update the patient entity directly
            masked_text = self.process_note(text)
            
            # Update the event text with the deidentified version
            event.attr_dict["text"] = masked_text
            event.attr_dict["phimasked"] = True  # Add a flag to indicate this note has been deidentified
            
            # Create a sample for return value

            sample = {
                "patient_id": patient.patient_id,
                "text": text,
                "masked_text": masked_text,
                "timestamp": event.timestamp,                
            }            


            samples.append(sample)
            
        # Log the number of processed notes
        #logger.info(f"Deidentified {len(samples)} notes for patient {patient.patient_id}")
        
        return samples
    
    def is_date(self, s: str) -> bool:
        """Check if a string contains a date or time-related information.
        Args:
            s (str): The string to check.
        Returns:
            bool: True if the string contains date or time-related information, False otherwise.
        """
        return bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}", s) or
                    "month" in s or "year" in s)
    
    def replace_deid(self, text: str) -> str:
        """Replaces [** ... **] tags with PHI placeholders.
        Args:
            text (str): Clinical note text.
        Returns:
            str: Text with PHI replaced by placeholder tokens.
        """
        
        return re.sub(r"\[\*\*(.*?)\*\*\]", self.repl_deid, text)
    
    def repl_deid(self, match: re.Match) -> str:
        """De-identify text by replacing PHI with mask strings.
        Args:
            match (re.Match): Regex match object containing the text to be replaced.
        Returns:
            str: The replacement string based on the matched text.
        """

        low_label = match.group(1).lower().strip()
        date = self.is_date(low_label)

        if date or 'holiday' in low_label:
            label = 'PHIDATEPHI'

        elif 'hospital' in low_label:
            label = 'PHIHOSPITALPHI'

        elif ('location' in low_label
            or 'url ' in low_label
            or 'university' in low_label
            or 'address' in low_label
            or 'po box' in low_label
            or 'state' in low_label
            or 'country' in low_label
            or 'company' in low_label):
            label = 'PHILOCATIONPHI'


        elif ('name' in low_label
            or 'dictator info' in low_label
            or 'contact info' in low_label
            or 'attending info' in low_label):
            label = 'PHINAMEPHI'

        elif 'telephone' in low_label:
            label = 'PHICONTACTPHI'

        elif ('job number' in low_label
                or 'number' in low_label
                or 'numeric identifier' in low_label
                or re.search('^\d+$', low_label)
                or re.search('^[\d-]+$', low_label)
                or re.search('^[-\d/]+$', low_label)):
            label = 'PHINUMBERPHI'

        elif 'age over 90' in low_label:
            label = 'PHIAGEPHI'

        else:
            label = 'PHIOTHERPHI'

        return label
   



    
    def mask_gendered_terms(self, text: str) -> str:
        
        """Replace gendered terms with a mask.
            Args:
            text (str): Clinical note text.     
        
            Returns:
            str: Text with gendered terms replaced by a placeholder.
        """

        text = re.sub(
            r"\b(male|gentleman|man|he|female|woman|she|his|her|him)\b",
            "[GEND]",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r"Sex:\s*[MFmf]", "Sex: [GEND]", text, flags=re.IGNORECASE)
        return text

    def clean_note_format(self, note: str) -> str:
        """Clean and standardize the format of clinical notes.
        Args:
            note (str): Clinical note text.
        Returns:
            str: Cleaned and standardized clinical note text.
        """
        note = re.sub(r'\n', ' ', note)
        note = re.sub(r'[0-9]+\.', '', note)

        note = re.sub(r'[-_=]{2,}', '', note)
        note = re.sub(r'\bdr\.', 'doctor', note, flags=re.IGNORECASE)
        note = re.sub(r'\bm\.d\.', 'md', note, flags=re.IGNORECASE)
        #note = re.sub(r'[ -_=]{2,}', '', note)
        note = re.sub(r"\s+", " ", note)
        
        return note.strip()

    def process_note(self, note: str) -> str:
        """Process a clinical note through the complete pipeline.
            Args:
            note (str): Clinical note text.     
            Returns:
            str: Processed clinical note text.  
        """
        # Apply all text processing steps in sequence
        #sections = process_note_helper(note)
        note = self.replace_deid(note)
        note = self.mask_gendered_terms(note)
        note = self.clean_note_format(note)
        return note

if __name__ == "__main__":
    task = MIMIC3NoteDeidentificationTask()
