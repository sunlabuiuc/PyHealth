from typing import Any, Dict, List
import polars as pl
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import BaseTask

class MIMIC3ConceptEmbedding(BaseTask):
    """
    A task for generating concept sequences from MIMIC-III patient visits for
    use in unsupervised or self-supervised medical concept embedding.

    For each visit in a patient's history, this task extracts a list of medical
    concepts including ICD-9 diagnosis codes, ICD-9 procedure codes, and prescribed drug names.

    Each output sample corresponds to one hospital visit, containing:
    - patient_id: Unique patient identifier
    - visit_id: Unique hospital admission identifier
    - concepts: A list of medical concept codes/strings (can include duplicates unless removed)

    This format is suitable for training concept embeddings using sequence-based
    models (e.g., Word2Vec, FastText, Transformer-based encoders).

    Input Schema:
        {
            "concepts": "sequence"  # A list of concept codes per visit
        }

    Output Schema:
        {
            "concepts": "sequence"  # Same format as input, used for embedding learning
        }

    Args:
        remove_duplicate_concepts (bool): If True, deduplicates concepts within a visit.
    """

    task_name = "concept_embedding"

    input_schema = {
    "concepts": "sequence"
    }
    output_schema = {
        "concepts": "sequence"
    }

    def __init__(self, remove_duplicate_concepts: bool = False):
        super().__init__()
        self.remove_duplicate_concepts = remove_duplicate_concepts

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generate concept embeddings for a single patient from MIMIC-III."""
        samples = []
        
        # Get all visits for the patient
        visits = patient.get_events(event_type="admissions")
        
        for visit in visits:
            # For each visit, extract relevant clinical concepts
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                start=visit.timestamp
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                start=visit.timestamp
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=visit.timestamp
            )
            
            # Extract ICD-9 codes, procedures, and drug names
            conditions = [event.icd9_code for event in diagnoses]
            procedures_list = [event.icd9_code for event in procedures]
            drugs = [event.drug for event in prescriptions]
            
            # Combine concepts into a single list
            all_concepts = conditions + procedures_list + drugs

            # Remove duplicate concepts
            if self.remove_duplicate_concepts:
                all_concepts = list(set(all_concepts))
            
            # Exclude visits with no concepts
            if len(all_concepts) == 0:
                continue
            
            # Create a sample with patient and visit info
            sample = {
                "patient_id": patient.patient_id,
                "visit_id": visit.hadm_id,
                "concepts": all_concepts,
            }
            samples.append(sample)
        
        return samples
