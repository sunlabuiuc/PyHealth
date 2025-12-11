from typing import List, Dict, Optional
from pyhealth.data import Patient


def next_visit_diagnosis_prediction_mimic4_fn(
    patient: Patient,
    time_aware: bool = True
) -> List[Dict]:
    """
    Processes a single patient for next-visit diagnosis prediction task.
    
    This task uses all previous hospital visits to predict diagnoses that
    will occur in the patient's next visit. Each sample represents a prediction
    point where we use visit history (0 to N-1) to predict visit N.
    
    The task is formulated as a multi-label classification problem where:
    - **Features**: Historical diagnosis codes from all previous visits
    - **Label**: Diagnosis codes from the next visit (multi-hot vector)
    - **Temporal information**: Days between visits (optional)

    Args:
        patient: A Patient object from MIMIC-IV dataset containing visit history
        time_aware: If True, includes time gaps between visits as features.
                   Time gaps are calculated as days between consecutive visits.
                   Default is True.
        
    Returns:
        samples: List of samples, one per prediction point. Each sample is a
                dictionary with:
                
                - **patient_id** (str): Unique patient identifier
                - **visit_id** (str): ID of the target visit being predicted
                - **conditions_history** (list of lists): Diagnosis codes from 
                  each previous visit. Each inner list contains codes from one visit.
                  Format: [[visit_1_codes], [visit_2_codes], ...]
                - **procedures_history** (list of lists): Procedure codes from
                  each previous visit (optional, for richer features)
                - **time_gaps** (list of int): Days between consecutive visits.
                  Only included if time_aware=True.
                  Format: [days_between_v1_v2, days_between_v2_v3, ...]
                - **label** (list of str): Diagnosis codes from the target visit
                  (what we're predicting)
    
    Note:
        - Requires patients with at least 2 visits
        - Patients with single visits are filtered out (return empty list)
        - Visits without any diagnosis codes in the target visit are skipped
        - Time gaps are computed only between consecutive visits with valid timestamps
        
    """
    samples = []
    
    # Filter: Need at least 2 visits (one for features, one for label)
    if len(patient) < 2:
        return samples
    
    # Create one sample for each possible prediction point
    # Use visits 0 to i-1 to predict visit i
    for target_idx in range(1, len(patient)):
        
        # Initialize feature lists
        conditions_history = []
        procedures_history = []
        time_gaps = []
        
        # Track previous visit time for time gap calculation
        prev_encounter_time = None
        
        # Collect features from all previous visits (0 to target_idx-1)
        for hist_idx in range(target_idx):
            visit = patient[hist_idx]
            
            # Extract diagnosis codes from this historical visit
            conditions = visit.get_code_list(table="diagnoses_icd")
            conditions_history.append(conditions)
            
            # Extract procedure codes (optional, provides richer context)
            procedures = visit.get_code_list(table="procedures_icd")
            procedures_history.append(procedures)
            
            # Calculate time gaps between consecutive visits
            if time_aware:
                current_time = visit.encounter_time
                
                if prev_encounter_time is not None and current_time is not None:
                    # Time gap in days between consecutive visits
                    days_gap = (current_time - prev_encounter_time).days
                    time_gaps.append(days_gap)
                
                if current_time is not None:
                    prev_encounter_time = current_time
        
        # Get the target visit (what we're predicting)
        target_visit = patient[target_idx]
        
        # Extract label: diagnosis codes from target visit
        label = target_visit.get_code_list(table="diagnoses_icd")
        
        #  Skip if target visit has no diagnoses
        if len(label) == 0:
            continue
        
        # Assemble the sample
        sample = {
            "patient_id": patient.patient_id,
            "visit_id": target_visit.visit_id,
            "conditions_history": conditions_history,
            "procedures_history": procedures_history,
            "label": label,
        }
        
        # Add time gaps if time-aware mode is enabled
        if time_aware:
            sample["time_gaps"] = time_gaps
        
        samples.append(sample)
    
    return samples


def next_visit_diagnosis_prediction_mimic3_fn(
    patient: Patient,
    time_aware: bool = True
) -> List[Dict]:
    """
    Next-visit diagnosis prediction for MIMIC-III dataset.
    
    This is identical to the MIMIC-IV version since both datasets use the same
    table structure (diagnoses_icd, procedures_icd).
    
    Args:
        patient: A Patient object from MIMIC-III dataset
        time_aware: If True, includes time gaps between visits
        
    Returns:
        samples: List of prediction samples

    """
    return next_visit_diagnosis_prediction_mimic4_fn(patient, time_aware=time_aware)


def next_visit_diagnosis_prediction_eicu_fn(
    patient: Patient,
    time_aware: bool = True
) -> List[Dict]:
    """
    Next-visit diagnosis prediction for eICU dataset.
    
    Adapted for eICU's table structure which uses "diagnosis" instead of
    "diagnoses_icd".
    
    Args:
        patient: A Patient object from eICU dataset
        time_aware: If True, includes time gaps between visits
        
    Returns:
        samples: List of prediction samples
        
    """
    samples = []
    
    if len(patient) < 2:
        return samples
    
    for target_idx in range(1, len(patient)):
        conditions_history = []
        time_gaps = []
        prev_encounter_time = None
        
        for hist_idx in range(target_idx):
            visit = patient[hist_idx]
            
            # eICU uses "diagnosis" table instead of "diagnoses_icd"
            conditions = visit.get_code_list(table="diagnosis")
            conditions_history.append(conditions)
            
            if time_aware:
                current_time = visit.encounter_time
                if prev_encounter_time is not None and current_time is not None:
                    days_gap = (current_time - prev_encounter_time).days
                    time_gaps.append(days_gap)
                if current_time is not None:
                    prev_encounter_time = current_time
        
        target_visit = patient[target_idx]
        label = target_visit.get_code_list(table="diagnosis")
        
        if len(label) == 0:
            continue
        
        sample = {
            "patient_id": patient.patient_id,
            "visit_id": target_visit.visit_id,
            "conditions_history": conditions_history,
            "label": label,
        }
        
        if time_aware:
            sample["time_gaps"] = time_gaps
        
        samples.append(sample)
    
    return samples