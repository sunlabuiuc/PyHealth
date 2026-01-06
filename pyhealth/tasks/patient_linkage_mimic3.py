from datetime import datetime
from collections import defaultdict
import math
from pyhealth.tasks import BaseTask

class PatientLinkageMIMIC3Task(BaseTask):
    """
    Patient linkage task for MIMIC-III.

    For each patient with >=2 admissions:
    - Query: last admission
    - Positive database record: all previous admissions concatenated
    
    This creates ONE positive pair per patient.
    Negatives are sampled later during training (in-batch + hard negatives).
    """

    task_name = "patient_linkage_mimic3"
    
    input_schema = {
        # Query side (last admission)
        "q_conditions": "sequence",
        "q_timestamp": "datetime",
        "q_visit_id": "string",
        
        # Positive database side (all previous admissions concatenated)
        "d_conditions": "sequence",  # concatenated with [SEP] tokens
        "d_timestamp": "datetime",   # timestamp of most recent prior admission
        "d_visit_ids": "string",     # pipe-separated list of hadm_ids
        
        # Metadata
        "patient_id": "string",      # ground truth for evaluation
        "time_gap_days": "integer",  # for analysis by time interval
    }
    
    output_schema = {}  # Task just creates pairs; model outputs matching scores

    def __call__(self, patient):
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        admissions = sorted(admissions, key=lambda e: e.timestamp)
        
        # Query: LAST admission
        q_visit = admissions[-1]
        
        # Database: ALL PREVIOUS admissions
        d_visits = admissions[:-1]

        # Get demographics
        patients_events = patient.get_events(event_type="patients")
        if not patients_events:
            return []
        demo = patients_events[0]

        dob_raw = demo.attr_dict.get("dob")
        birth_dt = None
        if isinstance(dob_raw, datetime):
            birth_dt = dob_raw
        elif dob_raw is not None:
            try:
                birth_dt = datetime.fromisoformat(str(dob_raw))
            except Exception:
                birth_dt = None

        def compute_age(ts):
            if birth_dt is None or ts is None:
                return None
            return int((ts - birth_dt).days // 365.25)

        # Age check for query
        q_age = compute_age(q_visit.timestamp)
        if q_age is None or q_age < 18:
            return []

        # Get all diagnosis codes
        diag_events = patient.get_events(event_type="diagnoses_icd")
        hadm_to_codes = defaultdict(list)
        for ev in diag_events:
            hadm = ev.attr_dict.get("hadm_id")
            code = ev.attr_dict.get("icd9_code")
            if hadm is None or code is None:
                continue
            hadm_to_codes[str(hadm)].append(str(code))

        # Query conditions
        q_hadm = str(q_visit.attr_dict.get("hadm_id"))
        q_conditions = hadm_to_codes.get(q_hadm, [])
        if len(q_conditions) == 0:
            return []

        # Database conditions: CONCATENATE all previous admissions
        d_conditions = []
        d_hadm_ids = []
        d_most_recent_timestamp = None
        
        for d_visit in d_visits:
            d_age = compute_age(d_visit.timestamp)
            if d_age is None or d_age < 18:
                continue
                
            d_hadm = str(d_visit.attr_dict.get("hadm_id"))
            d_codes = hadm_to_codes.get(d_hadm, [])
            
            if len(d_codes) > 0:
                # Add separator between admissions
                if len(d_conditions) > 0:
                    d_conditions.append("[SEP]")
                
                d_conditions.extend(d_codes)
                d_hadm_ids.append(d_hadm)
                d_most_recent_timestamp = d_visit.timestamp
        
        if len(d_conditions) == 0:
            return []

        # Calculate time gap between query and most recent database record
        time_gap_days = None
        if d_most_recent_timestamp and q_visit.timestamp:
            time_gap_days = (q_visit.timestamp - d_most_recent_timestamp).days

        sample = {
            "patient_id": patient.patient_id,
            
            # Query side
            "q_visit_id": q_hadm,
            "q_conditions": [""] + q_conditions,  # Add CLS token like logic (handled usually by tokenizer but keeping empty string consistency from prev implementation if needed, though typically done in processor. Keeping consistent with previous [""] prepend behavior for now)
            "q_timestamp": q_visit.timestamp,
            
            # Database side (concatenated)
            "d_visit_ids": "|".join(d_hadm_ids),
            "d_conditions": [""] + d_conditions,
            "d_timestamp": d_most_recent_timestamp,
            
            # Metadata
            "time_gap_days": time_gap_days,
        }
        
        return [sample]
