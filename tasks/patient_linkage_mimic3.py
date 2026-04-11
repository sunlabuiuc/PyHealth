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

    Example:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import patient_linkage_mimic3
        >>> dataset = MIMIC3Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...     tables=["DIAGNOSES_ICD", "ADMISSIONS", "PATIENTS"],
        ...     code_mapping={"ICD9CM": ("CCSCM", {"target": "dx"})},
        ... )
        >>> sample_dataset = dataset.set_task(patient_linkage_mimic3)
        >>> sample_dataset.samples[0]
        {
            'patient_id': '109',
            'visit_id': '173633',
            'conditions': ['', '989'],
            'age': 25,
            'identifiers': 'F+Government+English+NOT SPECIFIED+SINGLE+BLACK/AFRICAN AMERICAN',
            'timestamp': datetime.datetime(2141, 9, 18, 11, 23),
            'd_visit_id': '172335',
            'd_conditions': ['', '989', '[SEP]', '989'],
            'd_age': 25,
            'd_identifiers': 'F+Government+English+NOT SPECIFIED+SINGLE+BLACK/AFRICAN AMERICAN',
            'd_timestamp': datetime.datetime(2141, 9, 18, 11, 23),
            'time_gap_days': 0,
            'd_visit_ids': '172335|170258'
        }
    """

    task_name = "patient_linkage_mimic3"
    
    input_schema = {
        # Query side (last admission)
        "conditions": "sequence",
        "age": "integer",
        "identifiers": "string",
        "visit_id": "string",
        "timestamp": "datetime",
        
        # Positive database side (all previous admissions concatenated)
        "d_conditions": "sequence",  # concatenated with [SEP] tokens
        "d_age": "integer",
        "d_identifiers": "string",
        "d_visit_id": "string",      # hadm_id of the most recent prior admission
        "d_timestamp": "datetime",   # timestamp of most recent prior admission
        
        # Metadata
        "patient_id": "string",      # ground truth for evaluation
        "time_gap_days": "integer",  # for analysis by time interval
        "d_visit_ids": "string",     # pipe-separated list of all prior hadm_ids
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
        gender = str(demo.attr_dict.get("gender") or "")

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

        def clean(x):
            import math
            if x is None:
                return ""
            if isinstance(x, float) and math.isnan(x):
                return ""
            return str(x)

        def build_identifiers(adm_event):
            insurance = clean(adm_event.attr_dict.get("insurance"))
            language = clean(adm_event.attr_dict.get("language"))
            religion = clean(adm_event.attr_dict.get("religion"))
            marital_status = clean(adm_event.attr_dict.get("marital_status"))
            ethnicity = clean(adm_event.attr_dict.get("ethnicity"))
            return "+".join([gender, insurance, language, religion, marital_status, ethnicity])

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
        d_most_recent_visit = None
        
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
                d_most_recent_visit = d_visit
        
        if len(d_conditions) == 0:
            return []

        # Calculate time gap between query and most recent database record
        time_gap_days = None
        if d_most_recent_visit and q_visit.timestamp:
            time_gap_days = (q_visit.timestamp - d_most_recent_visit.timestamp).days

        sample = {
            "patient_id": patient.patient_id,
            
            # Query side (compatible keys)
            "visit_id": q_hadm,
            "conditions": [""] + q_conditions,
            "age": q_age,
            "identifiers": build_identifiers(q_visit),
            "timestamp": q_visit.timestamp,
            
            # Database side (concatenated, compatible keys)
            "d_visit_id": str(d_most_recent_visit.attr_dict.get("hadm_id")),
            "d_conditions": [""] + d_conditions,
            "d_age": compute_age(d_most_recent_visit.timestamp),
            "d_identifiers": build_identifiers(d_most_recent_visit),
            "d_timestamp": d_most_recent_visit.timestamp,
            
            # Metadata
            "time_gap_days": time_gap_days,
            "d_visit_ids": "|".join(d_hadm_ids),
        }
        
        return [sample]
