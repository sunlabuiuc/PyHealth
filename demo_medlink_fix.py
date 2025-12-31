from datetime import datetime
from collections import defaultdict
from unittest.mock import MagicMock
from pyhealth.tasks.patient_linkage_mimic3 import PatientLinkageMIMIC3Task
import json

def clean_for_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def run_demo():
    print("Setting up mock patient with 3 admissions...")
    # Mock patient
    patient = MagicMock()
    patient.patient_id = "TEST_PATIENT"

    # Mock admissions
    # Adm 1: 2010-01-01, hadm_id=101, codes=["DIAG_A"]
    # Adm 2: 2011-06-15, hadm_id=102, codes=["DIAG_B", "DIAG_C"]
    # Adm 3: 2012-12-31, hadm_id=103, codes=["DIAG_D"] (Query)
    
    adm1 = MagicMock(); adm1.timestamp = datetime(2010, 1, 1); adm1.attr_dict = {"hadm_id": "101"}
    adm2 = MagicMock(); adm2.timestamp = datetime(2011, 6, 15); adm2.attr_dict = {"hadm_id": "102"}
    adm3 = MagicMock(); adm3.timestamp = datetime(2012, 12, 31); adm3.attr_dict = {"hadm_id": "103"}

    # Mock demographics
    demo = MagicMock()
    demo.attr_dict = {"dob": datetime(1980, 1, 1), "gender": "M"}

    # Mock diagnosis events
    d1 = MagicMock(); d1.attr_dict = {"hadm_id": "101", "icd9_code": "DIAG_A"}
    d2 = MagicMock(); d2.attr_dict = {"hadm_id": "102", "icd9_code": "DIAG_B"}
    d3 = MagicMock(); d3.attr_dict = {"hadm_id": "102", "icd9_code": "DIAG_C"}
    d4 = MagicMock(); d4.attr_dict = {"hadm_id": "103", "icd9_code": "DIAG_D"}

    # Configure patient.get_events
    def get_events_side_effect(event_type):
        if event_type == "admissions": return [adm1, adm2, adm3]
        elif event_type == "patients": return [demo]
        elif event_type == "diagnoses_icd": return [d1, d2, d3, d4]
        return []
    
    patient.get_events.side_effect = get_events_side_effect

    print("Running PatientLinkageMIMIC3Task...")
    task = PatientLinkageMIMIC3Task()
    results = task(patient)
    
    if not results:
        print("No results returned!")
        return

    sample = results[0]
    
    print("\n=== Result Verification ===")
    print(f"Patient ID: {sample['patient_id']}")
    print(f"Query Visit ID: {sample['q_visit_id']} (Expected 103)")
    print(f"Query Conditions: {sample['q_conditions']}")
    print(f"Query Timestamp: {sample['q_timestamp']}")
    print("-" * 20)
    print(f"Database Visit IDs: {sample['d_visit_ids']} (Expected 101|102)")
    print(f"Database Conditions (concatenated): {sample['d_conditions']}")
    print(f"Database Timestamp (most recent): {sample['d_timestamp']}")
    print(f"Time Gap Days: {sample['time_gap_days']}")
    
    print("\n=== Checking for [SEP] token ===")
    if "[SEP]" in sample['d_conditions']:
        print("SUCCESS: [SEP] token found in database conditions.")
        print(f"Index of [SEP]: {sample['d_conditions'].index('[SEP]')}")
    else:
        print("FAILURE: [SEP] token NOT found.")

if __name__ == "__main__":
    run_demo()
