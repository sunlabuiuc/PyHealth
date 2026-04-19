import pytest
from datetime import datetime
from pyhealth.data import Patient, Event
from pyhealth.tasks import ReadmissionPredictionEICU

def test_readmission_prediction_eicu_task():
    """
    Tests the ReadmissionPredictionEICU task using synthetic, in-memory patient data 
    to ensure tests complete in milliseconds.
    """
    # 1. Initialize Task
    task = ReadmissionPredictionEICU(exclude_minors = True)
    
    # 2. Create a mock patient
    patient = Patient(patient_id = "test_pat_001")
    
    # Visit 1: ICU Stay 1 (in Hospital 1)
    patient.add_event(Event(
        event_type = "patient",
        timestamp=datetime(2025, 1, 1),
        patienthealthsystemstayid = "hosp_001",
        patientunitstayid = "icu_001",
        unitvisitnumber = 1,
        age = "65"
    ))
    patient.add_event(Event(
        event_type = "diagnosis",
        patientunitstayid = "icu_001",
        icd9code = "428.0"
    ))
    patient.add_event(Event(
        event_type = "medication",
        patientunitstayid = "icu_001",
        drugname = "Aspirin"
    ))

    # Visit 2: ICU Stay 2 (Readmitted to the SAME hospital, hosp_001)
    patient.add_event(Event(
        event_type = "patient",
        timestamp = datetime(2025, 1, 10),
        patienthealthsystemstayid = "hosp_001", 
        patientunitstayid = "icu_002",
        unitvisitnumber = 2,
        age = "65"
    ))
    patient.add_event(Event(
        event_type = "physicalexam",
        patientunitstayid = "icu_002",
        physicalexampath = "cardiovascular|murmur"
    ))

    # Visit 3: ICU Stay 3 (Admitted to a DIFFERENT hospital, hosp_002)
    patient.add_event(Event(
        event_type = "patient",
        timestamp = datetime(2025, 5, 1),
        patienthealthsystemstayid = "hosp_002", 
        patientunitstayid = "icu_003",
        unitvisitnumber = 1,
        age = "65"
    ))
    patient.add_event(Event(
        event_type = "diagnosis",
        patientunitstayid = "icu_003",
        icd9code = "250.00"
    ))

    # 3. Call the task
    samples = task(patient)

    # 4. Assertions
    # With 3 ICU stays, we expect 2 samples (1->2, 2->3)
    assert len(samples) == 2, "Task should generate exactly 2 samples"
    
    # Check Sample 1 (icu_001 -> icu_002)
    assert samples[0]["visit_id"] == "icu_001"
    assert samples[0]["readmission"] == 1
    assert "428.0" in samples[0]["conditions"][0]
    assert "Aspirin" in samples[0]["drugs"][0]
    assert len(samples[0]["procedures"][0]) == 0
    
    # Check Sample 2 (icu_002 -> icu_003)
    assert samples[1]["visit_id"] == "icu_002"
    assert samples[1]["readmission"] == 0
    assert "cardiovascular|murmur" in samples[1]["procedures"][0]

def test_exclude_minors():
    """Test that the task correctly excludes patients under 18."""
    task = ReadmissionPredictionEICU(exclude_minors = True)
    patient = Patient(patient_id = "test_minor")
    for i in [1, 2]:
        patient.add_event(Event(
            event_type="patient",
            timestamp=datetime(2025, 1, i),
            patienthealthsystemstayid = "hosp_001",
            patientunitstayid = f"icu_{i}",
            unitvisitnumber = i,
            age="10"
        ))
        patient.add_event(Event(
            event_type = "diagnosis",
            patientunitstayid = f"icu_{i}",
            icd9code = "test"
        ))
    samples = task(patient)
    assert len(samples) == 0, "Task should return 0 samples for minors when exclude_minors=True"