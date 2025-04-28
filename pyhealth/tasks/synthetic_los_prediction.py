"""
Synthetic ICU Length of Stay Prediction Task

This module defines a function to generate synthetic data for predicting
ICU length of stay, including time series vital signs, ICU interventions,
and LOS labels. This task enables rapid prototyping of ICU LOS prediction 
models through synthetic data generation that integrates seamlessly with 
PyHealth's data pipelines and model architectures.

Author: Jacob Hennig
NetId: jhennig3
"""

from typing import Dict, Any, List
import numpy as np
from pyhealth.data import Patient, Visit

def synthetic_los_prediction(patient: Patient) -> List[Dict[str, Any]]:
    """Processes a single patient for synthetic ICU Length of Stay prediction.

    Args:
        patient: A Patient object that contains patient information.

    Returns:
        samples: A list of samples, where each sample is a dictionary containing:
            - visit_id (str): Visit ID
            - patient_id (str): Patient ID
            - vital_signs (Dict[str, List[float]]): Hourly vital signs
            - interventions (List[str]): ICU interventions
            - los (float): Length of stay in hours (label)
    """
    samples = []
    
    # Common ICU intervention codes
    ICU_INTERVENTIONS = [
        "ventilator", "vasopressor", "crrt", 
        "antibiotics", "sedation", "iv_fluids"
    ]
    # Common vital signs to track
    VITAL_SIGNS = [
        "heart_rate", "sbp", "dbp", 
        "resp_rate", "spo2", "temperature"
    ]

    for visit in patient:
        # Generate time-series vital signs (24-168 hours of ICU stay)
        timesteps = np.random.randint(24, 168) # 1-7 days of data
        vital_signs = {
            vs: np.clip(
                np.random.normal(loc = 80 if vs == "heart_rate" else 120, 
                               scale = 10, 
                               size = timesteps),
                0, 200
            ).tolist()
            for vs in VITAL_SIGNS
        }
        
        # Generate ICU interventions (1-3 random interventions)
        num_interventions = np.random.randint(1, 4)
        interventions = np.random.choice(
            ICU_INTERVENTIONS, 
            num_interventions, 
            replace = False
        ).tolist()
        # Generate LOS label (skewed log-normal distribution)
        los = np.random.lognormal(mean = 4.5, sigma = 0.5)
        
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "vital_signs": vital_signs,
            "interventions": interventions,
            "los": float(los)
        })
    return samples

if __name__ == "__main__":
    # Create a Patient with 3 ICU stays
    patient = Patient(patient_id = "icu_001")
    
    for i in range(3):
        visit = Visit(
            visit_id = f"icu_stay_{i+1}",
            patient_id = "icu_001",
        )
        patient.add_visit(visit)
    
    # Generate samples
    samples = synthetic_los_prediction(patient)
    
    # Print first sample with vital sign preview
    sample = samples[0]
    print({
        "visit_id": sample["visit_id"],
        "patient_id": sample["patient_id"],
        "interventions": sample["interventions"],
        "los": f"{sample['los']:.1f} hours",
        "vital_preview": {
            k: f"{min(v)}-{max(v)} ({len(v)} hrs)" 
            for k, v in sample["vital_signs"].items()
        }
    })
    
# Test Cases (Uncomment to run)
if __name__ == "__main__":
    
    def test():
        """Tests"""
        # Create test patient
        patient = Patient(patient_id = "test_icu")
        for i in range(3):
            visit = Visit(visit_id = f"v{i}", patient_id = "test_icu")
            patient.add_visit(visit)
        
        # Generate samples
        samples = synthetic_los_prediction(patient)
        
        # Validation
        print("\nTest Output Preview:")
        sample = samples[0]
        print(f"Patient: {sample['patient_id']}")
        print(f"LOS: {sample['los']:.1f} hours")
        print(f"Interventions: {sample['interventions']}")
        print(f"Vitals Length: {len(sample['vital_signs']['heart_rate'])} hours")
        
        # Run tests
        print("\nRunning comprehensive tests...")
        test_synthetic_los_prediction()
        print("All tests passed!")

    def test_synthetic_los_prediction():
        """Checks/asserts"""
        patient = Patient(patient_id = "test_icu")
        visit = Visit(visit_id = "v1", patient_id = "test_icu")
        patient.add_visit(visit)
        
        samples = synthetic_los_prediction(patient)
        sample = samples[0]
        
        # Structural checks
        assert isinstance(sample["los"], float), "LOS must be float"
        assert (0 < sample["los"] < 1000), "LOS out of valid range"
        assert all(
            isinstance(intervention, str) 
            for intervention in sample["interventions"]
        ), "Invalid interventions, must be strings"
        
        # Temporal checks
        hr_data = sample["vital_signs"]["heart_rate"]
        assert (24 <= len(hr_data) <= 168), "Invalid vital sign length"
        assert all(0 <= hr <= 200 for hr in hr_data), "HR out of range"

        # Empty check
        empty_patient = Patient(patient_id = "empty")
        empty_samples = synthetic_los_prediction(empty_patient)
        assert empty_samples == [], "Empty patient should return empty list"

    test()