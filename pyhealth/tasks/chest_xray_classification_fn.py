import logging
from typing import List, Dict, Any

def chest_xray_classification_fn(patient: Any) -> List[Dict[str, Any]]:
    """Task function for classifying abnormalities in chest X-rays.

    This function processes patient visit data to identify chest X-ray images and labels them
    based on the presence of specific diagnoses (e.g., pneumonia, edema). It follows the
    template structure of readmission_prediction tasks in PyHealth.

    Args:
        patient: A PyHealth Patient object containing visit and event data.
                 Expected to have `visits` attribute with event data including
                 'xray_images', 'view_position', and 'diagnoses'.

    Returns:
        List[Dict[str, Any]]: A list of samples, each containing:
            - patient_id (str): Unique patient identifier.
            - visit_id (str): Unique visit identifier.
            - xray_path (str): Path to the chest X-ray image.
            - view_position (str): View position of the X-ray (e.g., PA, AP, LATERAL, or UNKNOWN).
            - label (int): Binary label (1 if pneumonia or edema is present, 0 otherwise).

    Raises:
        KeyError: If required event data (e.g., 'xray_images') is missing.
        ValueError: If the patient object structure is invalid.
    """
    samples = []
    try:
        if not hasattr(patient, 'visits') or not patient.visits:
            raise ValueError("Patient object has no visits or visits is empty.")
        
        for visit in patient.visits:
            if not hasattr(visit, 'events'):
                logging.warning(f"Visit {visit.visit_id} has no events data, skipping.")
                continue
            
            if "xray_images" not in visit.events:
                logging.warning(f"No xray_images in visit {visit.visit_id}, skipping.")
                continue
            
            diagnoses = visit.events.get("diagnoses", [])
            label = 1 if any(d in diagnoses for d in ["pneumonia", "edema"]) else 0
            
            sample = {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "xray_path": visit.events["xray_images"],
                "view_position": visit.events.get("view_position", "UNKNOWN"),
                "label": label
            }
            samples.append(sample)
        
        if not samples:
            logging.warning(f"No valid samples generated for patient {patient.patient_id}.")
        return samples
    
    except KeyError as e:
        logging.error(f"Missing required key in patient data: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in chest_xray_classification_fn: {e}")
        raise

# Example usage (for testing)
if __name__ == "__main__":
    # Mock patient object for testing
    class MockVisit:
        def __init__(self, visit_id, events):
            self.visit_id = visit_id
            self.events = events
    
    class MockPatient:
        def __init__(self, patient_id, visits):
            self.patient_id = patient_id
            self.visits = visits
    
    # Test data
    test_visits = [
        MockVisit("v1", {"xray_images": "/path/to/xray1.jpg", "view_position": "PA", "diagnoses": ["pneumonia"]}),
        MockVisit("v2", {"xray_images": "/path/to/xray2.jpg", "view_position": "AP", "diagnoses": ["fever"]}),
        MockVisit("v3", {"xray_images": "/path/to/xray3.jpg", "diagnoses": ["edema"]})
    ]
    test_patient = MockPatient("p1", test_visits)
    
    samples = chest_xray_classification_fn(test_patient)
    for sample in samples:
        print(sample)