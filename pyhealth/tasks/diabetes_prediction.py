from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class DiabetesPrediction(BaseTask):
    """Task for predicting whether or not a patient has diabetes.

    This task takes various biomarkers as input and predicts whether or not
    a patient will develop (or currently has) diabetes.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples. Each sample is a dict including patient_id, various biomarkers,
            as well as the predicted target (outcome)
            {
                "patient_id": xxx,
                "pregnancies": number of times the patient has gotten pregnant,
                "blood_pressure": diastolic blood pressure (mm Hg),
                "skin_thickness": triceps skin fold thickness (mm),
                "insulin": 2-hour serum insulin (mu U/ml),
                "bmi": body mass index,
                "diabetes_pedigree_function": diabetes pedigree function,
                "age": age in years
                "outcome": 0 or 1 - this is the predicted target
            }
    """
    task_name: str = "DiabetesPrediction"
    input_schema: Dict[str, str] = {
        "pregnancies": "raw",
        "glucose": "raw",
        "blood_pressure": "raw",
        "skin_thickness": "raw",
        "insulin": "raw",
        "bmi": "raw",
        "diabetes_pedigree_fn": "raw",
        "age": "raw"
    }
    output_schema: Dict[str, str] = {"outcome": "binary"}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Process a patient record to extract medical transcription samples.

        Args:
            patient (Patient): Patient record containing medical
                transcription events

        Returns:
            List[Dict[str, Any]]: List of samples containing transcription
                and medical specialty
        """
        event = patient.get_events(event_type="diabetes")
        # There should be only one event
        assert len(event) == 1
        event = event[0]

        try:
            pregnancies = int(event.Pregnancies)
            glucose = int(event.Glucose)
            blood_pressure = int(event.BloodPressure)
            skin_thickness = int(event.SkinThickness)
            insulin = int(event.Insulin)
            bmi = float(event.BMI)
            diabetes_pedigree_fn = float(event.DiabetesPedigreeFunction)
            age = int(event.Age)
            outcome = int(event.Outcome)
        except:
            return []

        sample = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetes_pedigree_fn": diabetes_pedigree_fn,
            "age": age,
            "outcome": outcome
        }
        return [sample]

if __name__ == "__main__":
    from pyhealth.datasets import DiabetesDataset

    # change the root as needed
    root = "/srv/local/data/diabetes"
    dataset = DiabetesDataset(root=root)

    samples = dataset.set_task()
    print(samples[0])