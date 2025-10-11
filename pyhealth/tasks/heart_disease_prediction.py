from typing import Any, Dict, List
from .base_task import BaseTask
from pyhealth.data import Patient






class HeartDiseasePrediction(BaseTask):

    """Task for predicting whether or not a patient has heart disease.

    Args:

        patient: a Patient object

    Returns:

        samples: a list of samples. Each sample is a dict that consists of various factors for heart disease as well as the final outcome.

        Here is the schema describing one particular sample:

            {

                "age": age of patient in years,

                "sex": sex of patient (1 = male, 0 = female),

                "trestbps": resting blood pressure (mm Hg),

                "chol": serum cholesterol in mg/dl,

                "fbs": fasting blood sugar > 120 mg/dl (1 = yes, 0 = no),

                "restecg": Resting electrocardiographic results (0-2),

                "thalach": Maximum heart rate achieved,
                "exang": Exercise-induced angina (1 = yes; 0 = no)

                "oldpeak": ST depression induced by exercise relative to rest

                "target": whether or not patient has heart disease (1 = yes 0 = no)

            }

    """

    task_name: str = "HeartDiseasePrediction"

    input_schema: Dict[str, str] = {

        "age": "raw",

        "sex": "raw",

        "trestbps": "raw",

        "chol": "raw",

        "fbs": "raw",

        "restecg": "raw",

        "thalach": "raw",

        "exang": "raw",
        "oldpeak": "raw",
        "slope": "raw",
        "ca": "raw",
        "thal": "raw",

    }


    output_schema: Dict[str, str] = {"target": "binary"}




    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:

        """Process one particular patient record



        Args:

            patient (Patient): One patient record



        Returns:

            List[Dict[str, Any]]: Return one particular sample from the dataset

        """

        event = patient.get_events(event_type="heart_disease")


        assert len(event) == 1

        event = event[0]

        try:

            age = int(event.age)

            sex = int(event.sex)

            cp = int(event.cp)

            trestbps = int(event.trestbps)

            chol = int(event.chol)

            fbs = int(event.fbs)

            restecg = int(event.restecg)

            thalach = int(event.thalach)

            exang = int(event.exang)
            oldpeak = float(event.oldpeak)
            slope = int(event.slope)
            ca = int(event.ca)
            thal = int(event.thal)
            target = int(event.target)

        except:

            return []




        sample = {

            "age": age,

            "sex": sex,

            "trestbps": trestbps,

            "chol": chol,

            "fbs": fbs,

            "restecg": restecg,

            "thalach": thalach,

            "exang": exang,

            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target

        }

        return [sample]




if __name__ == "__main__":

    from pyhealth.datasets import HeartDiseaseDataset


    # change the root as needed when testing

    root = "/srv/local/data/heart_disease"

    dataset = HeartDiseaseDataset(root=root)

    samples = dataset.set_task()
    print(samples[0])

    