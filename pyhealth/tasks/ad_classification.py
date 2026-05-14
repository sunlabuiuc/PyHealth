"""AlzheimersDiseaseClassification Task. 

Author: Bryan Lau (bryan16@illinois.edu)
Description:
    Task to parse records from an ADNIDataset for model processing.
"""
from .base_task import BaseTask

class AlzheimersDiseaseClassification(BaseTask):
    """Pyhealth task for classification of ADNI records for the purposes
    of Alzheimer's Disease classification.

    This task pre-processes ADNI records (patient data, MRI brain scans) 
    for the purposes of classification as described in the paper "On the 
    Design of Convolutional Neural Networks for Automatic Detection of 
    Alzheimer's Disease" by Liu et al. (https://arxiv.org/abs/1911.03740)

    The patients are classified as follows:

    - Cognitively Normal (CN)
    - Mild Cognitive Impairment (MCI)
    - Alzheimer's Disease (AD)

    The task performs the following actions:

    - Maps the labels (CN, MCI, AD) to numeric values
    - Maps gender to numeric values

    Returns:
        An image sample containing:
        - patient id
        - image data
        - label
        - age
        - gender
        - weight (Kg)
    """

    LABEL_MAPPING = {"CN": 0, "MCI": 1, "AD": 2}
    GENDER_MAPPING = {"M": 0, "F": 1}

    task_name: str = "AlzheimersDiseaseClassification"
    input_schema = {"image": "nifti_image",
                    "age": "tensor", "gender": "tensor"}
    output_schema = {"label": "multiclass"}

    def __call__(self, patient):
        """Generates MRI image data samples for the given patient.

        Args:
            patient: A patient object containing demographic data and MRI images.

        Returns:
            List[Dict]: A list of dictionaries for this patient, each containing:
                - patient_id: Patient's unique ID
                - image:      MRI image path
                - label:      This patient's classification (i.e. CN, MCI, AD)
                - age:        Subject's age
                - gender:     Subject's gender
                - weight:     Subject's weight in Kg
        """
        samples = []
        events = patient.get_events(event_type="adni")
        for event in events:

            # Add sample
            samples.append({
                "patient_id": patient.patient_id,
                "image": event["image_path"],
                "label": self.LABEL_MAPPING[event["group"]],
                "age": float(event["age"]),
                "gender": self.GENDER_MAPPING[event["gender"]],
                "weight": float(event["weight"]),
            })

        return samples
