from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks import MIMIC3NoteReplaceDeIdTask
import logging

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="/home/adafe/selected", # path to the dataset
        tables=[ "noteevents"], # specify the tables to load
        config_path="/home/adafe/DLH/PyHealth/pyhealth/datasets/configs/mimic3.yaml" # path to the config file

    )

    # STEP 2: set task

    task_instance = MIMIC3NoteReplaceDeIdTask()
    sample_dataset = base_dataset.set_task(task_instance)    

    print("After de-identification:")
    """
        for patient_id in base_dataset.unique_patient_ids[:5]:
            patient = base_dataset.get_patient(patient_id)
            print("------------------------------------------------")
            print(f"Patient ID: {patient.patient_id}")
            print(f"Patient Events: {patient.get_events(event_type='noteevents')}")
    """

    for i in range(5):
        patient = base_dataset.get_patient(base_dataset.unique_patient_ids[i])
        print("------------------------------------------------")
        #print(f"Patient ID: {patient.patient_id}")
        #print(f"Patient Events: {patient.get_events(event_type='noteevents')}")
        #print("------------------------------------------------")
        print(sample_dataset[i]['masked_text'])
        print("------------------------------------------------")\


        