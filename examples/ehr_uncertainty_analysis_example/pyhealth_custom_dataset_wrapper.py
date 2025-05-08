from pyhealth.datasets import SampleEHRDataset
import torch

def basic_task_fn(patient_sample_from_list):
    """
    Basic task function for SampleEHRDataset.
    Processes a single sample dictionary from the pre-cached list
    and converts relevant data back into PyTorch tensors.
    """
    return {
        "patient_id": patient_sample_from_list["patient_id"],
        "sequence_data": torch.tensor(patient_sample_from_list["sequence_data"], dtype=torch.float32),
        "label": torch.tensor([patient_sample_from_list["label"]], dtype=torch.float32),
    }

class CustomSequentialEHRDataPyHealth(SampleEHRDataset):
    """
    PyHealth Dataset wrapper for custom pre-processed sequential data.

    This class demonstrates how to integrate pre-processed sequential data 
    (provided as lists of sequences and labels) into the PyHealth ecosystem 
    by inheriting from `pyhealth.datasets.SampleEHRDataset`.

    It converts the input lists into the `samples` format expected by 
    `SampleEHRDataset` during initialization and uses a `task_fn` 
    (basic_task_fn) to process samples when they are accessed.

    Args:
        list_of_patient_sequences (list): List of PyTorch tensors, where each 
            tensor represents a patient's sequence (shape: [seq_len, feature_dim]).
        list_of_patient_labels (list): List of PyTorch tensors, where each 
            tensor is a patient's label (e.g., tensor([0.]) or tensor([1.])).
        root (str): Root directory path required by PyHealth datasets (e.g., ".").
        dataset_name (str): Name for this dataset instance.
    """
    def __init__(self, list_of_patient_sequences, list_of_patient_labels, root=".", dataset_name="custom_ehr_example"):
        pyhealth_samples = []
        if len(list_of_patient_sequences) != len(list_of_patient_labels):
            raise ValueError("Sequences and labels lists must have the same length.")
            
        for i in range(len(list_of_patient_labels)):
            pyhealth_samples.append({
                "patient_id": str(i), 
                "sequence_data": list_of_patient_sequences[i].tolist(), 
                "label": list_of_patient_labels[i].item() 
            })

        super().__init__(
            samples=pyhealth_samples,
            task_fn=basic_task_fn,
            dataset_name=dataset_name,
            root=root 
        )
