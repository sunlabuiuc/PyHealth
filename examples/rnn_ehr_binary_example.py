import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Dict, Any, List, Tuple
import random
from collections import defaultdict
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import RNN
from pyhealth.trainer import Trainer

TARGET_CONDITION_CODE_FOR_TASK = "C003"

# Function to generate synthetic EHR visit samples
def generate_synthetic_ehr_visit_samples(
    n_patients: int,
    avg_visits_per_patient: int,
    avg_codes_per_visit: int,
    condition_prefix: str = "C",
    procedure_prefix: str = "P",
    max_cond_code_idx: int = 100,
    max_proc_code_idx: int = 100,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    samples = []
    all_condition_codes_no_pad = [f"{condition_prefix}{i:03d}" for i in range(max_cond_code_idx)]
    all_procedure_codes_no_pad = [f"{procedure_prefix}{i:03d}" for i in range(max_proc_code_idx)]
    if not all_condition_codes_no_pad: all_condition_codes_no_pad = [f"{condition_prefix}000"]
    if not all_procedure_codes_no_pad: all_procedure_codes_no_pad = [f"{procedure_prefix}000"]
    condition_vocab_padded = {"<PAD>": 0}
    for i, code in enumerate(all_condition_codes_no_pad):
        condition_vocab_padded[code] = i + 1 
    procedure_vocab_padded = {"<PAD>": 0}
    for i, code in enumerate(all_procedure_codes_no_pad):
        procedure_vocab_padded[code] = i + 1
    code_vocs = {"conditions": condition_vocab_padded, "procedures": procedure_vocab_padded}
    visit_counter = 0
    for i in range(n_patients):
        patient_id = f"patient_{i}"
        num_visits = random.randint(1, max(1, 2 * avg_visits_per_patient -1 if avg_visits_per_patient > 1 else 1))
        for j in range(num_visits):
            visit_id = f"visit_{visit_counter}"
            visit_counter += 1
            num_cond_codes = random.randint(1, max(1, 2 * avg_codes_per_visit -1 if avg_codes_per_visit > 1 else 1))
            visit_conditions = random.sample(all_condition_codes_no_pad, min(num_cond_codes, len(all_condition_codes_no_pad)))
            num_proc_codes = random.randint(1, max(1, 2 * avg_codes_per_visit -1 if avg_codes_per_visit > 1 else 1))
            visit_procedures = random.sample(all_procedure_codes_no_pad, min(num_proc_codes, len(all_procedure_codes_no_pad)))
            sample = {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "conditions": visit_conditions,
                "procedures": visit_procedures,
            }
            samples.append(sample)

    return samples, code_vocs

def process_visits_to_patient_samples(
    visit_samples: List[Dict[str, Any]],
    target_condition_code: str
) -> List[Dict[str, Any]]:
    patients_data = defaultdict(lambda: {"conditions": [], "procedures": [], "original_visit_ids": []})
    for visit in visit_samples:
        pid = visit["patient_id"]
        patients_data[pid]["conditions"].append(visit.get("conditions", []))
        patients_data[pid]["procedures"].append(visit.get("procedures", []))
        patients_data[pid]["original_visit_ids"].append(visit["visit_id"])
    patient_level_samples = []
    patient_level_visit_id_counter = 0
    for patient_id, data in patients_data.items():
        has_target_condition_in_patient = False
        for visit_conditions in data["conditions"]: 
            if target_condition_code in visit_conditions:
                has_target_condition_in_patient = True
                break
        patient_level_dummy_visit_id = f"patient_summary_visit_{patient_level_visit_id_counter}"
        patient_level_visit_id_counter += 1
        patient_sample = {
            "patient_id": patient_id,
            "visit_id": patient_level_dummy_visit_id,
            "conditions": data["conditions"], 
            "procedures": data["procedures"], 
            "label": 1 if has_target_condition_in_patient else 0
        }
        patient_level_samples.append(patient_sample)

    return patient_level_samples

# Function to print dataset statistics
def print_dataset_statistics(dataset_obj: Any, dataset_name: str = "Dataset"):
    print(f"Statistics of {dataset_name}:")
    if hasattr(dataset_obj, "dataset_name") and dataset_obj.dataset_name:
        print(f"        - Dataset Name: {dataset_obj.dataset_name}")
    if hasattr(dataset_obj, "task_name") and dataset_obj.task_name:
        print(f"        - Task Name: {dataset_obj.task_name}")
    if hasattr(dataset_obj, 'samples') and dataset_obj.samples:
        num_samples = len(dataset_obj.samples)
        print(f"        - Number of samples (patients): {num_samples}")
        if num_samples > 0 and isinstance(dataset_obj.samples[0], dict) and "label" in dataset_obj.samples[0]:
            labels = [s["label"] for s in dataset_obj.samples]
            num_positive = sum(labels)
            print(f"        - Label distribution: {num_positive} positive cases out of {num_samples}")

# Simplified collate function for DataLoader
def simplified_collate_fn(
    batch: List[Dict[str, Any]], 
    feature_keys: List[str]
    ) -> Dict[str, Any]:
    collated_batch: Dict[str, Any] = {}
    for key in feature_keys:
        collated_batch[key] = []
    raw_labels = [] 
    for patient_sample in batch:
        for key in feature_keys:
            collated_batch[key].append(patient_sample.get(key, []))
        raw_labels.append(patient_sample["label"])
    collated_batch["label"] = raw_labels
    return collated_batch


def run_ehr_binary_prediction_example(
    n_patients_to_generate: int = 100,
    avg_visits_to_generate: int = 5,
    avg_codes_to_generate: int = 5,
    target_code_for_task: str = "C003",
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "cpu"
) -> Dict[str, float]:
    print(f"--- PyHealth EHR Binary Prediction Example (Adjusted Label Handling version) ---")

    global TARGET_CONDITION_CODE_FOR_TASK
    TARGET_CONDITION_CODE_FOR_TASK = target_code_for_task
    
    synthetic_visit_samples, synthetic_code_vocs = generate_synthetic_ehr_visit_samples(
        n_patients=n_patients_to_generate,
        avg_visits_per_patient=avg_visits_to_generate,
        avg_codes_per_visit=avg_codes_to_generate,
        max_cond_code_idx=100
    )
    if not synthetic_visit_samples:
        return {"error": "Synthetic visit sample generation failed."}

    patient_level_samples_with_labels = process_visits_to_patient_samples(
        visit_samples=synthetic_visit_samples,
        target_condition_code=TARGET_CONDITION_CODE_FOR_TASK
    )
    if not patient_level_samples_with_labels:
        return {"error": "Patient-level sample processing failed."}

    try:
        patient_dataset = SampleEHRDataset(
            samples=patient_level_samples_with_labels, 
            code_vocs=synthetic_code_vocs,
            dataset_name="SyntheticPatientEHR",
            task_name=f"Predict_{TARGET_CONDITION_CODE_FOR_TASK}"
        )
    except Exception as e:
        return {"error": f"Patient-level SampleEHRDataset instantiation error: {e}"}

    print_dataset_statistics(patient_dataset, "Patient-Level Dataset") # Keep this for a summary

    if not hasattr(patient_dataset, 'samples') or len(patient_dataset.samples) == 0: # type: ignore
        return {"error": "Patient dataset is empty or 'samples' attribute missing."}

    model = RNN(
       dataset=patient_dataset,
       feature_keys=["conditions", "procedures"],
       label_key="label", 
       mode="binary",
       embedding_dim=64,
       hidden_dim=64,
    )

    dataset_size = len(patient_dataset)
    indices = list(range(dataset_size))
    all_labels_in_dataset = [s['label'] for s in patient_dataset.samples] # type: ignore

    if dataset_size < 3:
        print(f"Warning: Dataset size ({dataset_size}) is very small, splits might be compromised.")

    stratify_first_split = None
    if len(set(all_labels_in_dataset)) > 1:
        overall_label_counts = Counter(all_labels_in_dataset)
        if all(count >= 2 for count in overall_label_counts.values()):
            stratify_first_split = all_labels_in_dataset
    
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, all_labels_in_dataset, test_size=0.3, random_state=42, stratify=stratify_first_split
    )
    
    stratify_second_split = None
    if len(temp_indices) >= 2:
        if len(set(temp_labels)) > 1: 
            label_counts_in_temp = Counter(temp_labels)
            if all(count >= 2 for count in label_counts_in_temp.values()):
                stratify_second_split = temp_labels

    val_indices, test_indices = [], []
    if len(temp_indices) == 1:
        val_indices = temp_indices
    elif len(temp_indices) >= 2:
        val_indices, test_indices, _, _ = train_test_split(
            temp_indices, temp_labels, test_size=0.5, random_state=42, stratify=stratify_second_split
        )

    if not train_indices and dataset_size > 0 :
        return {"error": "Training set empty after split."}

    train_subset = Subset(patient_dataset, train_indices) if train_indices else None
    val_subset = Subset(patient_dataset, val_indices) if val_indices else None
    test_subset = Subset(patient_dataset, test_indices) if test_indices else None
    
    from functools import partial
    current_collate_fn = partial(simplified_collate_fn, 
                                 feature_keys=model.feature_keys)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=current_collate_fn) if train_subset else None
    val_loader = None
    if val_subset and len(val_subset) > 0:
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=current_collate_fn)
    test_loader = None
    if test_subset and len(test_subset) > 0:
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=current_collate_fn)
    
    if not train_loader:
        return {"error": "Train loader is None. Increase dataset size."}

    trainer = Trainer(model=model, device=device)

    print(f"\nStarting model training for {epochs} epoch(s)...")
    if val_loader:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            monitor="roc_auc",
        )
        print("Model training completed.")
    else:
        return {"error": "Validation dataloader is None or empty. Required for trainer.train."}

    print(f"\nEvaluating the model on the test set...")
    evaluation_results: Dict[str, float] = {}
    if test_loader:
        evaluation_results = trainer.evaluate(test_loader)
        print("Evaluation results on the test set:")
        for metric, value in evaluation_results.items():
            print(f"  {metric}: {value:.4f}")
    else:
         evaluation_results = {"warning": "Test dataloader was None or empty. No test evaluation."}
         print(evaluation_results["warning"])
    
    print("\n--- PyHealth EHR Binary Prediction Example Finished ---")
    return evaluation_results

if __name__ == "__main__":
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda":
        try:
            torch.cuda.device_count() 
        except RuntimeError as e:
            if "cudaGetDeviceCount() failed" in str(e) or "forward compatibility" in str(e):
                print("CUDA detected but not fully working (RuntimeError). Forcing CPU.")
                selected_device = "cpu"
            else:
                print(f"Unexpected CUDA error, forcing CPU: {e}") 
                selected_device = "cpu"

    num_patients_config = 100
    avg_visits_config = 3 
    avg_codes_config = 2
    target_condition_config = "C003"
    training_epochs_config = 1
    data_batch_size_config = 16

    final_metrics = run_ehr_binary_prediction_example(
        n_patients_to_generate=num_patients_config,
        avg_visits_to_generate=avg_visits_config,
        avg_codes_to_generate=avg_codes_config,
        target_code_for_task=target_condition_config,
        epochs=training_epochs_config,
        batch_size=data_batch_size_config,
        device=selected_device
    )

    print("\nFinal reported metrics from the example run:")
    if "error" in final_metrics :
        print(f"  Error during execution: {final_metrics['error']}")
    elif "warning" in final_metrics:
        print(f"  Warning during execution: {final_metrics['warning']}")
    else:
        for metric_name, metric_value in final_metrics.items():
            print(f"  {metric_name}: {metric_value}")