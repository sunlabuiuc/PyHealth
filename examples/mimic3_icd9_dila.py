"""Ablation Study: DILA (Dictionary Label Attention) on MIMIC-III.

Paper: DILA: Dictionary Label Attention for Mechanistic Interpretability 
in High-dimensional Multi-label Medical Coding Prediction.

Experimental Setup:
This script performs an ablation study to quantify the trade-off between strict 
interpretability (enforced by sparsity) and downstream predictive accuracy. We 
evaluate the model's performance on the medical coding task by varying two key 
hyperparameters:

1. Dictionary Size (m): Controls the total number of sparse features allowed.
   - Values tested: [1000, 3000, 6088] 
2. Sparsity Penalty (lambda_saenc): Controls the L1/L2 penalty threshold.
   - Values tested: [1e-5, 1e-6]

Metrics Tracked:
- Micro F1
- Macro F1
- ROC-AUC

Actual Findings:
When trained on the restricted MIMIC-III demo dataset, modifying the dictionary 
size (m) and sparsity penalty (lambda_saenc) yielded negligible differences in 
performance. Micro F1 remained heavily bounded around ~0.056 across all 
configurations, and Macro F1 hovered around ~0.047. The ROC AUC metric 
evaluated to NaN due to the extremely small test split lacking positive samples 
for the vast majority of the 581 extracted ICD-9 classes. Running this on the 
full MIMIC-III dataset is required to observe the true sparsity/accuracy tradeoff.
"""

import torch
import pandas as pd
from typing import List, Dict, Any
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import get_dataloader
from pyhealth.models import DILA
from pyhealth.trainer import Trainer


class ICD9CodingTask:
    """Medical coding task definition for extracting diagnosis codes.
    
    Attributes:
        task_name (str): Identifier for the task.
        input_schema (Dict[str, str]): Schema definition for input features.
        output_schema (Dict[str, str]): Schema definition for output labels.
    """
    
    task_name = "ICD9_Coding_Task"
    input_schema = {"conditions": "sequence"}
    output_schema = {"label": "multilabel"}

    def pre_filter(self, global_event_df: pd.DataFrame) -> pd.DataFrame:
        """Applies filtering to the global event dataframe before patient parsing.
        
        Args:
            global_event_df (pd.DataFrame): Raw event dataframe.
            
        Returns:
            pd.DataFrame: Unmodified event dataframe.
        """
        return global_event_df

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extracts ICD-9 diagnosis codes for each hospital admission.

        Args:
            patient (Any): Patient object containing historical hospital visits.

        Returns:
            List[Dict[str, Any]]: A list of parsed samples ready for processing.
        """
        samples = []
        admissions = patient.get_events(event_type="admissions")
        
        for admission in admissions:
            diagnoses_events = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)]
            )
            
            diagnoses = [event.icd9_code for event in diagnoses_events]
            
            if not diagnoses:
                continue
                
            samples.append({
                "visit_id": admission.hadm_id,
                "patient_id": patient.patient_id,
                "conditions": diagnoses, 
                "label": diagnoses         
            })
            
        return samples


if __name__ == "__main__":
    print("Loading MIMIC-III Demo Dataset...")
    dataset = MIMIC3Dataset(
        root="./data/mimic-iii-clinical-database-demo-1.4/", 
        tables=["DIAGNOSES_ICD", "ADMISSIONS", "PATIENTS"]
    )

    dataset = dataset.set_task(ICD9CodingTask())

    train_dataset, val_dataset, test_dataset = split_by_patient(
        dataset,
        [0.8, 0.1, 0.1], 
        seed=42 
    )

    train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=8, shuffle=False)

    dictionary_sizes = [1000, 3000, 6088]
    sparsity_penalties = [1e-5, 1e-6]
    results_log = {}

    print("\nStarting DILA Ablation Study...")
    print("=" * 50)

    for m in dictionary_sizes:
        for penalty in sparsity_penalties:
            config_name = f"DictSize_{m}_Penalty_{penalty}"
            print(f"\nTraining configuration: {config_name}")
            
            model = DILA(
                dataset=dataset,
                feature_keys=["conditions"],
                label_key="label",
                mode="multilabel",
                embedding_dim=768,       
                dictionary_size=m,       
                sparsity_penalty=penalty 
            )
            
            trainer = Trainer(
                model=model,
                metrics=["roc_auc_macro", "f1_macro", "f1_micro"]
            )
            
            trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=3, 
                optimizer_class=torch.optim.AdamW,
                optimizer_params={"lr": 5e-5},
                weight_decay=0.01
            )
            
            print(f"Evaluating {config_name} on Test Set...")
            eval_results = trainer.evaluate(dataloader=test_loader)
            results_log[config_name] = eval_results

    print("\n" + "=" * 50)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Configuration':<30} | {'Micro F1':<10} | {'Macro F1':<10} | {'ROC AUC (Macro)':<15}")
    print("-" * 75)

    for config, metrics in results_log.items():
        micro_f1 = metrics.get('f1_micro', 0.0)
        macro_f1 = metrics.get('f1_macro', 0.0)
        roc_auc = metrics.get('roc_auc_macro', 0.0)
        
        print(f"{config:<30} | {micro_f1:<10.4f} | {macro_f1:<10.4f} | {roc_auc:<15.4f}")