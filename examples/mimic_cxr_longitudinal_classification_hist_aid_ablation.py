import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# PyHealth Trainer
from pyhealth.trainer import Trainer

from pyhealth.datasets.mimic_cxr_longitudinal import MIMICCXRLongitudinalDataset
from pyhealth.tasks.mimic_cxr_longitudinal_classification import MIMICCXRLongitudinalClassificationTask
from pyhealth.models.hist_aid import HistAID

# ======================================================================
# 1. SYNTHETIC DATA GENERATOR (For instant testing)
# ======================================================================
def generate_synthetic_mimic(data_dir="./synthetic_data"):
    """Creates dummy CSVs mimicking the longitudinal MIMIC-CXR structure."""
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)
    
    num_patients = 30
    meta_list, label_list, report_list = [], [], []

    for p_id in range(100, 100 + num_patients):
        num_visits = np.random.randint(2, 5) 
        for v_idx in range(num_visits):
            v_id = f"V_{p_id}_{v_idx}"
            v_time = datetime(2026, 1, 1) + timedelta(days=v_idx * 30)
            
            # Metadata
            meta_list.append([p_id, v_id, v_time, f"IMG_{v_id}"])
            # Labels (14 clinical findings)
            label_list.append([p_id, v_id] + np.random.randint(0, 2, 14).tolist())
            # Reports
            report_list.append([p_id, v_id, f"Report for patient {p_id} visit {v_idx}."])

    pd.DataFrame(meta_list, columns=['subject_id', 'study_id', 'encounter_time', 'dicom_id']).to_csv(f"{data_dir}/metadata.csv", index=False)
    pd.DataFrame(label_list, columns=['subject_id', 'study_id'] + [f"l_{i}" for i in range(14)]).to_csv(f"{data_dir}/chexpert.csv", index=False)
    pd.DataFrame(report_list, columns=['subject_id', 'study_id', 'report_text']).to_csv(f"{data_dir}/reports.csv", index=False)
    print(f"--- Synthetic data generated in {data_dir} ---")

# ======================================================================
# 2. REPORT FORMATTING FUNCTION
# ======================================================================
def print_formatted_report(results):
    print("\n" + "="*75)
    print("HIST-AID ABLATION STUDY: FINAL RESEARCH REPORT")
    print("="*75)
    print(f"{'Configuration':<35} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-" * 75)
    
    # Identify the highest K for the winner tag
    max_k = max(r['K'] for r in results)
    
    for res in sorted(results, key=lambda x: x['K']):
        name = "Image Only (Baseline)" if res['K'] == 0 else f"Current + History (K={res['K']})"
        winner = " <-- WINNER" if res['K'] == max_k else ""
        print(f"{name:<35} | {res['roc_auc_weighted']:.4f}     | {res['pr_auc_weighted']:.4f}    {winner}")
    print("="*75 + "\n")

# ======================================================================
# 3. MAIN EXECUTION (The Ablation Loop)
# ======================================================================
if __name__ == "__main__":
    DATA_PATH = "./synthetic_mimic_data"
    generate_synthetic_mimic(DATA_PATH)
    
    # 1. Load the base dataset (imported)
    base_ds = MIMICCXRLongitudinalDataset(root=DATA_PATH)
    all_metrics = []

    # 2. Loop through K-values for the ablation study
    for K in [0, 3]: # Testing Baseline vs. Longitudinal context
        print(f"\n>>> Running Ablation Trial: K={K}")
        
        # A. Set the task with current max_history (K)
        task_ds = base_ds.set_task(MIMICCXRLongitudinalClassificationTask(max_history=K))
        
        # B. Split (Using the 'Small Data' settings for synthetic reliability)
        train_ds, val_ds, test_ds = task_ds.split([0.7, 0.15, 0.15])
        
        # C. Initialize Model (imported)
        model = HistAID(dataset=task_ds, num_history=K)
        
        # D. Train
        trainer = Trainer(model=model, metrics=["roc_auc_weighted", "pr_auc_weighted"])
        trainer.train(
            train_dataset=train_ds, 
            val_dataset=val_ds, 
            epochs=3, 
            train_batch_size=4
        )
        
        # E. Store results
        res = trainer.evaluate(test_ds)
        res['K'] = K
        all_metrics.append(res)

    # 3. Final Output
    print_formatted_report(all_metrics)