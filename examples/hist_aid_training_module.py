import torch
from pyhealth.trainer import Trainer
from pyhealth.datasets.mimic_cxr_longitudinal import MIMICCXRLongitudinalDataset
from pyhealth.tasks.mimic_cxr_longitudinal_classification import MIMICCXRLongitudinalClassificationTask
from pyhealth.models.hist_aid import HistAID

def run_longitudinal_experiment(data_path="./mimic_data", k_window=3):
    # 1. Initialize the Dataset
    # This reads the metadata, chexpert, and reports tables
    base_ds = MIMICCXRLongitudinalDataset(root=data_path)
    
    # 2. Define the Longitudinal Task
    # max_history=k_window controls the K-report ablation logic
    task_fn = MIMICCXRLongitudinalClassificationTask(max_history=k_window)
    task_ds = base_ds.set_task(task_fn)
    
    # 3. Data Split (70/15/15 for robust medical evaluation)
    train_ds, val_ds, test_ds = task_ds.split([0.7, 0.15, 0.15])

    # 4. Model Initialization
    # Pass the task_ds so the model knows the label mapping and feature dimensions
    model = HistAID(dataset=task_ds, num_history=k_window)
    
    # 5. Trainer Configuration
    # We focus on roc_auc_weighted as our primary success metric
    trainer = Trainer(
        model=model,
        metrics=["roc_auc_weighted", "pr_auc_weighted"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        exp_name=f"hist_aid_longitudinal_k{k_window}"
    )

    # 6. Training Phase
    print(f"\n>>> Training HIST-AID with Longitudinal Window K={k_window}")
    trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_batch_size=8,      # Batch size adjusted for longitudinal memory
        epochs=10,               # Sufficient epochs for transformer convergence
        optimizer="adamw",
        optimizer_params={"lr": 1e-4, "weight_decay": 1e-2},
        monitor="roc_auc_weighted" 
    )

    # 7. Final AUROC Evaluation
    # This evaluates on the unseen test set
    print("\n>>> Final Evaluation on Hold-out Test Set")
    performance = trainer.evaluate(test_ds)
    
    print("-" * 30)
    print(f"Test AUROC (Weighted): {performance['roc_auc_weighted']:.4f}")
    print(f"Test PR-AUC (Weighted): {performance['pr_auc_weighted']:.4f}")
    print("-" * 30)

    return performance

if __name__ == "__main__":
    # Run the experiment with the paper's standard K=3 setting
    results = run_longitudinal_experiment(k_window=3)