"""
Contributor: Adrianne Sun, Ruoyi Xie
NetID: ajsun2, ruoyix2
Paper Title: Reproducible Survival Prediction with SEER Cancer Data
Paper Link: https://proceedings.mlr.press/v85/hegselmann18a/hegselmann18a.pdf

Ablation: Evaluating Time-based Feature Significance in SEER Survival Prediction

Purpose:
This script performs a comparative ablation study to quantify the predictive 
contribution of the 'Diagnosis Year' (year_dx) feature within the SEER 
Cancer dataset. By comparing a baseline model against a reduced feature set, 
we evaluate whether chronological trends offer independent signal beyond 
standard clinical and pathological markers.

Methodology:
- Model Architecture: Multi-Layer Perceptron (MLP) implemented via the 
  'pyhealth.models.MLP' module.
- Training Framework: 'pyhealth.trainer.Trainer' with Adam optimization 
  ($LR=0.001$) and Binary Cross-Entropy loss.
- Dataset: SEER Cancer cohort (n=288,818) with a 5-year survival 
  binary classification task.
- Experimental Design: 
    1. Baseline: Model trained on the complete 55-feature set.
    2. Ablation: Model trained on 54 features, specifically excluding 
       the time 'year_dx' variable.

Findings:
The results indicate that the Diagnosis Year provides a statistically measurable 
but practically marginal contribution to model performance.

| Configuration      | Feature Count | AUROC  | Accuracy |
|--------------------|---------------|--------|----------|
| Baseline (Full)    | 55            | 0.8443 | 0.8933   |
| Ablation (Reduced) | 54            | 0.8428 | 0.8951   |
| **Difference** | **-1** | **-0.0015** | **+0.0018** |

Discussion:
The observed decrease of AUROC by 0.0015 suggests that clinical pathology (tumor 
stage, grade, and receptor status) remains the primary driver of survival 
outcomes. The marginal impact of the time-based feature implies high redundancy 
with existing clinical variables, likely due to consistent standard-of-care 
protocols across the observed timeframe.

Example:
    PYTHONPATH=.. python examples/seer_survival_prediction_lr.py --root "/path/to/data"
"""

import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from pyhealth.datasets import SEERDataset, split_by_patient
from pyhealth.tasks.seer_survival_prediction import SEERSurvivalPrediction
from pyhealth.trainer import Trainer
from pyhealth.models import MLP

class Processor:
    def __init__(self, size): self._size = size
    def size(self): return self._size
    def schema(self): return None

class SEERModelWrapper(nn.Module):
    def __init__(self, dataset, input_dim: int):
        super().__init__()
        # Placeholder metadata
        dataset.input_schema = {"features": None}
        dataset.output_schema = {"label": None}
        dataset.input_processors = {"features": Processor(input_dim)}
        dataset.output_processors = {"label": Processor(1)}
        dataset.label_processors = {"label": Processor(1)}
        self.mode = "binary"
        
        self.mlp_core = MLP(
            dataset=dataset,
            feature_keys=["features"],
            label_key="label",
            mode="binary",
            hidden_dim=64,
        )
        self.mlp_core.mlp['features'][0] = nn.Linear(input_dim, 64)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, features, label, **kwargs):
        device = next(self.parameters()).device
        features = features.to(device).float()
        label = label.to(device).float().view(-1)
        
        x = self.mlp_core.mlp['features'](features)
        logits = self.mlp_core.fc(x).view(-1)
        
        loss = self.loss_fn(logits, label)
        return {"loss": loss, "y_prob": torch.sigmoid(logits), "y_true": label}

def collate_fn(batch):
    return {
        "features": torch.stack(
            [torch.as_tensor(s["features"], dtype=torch.float32) for s in batch]
        ),
        "label": torch.as_tensor([s["label"] for s in batch], dtype=torch.float32)
    }

def run_experiment(dataset, split_data, input_dim, experiment_name):
    print(f"\n" + "-"*50)
    print(f"RUNNING: {experiment_name} (Input Dim: {input_dim})")
    print("-"*50)
    
    train_loader = DataLoader(
        list(split_data[0]), 
        batch_size=128, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        list(split_data[1]), 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        list(split_data[2]), 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    model = SEERModelWrapper(dataset=dataset, input_dim=input_dim)
    trainer = Trainer(model=model, metrics=["roc_auc", "accuracy"])
    trainer.train(
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        epochs=3, 
        monitor="roc_auc"
    )
    
    results = trainer.evaluate(test_loader)
    print(f"Results for {experiment_name}: {results}")
    return results["roc_auc"], results["accuracy"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    
    dataset = SEERDataset(root=args.root)
    task = SEERSurvivalPrediction()
    samples = dataset.set_task(task)
    
    print("Splitting dataset by patient...")
    baseline_splits = split_by_patient(samples, [0.8, 0.1, 0.1])
    full_dim = baseline_splits[0][0]["features"].shape[0]
    
    # Baseline
    auc_full, _ = run_experiment(dataset, baseline_splits, full_dim, "Baseline")
    
    # Ablation
    print("\nAblating features (removing Diagnosis Year)...")
    ab_splits = []
    for split in baseline_splits:
        new_split = []
        for s in split:
            new_s = copy.deepcopy(s)
            new_s["features"] = np.delete(np.array(s["features"]), -1, axis=0)
            new_split.append(new_s)
        ab_splits.append(new_split)
    
    ab_dim = ab_splits[0][0]["features"].shape[0]
    auc_ab, _ = run_experiment(dataset, ab_splits, ab_dim, "Ablation")
    
    print("\n" + "======================================")
    print("\t\t FINAL ABLATION RESULTS")
    print("======================================")
    print(f"Baseline AUROC: {auc_full:.4f}")
    print(f"Ablated AUROC:  {auc_ab:.4f}")
    print(f"AUROC Delta:    {auc_full - auc_ab:.4f}")
    print("======================================")

if __name__ == "__main__":
    main()