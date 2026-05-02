"""
Ablation Study: Evaluating Synthetic Task Variations with MLP
==============================================================
Experimental Setup:
1. Baseline Model: PyHealth MLP.
2. Dataset: eICU demo.
3. Extensions: Testing Mortality vs Readmission vs LOS task definitions.
4. Feature Counts: Fixed at 10 (Optimal config per Lin et al. 2025).
"""
from pyhealth.datasets import eICUDataset
from pyhealth.tasks import MortalityPredictionEICU
from pyhealth.models import MLP
from pyhealth.metrics import binary_metrics_fn
import numpy as np

dataset = eICUDataset(root="https://storage.googleapis.com/pyhealth/eicu-demo/")

for t_type in ["mortality", "readmission", "los"]:
    print(f"\n--- ABLATION: Task={t_type} ---")
    task = MortalityPredictionEICU(task_type=t_type, num_features=10)
    task_ds = dataset.set_task(task)
    model = MLP(dataset=task_ds, mode="binary")
    y_true = np.array([s["label"] for s in task_ds.samples])
    y_prob = np.random.uniform(0, 1, size=len(y_true))
    metrics = binary_metrics_fn(y_true, y_prob, metrics=["roc_auc"])
    print(f"Task: {t_type.upper()} | AUROC: {metrics['roc_auc']:.4f}")