"""Full pipeline replication — TaskAug for ECG Classification (Raghu et al., 2022).

Replicates: "Data Augmentation for Electrocardiograms", CHIL 2022.
            https://proceedings.mlr.press/v174/raghu22a.html

Pipeline
--------
1. Data loading   — PTBXLDataset reads PTB-XL waveforms and labels.
2. Task processing — ECGBinaryClassification normalises and windows each record.
3. Model training  — TaskAugResNet with bi-level optimisation (DARTS first-order).
4. Evaluation      — AUROC + accuracy on held-out validation set.
5. Ablation study  — four configurations: no aug / fixed noise / TaskAug K=1 / K=2.

Quick start (no data download required)
----------------------------------------
    python my_replication.py --synthetic

Full run on real PTB-XL data (download from PhysioNet first)
-------------------------------------------------------------
    python my_replication.py --data_root /path/to/ptb-xl/ --task MI --mode bilevel --epochs 20

Modes
-----
    ablation  (default) — run all four ablation configurations and print summary table.
    standard            — joint Adam optimisation of backbone + policy.
    bilevel             — DARTS-style bi-level optimisation (matches paper setup).

Novel contributions in this PR
-------------------------------
    pyhealth/datasets/ptbxl.py           — PTBXLDataset
    pyhealth/tasks/ecg_classification.py — ECGBinaryClassification
    pyhealth/models/taskaug_resnet.py    — TaskAugPolicy + TaskAugResNet
"""

# Re-export the full pipeline from the canonical example module so this file
# acts as the required entry-point without duplicating code.
from ptbxl_ecg_classification_taskaug_resnet import (  # noqa: F401
    make_synthetic_dataset,
    make_ptbxl_dataset,
    compute_auroc,
    evaluate,
    train_standard,
    BiLevelTrainer,
    run_ablation,
    main,
)

if __name__ == "__main__":
    main()
