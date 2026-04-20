"""
examples/mimic4_readmission_keep.py

Ablation Study for KEEP on MIMIC-IV Readmission Prediction
===========================================================

This script evaluates the KEEP model and performs a structured
ablation study to analyze the impact of ontology regularization
and frequency-aware regularization on readmission prediction.

----------------------------------------------------------------------
RESEARCH QUESTION
----------------------------------------------------------------------

Does ontology-based regularization improve readmission prediction
performance, and does frequency-aware regularization further improve
robustness compared to uniform regularization?

----------------------------------------------------------------------
EXPERIMENTAL VARIABLES
----------------------------------------------------------------------

We systematically vary two factors:

1) Regularization Strength (lambda_base)
   - 0.0  → No ontology regularization
   - 0.1  → Standard KEEP regularization

2) Frequency-Aware Regularization
   - False → Uniform λ for all codes
   - True  → λ_i = lambda_base / sqrt(freq_i + 1)

Additionally, we vary embedding dimensionality:
   - 64
   - 128

----------------------------------------------------------------------
DATASET
----------------------------------------------------------------------

This script uses the official MIMIC-IV demo dataset
(mimic-iv-clinical-database-demo-2.2).

The demo dataset is sufficient for:
    - Verifying model integration
    - Running structured ablation
    - Demonstrating performance comparison

----------------------------------------------------------------------
EVALUATION METRICS
----------------------------------------------------------------------

We report:
    - AUROC
    - AUPRC
    - F1 Score
    - Accuracy

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------

Run with MIMIC-IV demo data:

    python examples/mimic4_readmission_keep.py \
        --mimic_root /path/to/mimic-iv-demo
"""

import os
import random
import argparse
import numpy as np
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.tasks import ReadmissionPredictionMIMIC4
from pyhealth.trainer import Trainer
from pyhealth.models import KEEP


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Single Experiment Runner
# ---------------------------------------------------------------------
def run_experiment(
    train_dataset,
    val_dataset,
    test_dataset,
    lambda_base: float,
    use_frequency_regularization: bool,
    embedding_dim: int,
):
    """
    Train and evaluate KEEP under a specific configuration.

    Each experiment includes:
        1) Unsupervised embedding pretraining
        2) Supervised readmission training
        3) Evaluation on held-out test set
    """

    print("=" * 80)
    print(
        f"Config | lambda_base={lambda_base} | "
        f"use_freq_reg={use_frequency_regularization} | "
        f"embedding_dim={embedding_dim}"
    )

    model = KEEP(
        dataset=train_dataset,
        embedding_dim=embedding_dim,
        lambda_base=lambda_base,
        use_frequency_regularization=use_frequency_regularization,
    )

    # ----------------------------------------------------------
    # Stage 1: Co-occurrence Pretraining
    # ----------------------------------------------------------
    samples = [train_dataset[i] for i in range(len(train_dataset))]
    model.pretrain_embeddings(
        samples=samples,
        epochs=1,      # Reduced for demo speed
        lr=1e-3,
    )

    # ----------------------------------------------------------
    # Stage 2: Supervised Fine-tuning
    # ----------------------------------------------------------
    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "f1", "accuracy"],
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    test_loader = get_dataloader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=2,
    )

    results = trainer.evaluate(test_loader)

    return {
        "auroc": results.get("roc_auc", float("nan")),
        "auprc": results.get("pr_auc", float("nan")),
        "f1": results.get("f1", float("nan")),
        "accuracy": results.get("accuracy", float("nan")),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:

    set_seed(42)

    # ----------------------------------------------------------
    # Argument Parsing
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mimic_root",
        type=str,
        required=True,
        help="Path to MIMIC-IV demo dataset.",
    )
    args = parser.parse_args()

    print(f"Using MIMIC-IV demo dataset at: {args.mimic_root}")

    # ----------------------------------------------------------
    # Dataset Loading
    # ----------------------------------------------------------
    dataset = MIMIC4Dataset(
        ehr_root=args.mimic_root,
        ehr_tables=[
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
        dev=False,
    )

    print("Setting up readmission prediction task...")
    task_dataset = dataset.set_task(
        ReadmissionPredictionMIMIC4()
    )

    # ----------------------------------------------------------
    # Train / Validation / Test Split
    # ----------------------------------------------------------
    print("Splitting dataset by patient...")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        task_dataset,
        ratios=[0.7, 0.1, 0.2],
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    # ----------------------------------------------------------
    # Ablation Configurations
    # ----------------------------------------------------------
    configs = [
        (0.0, False),   # No regularization
        (0.1, False),   # Standard KEEP
        (0.1, True),    # Frequency-aware KEEP
    ]

    embedding_dims = [64, 128]
    all_results = []

    # ----------------------------------------------------------
    # Run Experiments
    # ----------------------------------------------------------
    for embedding_dim in embedding_dims:
        for lambda_base, use_freq_reg in configs:

            metrics = run_experiment(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                lambda_base=lambda_base,
                use_frequency_regularization=use_freq_reg,
                embedding_dim=embedding_dim,
            )

            all_results.append(
                {
                    "embedding_dim": embedding_dim,
                    "lambda_base": lambda_base,
                    "use_freq_reg": use_freq_reg,
                    **metrics,
                }
            )

    # ----------------------------------------------------------
    # Print Results Table
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL ABLATION RESULTS")
    print("Comparison across regularization strategies and embedding sizes")
    print("=" * 80)

    header = (
        f"{'emb_dim':<8} | {'lambda':<8} | {'freq_reg':<8} | "
        f"{'AUROC':<8} | {'AUPRC':<8} | {'F1':<8} | {'Accuracy':<8}"
    )
    print(header)
    print("-" * len(header))

    for result in all_results:
        print(
            f"{result['embedding_dim']:<8} | "
            f"{result['lambda_base']:<8} | "
            f"{str(result['use_freq_reg']):<8} | "
            f"{result['auroc']:<8.4f} | "
            f"{result['auprc']:<8.4f} | "
            f"{result['f1']:<8.4f} | "
            f"{result['accuracy']:<8.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()